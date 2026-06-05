"""
Utilities for building solid ellipsoid meshes from hexahedral elements.
"""
from cmlibs.maths.vectorops import add, cross, div, dot, magnitude, mult, normalize, rejection, set_magnitude, sub
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node

from scaffoldmaker.utils.eft_utils import determineCubicHermiteSerendipityEft, HermiteNodeLayoutManager
from scaffoldmaker.utils.geometry import (
    getEllipsePointAtTrueAngle, getEllipseTangentAtPoint, moveCoordinatesToEllipsoidSurface,
    moveDerivativeToEllipsoidSurface, moveDerivativeToEllipsoidSurfaceInPlane, sampleCurveOnEllipsoid)
from scaffoldmaker.utils.interpolation import DerivativeScalingMode, linearlyInterpolateVectors, sampleHermiteCurve
from scaffoldmaker.utils.hextetrahedronmesh import HexTetrahedronMesh
from scaffoldmaker.utils.quadtrianglemesh import QuadTriangleMesh
import copy
from enum import Enum
import math


class EllipsoidSurfaceD3Mode(Enum):
    SURFACE_NORMAL = 1  # surface D3 are exact surface normals to ellipsoid
    OBLIQUE_DIRECTION = 2  # surface D3 are in direction of surface point on ellipsoid, gives flat oblique planes
    # Surface D3 are surface normals to ellipsoid projected onto radial planes transitioning between axis2 and axis3
    SURFACE_NORMAL_PLANE_PROJECTION = 3


class EllipsoidMesh:
    """
    Generates a solid ellipsoid of hexahedral elements with oblique cross axes suited to describing lung geometry.
    """

    def __init__(self, a, b, c, element_counts, transition_element_count, surface_only=False):
        """
        :param a: Axis length (radius) in x direction.
        :param b: Axis length (radius) in y direction.
        :param c: Axis length (radius) in z direction.
        :param element_counts: Number of elements across full ellipse in a, b, c.
        :param transition_element_count: Number of transition elements around outside >= 1.
        :param surface_only: Set to True to only make nodes and 2-D elements on the surface.
        """
        assert all((count >= 4) and (count % 2 == 0) for count in element_counts)
        assert 1 <= transition_element_count <= (min(element_counts) // 2 - 1)
        self._a = a
        self._b = b
        self._c = c
        self._element_counts = element_counts
        self._trans_count = transition_element_count
        self._surface_only = surface_only
        self._nway_d_factor = 0.6
        self._surface_d3_mode = EllipsoidSurfaceD3Mode.SURFACE_NORMAL
        self._box_group = None
        self._transition_group = None
        self._octant_group_lists = None
        none_parameters = [None] * 4  # x, d1, d2, d3
        self._nx = []  # shield mesh with holes over n3, n2, n1, d
        self._nids = []
        half_counts = [count // 2 for count in self._element_counts]
        for n3 in range(self._element_counts[2] + 1):
            # index into transition zone
            trans3 = (self._trans_count - n3) if (n3 < half_counts[2]) else \
                (self._trans_count + n3 - self._element_counts[2])
            nx_layer = []
            nids_layer = []
            # print(n3, trans3)
            for n2 in range(self._element_counts[1] + 1):
                # index into transition zone
                trans2 = (self._trans_count - n2) if (n2 < half_counts[1]) else \
                    (self._trans_count + n2 - self._element_counts[1])
                nx_row = []
                nids_row = []
                # s = ""
                for n1 in range(self._element_counts[0] + 1):
                    # index into transition zone
                    trans1 = (self._trans_count - n1) if (n1 < half_counts[0]) else \
                        (self._trans_count + n1 - self._element_counts[0])
                    if (((trans1 <= 0) and (trans2 <= 0) and (trans3 <= 0)) or
                            (trans1 == trans2 == trans3) or
                            ((trans1 < 0) and ((trans2 == trans3) or (trans2 < 0))) or
                            ((trans2 < 0) and ((trans3 == trans1) or (trans3 < 0))) or
                            ((trans3 < 0) and ((trans1 == trans2) or (trans1 < 0)))):
                        parameters = copy.copy(none_parameters)
                        # s += "[]"
                    else:
                        parameters = None
                        # s += "  "
                    nx_row.append(parameters)
                    nids_row.append(None)
                nx_layer.append(nx_row)
                nids_layer.append(nids_row)
                # print(s)
            self._nx.append(nx_layer)
            self._nids.append(nids_layer)
        self._node_layout_manager = HermiteNodeLayoutManager()
        self._prescribed_node_layouts = []  # list of (n1, n2, n3, node_layout)

    def set_box_transition_groups(self, box_group, transition_group):
        """
        Set zinc groups to fill with elements in box and transition regions, if not surface_only.
        :param box_group: Group field to add elements from box region to.
        :param transition_group: Group field to add elements from transition region to.
        """
        self._box_group = box_group
        self._transition_group = transition_group

    def set_octant_group_lists(self, octant_group_lists):
        """
        Set lists of zinc groups to add elements in each of the 8 octants to.
        :param octant_group_lists: List of 8 lists of group fields to put elements into. For example, the
        elements of octant 0 (negative 3 axis, negative 2 axis, negative 1 axis) will be put in the mesh group of
        the appropriate dimension for each group in octant_groups_lists[0]. Order of octants for N (negative) or
        P (positive) 321 axes: NNN, NNP, NPN, NPP, PNN, PNP, PPN, PPP.
        """
        assert (octant_group_lists is None) or (len(octant_group_lists) == 8)
        self._octant_group_lists = octant_group_lists

    def set_nway_derivative_factor(self, nway_derivative_factor):
        """
        Set factor controlling the shape of 3-way and 4-way points in ellipsoid mesh.
        :param nway_d_factor: Value, normally from 0.5 to 1.0 giving n-way derivative magnitude as a proportion
        of the minimum regular magnitude sampled to the n-way point. This reflects that distances from the mid-side
        of a triangle to the centre are shorter, so the derivative in the middle must be smaller.
        """
        self._nway_d_factor = nway_derivative_factor

    def set_surface_d3_mode(self, surface_d3_mode: EllipsoidSurfaceD3Mode):
        """
        Set mode controlling how surface d3 values are calculated.
        :param surface_d3_mode: Value from EllipsoidSurfaceD3Mode.
        """
        self._surface_d3_mode = surface_d3_mode

    def build(self, axis2_x_rotation_radians, axis3_x_rotation_radians):
        """
        Determine coordinates and derivatives over and within the full ellipsoid.
        :param axis2_x_rotation_radians: Rotation of axis 2 about +x direction
        :param axis3_x_rotation_radians: Rotation of axis 3 about +x direction.
        """
        half_counts = [count // 2 for count in self._element_counts]

        octant1 = self.build_octant(half_counts, axis2_x_rotation_radians, axis3_x_rotation_radians)
        self.merge_octant(octant1, quadrant=0)
        octant1.mirror_yz()
        self.merge_octant(octant1, quadrant=2)

        octant2 = self.build_octant([half_counts[0], half_counts[2], half_counts[1]],
                                    axis3_x_rotation_radians, axis2_x_rotation_radians + math.pi)
        self.merge_octant(octant2, quadrant=1)
        octant2.mirror_yz()
        self.merge_octant(octant2, quadrant=3)

        self.copy_to_negative_axis1()

    def build_octant(self, half_counts, axis2_x_rotation_radians, axis3_x_rotation_radians,
                     axis2_extension=0.0, axis2_extension_elements_count=0):
        """
        Get coordinates of top, right, front octant with supplied angles.
        :param half_counts: Numbers of elements across octant 1, 2 and 3 directions.
        :param axis2_x_rotation_radians: Rotation of axis 2 about +x direction
        :param axis3_x_rotation_radians: Rotation of axis 3 about +x direction.
        :param axis2_extension: Extension distance along axis2 beyond origin [0.0, 0.0, 0.0].
        :param axis2_extension_elements_count: If axis2_extension: number of elements beyond origin.
        Note: included in half_counts[1].
        :return: HexTetrahedronMesh
        """
        assert ((axis2_extension == 0.0) and (axis2_extension_elements_count == 0)) or (
                (axis2_extension > 0.0) and (0 < axis2_extension_elements_count))
        box_counts = [half_counts[i] - self._trans_count for i in range(3)]

        cos_axis2 = math.cos(axis2_x_rotation_radians)
        sin_axis2 = math.sin(axis2_x_rotation_radians)
        cos_axis3 = math.cos(axis3_x_rotation_radians)
        sin_axis3 = math.sin(axis3_x_rotation_radians)

        origin = [0.0, 0.0, 0.0]
        ext_origin = [0.0, -axis2_extension * cos_axis2, -axis2_extension * sin_axis2]
        ext_axis1 = axis1 = [self._a, 0.0, 0.0]
        axis2 = [0.0] + getEllipsePointAtTrueAngle(self._b, self._c, axis2_x_rotation_radians)
        axis2_mag = magnitude(axis2)
        axis2_normal = normalize([0.0, axis2[2], -axis2[1]])
        ext_axis3 = axis3 = [0.0] + getEllipsePointAtTrueAngle(self._b, self._c, axis3_x_rotation_radians)
        axis3_normal = normalize([0.0, axis3[2], -axis3[1]])
        if axis2_extension_elements_count:
            assert axis2_extension < axis2_mag  # extension must not go outside ellipsoid
            xb, xa = getEllipsePointAtTrueAngle(axis2_mag, self._a, math.pi / 2.0, [-axis2_extension, 0.0])
            ext_axis1 = [xa, xb * cos_axis2, xb * sin_axis2]
            ext_axis3 = [0.0] + getEllipsePointAtTrueAngle(self._b, self._c, axis3_x_rotation_radians, ext_axis1[1:])
            ext_axis3m = [0.0] + getEllipsePointAtTrueAngle(self._b, self._c, axis3_x_rotation_radians + math.pi, ext_axis1[1:])
            centre_mod_axis3 = mult(add(ext_axis3, ext_axis3m), 0.5)
            mod_axis3 = sub(ext_axis3, centre_mod_axis3)
            mag_mod_axis3 = magnitude(mod_axis3)
        else:
            centre_mod_axis3 = origin
            mag_mod_axis3 = magnitude(axis3)

        axis_d1 = div(axis1, half_counts[0])
        ext_axis_d1 = div(sub(ext_axis1, ext_origin), half_counts[0])
        axis_d2 = div(axis2, half_counts[1])
        axis_d3 = div(axis3, half_counts[2])
        ext_axis_d3 = div(sub(ext_axis3, ext_origin), half_counts[2])
        axis_md1 = [-d for d in axis_d1]
        axis_md2 = [-d for d in axis_d2]
        axis_md3 = [-d for d in axis_d3]
        ext_axis_md1 = [-d for d in ext_axis_d1]

        # most derivatives indicate only direction, so magnitude not known
        dir_mag = min(magnitude(axis_d1), magnitude(axis_d2), magnitude(axis_d3))
        axis2_dt = set_magnitude([0.0] + getEllipseTangentAtPoint(self._b, self._c, axis2[1:]), magnitude(axis_d3))
        ext_axis3_dt = set_magnitude([0.0] + getEllipseTangentAtPoint(self._b, self._c, ext_axis3[1:]), magnitude(ext_axis_d3))
        ext_axis3_mdt = [-d for d in ext_axis3_dt]
        axis2_mag = magnitude(axis2)
        axis3_mag = magnitude(axis3)

        sample_curve_on_ellipsoid = (
            lambda start_x, start_d1, start_d2, end_x, end_d1, end_d2, elements_count,
                   start_weight=None, end_weight=None, overweighting=1.0, end_transition=False:
            sampleCurveOnEllipsoid(
                self._a, self._b, self._c, start_x, start_d1, start_d2, end_x, end_d1, end_d2, elements_count,
                start_weight, end_weight, overweighting, end_transition))
        move_x_to_ellipsoid_surface = lambda x: moveCoordinatesToEllipsoidSurface(self._a, self._b, self._c, x)
        move_d_to_ellipsoid_surface = lambda x, d: moveDerivativeToEllipsoidSurface(self._a, self._b, self._c, x, d)
        def evaluate_surface_d3_ellipsoid_plane(tx, td1, td2):
            """
            Restrict d3 to be the ellipsoid normal constrained to be in radial planes from ext_origin through tx,
            varying between axis_d2 and axis_d3.
            :param tx: Coordinates of a point on the ellipsoid surface in the octant.
            :param td1: Unused point d1.
            :param td2: Unused point d2.
            :return: Radial plane constrained ellipsoid normal d3 with magnitude dir_mag.
            """
            n = [tx[0] / (self._a * self._a), tx[1] / (self._b * self._b), tx[2] / (self._c * self._c)]
            if dot(tx, axis3_normal) <= 1.0E-5:
                if dot(tx, axis2_normal) >= -1.0E-5:
                    return set_magnitude(axis1, dir_mag)
                else:
                    plane_normal = [0.0, axis3[2], -axis3[1]]
            else:
                plane_normal = [0.0, tx[2], -tx[1]]
            normal = rejection(n, plane_normal)
            return set_magnitude(normal, dir_mag)
        if self._surface_d3_mode == EllipsoidSurfaceD3Mode.SURFACE_NORMAL:
            evaluate_surface_d3_ellipsoid = lambda tx, td1, td2: set_magnitude(
                [tx[0] / (self._a * self._a), tx[1] / (self._b * self._b), tx[2] / (self._c * self._c)], dir_mag)
        elif self._surface_d3_mode == EllipsoidSurfaceD3Mode.OBLIQUE_DIRECTION:
            evaluate_surface_d3_ellipsoid=lambda tx, td1, td2: set_magnitude(tx, dir_mag)
        else:  # EllipsoidSurfaceD3Mode.SURFACE_NORMAL_PLANE_PROJECTION
            evaluate_surface_d3_ellipsoid = evaluate_surface_d3_ellipsoid_plane

        ext_half_counts = [
            half_counts[0],
            half_counts[1] + axis2_extension_elements_count,
            half_counts[2]
        ]
        diag_counts = [
            half_counts[0] + half_counts[1] - 2 * self._trans_count,
            half_counts[0] + half_counts[2] - 2 * self._trans_count,
            half_counts[1] + half_counts[2] - 2 * self._trans_count
        ]
        ext_diag_counts = [
            ext_half_counts[0] + ext_half_counts[1] - 2 * self._trans_count,
            ext_half_counts[0] + ext_half_counts[2] - 2 * self._trans_count,
            ext_half_counts[1] + ext_half_counts[2] - 2 * self._trans_count
        ]
        octant = HexTetrahedronMesh(ext_half_counts, ext_diag_counts, nway_d_factor=self._nway_d_factor)

        # get outside curve from axis 1 to axis 2
        abx, abd1, abd2 = sampleCurveOnEllipsoid(
            self._a, self._b, self._c,
            axis1, axis_d2, axis_d3,
            axis2, axis_md1, axis2_dt,
            diag_counts[0])
        if axis2_extension_elements_count:
            end_axis_d3 = moveDerivativeToEllipsoidSurfaceInPlane(
                self._a, self._b, self._c, ext_axis1, [axis_d3[0], -axis_d3[2], axis_d3[1]], axis_d3)
            ext_abx, ext_abd1, ext_abd2 = sampleCurveOnEllipsoid(
                self._a, self._b, self._c,
                axis1, [-d for d in axis_d2], axis_d3,
                ext_axis1, None, end_axis_d3,  # axis_d3
                axis2_extension_elements_count)
            for i in range(1, axis2_extension_elements_count + 1):
                abx.insert(0, ext_abx[i])
                abd1.insert(0, [-d for d in ext_abd1[i]])
                abd2.insert(0, ext_abd2[i])
        # get outside curve from axis 1 to axis 3
        acx, acd2, acd1 = sampleCurveOnEllipsoid(
            self._a, self._b, self._c,
            abx[0], abd2[0], abd1[0],
            ext_axis3, ext_axis_md1, ext_axis3_mdt,
            ext_diag_counts[1])
        # get outside curve from axis 2 to axis 3
        bcx, bcd2, bcd1 = sampleCurveOnEllipsoid(
            self._a, self._b, self._c,
            abx[-1], abd2[-1], abd1[-1],
            acx[-1], [-d for d in acd1[-1]], acd2[-1],
            ext_diag_counts[2])
        # fix first/last derivatives
        abd2[0] = acd2[0]
        abd2[-1] = bcd2[0]
        acd1[-1] = [-d for d in bcd2[-1]]

        # make outer surface triangle of octant 1
        triangle_abc = QuadTriangleMesh(
            box_counts[0], box_counts[1] + axis2_extension_elements_count, box_counts[2],
            sample_curve_on_ellipsoid, move_x_to_ellipsoid_surface, move_d_to_ellipsoid_surface, self._nway_d_factor)
        triangle_abc.set_edge_parameters12(abx, abd1, abd2)
        triangle_abc.set_edge_parameters13(acx, acd1, acd2)
        triangle_abc.set_edge_parameters23(bcx, bcd1, bcd2)
        triangle_abc.build()
        if not self._surface_only:
            triangle_abc.assign_d3(evaluate_surface_d3_ellipsoid)
        octant.set_triangle_abc(triangle_abc)

        if not self._surface_only:
            # extract exact derivatives
            abd2 = triangle_abc.get_edge_parameters12()[2]
            acd1 = triangle_abc.get_edge_parameters13()[1]
            bcd1 = triangle_abc.get_edge_parameters23()[1]

            # build interior lines from axis1, axis2, axis3 to origin
            aox, aod2, aod1 = sampleHermiteCurve(
                ext_axis1, ext_axis_md1, abd1[0], ext_origin, ext_axis_md1, axis_d2, elements_count=half_counts[0])
            box, bod2, bod3 = sampleHermiteCurve(
                axis2, axis_md2, bcd2[0], origin, axis_md2, axis_d3, elements_count=half_counts[1])
            if axis2_extension_elements_count:
                ext_box, ext_bod2, ext_bod3 = sampleHermiteCurve(
                    box[-1], bod2[-1], bod3[-1], ext_origin, None, ext_axis_d3,
                    elements_count=axis2_extension_elements_count)
                for i in range(1, axis2_extension_elements_count + 1):
                    box.append(ext_box[i])
                    bod2.append(ext_bod2[i])
                    bod3.append(ext_bod3[i])
            bod1 = [abd1[-1]] + [axis_md1] * (len(box) - 1)
            cox, cod2, cod1 = sampleHermiteCurve(
                ext_axis3, axis_md3, [-d for d in acd1[-1]], ext_origin, axis_md3, bod2[-1],
                elements_count=half_counts[2])

            # make inner surface triangle 1-2-origin
            triangle_abo = QuadTriangleMesh(
                box_counts[0], box_counts[1] + axis2_extension_elements_count, self._trans_count, sampleHermiteCurve,
                nway_d_factor=self._nway_d_factor)
            abd3 = [[-d for d in evaluate_surface_d3_ellipsoid(x, None, None)] for x in abx]
            triangle_abo.set_edge_parameters12(abx, abd1, abd3, abd2)
            count = len(aox) - 1
            aod3 = [linearlyInterpolateVectors(abd2[0], axis_d3, i / count) for i in range(count + 1)]

            triangle_abo.set_edge_parameters13(aox, aod1, aod2, aod3)
            triangle_abo.set_edge_parameters23(box, bod1, bod2, bod3)
            triangle_abo.build(regular_count2=axis2_extension_elements_count)
            aa = self._a * self._a
            bb = axis2_mag * axis2_mag  # of axis2 ellipse
            def evaluate_surface_d3_abo(tx, td1, td2):
                y = tx[1] * cos_axis2 + tx[2] * sin_axis2
                yy = y * y
                if yy >= bb:
                    return axis2_dt  # tip of ellipse
                centre_d3 = (linearlyInterpolateVectors(axis_d3, axis2_dt, magnitude(tx[1:]) / axis2_mag)
                             if (y >= 0.0) else axis_d3)
                xx = aa * (1.0 - yy / bb)
                x = math.sqrt(xx)
                side_x = [x, tx[1], tx[2]]
                side_d3 = moveDerivativeToEllipsoidSurface(self._a, self._b, self._c, side_x, centre_d3)
                return linearlyInterpolateVectors(centre_d3, side_d3, tx[0] / x)
            triangle_abo.assign_d3(evaluate_surface_d3_abo)
            octant.set_triangle_abo(triangle_abo)
            # extract exact derivatives
            aod1 = triangle_abo.get_edge_parameters13()[1]
            bod1 = triangle_abo.get_edge_parameters23()[1]

            # make inner surface triangle 1-3-origin
            triangle_aco = QuadTriangleMesh(
                box_counts[0], box_counts[2], self._trans_count, sampleHermiteCurve,
                nway_d_factor=self._nway_d_factor)
            acd3 = [[-d for d in evaluate_surface_d3_ellipsoid(x, None, None)] for x in acx]
            acmd1 = [[-d for d in d1] for d1 in acd1]
            triangle_aco.set_edge_parameters12(acx, acd2, acd3, acmd1)
            aomd1 = [[-d for d in d1] for d1 in aod1]
            triangle_aco.set_edge_parameters13(aox, aod3, aod2, aomd1)
            cod3 = [acd2[-1]] + [axis_md1] * (len(cox) - 1)
            triangle_aco.set_edge_parameters23(cox, cod3, cod2, cod1)
            triangle_aco.build()
            aa = self._a * self._a
            bb = mag_mod_axis3 * mag_mod_axis3  # mod_axis3 ellipse
            def evaluate_surface_d3_aco(tx, td1, td2):
                mx = sub(tx, centre_mod_axis3)
                y = mx[1] * cos_axis3 + mx[2] * sin_axis3
                yy = y * y
                if yy >= bb:
                    return ext_axis3_dt  # tip of ellipse
                centre_d3 = (linearlyInterpolateVectors(axis_md2, ext_axis3_dt, magnitude(mx[1:]) / axis3_mag)
                             if (y >= 0.0) else axis_md2)
                xx = aa * (1.0 - yy / bb)
                x = math.sqrt(xx)
                side_x = [x, tx[1], tx[2]]
                side_d3 = moveDerivativeToEllipsoidSurface(self._a, self._b, self._c, side_x, centre_d3)
                return linearlyInterpolateVectors(centre_d3, side_d3, tx[0] / x)
            triangle_aco.assign_d3(evaluate_surface_d3_aco)
            octant.set_triangle_aco(triangle_aco)
            # extract exact derivatives
            cod3 = triangle_aco.get_edge_parameters23()[1]

            # make inner surface 2-3-origin
            triangle_bco = QuadTriangleMesh(
                box_counts[1] + axis2_extension_elements_count, box_counts[2], self._trans_count, sampleHermiteCurve,
                nway_d_factor=self._nway_d_factor)
            bcd3 = [bod2[0]] + [[-d for d in evaluate_surface_d3_ellipsoid(x, None, None)] for x in bcx[1:-1]] \
                   + [cod2[-1]]
            bcmd1 = [[-d for d in d1] for d1 in bcd1]
            triangle_bco.set_edge_parameters12(bcx, bcd2, bcd3, bcmd1)
            bomd1 = [[-d for d in d1] for d1 in bod1]
            triangle_bco.set_edge_parameters13(box, bod3, bod2, bomd1)
            comd3 = [[-d for d in d3] for d3 in cod3]
            triangle_bco.set_edge_parameters23(cox, cod1, cod2, comd3)
            triangle_bco.build()
            triangle_bco.assign_d3(lambda tx, td1, td2: axis_d1)
            octant.set_triangle_bco(triangle_bco)

            octant.build_interior()

        return octant

    def merge_octant(self, octant: HexTetrahedronMesh, quadrant: int):
        """
        Merge octant parameters into ellipsoid in one of the 4 quadrants on +axis1.
        Octant can be extended into its axis2 over regular elements of ellipsoid.
        :param octant: HexTetrahedronMesh
        :param quadrant: 0 for +axis2, +axis3 increasing anticlockwise around +axis1 up to 3.
        """
        assert 0 <= quadrant <= 3
        half_counts = [count // 2 for count in self._element_counts]
        axis_counts = octant.get_axis_counts()
        even_quadrant = quadrant in (0, 2)
        assert half_counts[0] == axis_counts[0]
        ext_count2 = (axis_counts[1] - half_counts[1]) if even_quadrant else 0
        ext_count3 = 0 if even_quadrant else (axis_counts[1] - half_counts[2])
        assert 0 <= ext_count2 < (half_counts[1] - self._trans_count)
        assert 0 <= ext_count3 < (half_counts[2] - self._trans_count)
        obox_counts = octant.get_box_counts()
        # box_counts = [half_counts[i] - self._trans_count for i in range(3)]
        octant_parameters = octant.get_parameters()

        for o3 in range(axis_counts[2] + 1):
            for o2 in range(axis_counts[1] + 1):
                ox_row = octant_parameters[o3][o2]
                if even_quadrant:
                    if quadrant == 0:
                        n3 = half_counts[2] + o3 - ext_count3
                        n2 = half_counts[1] + o2 - ext_count2
                    else:  # quadrant == 2:
                        n3 = half_counts[2] - o3 + ext_count3
                        n2 = half_counts[1] - o2 + ext_count2
                else:
                    if quadrant == 1:
                        n3 = half_counts[2] + o2 - ext_count3
                        n2 = half_counts[1] - o3 + ext_count2
                    else:  # if quadrant == 3:
                        n3 = half_counts[2] - o2 + ext_count3
                        n2 = half_counts[1] + o3 - ext_count2
                transition3 = (n3 < self._trans_count) or (n3 > (self._element_counts[2] - self._trans_count))
                transition2 = (n2 < self._trans_count) or (n2 > (self._element_counts[1] - self._trans_count))
                # bottom_transition = n3 < self._trans_count
                nx_row = self._nx[n3][n2]
                obox_row = (o3 <= obox_counts[2]) and (o2 <= obox_counts[1])
                for o1 in range(axis_counts[0] + 1):
                    n1 = half_counts[0] + o1
                    transition1 = n1 > (self._element_counts[0] - self._trans_count)
                    obox = obox_row and (o1 <= obox_counts[0])
                    ox = ox_row[o1]
                    if ox and ox[0]:
                        x = copy.copy(ox[0])
                        perm = None
                        if even_quadrant:
                            if quadrant == 0:
                                perm = [1, 2, 3]
                            else:  # quadrant == 2:
                                perm = [1, -2, -3] if obox else [-1, -2, 3]
                        else:
                            if obox:
                                perm = [1, -3, 2]
                            elif transition3:
                                if transition2:
                                    if transition1:
                                        # fix 3-way point
                                        ox = [ox[0], ox[1], add(ox[1], ox[2]), ox[3]]
                                    perm = [1, 2, 3]
                                else:
                                    perm = [-1, -2, 3]
                            else:
                                if transition1 and not transition2:
                                    perm = [-2, 1, 3]
                                else:
                                    perm = [1, 2, 3]
                            if quadrant == 3:
                                perm = [perm[0], -perm[1], -perm[2]] if obox else [-perm[0], -perm[1], perm[2]]
                        d1, d2, d3 = [copy.copy(ox[i]) if (i > 0) else [-d for d in ox[-i]] for i in perm]
                        new_nx = [x, d1, d2, d3]
                        # merge:
                        nx = nx_row[n1]
                        for i in range(4):
                            d = new_nx[i]
                            if i and nx[i]:
                                # blend derivatives with harmonic mean magnitude; should already be in same direction
                                d = linearlyInterpolateVectors(
                                    nx[i], d, 0.5, magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
                            nx[i] = d

    def copy_to_negative_axis1(self):
        """
        Copy parameters from +axis1 to -axis1.
        """
        half_counts0 = self._element_counts[0] // 2
        for n3 in range(self._element_counts[2] + 1):
            nx_layer = self._nx[n3]
            for n2 in range(self._element_counts[1] + 1):
                nx_row = nx_layer[n2]
                for n1 in range(half_counts0 + 1, self._element_counts[0] + 1):
                    nx = self._nx[n3][n2][n1]
                    if nx and nx[0]:
                        x, d1, d2, d3 = nx
                        x = [-x[0], x[1], x[2]]
                        d1 = [d1[0], -d1[1], -d1[2]] if d1 else None
                        d2 = [-d2[0], d2[1], d2[2]] if d2 else None
                        d3 = [-d3[0], d3[1], d3[2]] if d3 else None
                        nx_row[self._element_counts[0] - n1] = [x, d1, d2, d3]

    def _next_increment_out_of_bounds(self, indexes, index_increment):
        """
        :param indexes: List of 3 current indexes in the EllipsoidMesh nodes block: [n1, n2, n3].
        :param index_increment: Increments to the 3 indexes.
        :return: True if adding index_increment to indexes puts the index outside the nodes block.
        """
        for c in range(3):
            index = indexes[c] + index_increment[c]
            if (index < 0) or (index > self._element_counts[c]):
                return True
        return False

    def _get_nid_to_node_layout_map_3d(self, node_layout_manager):
        """
        Get map from node identifier to special node layout.
        :param node_layout_manager: Manager of node layouts for getting standard layouts from.
        :return: map from node identifier to special node layout. No entry made if default=None layout.
        """
        node_layout_permuted = node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=True)
        node_layout_3way12 = node_layout_manager.getNodeLayout3WayPoints12()
        node_layout_3way13 = node_layout_manager.getNodeLayout3WayPoints13()
        node_layout_3way23 = node_layout_manager.getNodeLayout3WayPoints23()
        node_layout_4way = node_layout_manager.getNodeLayout4WayPoints()
        nid_to_node_layout = {}
        upper_trans_counts = [self._element_counts[i] - self._trans_count for i in range(3)]
        # bottom and top transition side face nodes are fully permuted
        for nt in range(1, self._trans_count + 1):
            # bottom transition
            for n3 in (self._trans_count - nt, upper_trans_counts[2] + nt):
                for n2 in range(self._trans_count + 1, upper_trans_counts[1]):
                    for n1 in (self._trans_count - nt, upper_trans_counts[0] + nt):
                        nid = self._nids[n3][n2][n1]
                        nid_to_node_layout[nid] = node_layout_permuted
                for n1 in range(self._trans_count + 1, upper_trans_counts[0]):
                    for n2 in (self._trans_count - nt, upper_trans_counts[1] + nt):
                        nid = self._nids[n3][n2][n1]
                        nid_to_node_layout[nid] = node_layout_permuted
        for n2 in range(self._element_counts[1] + 1):
            for n1 in range(self._element_counts[0] + 1):
                # nodes on boundary between bottom transition and box are 4-way on corners, 3-way on edges,
                # fully permuted in between
                nid = self._nids[self._trans_count][n2][n1]
                if nid:
                    node_layout = node_layout_permuted
                    if n2 == self._trans_count:
                        node_layout = node_layout_3way23[0]
                        if n1 == self._trans_count:
                            node_layout = node_layout_4way[0]
                        elif n1 == upper_trans_counts[0]:
                            node_layout = node_layout_4way[1]
                    elif n2 == upper_trans_counts[1]:
                        node_layout = node_layout_3way23[1]
                        if n1 == self._trans_count:
                            node_layout = node_layout_4way[2]
                        elif n1 == upper_trans_counts[0]:
                            node_layout = node_layout_4way[3]
                    elif n1 == self._trans_count:
                        node_layout = node_layout_3way13[0]
                    elif n1 == upper_trans_counts[0]:
                        node_layout = node_layout_3way13[1]
                    nid_to_node_layout[nid] = node_layout
                # nodes on boundary between top transition and box are 4-way on corners, 3-way on edges, None in between
                nid = self._nids[upper_trans_counts[2]][n2][n1]
                if nid:
                    node_layout = None
                    if n2 == self._trans_count:
                        node_layout = node_layout_3way23[2]
                        if n1 == self._trans_count:
                            node_layout = node_layout_4way[4]
                        elif n1 == upper_trans_counts[0]:
                            node_layout = node_layout_4way[5]
                    elif n2 == upper_trans_counts[1]:
                        node_layout = node_layout_3way23[3]
                        if n1 == self._trans_count:
                            node_layout = node_layout_4way[6]
                        elif n1 == upper_trans_counts[0]:
                            node_layout = node_layout_4way[7]
                    elif n1 == self._trans_count:
                        node_layout = node_layout_3way13[2]
                    elif n1 == upper_trans_counts[0]:
                        node_layout = node_layout_3way13[3]
                    if node_layout:
                        nid_to_node_layout[nid] = node_layout
        # 3-way points on box edges up middle, fully permuted on faces
        for n3 in range(self._trans_count + 1, upper_trans_counts[2]):
            for n2 in range(self._trans_count + 1, upper_trans_counts[1]):
                for n1 in (self._trans_count, upper_trans_counts[0]):
                    nid = self._nids[n3][n2][n1]
                    nid_to_node_layout[nid] = node_layout_permuted
            for n1 in range(self._trans_count + 1, upper_trans_counts[0]):
                for n2 in (self._trans_count, upper_trans_counts[1]):
                    nid = self._nids[n3][n2][n1]
                    nid_to_node_layout[nid] = node_layout_permuted
            nid = self._nids[n3][self._trans_count][self._trans_count]
            nid_to_node_layout[nid] = node_layout_3way12[0]
            nid = self._nids[n3][self._trans_count][upper_trans_counts[0]]
            nid_to_node_layout[nid] = node_layout_3way12[1]
            nid = self._nids[n3][upper_trans_counts[1]][self._trans_count]
            nid_to_node_layout[nid] = node_layout_3way12[2]
            nid = self._nids[n3][upper_trans_counts[1]][upper_trans_counts[0]]
            nid_to_node_layout[nid] = node_layout_3way12[3]
        # 3-way points on 8 corner transitions out from 4-way points
        for nt in range(self._trans_count):
            nid = self._nids[nt][nt][nt]
            nid_to_node_layout[nid] = node_layout_3way12[1]
            nid = self._nids[nt][nt][self._element_counts[0] - nt]
            nid_to_node_layout[nid] = node_layout_3way12[0]
            nid = self._nids[nt][self._element_counts[1] - nt][nt]
            nid_to_node_layout[nid] = node_layout_3way12[3]
            nid = self._nids[nt][self._element_counts[1] - nt][self._element_counts[0] - nt]
            nid_to_node_layout[nid] = node_layout_3way12[2]
            nid = self._nids[self._element_counts[2] - nt][nt][nt]
            nid_to_node_layout[nid] = node_layout_3way12[0]
            nid = self._nids[self._element_counts[2] - nt][nt][self._element_counts[0] - nt]
            nid_to_node_layout[nid] = node_layout_3way12[1]
            nid = self._nids[self._element_counts[2] - nt][self._element_counts[1] - nt][nt]
            nid_to_node_layout[nid] = node_layout_3way12[2]
            nid = self._nids[self._element_counts[2] - nt][self._element_counts[1] - nt][self._element_counts[0] - nt]
            nid_to_node_layout[nid] = node_layout_3way12[3]
        # add prescribed node layouts
        for n1, n2, n3, node_layout in self._prescribed_node_layouts:
            nid = self._nids[n3][n2][n1]
            nid_to_node_layout[nid] = node_layout
        return nid_to_node_layout

    def get_node_layout_manager(self):
        return self._node_layout_manager

    def get_node_identifier(self, n1, n2, n3):
        assert 0 <= n1 <= self._element_counts[0]
        assert 0 <= n2 <= self._element_counts[1]
        assert 0 <= n3 <= self._element_counts[2]
        return self._nids[n3][n2][n1]

    def get_node_parameters(self, n1, n2, n3):
        assert 0 <= n1 <= self._element_counts[0]
        assert 0 <= n2 <= self._element_counts[1]
        assert 0 <= n3 <= self._element_counts[2]
        return self._nx[n3][n2][n1]

    def set_node_parameters(self, n1, n2, n3, parameters, nid=None, node_layout=None):
        assert 0 <= n1 <= self._element_counts[0]
        assert 0 <= n2 <= self._element_counts[1]
        assert 0 <= n3 <= self._element_counts[2]
        assert self._nx[n3][n2][n1] is not None
        assert self._nids[n3][n2][n1] is None
        self._nx[n3][n2][n1] = copy.deepcopy(parameters)
        self._nids[n3][n2][n1] = nid
        self._prescribed_node_layouts.append((n1, n2, n3, node_layout))

    def generate_mesh(self, fieldmodule, coordinates, start_node_identifier=1, start_element_identifier=1):
        """
        After build() has been called, generate nodes and elements of ellipsoid.
        Client is expected to run within ChangeManager(fieldmodule).
        :param fieldmodule: Owning fieldmodule to create mesh in.
        :param coordinates: Coordinate field to define.
        :param start_node_identifier: Identifier to define for first node and incremented thereafter.
        :param start_element_identifier: Identifier to define for first element and incremented thereafter.
        Note this is for mesh2d if surface_only, otherwise mesh3d.
        return next node identifier, next element identifier for objects after this.
        """
        fieldcache = fieldmodule.createFieldcache()

        # create nodes

        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        value_labels = [Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2]
        if not self._surface_only:
            value_labels.append(Node.VALUE_LABEL_D_DS3)
        for value_label in value_labels:
            nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)

        node_identifier = start_node_identifier
        for n3 in range(self._element_counts[2] + 1):
            for n2 in range(self._element_counts[1] + 1):
                for n1 in range(self._element_counts[0] + 1):
                    if self._nids[n3][n2][n1] is not None:
                        continue  # prescribed node
                    parameters = self._nx[n3][n2][n1]
                    if not parameters:
                        continue
                    x, d1, d2, d3 = parameters
                    if not x:
                        continue  # while in development
                    node = nodes.createNode(node_identifier, nodetemplate)
                    self._nids[n3][n2][n1] = node_identifier
                    fieldcache.setNode(node)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                    if d1:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, d1)
                    if d2:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, d2)
                    if d3:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)
                    node_identifier += 1

        # create elements

        mesh_dimension = 2 if self._surface_only else 3
        mesh = fieldmodule.findMeshByDimension(mesh_dimension)
        element_identifier = start_element_identifier
        # return node_identifier, element_identifier
        half_counts = [count // 2 for count in self._element_counts]
        octant_mesh_group_lists = None
        if self._octant_group_lists:
            octant_mesh_group_lists = []
            for octant_group_list in self._octant_group_lists:
                octant_mesh_group_list = []
                for octant_group in octant_group_list:
                    octant_mesh_group_list.append(octant_group.getOrCreateMeshGroup(mesh))
                octant_mesh_group_lists.append(octant_mesh_group_list)
        box_mesh_group = None
        transition_mesh_group = None
        if not self._surface_only:
            if self._box_group:
                box_mesh_group = self._box_group.getOrCreateMeshGroup(mesh)
            if self._transition_group:
                transition_mesh_group = self._transition_group.getOrCreateMeshGroup(mesh)

        if self._surface_only:
            # 2-D mesh
            elementtemplate_regular = mesh.createElementtemplate()
            elementtemplate_regular.setElementShapeType(Element.SHAPE_TYPE_SQUARE)
            bicubic_hermite_serendipity_basis = (
                fieldmodule.createElementbasis(2, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY))
            eft_regular = mesh.createElementfieldtemplate(bicubic_hermite_serendipity_basis)
            elementtemplate_regular.defineField(coordinates, -1, eft_regular)
            elementtemplate_special = mesh.createElementtemplate()
            elementtemplate_special.setElementShapeType(Element.SHAPE_TYPE_SQUARE)
            # get actual indexes used on rim in 1, 2, 3 directions
            rim_indexes = [[0] + [self._trans_count + 1 + j
                                  for j in range(self._element_counts[i] - 2 * self._trans_count - 1)] +
                           [self._element_counts[i]] for i in range(3)]
            # bottom rectangle
            bottom_nids = self._nids[0]
            last_nids_row = None
            octant_n3 = 0
            for i2, n2 in enumerate(rim_indexes[1]):
                octant_n2 = 2 if (n2 > half_counts[1]) else 0
                nids_row = []
                for i1, n1 in enumerate(reversed(rim_indexes[0])):
                    octant_n1 = 1 if (n1 >= half_counts[0]) else 0
                    nids_row.append(bottom_nids[n2][n1])
                    if (i2 > 0) and (i1 > 0):
                        nids = [last_nids_row[i1 - 1], last_nids_row[i1], nids_row[i1 - 1], nids_row[i1]]
                        if None in nids:
                            continue
                        element = mesh.createElement(element_identifier, elementtemplate_regular)
                        element.setNodesByIdentifier(eft_regular, nids)
                        # print("Element", element_identifier, "nids", nids)
                        if octant_mesh_group_lists:
                            octant = octant_n3 + octant_n2 + octant_n1
                            for mesh_group in octant_mesh_group_lists[octant]:
                                mesh_group.addElement(element)
                        element_identifier += 1
                last_nids_row = nids_row
            # around sides
            node_layout_permuted = self._node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=False)
            node_layout_triple_points = self._node_layout_manager.getNodeLayoutTriplePoint2D()
            index_increments = [[0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]]
            increment_number = 0
            index_increment = index_increments[0]
            elements_count_around12 = \
                2 * (self._element_counts[0] + self._element_counts[1] - 4 * self._trans_count)
            last_nids_row = None
            last_parameters_row = None
            last_corners_row = None
            for n3 in rim_indexes[2]:
                octant_n3 = 4 if (n3 > half_counts[2]) else 0
                indexes = [self._element_counts[0], half_counts[1], n3]
                nids_row = []
                parameters_row = []
                corners_row = []
                for nc in range(elements_count_around12):
                    if nc > 0:
                        while True:
                            indexes = [indexes[c] + index_increment[c] for c in range(3)]
                            # skip over blank transition coordinates
                            if self._nx[indexes[2]][indexes[1]][indexes[0]]:
                                break
                    nids_row.append(self._nids[indexes[2]][indexes[1]][indexes[0]])
                    parameters_row.append(self._nx[indexes[2]][indexes[1]][indexes[0]])
                    if self._next_increment_out_of_bounds(indexes, index_increment):
                        corners_row.append(True)
                        increment_number += 1
                        if increment_number == len(index_increments):
                            increment_number = 0
                        index_increment = index_increments[increment_number]
                    else:
                        corners_row.append(False)
                if n3 > 0:
                    quarter_elements_count_around12 = elements_count_around12 // 4
                    octant_nc = []
                    for nc in range(elements_count_around12):
                        ncp = (nc + 1) % elements_count_around12
                        nids = [last_nids_row[nc], last_nids_row[ncp],
                                nids_row[nc], nids_row[ncp]]
                        if None in nids:
                            continue
                        q = nc // quarter_elements_count_around12
                        octant_nc.append(3 if (q == 0) else (2 if (q == 1) else (0 if (q == 2) else 1)))
                        elementtemplate = elementtemplate_regular
                        eft = eft_regular
                        scalefactors = None
                        if n3 in (rim_indexes[2][1], self._element_counts[2]):
                            node_parameters = [
                                last_parameters_row[nc], last_parameters_row[ncp],
                                parameters_row[nc], parameters_row[ncp]]
                            if n3 == rim_indexes[2][1]:
                                node_layouts = [node_layout_permuted, node_layout_permuted, None, None]
                                if last_corners_row[nc]:
                                    node_layouts[0] = node_layout_triple_points[(5 - q) % 4]
                                elif last_corners_row[ncp]:
                                    node_layouts[1] = node_layout_triple_points[(5 - q) % 4]
                            else:
                                node_layouts = [None, None, node_layout_permuted, node_layout_permuted]
                                if corners_row[nc]:
                                    node_layouts[2] = node_layout_triple_points[q]
                                elif corners_row[ncp]:
                                    node_layouts[3] = node_layout_triple_points[q]
                            eft, scalefactors = \
                                determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                            elementtemplate_special.defineField(coordinates, -1, eft)
                            elementtemplate = elementtemplate_special
                        element = mesh.createElement(element_identifier, elementtemplate)
                        element.setNodesByIdentifier(eft, nids)
                        if scalefactors:
                            element.setScaleFactors(eft, scalefactors)
                        # print("Element", element_identifier, "nids", nids)
                        if octant_mesh_group_lists:
                            octant = octant_n3 + octant_nc[nc]
                            for mesh_group in octant_mesh_group_lists[octant]:
                                mesh_group.addElement(element)
                        element_identifier += 1
                last_nids_row = nids_row
                last_parameters_row = parameters_row
                last_corners_row = corners_row
            # top rectangle
            top_nids = self._nids[self._element_counts[2]]
            last_nids_row = None
            octant_n3 = 4
            for i2, n2 in enumerate(rim_indexes[1]):
                octant_n2 = 2 if (n2 > half_counts[1]) else 0
                nids_row = []
                for i1, n1 in enumerate(rim_indexes[0]):
                    octant_n1 = 1 if (n1 > half_counts[0]) else 0
                    nids_row.append(top_nids[n2][n1])
                    if (i2 > 0) and (i1 > 0):
                        nids = [last_nids_row[i1 - 1], last_nids_row[i1], nids_row[i1 - 1], nids_row[i1]]
                        if None in nids:
                            continue
                        element = mesh.createElement(element_identifier, elementtemplate_regular)
                        element.setNodesByIdentifier(eft_regular, nids)
                        # print("Element", element_identifier, "nids", nids)
                        if octant_mesh_group_lists:
                            octant = octant_n3 + octant_n2 + octant_n1
                            for mesh_group in octant_mesh_group_lists[octant]:
                                mesh_group.addElement(element)
                        element_identifier += 1
                last_nids_row = nids_row
        else:
            # 3-D mesh
            elementtemplate_regular = mesh.createElementtemplate()
            elementtemplate_regular.setElementShapeType(Element.SHAPE_TYPE_CUBE)
            tricubic_hermite_serendipity_basis = (
                fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY))
            eft_regular = mesh.createElementfieldtemplate(tricubic_hermite_serendipity_basis)
            elementtemplate_regular.defineField(coordinates, -1, eft_regular)
            elementtemplate_special = mesh.createElementtemplate()
            elementtemplate_special.setElementShapeType(Element.SHAPE_TYPE_CUBE)
            box_counts = [half_counts[i] - self._trans_count for i in range(3)]
            dbox_counts = [2 * box_counts[i] for i in range(3)]
            nid_to_node_layout = self._get_nid_to_node_layout_map_3d(self._node_layout_manager)
            # bottom transition
            last_nids_layer = None
            last_nx_layer = None
            for nt in range(self._trans_count + 1):
                n3 = nt
                octant_n3 = 0
                nids_layer = []
                nx_layer = []
                last_nids_row = None
                last_nx_row = None
                for i2 in range(dbox_counts[1] + 1):
                    n2 = (nt if (i2 == 0)
                          else (self._element_counts[1] - nt) if (i2 == dbox_counts[1])
                          else (self._trans_count + i2))
                    octant_n2 = 2 if (n2 > half_counts[1]) else 0
                    nids_row = []
                    nx_row = []
                    for i1 in range(dbox_counts[0] + 1):
                        n1 = (nt if (i1 == 0)
                              else (self._element_counts[0] - nt) if (i1 == dbox_counts[0])
                              else (self._trans_count + i1))
                        octant_n1 = 1 if (n1 > half_counts[0]) else 0
                        nids_row.append(self._nids[n3][n2][n1])
                        nx_row.append(self._nx[n3][n2][n1])
                        if (nt > 0) and (i2 > 0) and (i1 > 0):
                            nids = [last_nids_row[i1], last_nids_row[i1 - 1],
                                    nids_row[i1], nids_row[i1 - 1],
                                    last_nids_layer[i2 - 1][i1], last_nids_layer[i2 - 1][i1 - 1],
                                    last_nids_layer[i2][i1], last_nids_layer[i2][i1 - 1]]
                            if None in nids:
                                continue
                            elementtemplate = elementtemplate_regular
                            eft = eft_regular
                            scalefactors = None
                            node_layouts = [nid_to_node_layout.get(nid) for nid in nids]
                            if any(node_layout is not None for node_layout in node_layouts):
                                node_parameters = [last_nx_row[i1], last_nx_row[i1 - 1],
                                                   nx_row[i1], nx_row[i1 - 1],
                                                   last_nx_layer[i2 - 1][i1], last_nx_layer[i2 - 1][i1 - 1],
                                                   last_nx_layer[i2][i1], last_nx_layer[i2][i1 - 1]]
                                eft, scalefactors = \
                                    determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                                elementtemplate_special.defineField(coordinates, -1, eft)
                                elementtemplate = elementtemplate_special
                            element = mesh.createElement(element_identifier, elementtemplate)
                            element.setNodesByIdentifier(eft, nids)
                            # print("Element", element_identifier, "nids", nids)
                            if scalefactors:
                                element.setScaleFactors(eft, scalefactors)
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_n2 + octant_n1
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                            if transition_mesh_group:
                                transition_mesh_group.addElement(element)
                            element_identifier += 1
                    nids_layer.append(nids_row)
                    nx_layer.append(nx_row)
                    last_nids_row = nids_row
                    last_nx_row = nx_row
                last_nids_layer = nids_layer
                last_nx_layer = nx_layer
            # middle
            upper_trans_counts = [self._element_counts[i] - self._trans_count for i in range(3)]
            last_nids_layer = None
            last_nx_layer = None
            last_rim_nids_layer = None
            last_rim_nx_layer = None
            for i3 in range(dbox_counts[2] + 1):
                n3 = self._trans_count + i3
                octant_n3 = 4 if (n3 > half_counts[2]) else 0
                nids_layer = []
                nx_layer = []
                elementIdentifiers = []
                last_nids_row = None
                last_nx_row = None
                for i2 in range(dbox_counts[1] + 1):
                    n2 = self._trans_count + i2
                    octant_n2 = 2 if (n2 > half_counts[1]) else 0
                    nids_row = []
                    nx_row = []
                    for i1 in range(dbox_counts[0] + 1):
                        n1 = self._trans_count + i1
                        octant_n1 = 1 if (n1 > half_counts[0]) else 0
                        nids_row.append(self._nids[n3][n2][n1])
                        nx_row.append(self._nx[n3][n2][n1])
                        if (i3 > 0) and (i2 > 0) and (i1 > 0):
                            nids = [last_nids_layer[i2 - 1][i1 - 1], last_nids_layer[i2 - 1][i1],
                                    last_nids_layer[i2][i1 - 1], last_nids_layer[i2][i1],
                                    last_nids_row[i1 - 1], last_nids_row[i1],
                                    nids_row[i1 - 1], nids_row[i1]]
                            if None in nids:
                                continue
                            elementtemplate = elementtemplate_regular
                            eft = eft_regular
                            scalefactors = None
                            node_layouts = [nid_to_node_layout.get(nid) for nid in nids]
                            if any(node_layout is not None for node_layout in node_layouts):
                                node_parameters = [last_nx_layer[i2 - 1][i1 - 1], last_nx_layer[i2 - 1][i1],
                                                   last_nx_layer[i2][i1 - 1], last_nx_layer[i2][i1],
                                                   last_nx_row[i1 - 1], last_nx_row[i1],
                                                   nx_row[i1 - 1], nx_row[i1]]
                                eft, scalefactors = \
                                    determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                                elementtemplate_special.defineField(coordinates, -1, eft)
                                elementtemplate = elementtemplate_special
                            element = mesh.createElement(element_identifier, elementtemplate)
                            element.setNodesByIdentifier(eft, nids)
                            # print("Element", element_identifier, "nids", nids)
                            if scalefactors:
                                element.setScaleFactors(eft, scalefactors)
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_n2 + octant_n1
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                            if box_mesh_group:
                                box_mesh_group.addElement(element)
                            elementIdentifiers.append(element_identifier)
                            element_identifier += 1
                    nids_layer.append(nids_row)
                    nx_layer.append(nx_row)
                    last_nids_row = nids_row
                    last_nx_row = nx_row
                last_nids_layer = nids_layer
                last_nx_layer = nx_layer

                rim_nids_layer = []
                rim_nx_layer = []
                last_rim_nids_row = None
                last_rim_nx_row = None
                for nt in range(self._trans_count + 1):
                    n3 = ((self._trans_count - nt) if (i3 == 0) else (
                        (upper_trans_counts[2] + nt) if (i3 == dbox_counts[2]) else (self._trans_count + i3)))
                    rim_nids_row = []
                    rim_nx_row = []
                    octant_nc = []
                    elementIdentifiers = []
                    n2 = self._trans_count - nt
                    for i1 in range(dbox_counts[0]):
                        n1 = ((self._trans_count - nt) if (i1 == 0) else (
                            (upper_trans_counts[0] + nt) if (i1 == dbox_counts[0]) else (self._trans_count + i1)))
                        rim_nids_row.append(self._nids[n3][n2][n1])
                        rim_nx_row.append(self._nx[n3][n2][n1])
                        octant_nc.append(1 if n1 >= half_counts[0] else 0)
                    n1 = upper_trans_counts[0] + nt
                    for i2 in range(dbox_counts[1]):
                        n2 = ((self._trans_count - nt) if (i2 == 0) else (
                            (upper_trans_counts[1] + nt) if (i2 == dbox_counts[1]) else (self._trans_count + i2)))
                        rim_nids_row.append(self._nids[n3][n2][n1])
                        rim_nx_row.append(self._nx[n3][n2][n1])
                        octant_nc.append(3 if n2 >= half_counts[1] else 1)
                    n2 = upper_trans_counts[1] + nt
                    for i1 in range(dbox_counts[0]):
                        n1 = ((upper_trans_counts[0] + nt) if (i1 == 0) else (
                            (self._trans_count - nt) if (i1 == dbox_counts[0]) else (upper_trans_counts[0] - i1)))
                        rim_nids_row.append(self._nids[n3][n2][n1])
                        rim_nx_row.append(self._nx[n3][n2][n1])
                        octant_nc.append(3 if n1 > half_counts[0] else 2)
                    n1 = self._trans_count - nt
                    for i2 in range(dbox_counts[1]):
                        n2 = ((upper_trans_counts[1] + nt) if (i2 == 0) else (
                             (self._trans_count - nt) if (i2 == dbox_counts[1]) else (upper_trans_counts[1] - i2)))
                        rim_nids_row.append(self._nids[n3][n2][n1])
                        rim_nx_row.append(self._nx[n3][n2][n1])
                        octant_nc.append(2 if n2 > half_counts[1] else 0)
                    if (i3 > 0) and (nt > 0):
                        rim_count = len(rim_nids_row)
                        for nc in range(rim_count):
                            ncp = (nc + 1) % rim_count
                            nids = [last_rim_nids_layer[nt - 1][nc], last_rim_nids_layer[nt - 1][ncp],
                                    last_rim_nids_row[nc], last_rim_nids_row[ncp],
                                    last_rim_nids_layer[nt][nc], last_rim_nids_layer[nt][ncp],
                                    rim_nids_row[nc], rim_nids_row[ncp]]
                            if None in nids:
                                continue
                            elementtemplate = elementtemplate_regular
                            eft = eft_regular
                            scalefactors = None
                            node_layouts = [nid_to_node_layout.get(nid) for nid in nids]
                            if any(node_layout is not None for node_layout in node_layouts):
                                node_parameters = [last_rim_nx_layer[nt - 1][nc], last_rim_nx_layer[nt - 1][ncp],
                                                   last_rim_nx_row[nc], last_rim_nx_row[ncp],
                                                   last_rim_nx_layer[nt][nc], last_rim_nx_layer[nt][ncp],
                                                   rim_nx_row[nc], rim_nx_row[ncp]]
                                eft, scalefactors = \
                                    determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                                elementtemplate_special.defineField(coordinates, -1, eft)
                                elementtemplate = elementtemplate_special
                            element = mesh.createElement(element_identifier, elementtemplate)
                            element.setNodesByIdentifier(eft, nids)
                            # print("Element", element_identifier, "nids", nids)
                            if scalefactors:
                                element.setScaleFactors(eft, scalefactors)
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_nc[nc]
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                            if transition_mesh_group:
                                transition_mesh_group.addElement(element)
                            elementIdentifiers.append(element_identifier)
                            element_identifier += 1
                    rim_nids_layer.append(rim_nids_row)
                    rim_nx_layer.append(rim_nx_row)
                    last_rim_nids_row = rim_nids_row
                    last_rim_nx_row = rim_nx_row
                last_rim_nids_layer = rim_nids_layer
                last_rim_nx_layer = rim_nx_layer

            # top transition
            last_nids_layer = None
            last_nx_layer = None
            for nt in range(self._trans_count, -1, -1):
                n3 = self._element_counts[2] - nt
                octant_n3 = 4
                nids_layer = []
                nx_layer = []
                last_nids_row = None
                last_nx_row = None
                for i2 in range(dbox_counts[1] + 1):
                    n2 = (nt if (i2 == 0)
                          else (self._element_counts[1] - nt) if (i2 == dbox_counts[1])
                          else (self._trans_count + i2))
                    octant_n2 = 2 if (n2 > half_counts[1]) else 0
                    nids_row = []
                    nx_row = []
                    for i1 in range(dbox_counts[0] + 1):
                        n1 = (nt if (i1 == 0)
                              else (self._element_counts[0] - nt) if (i1 == dbox_counts[0])
                              else (self._trans_count + i1))
                        octant_n1 = 1 if (n1 > half_counts[0]) else 0
                        nids_row.append(self._nids[n3][n2][n1])
                        nx_row.append(self._nx[n3][n2][n1])
                        if (nt < self._trans_count) and (i2 > 0) and (i1 > 0):
                            nids = [last_nids_layer[i2 - 1][i1 - 1], last_nids_layer[i2 - 1][i1],
                                    last_nids_layer[i2][i1 - 1], last_nids_layer[i2][i1],
                                    last_nids_row[i1 - 1], last_nids_row[i1],
                                    nids_row[i1 - 1], nids_row[i1]]
                            if None in nids:
                                continue
                            elementtemplate = elementtemplate_regular
                            eft = eft_regular
                            scalefactors = None
                            node_layouts = [nid_to_node_layout.get(nid) for nid in nids]
                            if any(node_layout is not None for node_layout in node_layouts):
                                node_parameters = [last_nx_layer[i2 - 1][i1 - 1], last_nx_layer[i2 - 1][i1],
                                                   last_nx_layer[i2][i1 - 1], last_nx_layer[i2][i1],
                                                   last_nx_row[i1 - 1], last_nx_row[i1],
                                                   nx_row[i1 - 1], nx_row[i1]]
                                eft, scalefactors = \
                                    determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                                elementtemplate_special.defineField(coordinates, -1, eft)
                                elementtemplate = elementtemplate_special
                            element = mesh.createElement(element_identifier, elementtemplate)
                            element.setNodesByIdentifier(eft, nids)
                            # print("Element", element_identifier, "nids", nids)
                            if scalefactors:
                                element.setScaleFactors(eft, scalefactors)
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_n2 + octant_n1
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                            if transition_mesh_group:
                                transition_mesh_group.addElement(element)
                            element_identifier += 1
                    nids_layer.append(nids_row)
                    nx_layer.append(nx_row)
                    last_nids_row = nids_row
                    last_nx_row = nx_row
                last_nids_layer = nids_layer
                last_nx_layer = nx_layer

        return node_identifier, element_identifier
