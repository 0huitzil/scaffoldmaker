"""
Utilities for building solid ellipsoid meshes from hexahedral elements.
"""
from cmlibs.maths.vectorops import add, cross, div, dot, magnitude, mult, normalize, rejection, set_magnitude, sub
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node

from scaffoldmaker.utils.eft_utils import (
    determineCubicHermiteSerendipityEft, HermiteNodeLayoutManager, resolveEftCoreBoundaryScaling)
from scaffoldmaker.utils.geometry import (
    getEllipsePointAtTrueAngle, getEllipseTangentAtPoint, moveCoordinatesToEllipsoidSurface,
    moveDerivativeToEllipsoidSurface, moveDerivativeToEllipsoidSurfaceInPlane, sampleCurveOnEllipsoid)
from scaffoldmaker.utils.interpolation import DerivativeScalingMode, linearlyInterpolateVectors, sampleHermiteCurve
from scaffoldmaker.utils.hextetrahedronmesh import HexTetrahedronMesh
from scaffoldmaker.utils.meshgeneratedata import MeshGenerateData
from scaffoldmaker.utils.quadtrianglemesh import QuadTriangleMesh
import copy
from enum import Enum
import math

from scaffoldmaker.utils.zinc_utils import generate_datapoints


class EllipsoidSurfaceD3Mode(Enum):
    SURFACE_NORMAL = 1  # surface D3 are exact surface normals to ellipsoid
    OBLIQUE_DIRECTION = 2  # surface D3 are in direction of surface point on ellipsoid, gives flat oblique planes
    # Surface D3 are surface normals to ellipsoid projected onto radial planes transitioning between axis2 and axis3
    SURFACE_NORMAL_PLANE_PROJECTION = 3


class EllipsoidMesh:
    """
    Generates a solid ellipsoid of hexahedral elements with oblique cross axes suited to describing lung geometry.
    Exception: if not core and 0 shell elements, a 2-D surface mesh of quad elements is created.
    """

    def __init__(self, element_counts, shell_element_count=0, transition_element_count=1, core=True,
                 core_shell_scaling_mode: int=1):
        """
        :param element_counts: Number of elements across full ellipsoid in 1, 2, 3 directions.
        :param shell_element_count: Number of shell elements >= 0.
        :param transition_element_count: Number of transition elements around outside >= 1.
        :param core: Set to True to fill the core. If False and no shell elements, only makes nodes and 2-D elements
        on the surface.
        :param core_shell_scaling_mode: Mode controlling how core-shell derivative differences are scaled:
        1 = use scale factors on last transition element, 2 = use separate version on last transition element, or
        0 to replace existing d3 to suit core.
        """
        assert all((count >= 4) and (count % 2 == 0) for count in element_counts)
        assert 1 <= transition_element_count <= (min(element_counts) // 2 - 1)
        self._element_counts = element_counts
        self._rim_count = shell_element_count + transition_element_count
        self._shell_count = shell_element_count
        self._transition_count = transition_element_count
        self._core = core
        self._core_shell_scaling_mode = 0 if (shell_element_count == 0) else core_shell_scaling_mode
        self._box_group = None
        self._transition_group = None
        self._core_group = None
        self._shell_group = None
        self._octant_group_lists = None
        self._tube_core_box_layout = False  # set to true to layout core box derivatives for TubeNetworkMesh
        none_parameters = [None] * 4  # x, d1, d2, d3
        self._nx = []  # shield mesh with holes over n3, n2, n1, d
        self._nids = []
        self._eids = []
        self._merge_counts = []  # Number of already merged nodes at location for weighting next nodes
        half_counts = [count // 2 for count in self._element_counts]
        for n3 in range(self._element_counts[2] + 1):
            # index into transition zone
            trans3 = (self._rim_count - n3) if (n3 < half_counts[2]) else \
                (self._rim_count + n3 - self._element_counts[2])
            nx_layer = []
            nids_layer = []
            eids_layer = []
            merge_counts_layer = []
            # print(n3, trans3)
            for n2 in range(self._element_counts[1] + 1):
                # index into transition zone
                trans2 = (self._rim_count - n2) if (n2 < half_counts[1]) else \
                    (self._rim_count + n2 - self._element_counts[1])
                nx_row = []
                nids_row = []
                # s = ""
                for n1 in range(self._element_counts[0] + 1):
                    # index into transition zone
                    trans1 = (self._rim_count - n1) if (n1 < half_counts[0]) else \
                        (self._rim_count + n1 - self._element_counts[0])
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
                if n2 < self._element_counts[1]:
                    eids_layer.append([None] * self._element_counts[0])
                merge_counts_layer.append([0 for _ in range(self._element_counts[0] + 1)])
                # print(s)
            self._nx.append(nx_layer)
            self._nids.append(nids_layer)
            if n3 < self._element_counts[2]:
                self._eids.append(eids_layer)
            self._merge_counts.append(merge_counts_layer)
        self._prescribed_node_layouts = []  # list of (n1, n2, n3, node_layout), sets node layout before node created

    def get_element_counts(self):
        return self._element_counts

    def set_box_transition_groups(self, box_group, transition_group):
        """
        Set zinc groups to fill with elements in box and transition regions, if core being created.
        :param box_group: Group field to add elements from box region to.
        :param transition_group: Group field to add elements from transition region to.
        """
        self._box_group = box_group
        self._transition_group = transition_group

    def set_core_shell_groups(self, core_group, shell_group):
        """
        Set zinc groups to fill with elements in core and shell regions, if core is being created.
        :param core_group: Group field to add elements from core region to.
        :param shell_group: Group field to add elements from shell region to.
        """
        self._core_group = core_group
        self._shell_group = shell_group

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

    def is_tube_core_box_layout(self):
        """
        Check whether layout of core derivatives is for TubeNetworkMesh.
        :return: True if core derivatives layout is for TubeNetworkMesh i.e -d1, d3, d2
        """
        return self._tube_core_box_layout

    def set_tube_core_box_layout(self, tube_core_box_layout):
        """
        Set whether layout of core derivatives is for TubeNetworkMesh.
        :param tube_core_box_layout: Set to True to use core derivatives layout TubeNetworkMesh i.e -d1, d3, d2.
        """
        self._tube_core_box_layout = tube_core_box_layout

    def build(self, axes_lengths, axis2_x_rotation_radians=0, axis3_x_rotation_radians=math.pi/2.0,
              axes_shell_thicknesses=[0.0, 0.0, 0.0], nway_d_factor=0.6,
              surface_d3_mode=EllipsoidSurfaceD3Mode.SURFACE_NORMAL):
        """
        Determine coordinates and derivatives over and within the full ellipsoid.
        :param axes_lengths: Axes lengths (radius) in x, y, z directions: [a, b, c].
        :param axis2_x_rotation_radians: Rotation of axis 2 about +x direction
        :param axis3_x_rotation_radians: Rotation of axis 3 about +x direction.
        :param axes_shell_thicknesses: If there are shell elements, shell thicknesses in x, y, z directions,
        subtracted from axes lengths. Must all be > 0.0 if there are shell elements.
        :param nway_d_factor: Value, normally from 0.5 to 1.0 giving n-way derivative magnitude as a proportion
        of the minimum regular magnitude sampled to the n-way point. This reflects that distances from the mid-side
        of a triangle to the centre are shorter, so the derivative in the middle must be smaller.
        :param surface_d3_mode: Value from EllipsoidSurfaceD3Mode controlling d3 on the ellipsoid surface. Only used
        if there are no shell elements.
        """
        half_counts = [count // 2 for count in self._element_counts]

        octant1 = self.build_octant(axes_lengths, half_counts, axis2_x_rotation_radians, axis3_x_rotation_radians,
                                    axes_shell_thicknesses, nway_d_factor=nway_d_factor,
                                    surface_d3_mode=surface_d3_mode)
        self.merge_octant_plus1_quadrant(octant1, quadrant=0)
        octant1.mirror_yz()
        self.merge_octant_plus1_quadrant(octant1, quadrant=2)

        octant2 = self.build_octant(axes_lengths, [half_counts[0], half_counts[2], half_counts[1]],
                                    axis3_x_rotation_radians, axis2_x_rotation_radians + math.pi,
                                    axes_shell_thicknesses, nway_d_factor=nway_d_factor,
                                    surface_d3_mode=surface_d3_mode)
        self.merge_octant_plus1_quadrant(octant2, quadrant=1)
        octant2.mirror_yz()
        self.merge_octant_plus1_quadrant(octant2, quadrant=3)

        self.copy_to_negative_axis1()

    def build_octant(self, axes_lengths, full_half_counts, axis2_x_rotation_radians=0, axis3_x_rotation_radians=math.pi/2.0,
                     axes_shell_thicknesses=[0.0, 0.0, 0.0], axis2_extension=0.0,
                     axis2_extension_elements_count=0, nway_d_factor=0.6,
                     surface_d3_mode=EllipsoidSurfaceD3Mode.SURFACE_NORMAL):
        """
        Get coordinates of top-right-front octant with supplied angles.
        If there are shell elements the core is built to the axes lengths less the shell thicknesses and the
        shell elements are linearly between
        :param axes_lengths: Axes lengths (radius) in x, y, z directions: [a, b, c].
        :param full_half_counts: Numbers of elements across octant 1, 2 and 3 directions.
        :param axis2_x_rotation_radians: Rotation of axis 2 about +x direction
        :param axis3_x_rotation_radians: Rotation of axis 3 about +x direction.
        :param axes_shell_thicknesses: If there are shell elements, shell thicknesses in x, y, z directions,
        subtracted from axes lengths. Must all be > 0.0 if there are shell elements.
        :param axis2_extension: Extension distance along axis2 beyond origin [0.0, 0.0, 0.0].
        :param axis2_extension_elements_count: If axis2_extension: number of elements beyond origin.
        Note: included in half_counts[1].
        :param nway_d_factor: Value, normally from 0.5 to 1.0 giving n-way derivative magnitude as a proportion
        of the minimum regular magnitude sampled to the n-way point. This reflects that distances from the mid-side
        of a triangle to the centre are shorter, so the derivative in the middle must be smaller.
        :param surface_d3_mode: Value from EllipsoidSurfaceD3Mode controlling d3 on the ellipsoid surface. Only used
        if there are no shell elements.
        :return: HexTetrahedronMesh
        """
        assert (0 == self._shell_count) or all(axis_length > 0 for axis_length in axes_lengths)
        assert ((axis2_extension == 0.0) and (axis2_extension_elements_count == 0)) or (
                (axis2_extension > 0.0) and (0 < axis2_extension_elements_count))
        half_counts = copy.copy(full_half_counts)
        box_counts = [half_counts[i] - self._rim_count for i in range(3)]

        cos_axis2 = math.cos(axis2_x_rotation_radians)
        sin_axis2 = math.sin(axis2_x_rotation_radians)
        cos_axis3 = math.cos(axis3_x_rotation_radians)
        sin_axis3 = math.sin(axis3_x_rotation_radians)

        origin = [0.0, 0.0, 0.0]
        ext_origin = [0.0, -axis2_extension * cos_axis2, -axis2_extension * sin_axis2]
        ext_half_counts = [
            half_counts[0],
            half_counts[1] + axis2_extension_elements_count,
            half_counts[2]
        ]
        diag_counts = [
            half_counts[0] + half_counts[1] - 2 * self._rim_count,
            half_counts[0] + half_counts[2] - 2 * self._rim_count,
            half_counts[1] + half_counts[2] - 2 * self._rim_count
        ]
        ext_diag_counts = [
            ext_half_counts[0] + ext_half_counts[1] - 2 * self._rim_count,
            ext_half_counts[0] + ext_half_counts[2] - 2 * self._rim_count,
            ext_half_counts[1] + ext_half_counts[2] - 2 * self._rim_count
        ]
        octant = HexTetrahedronMesh(ext_half_counts, ext_diag_counts, nway_d_factor=nway_d_factor)

        a, b, c = axes_lengths
        rim_count = self._rim_count
        outer_triangle_abc = None
        triangle_abc = None  # only or inner surface triangle

        # outer/only surface then inner surface (if shell)
        for surface_index in range(2 if self._shell_count else 1):
            if surface_index == 1:
                a, b, c = sub(axes_lengths, axes_shell_thicknesses)
                for i in range(3):
                    half_counts[i] -= self._shell_count
                    ext_half_counts[i] -= self._shell_count
                rim_count -= self._shell_count

            ext_axis1 = axis1 = [a, 0.0, 0.0]
            axis2 = [0.0] + getEllipsePointAtTrueAngle(b, c, axis2_x_rotation_radians)
            axis2_mag = magnitude(axis2)
            axis2_normal = normalize([0.0, axis2[2], -axis2[1]])
            ext_axis3 = axis3 = [0.0] + getEllipsePointAtTrueAngle(b, c, axis3_x_rotation_radians)
            axis3_normal = normalize([0.0, axis3[2], -axis3[1]])
            if axis2_extension_elements_count:
                assert axis2_extension < axis2_mag  # extension must not go outside ellipsoid
                xb, xa = getEllipsePointAtTrueAngle(axis2_mag, a, math.pi / 2.0, [-axis2_extension, 0.0])
                ext_axis1 = [xa, xb * cos_axis2, xb * sin_axis2]
                ext_axis3 = [0.0] + getEllipsePointAtTrueAngle(b, c, axis3_x_rotation_radians, ext_axis1[1:])
                ext_axis3m = [0.0] + getEllipsePointAtTrueAngle(b, c, axis3_x_rotation_radians + math.pi, ext_axis1[1:])
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
            axis2_dt = set_magnitude([0.0] + getEllipseTangentAtPoint(b, c, axis2[1:]), magnitude(axis_d3))
            ext_axis3_dt = set_magnitude([0.0] + getEllipseTangentAtPoint(b, c, ext_axis3[1:]), magnitude(ext_axis_d3))
            ext_axis3_mdt = [-d for d in ext_axis3_dt]
            axis2_mag = magnitude(axis2)
            axis3_mag = magnitude(axis3)

            sample_curve_on_ellipsoid = (
                lambda start_x, start_d1, start_d2, end_x, end_d1, end_d2, elements_count,
                       start_weight=None, end_weight=None, overweighting=1.0, end_transition=False:
                sampleCurveOnEllipsoid(
                    a, b, c, start_x, start_d1, start_d2, end_x, end_d1, end_d2, elements_count,
                    start_weight, end_weight, overweighting, end_transition))
            move_x_to_ellipsoid_surface = lambda x: moveCoordinatesToEllipsoidSurface(a, b, c, x)
            move_d_to_ellipsoid_surface = lambda x, d: moveDerivativeToEllipsoidSurface(a, b, c, x, d)

            # get outside curve from axis 1 to axis 2
            abx, abd1, abd2 = sampleCurveOnEllipsoid(
                a, b, c,
                axis1, axis_d2, axis_d3,
                axis2, axis_md1, axis2_dt,
                diag_counts[0])
            if axis2_extension_elements_count:
                end_axis_d3 = moveDerivativeToEllipsoidSurfaceInPlane(
                    a, b, c, ext_axis1, [axis_d3[0], -axis_d3[2], axis_d3[1]], axis_d3)
                ext_abx, ext_abd1, ext_abd2 = sampleCurveOnEllipsoid(
                    a, b, c,
                    axis1, [-d for d in axis_d2], axis_d3,
                    ext_axis1, None, end_axis_d3,  # axis_d3
                    axis2_extension_elements_count)
                for i in range(1, axis2_extension_elements_count + 1):
                    abx.insert(0, ext_abx[i])
                    abd1.insert(0, [-d for d in ext_abd1[i]])
                    abd2.insert(0, ext_abd2[i])
            # get outside curve from axis 1 to axis 3
            acx, acd2, acd1 = sampleCurveOnEllipsoid(
                a, b, c,
                abx[0], abd2[0], abd1[0],
                ext_axis3, ext_axis_md1, ext_axis3_mdt,
                ext_diag_counts[1])
            # get outside curve from axis 2 to axis 3
            bcx, bcd2, bcd1 = sampleCurveOnEllipsoid(
                a, b, c,
                abx[-1], abd2[-1], abd1[-1],
                acx[-1], [-d for d in acd1[-1]], acd2[-1],
                ext_diag_counts[2])
            # fix first/last derivatives
            abd2[0] = acd2[0]
            abd2[-1] = bcd2[0]
            acd1[-1] = [-d for d in bcd2[-1]]

            # make outer surface triangle of octant
            triangle_abc = QuadTriangleMesh(
                box_counts[0], box_counts[1] + axis2_extension_elements_count, box_counts[2],
                sample_curve_on_ellipsoid, move_x_to_ellipsoid_surface, move_d_to_ellipsoid_surface,
                nway_d_factor=nway_d_factor)
            triangle_abc.set_edge_parameters12(abx, abd1, abd2)
            triangle_abc.set_edge_parameters13(acx, acd1, acd2)
            triangle_abc.set_edge_parameters23(bcx, bcd1, bcd2)
            triangle_abc.build()

            if surface_index == 0:
                outer_triangle_abc = triangle_abc

        if self._shell_count:
            triangle_abc.assign_d3_difference(outer_triangle_abc, 1.0 / self._shell_count)
            outer_triangle_abc.assign_d3_difference(triangle_abc, -1.0 / self._shell_count)
        else:
            if surface_d3_mode == EllipsoidSurfaceD3Mode.SURFACE_NORMAL:
                evaluate_surface_d3_ellipsoid = lambda tx, td1, td2: set_magnitude(
                    [tx[0] / (a * a), tx[1] / (b * b), tx[2] / (c * c)], dir_mag)
            elif surface_d3_mode == EllipsoidSurfaceD3Mode.OBLIQUE_DIRECTION:
                evaluate_surface_d3_ellipsoid=lambda tx, td1, td2: set_magnitude(tx, dir_mag)
            else:  # EllipsoidSurfaceD3Mode.SURFACE_NORMAL_PLANE_PROJECTION
                def evaluate_surface_d3_ellipsoid_plane(tx, td1, td2):
                    """
                    Restrict d3 to be the ellipsoid normal constrained to be in radial planes from ext_origin through tx,
                    varying between axis_d2 and axis_d3.
                    :param tx: Coordinates of a point on the ellipsoid surface in the octant.
                    :param td1: Unused point d1.
                    :param td2: Unused point d2.
                    :return: Radial plane constrained ellipsoid normal d3 with magnitude dir_mag.
                    """
                    n = [tx[0] / (a * a), tx[1] / (b * b), tx[2] / (c * c)]
                    if dot(tx, axis3_normal) <= 1.0E-5:
                        if dot(tx, axis2_normal) >= -1.0E-5:
                            return set_magnitude(axis1, dir_mag)
                        else:
                            plane_normal = [0.0, axis3[2], -axis3[1]]
                    else:
                        plane_normal = [0.0, tx[2], -tx[1]]
                    normal = rejection(n, plane_normal)
                    return set_magnitude(normal, dir_mag)
                evaluate_surface_d3_ellipsoid = evaluate_surface_d3_ellipsoid_plane
            triangle_abc.assign_d3(evaluate_surface_d3_ellipsoid)

        core_octant = None
        if self._core:
            if self._shell_count:
                # ext_half_counts was reduced in earlier loop for inner surface
                core_octant = HexTetrahedronMesh(ext_half_counts, ext_diag_counts, nway_d_factor=nway_d_factor)
            else:
                core_octant = octant
            core_octant.set_triangle_abc(triangle_abc)

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
                box_counts[0], box_counts[1] + axis2_extension_elements_count, rim_count, sampleHermiteCurve,
                nway_d_factor=nway_d_factor)
            abd3 = [[-d for d in d3] for d3 in triangle_abc.get_edge_parameters12()[3]]
            triangle_abo.set_edge_parameters12(abx, abd1, abd3, abd2)
            count = len(aox) - 1
            aod3 = [linearlyInterpolateVectors(abd2[0], axis_d3, i / count) for i in range(count + 1)]

            triangle_abo.set_edge_parameters13(aox, aod1, aod2, aod3)
            triangle_abo.set_edge_parameters23(box, bod1, bod2, bod3)
            triangle_abo.build(regular_count2=axis2_extension_elements_count)
            aa = a * a
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
                side_d3 = moveDerivativeToEllipsoidSurface(a, b, c, side_x, centre_d3)
                return linearlyInterpolateVectors(centre_d3, side_d3, tx[0] / x)
            triangle_abo.assign_d3(evaluate_surface_d3_abo)
            core_octant.set_triangle_abo(triangle_abo)
            # extract exact derivatives
            aod1 = triangle_abo.get_edge_parameters13()[1]
            bod1 = triangle_abo.get_edge_parameters23()[1]

            # make inner surface triangle 1-3-origin
            triangle_aco = QuadTriangleMesh(
                box_counts[0], box_counts[2], rim_count, sampleHermiteCurve,
                nway_d_factor=nway_d_factor)
            acd3 = [[-d for d in d3] for d3 in triangle_abc.get_edge_parameters13()[3]]
            acmd1 = [[-d for d in d1] for d1 in acd1]
            triangle_aco.set_edge_parameters12(acx, acd2, acd3, acmd1)
            aomd1 = [[-d for d in d1] for d1 in aod1]
            triangle_aco.set_edge_parameters13(aox, aod3, aod2, aomd1)
            cod3 = [acd2[-1]] + [axis_md1] * (len(cox) - 1)
            triangle_aco.set_edge_parameters23(cox, cod3, cod2, cod1)
            triangle_aco.build()
            aa = a * a
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
                side_d3 = moveDerivativeToEllipsoidSurface(a, b, c, side_x, centre_d3)
                return linearlyInterpolateVectors(centre_d3, side_d3, tx[0] / x)
            triangle_aco.assign_d3(evaluate_surface_d3_aco)
            core_octant.set_triangle_aco(triangle_aco)
            # extract exact derivatives
            cod3 = triangle_aco.get_edge_parameters23()[1]

            # make inner surface 2-3-origin
            triangle_bco = QuadTriangleMesh(
                box_counts[1] + axis2_extension_elements_count, box_counts[2], rim_count, sampleHermiteCurve,
                nway_d_factor=nway_d_factor)
            bcd3 = [[-d for d in d3] for d3 in triangle_abc.get_edge_parameters23()[3]]
            # substitute end derivatives for which magnitude is known:
            bcd3[0] = bod2[0]
            bcd3[-1] = cod2[-1]
            bcmd1 = [[-d for d in d1] for d1 in bcd1]
            triangle_bco.set_edge_parameters12(bcx, bcd2, bcd3, bcmd1)
            bomd1 = [[-d for d in d1] for d1 in bod1]
            triangle_bco.set_edge_parameters13(box, bod3, bod2, bomd1)
            comd3 = [[-d for d in d3] for d3 in cod3]
            triangle_bco.set_edge_parameters23(cox, cod1, cod2, comd3)
            triangle_bco.build()
            triangle_bco.assign_d3(lambda tx, td1, td2: axis_d1)
            core_octant.set_triangle_bco(triangle_bco)

            core_octant.build_interior()

        if core_octant and (octant is not core_octant):
            octant.copy_parameters(core_octant)
        if self._shell_count:
            octant.set_linear_shell(self._shell_count, triangle_abc, outer_triangle_abc)
        elif not core_octant:
            octant.set_triangle_abc(triangle_abc)

        return octant

    def merge_octant_plus1_quadrant(self, octant: HexTetrahedronMesh, quadrant: int):
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
        assert 0 <= ext_count2 < (half_counts[1] - self._rim_count)
        assert 0 <= ext_count3 < (half_counts[2] - self._rim_count)
        obox_counts = octant.get_box_counts()
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
                rim3 = (n3 < self._rim_count) or (n3 > (self._element_counts[2] - self._rim_count))
                rim2 = (n2 < self._rim_count) or (n2 > (self._element_counts[1] - self._rim_count))
                # bottom_transition = n3 < self._rim_count
                nx_row = self._nx[n3][n2]
                obox_row23 = (o2 <= obox_counts[1]) and (o3 <= obox_counts[2])
                for o1 in range(axis_counts[0] + 1):
                    n1 = half_counts[0] + o1
                    rim1 = n1 > (self._element_counts[0] - self._rim_count)
                    obox = obox_row23 and (o1 <= obox_counts[0])
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
                            elif rim3:
                                if rim2:
                                    if rim1:
                                        # fix 3-way point
                                        ox = [ox[0], ox[1], add(ox[1], ox[2]), ox[3]]
                                    perm = [1, 2, 3]
                                else:
                                    perm = [-1, -2, 3]
                            else:
                                if rim1 and not rim2:
                                    perm = [-2, 1, 3]
                                else:
                                    perm = [1, 2, 3]
                            if quadrant == 3:
                                perm = [perm[0], -perm[1], -perm[2]] if obox else [-perm[0], -perm[1], perm[2]]
                        d1, d2, d3 = [copy.copy(ox[i]) if (i > 0) else [-d for d in ox[-i]] for i in perm]
                        new_nx = [x, d1, d2, d3]
                        # merge:
                        nx = nx_row[n1]
                        xi = 1.0 / (self._merge_counts[n3][n2][n1] + 1)
                        for i in range(4):
                            d = new_nx[i]
                            if i and nx[i]:
                                # blend derivatives with harmonic mean magnitude; should already be in same direction
                                d = linearlyInterpolateVectors(
                                    nx[i], d, xi, magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
                            nx[i] = d
                        self._merge_counts[n3][n2][n1] += 1

    def merge_octant_minus3_quadrant(self, octant: HexTetrahedronMesh, quadrant: int):
        """
        Merge octant parameters into ellipsoid in one of the 4 quadrants on -axis3.
        :param octant: HexTetrahedronMesh
        :param quadrant: 0 for +axis1, -axis2 increasing anticlockwise around -axis3 up to 3.
        """
        assert 0 <= quadrant <= 3
        half_counts = [count // 2 for count in self._element_counts]
        axis_counts = octant.get_axis_counts()
        even_quadrant = quadrant in (0, 2)
        assert half_counts[2] == axis_counts[2]
        obox_counts = octant.get_box_counts()
        box_counts = [half_counts[i] - self._rim_count for i in range(3)]
        if even_quadrant:
            assert box_counts == obox_counts
        else:
            assert box_counts == [obox_counts[1], obox_counts[0], obox_counts[2]]
        octant_parameters = octant.get_parameters()

        for o3 in range(axis_counts[2] + 1):
            n3 = half_counts[2] - o3
            obox_layer = (o3 <= obox_counts[2])
            ox_layer = octant_parameters[o3]
            nx_layer = self._nx[n3]
            rim3 = n3 < self._rim_count
            for o2 in range(axis_counts[1] + 1):
                ox_row = ox_layer[o2]
                for o1 in range(axis_counts[0] + 1):
                    if even_quadrant:
                        if quadrant == 0:
                            n1 = half_counts[0] + o1
                            n2 = half_counts[1] - o2
                        else:  # quadrant == 2:
                            n1 = half_counts[0] - o1
                            n2 = half_counts[1] + o2
                    else:
                        if quadrant == 1:
                            n1 = half_counts[0] - o2
                            n2 = half_counts[1] - o1
                        else:  # if quadrant == 3:
                            n1 = half_counts[0] + o2
                            n2 = half_counts[1] + o1
                    rim1 = (n1 < self._rim_count) or (n1 > (self._element_counts[0] - self._rim_count))
                    rim2 = (n2 < self._rim_count) or (n2 > (self._element_counts[1] - self._rim_count))
                    obox = obox_layer and (o1 <= obox_counts[0]) and (o2 <= obox_counts[1])
                    ox = ox_row[o1]
                    if ox and ox[0]:
                        x = copy.copy(ox[0])
                        perm = None
                        if even_quadrant:
                            if quadrant == 0:
                                if obox:
                                    perm = [1, -2, -3]
                                else:
                                    perm = [-1, -2, 3]
                            else:  # quadrant == 2:
                                if obox:
                                    perm = [-1, 2, -3]
                                elif rim3:
                                    perm = [1, 2, 3]
                                else:
                                    perm = [-1, -2, 3]
                        else:
                            if quadrant == 1:
                                if obox:
                                    perm = [-2, -1, -3]
                                elif rim3:
                                    perm = [2, -1, 3]
                                else:
                                    perm = [-1, -2, 3]
                            else:  # quadrant == 3:
                                if obox:
                                    perm = [2, 1, -3]
                                elif rim3:
                                    perm = [-2, 1, 3]
                                else:
                                    perm = [-1, -2, 3]
                        d1, d2, d3 = [copy.copy(ox[i]) if (i > 0) else [-d for d in ox[-i]] for i in perm]
                        new_nx = [x, d1, d2, d3]
                        # merge:
                        nx = nx_layer[n2][n1]
                        xi = 1.0 / (self._merge_counts[n3][n2][n1] + 1)
                        for i in range(4):
                            d = new_nx[i]
                            if i and nx[i]:
                                # blend derivatives with harmonic mean magnitude; should already be in same direction
                                d = linearlyInterpolateVectors(
                                    nx[i], d, xi, magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
                            nx[i] = d
                        self._merge_counts[n3][n2][n1] += 1

    def merge_octant_plus3_quadrant(self, octant: HexTetrahedronMesh, quadrant: int):
        """
        Merge octant parameters into ellipsoid in one of the 4 quadrants on +axis3.
        :param octant: HexTetrahedronMesh
        :param quadrant: 0 for +axis1, +axis2 increasing anticlockwise around +axis3 up to 3.
        """
        assert 0 <= quadrant <= 3
        half_counts = [count // 2 for count in self._element_counts]
        axis_counts = octant.get_axis_counts()
        even_quadrant = quadrant in (0, 2)
        assert half_counts[2] == axis_counts[2]
        obox_counts = octant.get_box_counts()
        box_counts = [half_counts[i] - self._rim_count for i in range(3)]
        if even_quadrant:
            assert box_counts == obox_counts
        else:
            assert box_counts == [obox_counts[1], obox_counts[0], obox_counts[2]]
        octant_parameters = octant.get_parameters()

        for o3 in range(axis_counts[2] + 1):
            n3 = half_counts[2] + o3
            obox_layer = (o3 <= obox_counts[2])
            ox_layer = octant_parameters[o3]
            nx_layer = self._nx[n3]
            rim3 = n3 > (self._element_counts[2] - self._rim_count)
            for o2 in range(axis_counts[1] + 1):
                ox_row = ox_layer[o2]
                for o1 in range(axis_counts[0] + 1):
                    if even_quadrant:
                        if quadrant == 0:
                            n1 = half_counts[0] + o1
                            n2 = half_counts[1] + o2
                        else:  # quadrant == 2:
                            n1 = half_counts[0] - o1
                            n2 = half_counts[1] - o2
                    else:
                        if quadrant == 1:
                            n1 = half_counts[0] - o2
                            n2 = half_counts[1] + o1
                        else:  # if quadrant == 3:
                            n1 = half_counts[0] + o2
                            n2 = half_counts[1] - o1
                    rim1 = (n1 < self._rim_count) or (n1 > (self._element_counts[0] - self._rim_count))
                    rim2 = (n2 < self._rim_count) or (n2 > (self._element_counts[1] - self._rim_count))
                    obox = obox_layer and (o1 <= obox_counts[0]) and (o2 <= obox_counts[1])
                    ox = ox_row[o1]
                    if ox and ox[0]:
                        x = copy.copy(ox[0])
                        perm = None
                        if even_quadrant:
                            if quadrant == 0:
                                perm = [1, 2, 3]
                            else:  # quadrant == 2:
                                if obox:
                                    perm = [-1, -2, 3]
                                elif rim3:
                                    perm = [-1, -2, 3]
                                else:
                                    perm = [1, 2, 3]
                        else:
                            if quadrant == 1:
                                if obox:
                                    perm = [-2, 1, 3]
                                elif rim3:
                                    perm = [-2, 1, 3]
                                else:
                                    perm = [1, 2, 3]
                            else:  # quadrant == 3:
                                if obox:
                                    perm = [2, -1, 3]
                                elif rim3:
                                    perm = [2, -1, 3]
                                else:
                                    perm = [1, 2, 3]
                        d1, d2, d3 = [copy.copy(ox[i]) if (i > 0) else [-d for d in ox[-i]] for i in perm]
                        new_nx = [x, d1, d2, d3]
                        # merge:
                        nx = nx_layer[n2][n1]
                        xi = 1.0 / (self._merge_counts[n3][n2][n1] + 1)
                        for i in range(4):
                            d = new_nx[i]
                            if i and nx[i]:
                                # blend derivatives with harmonic mean magnitude; should already be in same direction
                                d = linearlyInterpolateVectors(
                                    nx[i], d, xi, magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
                            nx[i] = d
                        self._merge_counts[n3][n2][n1] += 1

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

    def get_node_layout(self, n1, n2, n3, generate_data):
        """
        Get node layout to use at a node location in the ellipsoid. Full 3-D only.
        :param n1: Index along axis 1.
        :param n2: Index along axis 2.
        :param n3: Index along axis 3.
        :return: Node layout or None.
        """
        mesh_dimension = generate_data.getMeshDimension()
        linear_through_shell = generate_data.isLinearThroughShell()
        d3_defined = (mesh_dimension == 3) and not linear_through_shell
        node_layout_manager = generate_data.getHermiteNodeLayoutManager()
        getShellNodeLayout3WayPoints12 = node_layout_manager.getNodeLayout3WayPoints12 if d3_defined \
            else node_layout_manager.getNodeLayout3WayPoints12_no_d3
        rim_count = self._rim_count
        rn1 = self._element_counts[0] - n1
        rn2 = self._element_counts[1] - n2
        rn3 = self._element_counts[2] - n3
        # bottom transition: 3-way along triple diagonals, permuted on double diagonals
        if n3 < rim_count:
            if n1 == n3:
                if n1 == n2:
                    return getShellNodeLayout3WayPoints12(1)
                if n1 == rn2:
                    return getShellNodeLayout3WayPoints12(3)
                if (n2 > rim_count) and (rn2 > rim_count):
                    return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
            elif rn1 == n3:
                if rn1 == n2:
                    return getShellNodeLayout3WayPoints12(0)
                if rn1 == rn2:
                    return getShellNodeLayout3WayPoints12(2)
                if (n2 > rim_count) and (rn2 > rim_count):
                    return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
            elif ((n2 == n3) or (rn2 == n3)) and (n1 > rim_count) and (rn1 > rim_count):
                return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
            return None
        # top transition: 3-way along triple diagonals, permuted on double diagonals
        if rn3 < rim_count:
            if n1 == rn3:
                if n1 == n2:
                    return getShellNodeLayout3WayPoints12(0)
                if n1 == rn2:
                    return getShellNodeLayout3WayPoints12(2)
                if (n2 > rim_count) and (rn2 > rim_count):
                    return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
            elif rn1 == rn3:
                if rn1 == n2:
                    return getShellNodeLayout3WayPoints12(1)
                if rn1 == rn2:
                    return getShellNodeLayout3WayPoints12(3)
                if (n2 > rim_count) and (rn2 > rim_count):
                    return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
            elif ((n2 == rn3) or (rn2 == rn3)) and (n1 > rim_count) and (rn1 > rim_count):
                return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
            return None
        # nodes on bottom of n3 box are 4-way on corners, 3-way on edges, permuted in between
        if n3 == rim_count:
            if n2 == rim_count:
                if n1 == rim_count:
                    return node_layout_manager.getNodeLayout4WayPoints(1 if self._tube_core_box_layout else 0)
                if rn1 == rim_count:
                    return node_layout_manager.getNodeLayout4WayPoints(0 if self._tube_core_box_layout else 1)
                if (n1 > rim_count) and (rn1 > rim_count):
                    return node_layout_manager.getNodeLayout3WayPoints23(0)
            elif rn2 == rim_count:
                if n1 == rim_count:
                    return node_layout_manager.getNodeLayout4WayPoints(5 if self._tube_core_box_layout else 2)
                if rn1 == rim_count:
                    return node_layout_manager.getNodeLayout4WayPoints(4 if self._tube_core_box_layout else 3)
                if (n1 > rim_count) and (rn1 > rim_count):
                    return node_layout_manager.getNodeLayout3WayPoints23(2 if self._tube_core_box_layout else 1)
            elif (n2 > rim_count) and (rn2 > rim_count):
                if n1 == rim_count:
                    return (node_layout_manager.getNodeLayout3WayPoints12(1) if self._tube_core_box_layout else
                            node_layout_manager.getNodeLayout3WayPoints13(0))
                if rn1 == rim_count:
                    return (node_layout_manager.getNodeLayout3WayPoints12(0) if self._tube_core_box_layout else
                            node_layout_manager.getNodeLayout3WayPoints13(1))
                if (n1 > rim_count) and (rn1 > rim_count):
                    return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
            return None
        # nodes on top of n3 box are 4-way on corners, 3-way on edges, permuted in between
        if rn3 == rim_count:
            if n2 == rim_count:
                if n1 == rim_count:
                    return node_layout_manager.getNodeLayout4WayPoints(3 if self._tube_core_box_layout else 4)
                if rn1 == rim_count:
                    return node_layout_manager.getNodeLayout4WayPoints(2 if self._tube_core_box_layout else 5)
                if (n1 > rim_count) and (rn1 > rim_count):
                    return node_layout_manager.getNodeLayout3WayPoints23(1 if self._tube_core_box_layout else 2)
            elif rn2 == rim_count:
                if n1 == rim_count:
                    return node_layout_manager.getNodeLayout4WayPoints(7 if self._tube_core_box_layout else 6)
                elif rn1 == rim_count:
                    return node_layout_manager.getNodeLayout4WayPoints(6 if self._tube_core_box_layout else 7)
                if (n1 > rim_count) and (rn1 > rim_count):
                    return node_layout_manager.getNodeLayout3WayPoints23(3)
            elif (n2 > rim_count) and (rn2 > rim_count):
                if n1 == rim_count:
                    return (node_layout_manager.getNodeLayout3WayPoints12(3) if self._tube_core_box_layout else
                            node_layout_manager.getNodeLayout3WayPoints13(2))
                if rn1 == rim_count:
                    return (node_layout_manager.getNodeLayout3WayPoints12(2) if self._tube_core_box_layout else
                            node_layout_manager.getNodeLayout3WayPoints13(3))
                if (n1 > rim_count) and (rn1 > rim_count):
                    return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
            return None
        # middle between +/- rim count, 3-way points on corners, permuted on faces
        if n2 == rim_count:
            if n1 == rim_count:
                return (node_layout_manager.getNodeLayout3WayPoints13(1) if self._tube_core_box_layout else
                        node_layout_manager.getNodeLayout3WayPoints12(0))
            if rn1 == rim_count:
                return (node_layout_manager.getNodeLayout3WayPoints13(0) if self._tube_core_box_layout else
                        node_layout_manager.getNodeLayout3WayPoints12(1))
            if (n1 > rim_count) and (rn1 > rim_count):
                return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
        elif rn2 == rim_count:
            if n1 == rim_count:
                return (node_layout_manager.getNodeLayout3WayPoints13(3) if self._tube_core_box_layout else
                        node_layout_manager.getNodeLayout3WayPoints12(2))
            if rn1 == rim_count:
                return (node_layout_manager.getNodeLayout3WayPoints13(2) if self._tube_core_box_layout else
                        node_layout_manager.getNodeLayout3WayPoints12(3))
            if (n1 > rim_count) and (rn1 > rim_count):
                return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
        elif ((n1 == rim_count) or (rn1 == rim_count)) and (n2 > rim_count) and (rn2 > rim_count):
            return node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=d3_defined)
        return None

    def get_node_identifier(self, n1, n2, n3):
        """
        Get node identifier for a node in the ellipsoid.
        :param n1: Index along axis 1.
        :param n2: Index along axis 2.
        :param n3: Index along axis 3.
        :return: Node identifier or None.
        """
        return self._nids[n3][n2][n1]

    def get_node_parameters(self, n1, n2, n3):
        """
        Get parameters for a node in the ellipsoid.
        :param n1: Index along axis 1.
        :param n2: Index along axis 2.
        :param n3: Index along axis 3.
        :return: x, d1, d2, d3, or None if not a valid node location.
        """
        return self._nx[n3][n2][n1]

    def prescribe_node_layout(self, n1, n2, n3, node_layout):
        """
        Set a node layout for the node at the supplied indexes.
        :param n1: Index of node in 1-direction.
        :param n2: Index of node in 2-direction.
        :param n3: Index of node in 3-direction.
        :param node_layout: NodeLayout to use at that location.
        """
        # GRC inefficient for large mesh:
        for i in range(len(self._prescribed_node_layouts)):
            tn1, tn2, tn3, _ = self._prescribed_node_layouts[i]
            if (tn1 == n1) and (tn2 == n2) and (tn3 == n3):
                self._prescribed_node_layouts[i] = (n1, n2, n3, node_layout)
                break
        else:
            self._prescribed_node_layouts.append((n1, n2, n3, node_layout))

    def prescribe_node_layout_if_new(self, n1, n2, n3, node_layout):
        """
        Set a node layout for the node at the supplied indexes, if not set already.
        :param n1: Index of node in 1-direction.
        :param n2: Index of node in 2-direction.
        :param n3: Index of node in 3-direction.
        :param node_layout: NodeLayout to use at that location.
        """
        # GRC inefficient for large mesh:
        for tn1, tn2, tn3, _ in self._prescribed_node_layouts:
            if (tn1 == n1) and (tn2 == n2) and (tn3 == n3):
                break
        else:
            self._prescribed_node_layouts.append((n1, n2, n3, node_layout))

    def set_node_parameters(self, n1, n2, n3, parameters, nid=None, node_layout=None):
        """
        Prescribe parameters, node identifier and layout at location in mesh.
        :param n1: Index of node in 1-direction.
        :param n2: Index of node in 2-direction.
        :param n3: Index of node in 3-direction.
        :param parameters: List [[x][d1][d2][d3 or None]]
        :param nid: Identifier of node to use at this location, or None if not created yet.
        :param node_layout: Optional NodeLayout for the supplied node or to be created here, or None to not set.
        """
        assert 0 <= n1 <= self._element_counts[0]
        assert 0 <= n2 <= self._element_counts[1]
        assert 0 <= n3 <= self._element_counts[2]
        assert self._nx[n3][n2][n1] is not None
        assert self._nids[n3][n2][n1] is None
        self._nx[n3][n2][n1] = copy.deepcopy(parameters)
        self._nids[n3][n2][n1] = nid
        if node_layout:
            self._prescribed_node_layouts.append((n1, n2, n3, node_layout))

    def get_box_node_identifiers12(self, n3):
        """
        Get layer of box node identifiers.
        :param n3: Layer to set ring on, must not be in n3 rim layer.
        :return: bnids[n2=minor][n1=major]
        """
        assert self._core
        assert self._rim_count <= n3 <= (self._element_counts[2] - n3)
        node_count1 = (self._element_counts[0] - 2 * self._rim_count) + 1
        node_count2 = (self._element_counts[1] - 2 * self._rim_count) + 1
        n1_start = self._rim_count
        n2_start = self._rim_count
        n1_limit = n1_start + node_count1
        n2_limit = n2_start + node_count2
        bnids = []
        for n2 in range(n2_start, n2_limit):
            bnids_row = []
            nids_row = self._nids[n3][n2]
            for n1 in range(n1_start, n1_limit):
                bnids_row.append(nids_row[n1])
            bnids.append(bnids_row)
        return bnids

    def set_box_node_parameters12(self, generate_data, n3, bparameters, bnids, node_layout=None):
        """
        Bulk set a slice of box node parameters, known node identifiers and node layouts.
        :param generate_data: MeshGenerateData with region, field, node/element identifier and node layout data.
        :param n3: Layer to set ring on, must not be in n3 rim layer.
        :param bparameters: Slice of box parameters [n2][n1][4: x, d1, d2, d3].
        :param bnids: Node identifiers across box, or None if none.
        :param node_layout: Optional NodeLayout for the supplied node identifiers. Not set for any node identifier
        or location which already has a node layout.
        """
        assert self._core
        assert self._rim_count <= n3 <= (self._element_counts[2] - n3)
        node_count1 = (self._element_counts[0] - 2 * self._rim_count) + 1
        node_count2 = (self._element_counts[1] - 2 * self._rim_count) + 1
        assert len(bparameters) == node_count2
        assert len(bparameters[0]) == node_count1
        assert len(bparameters[0][0]) == 4
        n1_start = self._rim_count
        n2_start = self._rim_count
        n1_limit = n1_start + node_count1
        n2_limit = n2_start + node_count2
        for i2, n2 in enumerate(range(n2_start, n2_limit)):
            sd = bparameters[i2]
            td = self._nx[n3][n2]
            for i1, n1 in enumerate(range(n1_start, n1_limit)):
                for i in range(4):
                    td[n1][i] = copy.copy(sd[i1][i])
                if bnids:
                    nid = bnids[i2][i1]
                    self._nids[n3][n2][n1] = nid
                else:
                    nid = None
                if node_layout:
                    # don't change an existing node layout
                    if nid:
                        generate_data.setNodeLayoutIfNew(nid, node_layout)
                    else:
                        self.prescribe_node_layout_if_new(n1, n2, n3, node_layout)

    def set_box_element_identifier(self, e1, e2, e3, element_identifier):
        """
        Set element node identifier in core box.
        :param e1: Index along axis 1. Must be a valid box e1.
        :param e2: Index along axis 2. Must be a valid box e2.
        :param e3: Index along axis 3. Must be a valid box e3.
        :param element_identifier: Element identifier to set.
        """
        self._eids[e3][e2][e1] = element_identifier

    def _get_rim_node_indexes12_list(self, ri):
        """
        Get list of (n1, n2) indexes around a ring.
        :param ri: Rim index where 0 is first layer outside core box.
        :return: List of (n1, n2) for each point around the 12 ring.
        """
        rim_count = self._rim_count
        assert 0 <= ri < rim_count
        half_counts = [count // 2 for count in self._element_counts]
        box_counts = [half_counts[i] - rim_count for i in range(3)]
        n1_lower = rim_count - ri - 1
        n1_upper = half_counts[0] + box_counts[0] + ri + 1
        n2_lower = n1_lower
        n2_upper = half_counts[1] + box_counts[1] + ri + 1
        n1_n2_list = (
            [(n1_upper, half_counts[1] + i) for i in range(box_counts[1])] +
            [(n1_upper, n2_upper)] +
            [(half_counts[0] + box_counts[0] - i - 1, n2_upper) for i in range(2 * box_counts[0] - 1)] +
            [(n1_lower, n2_upper)] +
            [(n1_lower, half_counts[1] + box_counts[1] - i - 1) for i in range(2 * box_counts[1] - 1)] +
            [(n1_lower, n2_lower)] +
            [(rim_count + i + 1, n2_lower) for i in range(2 * box_counts[0] - 1)] +
            [(n1_upper, n2_lower)] +
            [(n1_upper, rim_count + i + 1) for i in range(box_counts[1] - 1)])
        return n1_n2_list

    def _get_rim_node_indexes123(self, n3_box, ri, ai):
        """
        Get indexes of point around axis 3 on rim layer.
        :param n3_box: Index along axis 3. Must be a valid box n3.
        :param ri: Rim index where 0 is first layer outside core box.
        :param ai: Index around axis 3.
        :return: n1, n2, n3
        """
        rim_count = self._rim_count
        assert rim_count <= n3_box <= (self._element_counts[2] - rim_count)
        assert 0 <= ri < rim_count
        assert ai >= 0
        rp = ri + 1
        half_counts = [count // 2 for count in self._element_counts]
        box_counts = [half_counts[i] - rim_count for i in range(3)]
        n3 = n3_box
        if n3 == rim_count:
            n3 -= rp
        elif n3 == (self._element_counts[2] - rim_count):
            n3 += rp

        # start half way along first straight
        a = ai
        a_limit = box_counts[1]
        if a <= a_limit:
            n1 = self._element_counts[0] - rim_count + rp
            n2 = half_counts[1] + a
            if a == a_limit:
                n2 += rp
            return n1, n2, n3

        a -= a_limit
        a_limit = 2 * box_counts[0]
        if a <= a_limit:
            n1 = self._element_counts[0] - rim_count - a
            if a == a_limit:
                n1 -= rp
            n2 = self._element_counts[1] - rim_count + rp
            return n1, n2, n3

        a -= a_limit
        a_limit = 2 * box_counts[1]
        if a <= a_limit:
            n1 = rim_count - rp
            n2 = self._element_counts[1] - rim_count - a
            if a == a_limit:
                n2 -= rp
            return n1, n2, n3

        a -= a_limit
        a_limit = 2 * box_counts[0]
        if a <= a_limit:
            n1 = rim_count + a
            if a == a_limit:
                n1 += rp
            n2 = rim_count - rp
            return n1, n2, n3

        # remainder of first straight
        a -= a_limit
        a_limit = box_counts[1]
        assert a < a_limit
        n1 = self._element_counts[0] - rim_count + rp
        n2 = rim_count + a
        return n1, n2, n3

    def get_rim_node_identifiers12(self, n3, ri):
        """
        Get single ring of rim node node identifiers.
        :param n3: Layer to set ring on, must not be in n3 rim layer.
        :param ri: Rim index where 0 is first layer outside core box.
        :return: rnids[around count]
        """
        assert self._rim_count <= n3 <= (self._element_counts[2] - n3)
        rnids = []
        n1_n2_list = self._get_rim_node_indexes12_list(ri)
        for n1, n2 in n1_n2_list:
            rnids.append(self._nids[n3][n2][n1])
        return rnids

    def get_rim_parameters(self, n3_box, ri, ai, transform=False):
        """
        Get rim parameters from polar indexes around and radially from axis3.
        :param n3_box: Index along axis 3. Must be a valid box n3.
        :param ri: Rim index where 0 is first transition, or first shell row if no core
        :param ai: Index around rim starting at +axis1
        :param transform: If True, transform dervatives on start/end 3-way slice into tube rim orientation (from dome).
        :return: x, d1, d2, d3
        """
        n1, n2, n3 = self._get_rim_node_indexes123(n3_box, ri, ai)
        x, d1, d2, d3 = self._nx[n3][n2][n1]
        if transform and ((n3 < self._rim_count) or (n3 > (self._element_counts[2] - self._rim_count))):
            # only the end rim row needs transforming into tube rim orientation i.e. on the 3-way slice
            rn1 = self._element_counts[0] - n1
            rn2 = self._element_counts[1] - n2
            td1 = [0.0, 0.0, 0.0]
            td2 = [0.0, 0.0, 0.0]
            if n1 < self._rim_count:
                td1 = [-d for d in d2]
                td2 = d1
            elif n1 > (self._element_counts[0] - self._rim_count):
                td1 = d2
                td2 = [-d for d in d1]
            if n3 < self._rim_count:
                if n2 < self._rim_count:
                    td1 = sub(td1, d1)
                    td2 = sub(td2, d2)
                elif n2 > (self._element_counts[1] - self._rim_count):
                    td1 = add(td1, d1)
                    td2 = add(td2, d2)
            else:  # if (3 > (self._element_counts[2] - self._rim_count)
                if n2 < self._rim_count:
                    td1 = add(td1, d1)
                    td2 = add(td2, d2)
                elif n2 > (self._element_counts[1] - self._rim_count):
                    td1 = sub(td1, d1)
                    td2 = sub(td2, d2)
            d1, d2 = td1, td2
        return x, d1, d2, d3

    def get_rim_node_identifier(self, n3_box, ri, ai):
        """
        Get rim node identifier from polar indexes around and radially from axis3.
        :param n3_box: Index along axis 3. Must be a valid box n3.
        :param ri: Rim index where 0 is first transition, or first shell row if no core
        :param ai: Index around rim starting at +axis1
        :return: Node identifier, or None if not set.
        """
        n1, n2, n3 = self._get_rim_node_indexes123(n3_box, ri, ai)
        return self._nids[n3][n2][n1]

    def get_rim_node_layout(self, generate_data, n3_box, ri, ai):
        """
        Get rim node layout from polar indexes around and radially from axis3.
        :param generate_data: MeshGenerateData with region, field, node/element identifier and node layout data.
        :param n3_box: Index along axis 3. Must be a valid box n3.
        :param ri: Rim index where 0 is first transition, or first shell row if no core
        :param ai: Index around rim starting at +axis1
        :return: HermiteNodeLayout, or None if not used at that location.
        """
        n1, n2, n3 = self._get_rim_node_indexes123(n3_box, ri, ai)
        return self.get_node_layout(n1, n2, n3, generate_data)

    def set_rim_node_parameters12(self, generate_data, n3, ri, rparameters, rnids, node_layout=None):
        """
        Bulk set a ring of node parameters and known node identifiers starting at +axis1 and heading towards +axis2.
        :param generate_data: MeshGenerateData with region, field, node/element identifier and node layout data.
        :param n3: Layer to set ring on, must not be in n3 rim layer.
        :param ri: ring index where 0 is first layer outside core box.
        :param rparameters: Parameters [around][4: x, d1, d2, d3].
        :param rnids: Node identifiers around ring, or None if none
        :param node_layout: Optional NodeLayout for the supplied node identifiers. Not set for any node identifier
        or location which already has a node layout.
        """
        assert self._rim_count <= n3 <= (self._element_counts[2] - n3)
        half_counts = [count // 2 for count in self._element_counts]
        box_counts = [half_counts[i] - self._rim_count for i in range(3)]
        around_count = 4 * (box_counts[0] + box_counts[1])
        assert len(rparameters) == around_count
        assert all((rd is None or (len(rd) == 4)) for rd in rparameters)
        assert not rnids or (len(rnids) == around_count)
        n1_n2_list = self._get_rim_node_indexes12_list(ri)
        for ai, n1_n2 in enumerate(n1_n2_list):
            n1, n2 = n1_n2
            for i in range(4):
                self._nx[n3][n2][n1][i] = copy.copy(rparameters[ai][i])
            nid = rnids[ai] if rnids else None
            if nid:
                self._nids[n3][n2][n1] = rnids[ai]
            if node_layout:
                if nid:
                    generate_data.setNodeLayoutIfNew(rnids[ai], node_layout)
                else:
                    self.prescribe_node_layout_if_new(n1, n2, n3, node_layout)

    def _get_rim_element_indexes12(self, e3, ri, ai):
        """
        Get indexes of an element around axis 3 on rim layer.
        :param e3: Index along axis 3. Must be a valid box e3.
        :param ri: Rim index where 0 is first layer outside core box i.e. transition layer.
        :param ai: Index around axis 3, starting from +axis1 on +axis2 side.
        :return: e1, e2
        """
        rim_count = self._rim_count
        assert rim_count <= e3 < (self._element_counts[2] - rim_count)
        assert 0 <= ri < rim_count
        assert ai >= 0
        half_counts = [count // 2 for count in self._element_counts]
        box_counts = [half_counts[i] - rim_count for i in range(3)]

        # start half way along first straight
        a = ai
        a_limit = box_counts[1]
        if a < a_limit:
            e1 = self._element_counts[0] - rim_count + ri
            e2 = half_counts[1] + a
            return e1, e2

        a -= a_limit
        a_limit = 2 * box_counts[0]
        if a < a_limit:
            e1 = self._element_counts[0] - rim_count - a - 1
            e2 = self._element_counts[1] - rim_count + ri
            return e1, e2

        a -= a_limit
        a_limit = 2 * box_counts[1]
        if a < a_limit:
            e1 = rim_count - ri - 1
            e2 = self._element_counts[1] - rim_count - a - 1
            return e1, e2

        a -= a_limit
        a_limit = 2 * box_counts[0]
        if a < a_limit:
            e1 = rim_count + a
            e2 = rim_count - ri - 1
            return e1, e2

        # remainder of first straight
        a -= a_limit
        a_limit = box_counts[1]
        assert a < a_limit
        e1 = self._element_counts[0] - rim_count + ri
        e2 = rim_count + a
        return e1, e2

    def get_rim_element_identifier(self, e3, ri, ai):
        """
        Get rim element identifier from polar indexes around and radially from axis3.
        :param e3: Index along axis 3. Must be a valid box e3.
        :param ri: Rim index where 0 is first transition, or first shell row if no core
        :param ai: Index around rim starting at +axis1
        :return: Element identifier, or None if not set.
        """
        e1, e2 = self._get_rim_element_indexes12(e3, ri, ai)
        return self._eids[e3][e2][e1]

    def set_rim_element_identifier(self, e3, ri, ai, element_identifier):
        """
        Set element node identifier from polar indexes around and radially from axis3.
        :param e3: Index along axis 3. Must be a valid box e3.
        :param ri: Rim index where 0 is first transition, or first shell row if no core
        :param ai: Index around rim starting at +axis1
        :param element_identifier: Element identifier to set.
        """
        e1, e2 = self._get_rim_element_indexes12(e3, ri, ai)
        self._eids[e3][e2][e1] = element_identifier

    def generate_nodes(self, generate_data, n3_start=0, n3_limit=0):
        """
        After coordinates have been calculated with e.g. build(), generate nodes of ellipsoid.
        Allows nodes to be built for just a range of n3 indexes.
        Client is expected to run within ChangeManager(generate_data.getFieldmodule()).
        Note: If exactly one n3 is being generated and it is the last box layer, indexing proceeds diagonally
        into the corners of the box.
        :param generate_data: MeshGenerateData with region, field, node/element identifier and node layout data.
        :param n3_start: First n3 index in (0, self._element_counts[2]), or default to 0.
        :param n3_limit: Limit n3 index in (n3_start + 1, self._element_counts[2] + 1), or default to max limit.
        """
        assert 0 <= n3_start <= self._element_counts[2]
        if n3_limit:
            assert n3_start < n3_limit <= (self._element_counts[2] + 1)
        else:
            n3_limit = self._element_counts[2] + 1
        # work out whether on diagonal line
        box_diagonal = (n3_limit == (n3_start + 1)) and (
                (n3_start == self._rim_count) or (n3_start == (self._element_counts[2] - self._rim_count)))

        mesh_dimension = generate_data.getMeshDimension()
        linear_through_shell = generate_data.isLinearThroughShell()
        d3_defined = (mesh_dimension == 3) and not linear_through_shell

        fieldcache = generate_data.getFieldcache()
        coordinates = generate_data.getCoordinates()
        nodes = generate_data.getNodes()
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        value_labels = [Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2]
        if d3_defined:
            value_labels.append(Node.VALUE_LABEL_D_DS3)
        for value_label in value_labels:
            nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)

        for n3 in range(n3_start, n3_limit):
            n3_box = self._rim_count <= n3 <= (self._element_counts[2] - self._rim_count)
            for n2 in range(self._element_counts[1] + 1):
                n2_box = self._rim_count <= n2 <= (self._element_counts[1] - self._rim_count)
                for n1 in range(self._element_counts[0] + 1):
                    n1_box = self._rim_count <= n1 <= (self._element_counts[0] - self._rim_count)
                    n3_mod = n3
                    if box_diagonal:
                        # get rim index, 1 is first layer outside box
                        ri = 0
                        if n2 < self._rim_count:
                            ri = self._rim_count - n2
                        elif n2 > (self._element_counts[1] - self._rim_count):
                            ri = n2 - (self._element_counts[1] - self._rim_count)
                        elif n1 < self._rim_count:
                            ri = self._rim_count - n1
                        elif n1 > (self._element_counts[0] - self._rim_count):
                            ri = n1 - (self._element_counts[0] - self._rim_count)
                        if ri:
                            if n3_start == self._rim_count:
                                n3_mod -= ri
                            else:
                                n3_mod += ri
                    if self._nids[n3_mod][n2][n1] is not None:
                        continue  # existing node
                    parameters = self._nx[n3_mod][n2][n1]
                    if not parameters:
                        continue  # unusable location
                    x, d1, d2, d3 = parameters
                    if not x:
                        continue  # while in development
                    node_identifier = generate_data.nextNodeIdentifier()
                    node = nodes.createNode(node_identifier, nodetemplate)
                    self._nids[n3_mod][n2][n1] = node_identifier
                    fieldcache.setNode(node)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                    if self._tube_core_box_layout and n1_box and n2_box and n3_box:
                        d1 = [-d for d in d1] if d1 else None
                        d2, d3 = d3, d2
                        # put parameters back so picked up by elements when determining eft
                        self._nx[n3_mod][n2][n1] = x, d1, d2, d3
                    if d1:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, d1)
                    if d2:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, d2)
                    if d3_defined and d3:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)
                    node_layout = self.get_node_layout(n1, n2, n3, generate_data)
                    if node_layout:
                        generate_data.setNodeLayoutIfNew(node_identifier, node_layout)

    def generate_elements(self, generate_data, e3_start=0, e3_limit=0):
        """
        After coordinates have been calculated with e.g. build(), and nodes have been generated,
        generate elements of ellipsoid. Allows elements to be built for just a range of e3 indexes.
        Client is expected to run within ChangeManager(generate_data.getFieldmodule()).
        :param generate_data: MeshGenerateData with region, field, node/element identifier and node layout data.
        :param e3_start: First e3 index in (0, self._element_counts[2] - 1), or default to 0.
        :param e3_limit: Limit e3 index in (e3_start + 1, self._element_counts[2]), or default to max limit.
        """
        assert 0 <= e3_start < self._element_counts[2]
        if e3_limit:
            assert e3_start < e3_limit <= self._element_counts[2]
        else:
            e3_limit = self._element_counts[2]

        fieldmodule = generate_data.getFieldmodule()
        fieldcache = generate_data.getFieldcache()
        coordinates = generate_data.getCoordinates()

        mesh_dimension = generate_data.getMeshDimension()
        surface_only = (self._shell_count == 0) and not self._core
        assert mesh_dimension == (2 if surface_only else 3)

        # set prescribed node layouts
        prescribed_node_layouts = self._prescribed_node_layouts
        self._prescribed_node_layouts = []
        for prescribed_node_layout in prescribed_node_layouts:
            n1, n2, n3, node_layout = prescribed_node_layout
            nid = self._nids[n3][n2][n1]
            if nid:
                generate_data.setNodeLayout(nid, node_layout)
            else:
                self._prescribed_node_layouts.append((n1, n2, n3, node_layout))  # might be created later

        mesh = generate_data.getMesh()
        half_counts = [count // 2 for count in self._element_counts]
        box_counts = [half_counts[i] - self._rim_count for i in range(3)]
        dbox_counts = [2 * box_counts[i] for i in range(3)]

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
        core_mesh_group = None
        shell_mesh_group = None
        if self._core:
            if self._box_group:
                box_mesh_group = self._box_group.getOrCreateMeshGroup(mesh)
            if self._transition_group:
                transition_mesh_group = self._transition_group.getOrCreateMeshGroup(mesh)
            if self._core_group:
                core_mesh_group = self._core_group.getOrCreateMeshGroup(mesh)
            if self._shell_count and self._shell_group:
                shell_mesh_group = self._shell_group.getOrCreateMeshGroup(mesh)

        if surface_only:
            node_layout_manager = generate_data.getHermiteNodeLayoutManager()
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
            rim_indexes = [[0] + [self._rim_count + 1 + j
                                  for j in range(self._element_counts[i] - 2 * self._rim_count - 1)] +
                           [self._element_counts[i]] for i in range(3)]
            if e3_start == 0:
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
                            element_identifier = generate_data.nextElementIdentifier()
                            element = mesh.createElement(element_identifier, elementtemplate_regular)
                            element.setNodesByIdentifier(eft_regular, nids)
                            e1 = self._element_counts[0] - self._rim_count - i1
                            e2 = self._rim_count + i2 - 1
                            self._eids[0][e2][e1] = element_identifier
                            # print("Element", element_identifier, "nids", nids)
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_n2 + octant_n1
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                    last_nids_row = nids_row
            # around sides
            index_increments = [[0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]]
            increment_number = 0
            index_increment = index_increments[0]
            elements_count_around12 = \
                2 * (self._element_counts[0] + self._element_counts[1] - 4 * self._rim_count)
            quarter_elements_count_around12 = elements_count_around12 // 4
            last_nids_row = None
            last_parameters_row = None
            last_corners_row = None
            for i3, n3 in enumerate(rim_indexes[2]):
                if (n3 < e3_start) and (e3_start >= rim_indexes[2][1]):
                    continue
                if (n3 > e3_limit) and (e3_limit <= rim_indexes[2][-2]):
                    continue
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
                if (n3 > e3_start) and last_nids_row:
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
                            node_layouts = [generate_data.getNodeLayout(nid) for nid in nids]
                            eft, scalefactors = \
                                determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                            elementtemplate_special.defineField(coordinates, -1, eft)
                            elementtemplate = elementtemplate_special
                        element_identifier = generate_data.nextElementIdentifier()
                        element = mesh.createElement(element_identifier, elementtemplate)
                        element.setNodesByIdentifier(eft, nids)
                        if scalefactors:
                            element.setScaleFactors(eft, scalefactors)
                        e3 = self._rim_count + i3 - 1
                        e1, e2 = self._get_rim_element_indexes12(e3, self._rim_count - 1, nc)
                        self._eids[n3 - 1][e2][e1] = element_identifier
                        # print("Element", element_identifier, "nids", nids)
                        if octant_mesh_group_lists:
                            octant = octant_n3 + octant_nc[nc]
                            for mesh_group in octant_mesh_group_lists[octant]:
                                mesh_group.addElement(element)
                last_nids_row = nids_row
                last_parameters_row = parameters_row
                last_corners_row = corners_row
            if e3_limit == self._element_counts[2]:
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
                            element_identifier = generate_data.nextElementIdentifier()
                            element = mesh.createElement(element_identifier, elementtemplate_regular)
                            element.setNodesByIdentifier(eft_regular, nids)
                            e1 = self._rim_count + i1 - 1
                            e2 = self._rim_count + i2 - 1
                            self._eids[self._element_counts[2] - 1][e2][e1] = element_identifier
                            # print("Element", element_identifier, "nids", nids)
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_n2 + octant_n1
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                    last_nids_row = nids_row
        else:
            # 3-D mesh
            linear_through_shell = generate_data.isLinearThroughShell()
            assert not (linear_through_shell and self._core)
            elementtemplate_regular = mesh.createElementtemplate()
            elementtemplate_regular.setElementShapeType(Element.SHAPE_TYPE_CUBE)
            element_basis = (
                fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY))
            if linear_through_shell:
                element_basis.setFunctionType(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
            eft_regular = mesh.createElementfieldtemplate(element_basis)
            elementtemplate_regular.defineField(coordinates, -1, eft_regular)
            elementtemplate_special = mesh.createElementtemplate()
            elementtemplate_special.setElementShapeType(Element.SHAPE_TYPE_CUBE)

            # bottom transition
            last_nids_layer = None
            last_nx_layer = None
            for nt in range(self._rim_count + 1):
                n3 = nt
                e3 = nt - 1
                octant_n3 = 0
                nids_layer = []
                nx_layer = []
                last_nids_row = None
                last_nx_row = None
                for i2 in range(dbox_counts[1] + 1):
                    n2 = (nt if (i2 == 0)
                          else (self._element_counts[1] - nt) if (i2 == dbox_counts[1])
                          else (self._rim_count + i2))
                    e2 = self._rim_count + i2 - 1
                    octant_n2 = 2 if (n2 > half_counts[1]) else 0
                    nids_row = []
                    nx_row = []
                    for i1 in range(dbox_counts[0] + 1):
                        n1 = (nt if (i1 == 0)
                              else (self._element_counts[0] - nt) if (i1 == dbox_counts[0])
                              else (self._rim_count + i1))
                        e1 = self._rim_count + i1 - 1
                        octant_n1 = 1 if (n1 > half_counts[0]) else 0
                        nids_row.append(self._nids[n3][n2][n1])
                        nx_row.append(
                            (self._nx[n3][n2][n1][:3] + [None]) if linear_through_shell else self._nx[n3][n2][n1])
                        if (n3 > e3_start) and (i2 > 0) and (i1 > 0):
                            nids = [last_nids_row[i1], last_nids_row[i1 - 1],
                                    nids_row[i1], nids_row[i1 - 1],
                                    last_nids_layer[i2 - 1][i1], last_nids_layer[i2 - 1][i1 - 1],
                                    last_nids_layer[i2][i1], last_nids_layer[i2][i1 - 1]]
                            if None in nids:
                                continue
                            elementtemplate = elementtemplate_regular
                            eft = eft_regular
                            scalefactors = None
                            node_parameters = [
                                last_nx_row[i1], last_nx_row[i1 - 1],
                                nx_row[i1], nx_row[i1 - 1],
                                last_nx_layer[i2 - 1][i1], last_nx_layer[i2 - 1][i1 - 1],
                                last_nx_layer[i2][i1], last_nx_layer[i2][i1 - 1]]
                            node_layouts = [generate_data.getNodeLayout(nid) for nid in nids]
                            if any(node_layout is not None for node_layout in node_layouts):
                                eft, scalefactors = \
                                    determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                            if self._shell_count and (nt == (self._shell_count + 1)):
                                # apply scale factors or versions on core-shell boundary
                                if eft is eft_regular:
                                    # important: modify a non-shared eft
                                    eft = mesh.createElementfieldtemplate(element_basis)
                                eft, scalefactors = resolveEftCoreBoundaryScaling(
                                    eft, scalefactors, node_parameters, nids, self._core_shell_scaling_mode,
                                    generate_data.getNodes(), generate_data.getCoordinates(),
                                    generate_data.getFieldcache())
                            if eft is not eft_regular:
                                elementtemplate_special.defineField(coordinates, -1, eft)
                                elementtemplate = elementtemplate_special
                            element_identifier = generate_data.nextElementIdentifier()
                            element = mesh.createElement(element_identifier, elementtemplate)
                            element.setNodesByIdentifier(eft, nids)
                            # print("Element", element_identifier, "nids", nids)
                            if scalefactors:
                                element.setScaleFactors(eft, scalefactors)
                            self._eids[e3][e2][e1] = element_identifier
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_n2 + octant_n1
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                            if nt > self._shell_count:
                                if transition_mesh_group:
                                    transition_mesh_group.addElement(element)
                                if core_mesh_group:
                                    core_mesh_group.addElement(element)
                            else:
                                if shell_mesh_group:
                                    shell_mesh_group.addElement(element)
                    nids_layer.append(nids_row)
                    nx_layer.append(nx_row)
                    last_nids_row = nids_row
                    last_nx_row = nx_row
                last_nids_layer = nids_layer
                last_nx_layer = nx_layer

            # middle
            upper_trans_counts = [self._element_counts[i] - self._rim_count for i in range(3)]
            last_nids_layer = None
            last_nx_layer = None
            last_rim_nids_layer = None
            last_rim_nx_layer = None
            n3_first = max(e3_start, self._rim_count)
            for i3 in range(dbox_counts[2] + 1):
                n3 = self._rim_count + i3
                e3 = n3 - 1
                if n3 < e3_start:
                    continue
                if n3 > e3_limit:
                    break
                octant_n3 = 4 if (n3 > half_counts[2]) else 0
                nids_layer = []
                nx_layer = []
                last_nids_row = None
                last_nx_row = None
                for i2 in range(dbox_counts[1] + 1):
                    n2 = self._rim_count + i2
                    e2 = n2 - 1
                    octant_n2 = 2 if (n2 > half_counts[1]) else 0
                    nids_row = []
                    nx_row = []
                    for i1 in range(dbox_counts[0] + 1):
                        n1 = self._rim_count + i1
                        e1 = n1 - 1
                        octant_n1 = 1 if (n1 > half_counts[0]) else 0
                        nids_row.append(self._nids[n3][n2][n1])
                        nx_row.append(self._nx[n3][n2][n1])
                        if (n3 > n3_first) and (i2 > 0) and (i1 > 0):
                            nids = [last_nids_layer[i2 - 1][i1 - 1], last_nids_layer[i2 - 1][i1],
                                    last_nids_layer[i2][i1 - 1], last_nids_layer[i2][i1],
                                    last_nids_row[i1 - 1], last_nids_row[i1],
                                    nids_row[i1 - 1], nids_row[i1]]
                            if None in nids:
                                continue
                            elementtemplate = elementtemplate_regular
                            eft = eft_regular
                            scalefactors = None
                            node_layouts = [generate_data.getNodeLayout(nid) for nid in nids]
                            if self._tube_core_box_layout:
                                nids = [nids[1], nids[0], nids[5], nids[4], nids[3], nids[2], nids[7], nids[6]]
                                node_layouts = [
                                    node_layouts[1], node_layouts[0], node_layouts[5], node_layouts[4],
                                    node_layouts[3], node_layouts[2], node_layouts[7], node_layouts[6]]
                            if any(node_layout is not None for node_layout in node_layouts):
                                node_parameters = [last_nx_layer[i2 - 1][i1 - 1], last_nx_layer[i2 - 1][i1],
                                                   last_nx_layer[i2][i1 - 1], last_nx_layer[i2][i1],
                                                   last_nx_row[i1 - 1], last_nx_row[i1],
                                                   nx_row[i1 - 1], nx_row[i1]]
                                if self._tube_core_box_layout:
                                    node_parameters = [
                                        node_parameters[1], node_parameters[0], node_parameters[5], node_parameters[4],
                                        node_parameters[3], node_parameters[2], node_parameters[7], node_parameters[6]]
                                eft, scalefactors = \
                                    determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                                elementtemplate_special.defineField(coordinates, -1, eft)
                                elementtemplate = elementtemplate_special
                            element_identifier = generate_data.nextElementIdentifier()
                            element = mesh.createElement(element_identifier, elementtemplate)
                            element.setNodesByIdentifier(eft, nids)
                            # print("Element", element_identifier, "nids", nids)
                            if scalefactors:
                                element.setScaleFactors(eft, scalefactors)
                            self._eids[e3][e2][e1] = element_identifier
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_n2 + octant_n1
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                            if box_mesh_group:
                                box_mesh_group.addElement(element)
                            if core_mesh_group:
                                core_mesh_group.addElement(element)
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
                for nt in range(self._rim_count + 1):
                    n3_mod = n3
                    if i3 == 0:
                        n3_mod = self._rim_count - nt
                    elif i3 == dbox_counts[2]:
                        n3_mod = upper_trans_counts[2] + nt
                    rim_nids_row = []
                    rim_nx_row = []
                    rim_eindexes_row = []
                    octant_nc = []
                    n2 = self._rim_count - nt
                    e2 = n2
                    # GRC this should really start at +axis1
                    for i1 in range(dbox_counts[0]):
                        n1 = ((self._rim_count - nt) if (i1 == 0) else (
                            (upper_trans_counts[0] + nt) if (i1 == dbox_counts[0]) else (self._rim_count + i1)))
                        e1 = self._rim_count + i1
                        rim_nids_row.append(self._nids[n3_mod][n2][n1])
                        rim_nx_row.append(self._nx[n3_mod][n2][n1])
                        rim_eindexes_row.append((e1, e2, e3))
                        octant_nc.append(1 if n1 >= half_counts[0] else 0)
                    n1 = upper_trans_counts[0] + nt
                    e1 = n1 - 1
                    for i2 in range(dbox_counts[1]):
                        n2 = ((self._rim_count - nt) if (i2 == 0) else (
                            (upper_trans_counts[1] + nt) if (i2 == dbox_counts[1]) else (self._rim_count + i2)))
                        e2 = self._rim_count + i2
                        rim_nids_row.append(self._nids[n3_mod][n2][n1])
                        rim_nx_row.append(self._nx[n3_mod][n2][n1])
                        rim_eindexes_row.append((e1, e2, e3))
                        octant_nc.append(3 if n2 >= half_counts[1] else 1)
                    n2 = upper_trans_counts[1] + nt
                    e2 = n2 - 1
                    for i1 in range(dbox_counts[0]):
                        n1 = ((upper_trans_counts[0] + nt) if (i1 == 0) else (
                            (self._rim_count - nt) if (i1 == dbox_counts[0]) else (upper_trans_counts[0] - i1)))
                        e1 = upper_trans_counts[0] - i1 - 1
                        rim_nids_row.append(self._nids[n3_mod][n2][n1])
                        rim_nx_row.append(self._nx[n3_mod][n2][n1])
                        rim_eindexes_row.append((e1, e2, e3))
                        octant_nc.append(3 if n1 > half_counts[0] else 2)
                    n1 = self._rim_count - nt
                    e1 = n1
                    for i2 in range(dbox_counts[1]):
                        n2 = ((upper_trans_counts[1] + nt) if (i2 == 0) else (
                             (self._rim_count - nt) if (i2 == dbox_counts[1]) else (upper_trans_counts[1] - i2)))
                        e2 = upper_trans_counts[1] - i2 - 1
                        rim_nids_row.append(self._nids[n3_mod][n2][n1])
                        rim_nx_row.append(self._nx[n3_mod][n2][n1])
                        rim_eindexes_row.append((e1, e2, e3))
                        octant_nc.append(2 if n2 > half_counts[1] else 0)
                    if linear_through_shell:
                        rim_nx_row = [[params[0], params[1], params[2], None] for params in rim_nx_row]
                    if (n3 > n3_first) and (nt > 0):
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
                            node_parameters = [
                                last_rim_nx_layer[nt - 1][nc], last_rim_nx_layer[nt - 1][ncp],
                                last_rim_nx_row[nc], last_rim_nx_row[ncp],
                                last_rim_nx_layer[nt][nc], last_rim_nx_layer[nt][ncp],
                                rim_nx_row[nc], rim_nx_row[ncp]]
                            scalefactors = None
                            node_layouts = [generate_data.getNodeLayout(nid) for nid in nids]
                            if any(node_layout is not None for node_layout in node_layouts):
                                eft, scalefactors = \
                                    determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                            if self._shell_count and (nt == self._transition_count):
                                # apply scale factors or versions on core-shell boundary
                                if eft is eft_regular:
                                    # important: modify a non-shared eft
                                    eft = mesh.createElementfieldtemplate(element_basis)
                                eft, scalefactors = resolveEftCoreBoundaryScaling(
                                    eft, scalefactors, node_parameters, nids, self._core_shell_scaling_mode,
                                    generate_data.getNodes(), generate_data.getCoordinates(),
                                    generate_data.getFieldcache())
                            if eft is not eft_regular:
                                elementtemplate_special.defineField(coordinates, -1, eft)
                                elementtemplate = elementtemplate_special
                            element_identifier = generate_data.nextElementIdentifier()
                            element = mesh.createElement(element_identifier, elementtemplate)
                            element.setNodesByIdentifier(eft, nids)
                            # print("Element", element_identifier, "nids", nids)
                            if scalefactors:
                                element.setScaleFactors(eft, scalefactors)
                            eindexes = rim_eindexes_row[nc]
                            self._eids[eindexes[2]][eindexes[1]][eindexes[0]] = element_identifier
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_nc[nc]
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                            if nt <= self._transition_count:
                                if transition_mesh_group:
                                    transition_mesh_group.addElement(element)
                                if core_mesh_group:
                                    core_mesh_group.addElement(element)
                            else:
                                if shell_mesh_group:
                                    shell_mesh_group.addElement(element)
                    rim_nids_layer.append(rim_nids_row)
                    rim_nx_layer.append(rim_nx_row)
                    last_rim_nids_row = rim_nids_row
                    last_rim_nx_row = rim_nx_row
                last_rim_nids_layer = rim_nids_layer
                last_rim_nx_layer = rim_nx_layer

            # top transition
            last_nids_layer = None
            last_nx_layer = None
            for nt in range(self._rim_count, -1, -1):
                n3 = self._element_counts[2] - nt
                e3 = n3 - 1
                octant_n3 = 4
                nids_layer = []
                nx_layer = []
                last_nids_row = None
                last_nx_row = None
                for i2 in range(dbox_counts[1] + 1):
                    n2 = (nt if (i2 == 0)
                          else (self._element_counts[1] - nt) if (i2 == dbox_counts[1])
                          else (self._rim_count + i2))
                    e2 = self._rim_count + i2 - 1
                    octant_n2 = 2 if (n2 > half_counts[1]) else 0
                    nids_row = []
                    nx_row = []
                    for i1 in range(dbox_counts[0] + 1):
                        n1 = (nt if (i1 == 0)
                              else (self._element_counts[0] - nt) if (i1 == dbox_counts[0])
                              else (self._rim_count + i1))
                        e1 = self._rim_count + i1 - 1
                        octant_n1 = 1 if (n1 > half_counts[0]) else 0
                        nids_row.append(self._nids[n3][n2][n1])
                        nx_row.append(
                            (self._nx[n3][n2][n1][:3] + [None]) if linear_through_shell else self._nx[n3][n2][n1])
                        if (nt < self._rim_count) and (n3 <= e3_limit) and (i2 > 0) and (i1 > 0):
                            nids = [last_nids_layer[i2 - 1][i1 - 1], last_nids_layer[i2 - 1][i1],
                                    last_nids_layer[i2][i1 - 1], last_nids_layer[i2][i1],
                                    last_nids_row[i1 - 1], last_nids_row[i1],
                                    nids_row[i1 - 1], nids_row[i1]]
                            if None in nids:
                                continue
                            elementtemplate = elementtemplate_regular
                            eft = eft_regular
                            node_parameters = [
                                last_nx_layer[i2 - 1][i1 - 1], last_nx_layer[i2 - 1][i1],
                                last_nx_layer[i2][i1 - 1], last_nx_layer[i2][i1],
                                last_nx_row[i1 - 1], last_nx_row[i1],
                                nx_row[i1 - 1], nx_row[i1]]
                            scalefactors = None
                            node_layouts = [generate_data.getNodeLayout(nid) for nid in nids]
                            if any(node_layout is not None for node_layout in node_layouts):
                                eft, scalefactors = \
                                    determineCubicHermiteSerendipityEft(mesh, node_parameters, node_layouts)
                            if self._shell_count and (nt == self._shell_count):
                                # apply scale factors or versions on core-shell boundary
                                if eft is eft_regular:
                                    # important: modify a non-shared eft
                                    eft = mesh.createElementfieldtemplate(element_basis)
                                eft, scalefactors = resolveEftCoreBoundaryScaling(
                                    eft, scalefactors, node_parameters, nids, self._core_shell_scaling_mode,
                                    generate_data.getNodes(), generate_data.getCoordinates(),
                                    generate_data.getFieldcache())
                            if eft is not eft_regular:
                                elementtemplate_special.defineField(coordinates, -1, eft)
                                elementtemplate = elementtemplate_special
                            element_identifier = generate_data.nextElementIdentifier()
                            element = mesh.createElement(element_identifier, elementtemplate)
                            element.setNodesByIdentifier(eft, nids)
                            # print("Element", element_identifier, "nids", nids)
                            if scalefactors:
                                element.setScaleFactors(eft, scalefactors)
                            self._eids[e3][e2][e1] = element_identifier
                            if octant_mesh_group_lists:
                                octant = octant_n3 + octant_n2 + octant_n1
                                for mesh_group in octant_mesh_group_lists[octant]:
                                    mesh_group.addElement(element)
                            if nt < self._shell_count:
                                if shell_mesh_group:
                                    shell_mesh_group.addElement(element)
                            else:
                                if transition_mesh_group:
                                    transition_mesh_group.addElement(element)
                                if core_mesh_group:
                                    core_mesh_group.addElement(element)
                    nids_layer.append(nids_row)
                    nx_layer.append(nx_row)
                    last_nids_row = nids_row
                    last_nx_row = nx_row
                last_nids_layer = nids_layer
                last_nx_layer = nx_layer

    def generate_mesh(self, generate_data):
        """
        After coordinates have been calculated with e.g. build(), generate nodes and elements of ellipsoid.
        Client is expected to run within ChangeManager(generate_data.getFieldmodule()).
        :param generate_data: MeshGenerateData with region, field, node/element identifier and node layout data.
        """
        self.generate_nodes(generate_data)
        self.generate_elements(generate_data)

    def add_core_elements_to_mesh_group(self, mesh_group):
        master_mesh = mesh_group.getMasterMesh()
        for e3 in range(self._shell_count, self._element_counts[2] - self._shell_count):
            eids_layer = self._eids[e3]
            for e2 in range(self._shell_count, self._element_counts[1] - self._shell_count):
                eids_row = eids_layer[e2]
                for e1 in range(self._shell_count, self._element_counts[0] - self._shell_count):
                    eid = eids_row[e1]
                    if eid:
                        element = master_mesh.findElementByIdentifier(eid)
                        mesh_group.addElement(element)

    def add_shell_elements_to_mesh_group(self, mesh_group):
        """
        Add 3-D elements in the shell to the mesh group.
        :param mesh_group: 3-D Zinc MeshGroup.
        """
        assert mesh_group.getDimension() == 3
        master_mesh = mesh_group.getMasterMesh()
        for e3 in range(self._element_counts[2]):
            e3_shell = (e3 < self._shell_count) or (e3 >= (self._element_counts[2] - self._shell_count))
            eids_layer = self._eids[e3]
            for e2 in range(self._element_counts[1]):
                e2_shell = (e2 < self._shell_count) or (e2 >= (self._element_counts[1] - self._shell_count))
                eids_row = eids_layer[e2]
                for e1 in range(self._element_counts[0]):
                    e1_shell = (e1 < self._shell_count) or (e1 >= (self._element_counts[0] - self._shell_count))
                    if e3_shell or e2_shell or e1_shell:
                        eid = eids_row[e1]
                        if eid:
                            element = master_mesh.findElementByIdentifier(eid)
                            mesh_group.addElement(element)

    def add_axis1_elements_to_mesh_group(self, side, mesh_group):
        """
        Add elements from half of ellipsoid to mesh group.
        :param side: False for -axis1, True for +axis1.
        :param mesh_group: Mesh group to add to.
        """
        master_mesh = mesh_group.getMasterMesh()
        e1_start, e1_limit = (
            (self._element_counts[0] // 2, self._element_counts[0]) if side else (0, self._element_counts[0] // 2))
        for e3 in range(self._element_counts[2]):
            eids_layer = self._eids[e3]
            for e2 in range(self._element_counts[1]):
                eids_row = eids_layer[e2]
                for e1 in range(e1_start, e1_limit):
                    eid = eids_row[e1]
                    if eid:
                        element = master_mesh.findElementByIdentifier(eid)
                        mesh_group.addElement(element)

    def add_axis2_elements_to_mesh_group(self, side, mesh_group):
        """
        Add elements from half of ellipsoid to mesh group.
        :param side: False for -axis2, True for +axis2.
        :param mesh_group: Mesh group to add to.
        """
        master_mesh = mesh_group.getMasterMesh()
        e2_start, e2_limit = (
            (self._element_counts[1] // 2, self._element_counts[1]) if side else (0, self._element_counts[1] // 2))
        for e3 in range(self._element_counts[2]):
            eids_layer = self._eids[e3]
            for e2 in range(e2_start, e2_limit):
                eids_row = eids_layer[e2]
                for e1 in range(self._element_counts[0]):
                    eid = eids_row[e1]
                    if eid:
                        element = master_mesh.findElementByIdentifier(eid)
                        mesh_group.addElement(element)

    def add_axis3_elements_to_mesh_group(self, side, mesh_group):
        """
        Add elements from half of ellipsoid to mesh group.
        :param side: False for -axis3, True for +axis3.
        :param mesh_group: Mesh group to add to.
        """
        master_mesh = mesh_group.getMasterMesh()
        e3_start, e3_limit = (
            (self._element_counts[2] // 2, self._element_counts[2]) if side else (0, self._element_counts[2] // 2))
        for e3 in range(e3_start, e3_limit):
            eids_layer = self._eids[e3]
            for e2 in range(self._element_counts[1]):
                eids_row = eids_layer[e2]
                for e1 in range(self._element_counts[0]):
                    eid = eids_row[e1]
                    if eid:
                        element = master_mesh.findElementByIdentifier(eid)
                        mesh_group.addElement(element)
