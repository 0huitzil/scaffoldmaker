import math
import copy
from itertools import product

from cmlibs.maths.vectorops import (
    add,
    dot,
    magnitude,
    mult,
    rotate_vector_around_vector,
    sub,
)
from cmlibs.utils.zinc.field import find_or_create_field_coordinates
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.fieldcache import Fieldcache
from cmlibs.zinc.fieldmodule import Fieldmodule
from cmlibs.zinc.node import Node
from cmlibs.zinc.region import Region
from cmlibs.zinc.result import RESULT_OK

from scaffoldmaker.annotation.annotationgroup import (
    AnnotationGroup,
    findOrCreateAnnotationGroupForTerm,
)
from scaffoldmaker.annotation.hand_terms import get_hand_term
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.utils.eft_utils import (
    remapEftLocalNodes,
    remapEftNodeValueLabelWithNodes,
    setEftScaleFactorIds,
)
from scaffoldmaker.utils.geometry import sampleEllipsePoints
from scaffoldmaker.utils.interpolation import interpolateCubicHermite


class MeshType_3d_hand1(Scaffold_base):
    """
    Generates a hermite x bilinear 3-D box network mesh based on data supplied
    by an input file.
    """

    @classmethod
    def getName(cls):
        return "3D Hand 1"

    @classmethod
    def getParameterSetNames(cls):
        return ["Default"]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        baseParameterSetName = "Default"
        options = {
            "Base parameter set": baseParameterSetName,
            "Thumb angle": 25.0,
            "Splaying angle": 0,
            "Magnitude of major axis for finger ellipses": 2.0,
            "Magnitude of minor axis for finger ellipses": 2.0,
            "finger_d2": 0.15,
            "finger_d3": 0.15,
            "fingercap_d1": 0.6,
            "Create internal elements": True,
            "Create external elements": False,
            "Internal elements across hand": 1,
            "Elements along carpal": 1,
            "Elements along metacarpal": 2,
            "Elements along proximal phalanx": 2,
            "Elements along middle phalanx": 1,
            "Elements along distal phalanx": 2,
            "finger names": [
                "little finger",
                "ring finger",
                "middle finger",
                "index finger",
                "thumb",
            ],
            "bone names": [
                "carpal",
                "metacarpal",
                "proximal phalanx",
                "middle phalanx",
                "distal phalanx",
            ],
            "Number of matrix columns per finger": 5,
            "Internal elements along each finger": 1,
            "Scale factor for blending inner nodes of palm": 0.5,
            "Scale factor for blending outer nodes of palm": 0.75,
            "Magnitude of major axis for palm ellipses": 4.35,
            "Magnitude of minor axis for palm ellipses": 2.25,
        }
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Thumb angle",
            # "Splaying angle",
            # "Magnitude of major axis for finger ellipses",
            # "Magnitude of minor axis for finger ellipses",
            "fingercap_d1",
            "finger_d2",
            "finger_d3",
            "Create internal elements",
            "Create external elements",
            "Internal elements across hand",
            "Elements along carpal",
            "Elements along metacarpal",
            "Elements along proximal phalanx",
            "Elements along middle phalanx",
            "Elements along distal phalanx",
        ]

    @classmethod
    def checkOptions(cls, options):
        dependent_changes = False
        for key, angleRange in {
            "Thumb angle": (0.0, 80.0),
        }.items():
            if options[key] < angleRange[0]:
                options[key] = angleRange[0]
            elif options[key] > angleRange[1]:
                options[key] = angleRange[1]
        for key in [
            "Elements along carpal",
            "Elements along metacarpal",
            "Elements along proximal phalanx",
            "Elements along middle phalanx",
            "Elements along distal phalanx",
        ]:
            if options[key] < 1:
                options[key] = 1
            if options[key] > 3:
                options[key] = 3
        for key in [
            "Magnitude of major axis for finger ellipses",
            "Magnitude of minor axis for finger ellipses",
        ]:
            if options[key] < 1:
                options[key] = 1
        for key in [
            "fingercap_d1",
            "finger_d2",
            "finger_d3",
        ]:
            if options[key] < 0.1:
                options[key] = 0.1
        return dependent_changes

    @classmethod
    def generateBaseMesh(cls, region: Region, options):
        """
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        return: list of AnnotationGroup,
        """
        fieldmodule = region.getFieldmodule()
        fieldcache = fieldmodule.createFieldcache()
        create_internal = options.get("Create internal elements")
        create_external = options.get("Create external elements")
        node_identifier = 1
        hand_elements_along = [
            options["Elements along carpal"],
            options["Elements along metacarpal"],
            options["Elements along proximal phalanx"],
            options["Elements along middle phalanx"],
            options["Elements along distal phalanx"],
        ]

        node_network = Node_network(region, fieldcache)
        vnm = Virtual_node_matrix(options)

        node_network, node_identifier = generate_internal_node_network(
            node_network, hand_elements_along, node_identifier, options=options
        )
        if create_internal:
            vnm, node_network, node_identifier = generate_internal_node_matrix(
                hand_elements_along,
                node_identifier,
                vnm,
                node_network=node_network,
                options=options,
            )
            if create_external:
                vnm, node_network, node_identifier = generate_external_node_matrix(
                    hand_elements_along,
                    node_identifier,
                    vnm,
                    node_network=node_network,
                    options=options,
                )

        finger_names = options["finger names"]
        finger_bone_names = options["bone names"][2:]
        hand_group = AnnotationGroup(region, get_hand_term("hand"))
        carpal_group = AnnotationGroup(region, get_hand_term("carpal"))
        metacarpal_group = AnnotationGroup(region, get_hand_term("metacarpal"))
        phalanx_prox_group = AnnotationGroup(region, get_hand_term("proximal phalanx"))
        phalanx_mid_group = AnnotationGroup(region, get_hand_term("middle phalanx"))
        phalanx_dist_group = AnnotationGroup(region, get_hand_term("distal phalanx"))
        mesh3d = fieldmodule.findMeshByDimension(3)
        annotation_groups = [
            hand_group,
            carpal_group,
            metacarpal_group,
            phalanx_prox_group,
            phalanx_mid_group,
            phalanx_dist_group,
        ]
        for finger in finger_names:
            finger_group = AnnotationGroup(region, get_hand_term(finger))
            annotation_groups.append(finger_group)
            for bone in finger_bone_names:
                finger_bone_group = AnnotationGroup(
                    region, get_hand_term(bone + " of " + finger)
                )
                annotation_groups.append(finger_bone_group)

        element_identifier = 1
        # TODO remove these lines
        # elements_along_per_fingk,er = sum(hand_elements_along)
        # for i in range(4):
        #     node_network.remove_node(1 + i * elements_along_per_finger)
        node_network.set_zinc_nodes()
        [ix, iy, iz] = vnm.get_matrix_dimensions()
        for k in range(iz - 1):
            for j in range(iy - 1):
                for i in range(ix - 1):
                    result, element = create_cube_element(
                        fieldmodule,
                        element_identifier,
                        vnm,
                        [i, j, k],
                    )
                    if result == RESULT_OK:
                        # print('element: ' + str(element_identifier))
                        # virtual_node_matrix.print_cube_from_index([i, j, k])
                        element_annotations = [hand_group]
                        # find annotation groups
                        try:
                            node_annotations = vnm.get_vnode([i, j, k]).get_annotation()
                            for group in annotation_groups:
                                if group.getName() in node_annotations:
                                    element_annotations.append(group)
                        except KeyError:
                            continue
                        finally:
                            for group in element_annotations:
                                mesh_group = group.getMeshGroup(mesh3d)
                                mesh_group.addElement(element)
                            element_identifier += 1
        return annotation_groups, None

    @classmethod
    def defineFaceAnnotations(cls, region, options, annotationGroups):
        """
        :param region: Zinc region containing model.
        :param options: Dict containing options. See getDefaultOptions().
        :param annotationGroups: List of annotation groups for elements
            created in generateBaseMesh().
            New face/line annotation groups are appended to this list.
        """
        fieldmodule = region.getFieldmodule()
        mesh2d = fieldmodule.findMeshByDimension(2)
        is_exterior = fieldmodule.createFieldIsExterior()
        is_face_xi3_0 = fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0)

        skin_group = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_hand_term("skin of hand")
        )
        is_skin = fieldmodule.createFieldAnd(
            is_exterior, fieldmodule.createFieldNot(is_face_xi3_0)
        )
        skin_group.getMeshGroup(mesh2d).addElementsConditional(is_skin)

        finger_names = options["finger names"]
        finger_bone_names = options["bone names"][2:]
        for finger_name in finger_names:
            finger_group = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_hand_term(finger_name)
            )
            finger_skin_group = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_hand_term("skin of " + finger_name)
            )
            finger_skin_group.getMeshGroup(mesh2d).addElementsConditional(
                fieldmodule.createFieldAnd(finger_group.getGroup(), is_exterior)
            )
            for bone_name in finger_bone_names:
                finger_bone_group = findOrCreateAnnotationGroupForTerm(
                    annotationGroups, region, get_hand_term(bone_name + " of " + finger_name)
                )
                finger_bone_skin_group = findOrCreateAnnotationGroupForTerm(
                    annotationGroups,
                    region,
                    get_hand_term("skin of " + bone_name + " of " + finger_name),
                )
                finger_bone_skin_group.getMeshGroup(mesh2d).addElementsConditional(
                    fieldmodule.createFieldAnd(finger_bone_group.getGroup(), is_exterior)
                )


class Bone:
    def __init__(self):
        pass

    CARPAL = 0
    METACARPAL = 1
    PROX_PHALANX = 2
    MID_PHALANX = 3
    DIST_PHALANX = 4


class Ellipse:
    def __init__(
        self,
        parent_node_parameters,
        major_axis_mag,
        minor_axis_mag,
        n_internal_rows,
        n_internal_cols,
    ):
        center, d1, d2, d3 = parent_node_parameters
        major_axis = mult(d2, -major_axis_mag)
        minor_axis = mult(d3, minor_axis_mag)
        major_ax_mag = magnitude(major_axis)
        minor_ax_mag = magnitude(minor_axis)
        d2_sampling_coef = 2.0
        d3_sampling_coef = 1.15
        d2_box_ratio = magnitude(d2) / major_ax_mag
        d3_box_ratio = magnitude(d3) / minor_ax_mag
        internal_box_length = d2_sampling_coef * d2_box_ratio
        internal_box_width = d3_sampling_coef * d3_box_ratio
        n_half_rows = math.ceil(n_internal_rows / 2)
        n_half_cols = math.ceil(n_internal_cols / 2)
        if n_internal_cols < 2 or n_internal_rows < 2:
            n_sampling_nodes = (n_internal_rows + 1) * (n_internal_cols + 1)
        else:
            n_sampling_nodes = 2 * n_internal_cols + 2 * n_internal_rows

        self.center = center
        self.d1 = d1
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.d2_box_ratio = d2_box_ratio
        self.d3_box_ratio = d3_box_ratio
        self.internal_box_length = internal_box_length
        self.internal_box_width = internal_box_width
        self.n_internal_rows = n_internal_rows
        self.n_internal_cols = n_internal_cols
        self.n_sampling_nodes = n_sampling_nodes
        self.n_half_rows = n_half_rows
        self.n_half_cols = n_half_cols
        self.first_quadrant_angles = None
        self.list_ellipse_x = None
        self.ellipse_d1 = None
        self.list_ellipse_d2 = None

        self.sorted_fingercap_scale_factors = None

    ## TODO create set_sampling_method function, the only public function to be called

    def set_sampling_coef(self, d2_sampling_coef=None, d3_sampling_coef=None):
        if d2_sampling_coef:
            d2_box_ratio = self.d2_box_ratio
            internal_box_length = d2_sampling_coef * d2_box_ratio
            self.internal_box_length = internal_box_length
        if d3_sampling_coef:
            d3_box_ratio = self.d3_box_ratio
            internal_box_width = d3_sampling_coef * d3_box_ratio
            self.internal_box_width = internal_box_width

    def sampling_coefficient(k, n):
        """
        Samples the [-1, 1] interval in n segments.

        :param k: Integer from 1 to n + 1
        :param n: Total number of segments
        :return: Real number between [-1, 1]
        """
        return (2 * k - n - 2) / n

    def create_sampling_angles_along_axis(self):
        ## TODO rework this section, the 'dx_sampling_coeffients' is too unintuitive
        internal_box_length = self.internal_box_length
        internal_box_width = self.internal_box_width
        n_internal_rows = self.n_internal_rows
        n_internal_cols = self.n_internal_cols
        n_half_rows = self.n_half_rows
        n_half_cols = self.n_half_cols
        sampling_along_major_ax = [
            internal_box_length * sampling_coefficient(k, n_internal_cols - 2)
            for k in range(n_half_cols, n_internal_cols)
        ]
        sampling_along_minor_ax = [
            internal_box_width * sampling_coefficient(k, n_internal_rows)
            for k in range(n_half_rows + 1, n_internal_rows + 2)
        ]
        angles_along_major_ax = [math.acos(value) for value in sampling_along_major_ax]
        angles_along_minor_ax = [math.asin(value) for value in sampling_along_minor_ax]
        first_quadrant_angles = sorted(angles_along_major_ax + angles_along_minor_ax)
        self.first_quadrant_angles = first_quadrant_angles

    def create_sampling_angles_uniform(self):
        n_nodes = self.n_sampling_nodes
        pi = math.pi
        angle = pi / n_nodes if n_nodes % 4 == 0 else 0
        first_quadrant_angles = []
        while angle < pi / 2:
            first_quadrant_angles.append(angle)
            angle = angle + 2 * pi / n_nodes
        self.first_quadrant_angles = first_quadrant_angles

    def sample_scale_factors_along_angles(self):
        """
        From a list of angles on the first quadrant of an ellipse, calculate their reflection
        over the other three quadrants. Then calculate both the position and tangent angle at
        the updated list of angles.

        :param center: Description
        :param major_axis: Description
        :param minor_axis: Description
        :param sampling_angles: Description
        """
        angles = self.first_quadrant_angles
        pi = math.pi
        for angle in reversed(angles):
            new_angle = pi - angle
            if new_angle not in angles:
                angles.append(new_angle)
        for angle in angles[:]:
            new_angle = pi + angle
            if new_angle % (2 * pi) not in angles:
                angles.append(new_angle)
        angle = 2 * pi + angles[0]

        scalefactor_d2_internal = []
        scalefactor_d3_internal = []
        for angle in angles:
            scalefactor_d2_internal.append(math.cos(angle))
            scalefactor_d3_internal.append(math.sin(angle))
        self.list_scalefactor_d2_internal = scalefactor_d2_internal
        self.list_scalefactor_d3_internal = scalefactor_d3_internal

    def sample_ellipse_along_angles(self):
        """
        From a list of angles on the first quadrant of an ellipse, calculate their reflection
        over the other three quadrants. Then calculate both the position and tangent angle at
        the updated list of angles.

        :param center: Description
        :param major_axis: Description
        :param minor_axis: Description
        :param sampling_angles: Description
        """
        center = self.center
        major_axis = self.major_axis
        minor_axis = self.minor_axis
        list_angles = [*self.first_quadrant_angles]
        for angle in reversed(list_angles):
            new_angle = math.pi - angle
            if new_angle not in list_angles:
                list_angles.append(new_angle)
        for angle in list_angles[:]:
            new_angle = math.pi + angle
            if new_angle not in list_angles:
                list_angles.append(new_angle)
        angle = 2 * math.pi + list_angles[0]
        if angle not in list_angles:
            list_angles.append(2 * math.pi + list_angles[0])

        ellipse_x = []
        ellipse_d2 = []
        for i in range(len(list_angles) - 1):
            ellipse = sampleEllipsePoints(
                center,
                major_axis,
                minor_axis,
                list_angles[i],
                list_angles[i + 1],
                1,
            )
            ellipse_x.append(ellipse[0][0])
            ellipse_d2.append(ellipse[1][0])

        self.list_angles = list_angles
        self.list_ellipse_x = ellipse_x
        self.list_ellipse_d2 = ellipse_d2

    def create_fingercap_scale_factors(self):
        # TODO this fix was formatted with help from Claude, I am not quite happy with it.
        # TODO need to go back and revisit the logic in the future.
        """
        Calculate the scale factors of the fingercap at every node of the ring, where the
        fingercap derivatives are combinations of the ring derivatives D1 and D2:
            d1 = d11 * D1 + d12 * D2,    d2 = d21 * D1 + d22 * D2
        d21 = [upper, lower] is the D1 part of d2 in the fingercap elements above and below
        the node (the same for both at a corner, which has a single fingercap element).
        They are chosen so the halfway point of each fingercap edge lies on the half ellipsoid
        with the ellipse as its base and D1 as its apex (halfway rule).
        Going around the ring, every segment between two consecutive nodes is a fingercap
        edge. The tangents leaving node i are kept as [alpha, beta] = alpha * D1 + beta * D2,
        ahead[i] towards node i + 1 and behind[i] towards node i - 1 (D2 points towards i + 1):
        - Middle to middle: lamb * D1 +/- D2, lamb from the halfway rule (0 if the segment
          already lies outside the ellipsoid).
        - Middle to corner: the middle node keeps the lamb of its other segment, or 1 with
          2 internal rows, and the corner adapts to it.
        - Corner tangents: lamb * (D1 +/- c * D2). With 1 internal row the fingercap is a
          single element, whose tip rises by the sum of the across and side halfway heights,
          so lamb = 2 puts the tip on the apex and the halfway rule gives c. Otherwise
          c = |D1| / |D2| (equal parts of D1 and D2) and the halfway rule gives lamb. If the
          halfway rule has no solution, the corner keeps lamb = 1 and c = |D1| / |D2|.
        The edges across the middle nodes are not ring segments: they stay in the slice of the
        ellipsoid at the height of the node, a half ellipse whose apex |cos(angle)| * D1 is
        reached with d1 = 4 * cos(angle) * D1.
        """
        assert self.list_ellipse_x is not None, "List of ellipse samples not available"
        n_internal_rows = self.n_internal_rows
        n_internal_cols = self.n_internal_cols
        assert n_internal_cols == 1, "Current algorithm limited to 1 internal col per finger"
        list_x = self.list_ellipse_x
        list_d2 = self.list_ellipse_d2
        n_sampling_nodes = self.n_sampling_nodes
        center = self.center
        d1 = self.d1
        axes = [self.major_axis, self.minor_axis, d1]
        ellipse_cubic_factor = 4  # For the halfway point rule for an ellipse
        zero = [0.0, 0.0, 0.0]

        def leaving_tangent(i, factors):
            return add(mult(d1, factors[0]), mult(list_d2[i], factors[1]))

        def solve_on_ellipsoid(base, w):
            # s > 0 such that base + s * w lies on the ellipsoid, None if there is none
            p = [dot(sub(base, center), axis) / dot(axis, axis) for axis in axes]
            w = [dot(w, axis) / dot(axis, axis) for axis in axes]
            a, b, c = dot(w, w), 2.0 * dot(p, w), dot(p, p) - 1.0
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0.0:
                return None
            s = (-b + math.sqrt(discriminant)) / (2.0 * a)
            return s if s > 0.0 else None

        def solve_segment(i, j, fixed_i, varying_i, fixed_j, varying_j):
            # Halfway rule on the segment between nodes i and j, with the tangents leaving each
            # end given as fixed + s * varying factors. Returns s, or None if there is none.
            def halfway_point(x_i, factors_i, x_j, factors_j):
                return interpolateCubicHermite(
                    x_i,
                    leaving_tangent(i, factors_i),
                    x_j,
                    mult(leaving_tangent(j, factors_j), -1.0),
                    0.5,
                )

            return solve_on_ellipsoid(
                halfway_point(list_x[i], fixed_i, list_x[j], fixed_j),
                halfway_point(zero, varying_i, zero, varying_j),
            )

        def segments(kinds):
            # Consecutive nodes i, j = i + 1 around the ring whose (corner, corner) kinds match
            for i in range(n_sampling_nodes):
                j = (i + 1) % n_sampling_nodes
                if (is_corner[i], is_corner[j]) in kinds:
                    yield i, j

        sorted_index = self.sort_samples(list(range(n_sampling_nodes)))
        is_corner = [False] * n_sampling_nodes
        for k, col in product([1, n_internal_rows + 1], [0, n_internal_cols]):
            is_corner[sorted_index[k][col]] = True
        c_equal = [magnitude(d1) / magnitude(d2) for d2 in list_d2]
        ahead = [None] * n_sampling_nodes
        behind = [None] * n_sampling_nodes
        # Middle to middle
        for i, j in segments([(False, False)]):
            lamb = solve_segment(i, j, [0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [1.0, 0.0])
            lamb = 0.0 if lamb is None else lamb
            ahead[i] = [lamb, 1.0]
            behind[j] = [lamb, -1.0]
        # Middle nodes facing a corner
        for i in range(n_sampling_nodes):
            if not is_corner[i]:
                known = [t[0] for t in (ahead[i], behind[i]) if t is not None]
                lamb = known[0] if known else 1.0
                if ahead[i] is None:
                    ahead[i] = [lamb, 1.0]
                if behind[i] is None:
                    behind[i] = [lamb, -1.0]
        # Corner to corner
        for i, j in segments([(True, True)]):
            c_i, c_j = c_equal[i], c_equal[j]
            if n_internal_rows == 1:
                c = solve_segment(i, j, [2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, -2.0])
                lamb = None if c is None else 2.0
                if c is not None:
                    c_i = c_j = c
            else:
                lamb = solve_segment(i, j, [0.0, 0.0], [1.0, c_i], [0.0, 0.0], [1.0, -c_j])
            lamb = 1.0 if lamb is None else lamb
            ahead[i] = [lamb, lamb * c_i]
            behind[j] = [lamb, -lamb * c_j]
        # Corner to middle, with the tangent leaving the middle node fixed
        for i, j in segments([(True, False), (False, True)]):
            corner, middle = (i, j) if is_corner[i] else (j, i)
            towards_d2 = 1.0 if corner == i else -1.0
            leaving_middle = behind[j] if corner == i else ahead[i]
            c = towards_d2 * c_equal[corner]
            lamb = solve_segment(
                corner, middle, [0.0, 0.0], [1.0, c], leaving_middle, [0.0, 0.0]
            )
            lamb = 1.0 if lamb is None else lamb
            if corner == i:
                ahead[i] = [lamb, lamb * c]
            else:
                behind[j] = [lamb, lamb * c]

        # Fingercap d1 runs across (+j), d2 along the side (+k): sort and orient the tangents
        sorted_ahead = self.sort_samples(ahead)
        sorted_behind = self.sort_samples(behind)
        sorted_angles = self.sort_samples(self.list_angles[:-1])
        d11, d12, d21, d22 = [
            [[None for f in range(n_internal_cols + 1)] for k in range(n_internal_rows + 3)]
            for i in range(4)
        ]
        for k, col in product(range(1, n_internal_rows + 2), [0, n_internal_cols]):
            a2 = -1 if col == 0 else 1
            a3 = -1 if k == 1 else 1
            if k in [1, n_internal_rows + 1]:
                # The side edge is ahead of the bottom-left and top-right corners
                side, across = sorted_ahead[k][col], sorted_behind[k][col]
                if a2 * a3 != 1:
                    side, across = across, side
                d11[k][col], d12[k][col] = [-a2 * f for f in across]
                d21_side, d22[k][col] = [-a3 * f for f in side]
                d21[k][col] = [d21_side, d21_side]
            else:
                # Upper half towards k + 1 is ahead on the first column, behind on the last
                upper, lower = sorted_ahead[k][col], sorted_behind[k][col]
                if a2 == 1:
                    upper, lower = lower, upper
                d11[k][col] = ellipse_cubic_factor * math.cos(sorted_angles[k][col])
                d12[k][col] = 0.0
                d21[k][col] = [upper[0], -lower[0]]
                d22[k][col] = -a2
        self.sorted_fingercap_scale_factors = [d11, d12, d21, d22]

    def sort_samples(self, list_samples):
        n_internal_rows = self.n_internal_rows
        n_internal_cols = self.n_internal_cols
        n_half_rows = self.n_half_rows
        sorted_samples = [
            [None for f in range(n_internal_cols + 1)] for k in range(n_internal_rows + 3)
        ]
        i = -n_half_rows
        for k in range(1, n_internal_rows + 2):
            sorted_samples[k][0] = list_samples[i]
            i += 1
        for f in range(1, n_internal_cols):
            sorted_samples[n_internal_rows + 1][f] = list_samples[i]
            i += 1
        for k in reversed(range(1, n_internal_rows + 2)):
            sorted_samples[k][n_internal_cols] = list_samples[i]
            i += 1
        for f in reversed(range(1, n_internal_cols)):
            sorted_samples[1][f] = list_samples[i]
            i += 1
        return sorted_samples

    def sort_ellipse_samples(self):
        assert self.list_ellipse_x is not None, "List of ellipse samples not available"
        ellipse_x = self.list_ellipse_x
        ellipse_d2 = self.list_ellipse_d2
        ellipse_d1 = [self.d1 for _ in range(len(ellipse_x))]
        sorted_ellipse_x = self.sort_samples(ellipse_x)
        sorted_ellipse_d1 = self.sort_samples(ellipse_d1)
        sorted_ellipse_d2 = self.sort_samples(ellipse_d2)
        self.sorted_ellipse_x = sorted_ellipse_x
        self.sorted_ellipse_d1 = sorted_ellipse_d1
        self.sorted_ellipse_d2 = sorted_ellipse_d2

    def sort_internal_scale_factors(self):
        assert self.list_scalefactor_d2_internal is not None, (
            "List of internal scale factors not available"
        )
        scalefactor_d2_internal = self.list_scalefactor_d2_internal
        scalefactor_d3_internal = self.list_scalefactor_d3_internal
        sorted_scalefactor_d2 = self.sort_samples(scalefactor_d2_internal)
        sorted_scalefactor_d3 = self.sort_samples(scalefactor_d3_internal)
        self.sorted_scalefactor_d2_internal = sorted_scalefactor_d2
        self.sorted_scalefactor_d3_internal = sorted_scalefactor_d3

    def get_ellipse_samples(self):
        ## TODO call the sample and sort functions from here
        return [self.sorted_ellipse_x, self.sorted_ellipse_d1, self.sorted_ellipse_d2]

    def get_internal_scale_factors(self):
        ## TODO call the sample and sort functions from here
        return [self.sorted_scalefactor_d2_internal, self.sorted_scalefactor_d3_internal]

    def get_fingercap_scale_factors(self):
        ## TODO call the sample and sort functions from here
        self.create_fingercap_scale_factors()
        return self.sorted_fingercap_scale_factors


class Node_network:
    def __init__(self, region: Region, fieldcache: Fieldcache):
        self.fieldmodule = region.getFieldmodule()
        self.fieldcache = fieldcache
        self.coordinates = find_or_create_field_coordinates(self.fieldmodule, "coordinates")
        self.nodes = self.fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.node_identifier = 1
        self.nodes_dict = {}
        external_nodetemplate_labels = [
            Node.VALUE_LABEL_VALUE,
            Node.VALUE_LABEL_D_DS1,
            Node.VALUE_LABEL_D_DS2,
        ]
        nodes = self.nodes
        coordinates = self.coordinates
        external_nodetemplate = nodes.createNodetemplate()
        external_nodetemplate.defineField(coordinates)
        for value_label in external_nodetemplate_labels[1:]:
            external_nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)
        self.external_nodetemplate = external_nodetemplate

        internal_nodetemplate_labels = [
            Node.VALUE_LABEL_VALUE,
            # Node.VALUE_LABEL_D_DS1,
            Node.VALUE_LABEL_D_DS2,
            Node.VALUE_LABEL_D_DS3,
        ]
        internal_nodetemplate = nodes.createNodetemplate()
        internal_nodetemplate.defineField(coordinates)
        for value_label in internal_nodetemplate_labels[1:]:
            internal_nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)
        self.internal_nodetemplate = internal_nodetemplate

        internal_palm_nodetemplate_labels = [
            Node.VALUE_LABEL_VALUE,
            Node.VALUE_LABEL_D_DS1,
            Node.VALUE_LABEL_D_DS2,
            Node.VALUE_LABEL_D_DS3,
        ]
        internal_palm_nodetemplate = nodes.createNodetemplate()
        internal_palm_nodetemplate.defineField(coordinates)
        for value_label in internal_palm_nodetemplate_labels[1:]:
            internal_palm_nodetemplate.setValueNumberOfVersions(
                coordinates, -1, value_label, 1
            )
        self.internal_palm_nodetemplate = internal_palm_nodetemplate

    def set_node_parameters(
        self,
        node_coordinates: list,
        node_identifier: int = 0,
        internal_node=True,
        palm_node=False,
    ) -> None:
        ## TODO change this function to have x, d1, d2 and d3 as individual arguments
        nodes_dict = self.nodes_dict
        assert len(node_coordinates) == 4
        x, d1, d2, d3 = node_coordinates
        nodes_dict[node_identifier] = {
            "parameters": [x, d1, d2, d3],
            "internal": internal_node,
            "palm": palm_node,
        }

    def set_zinc_nodes(self):
        node_dict = self.nodes_dict
        coordinates = self.coordinates
        fieldcache = self.fieldcache
        for node_identifier, node_params in node_dict.items():
            is_internal = node_params.get("internal")
            is_palm = node_params.get("palm")
            node_coordinates = node_params.get("parameters")
            if is_internal:
                x, d1, d2, d3 = node_coordinates
                # d1 = None
                nodetemplate = self.internal_nodetemplate
                if is_palm:
                    d2 = None
                    nodetemplate = self.internal_palm_nodetemplate
            else:
                x, d1, d2, d3 = node_coordinates
                d3 = None
                nodetemplate = self.external_nodetemplate
            node = self.nodes.createNode(node_identifier, nodetemplate)
            fieldcache.setNode(node)
            setNodeFieldParameters(coordinates, fieldcache, x, d1=d1, d2=d2, d3=d3)

    def get_node_parameters(self, node_identifier):
        nodes_dict = self.nodes_dict
        node_dict = nodes_dict[node_identifier].get("parameters")
        return node_dict

    def get_node_identifier(self) -> int:
        return self.node_identifier

    def remove_node(self, node_identifier):
        nodes_dict = self.nodes_dict
        nodes_dict.pop(node_identifier)


class Virtual_node:
    """
    A class defining a virtual (also called hanging) node.
    A node can be internal (with only a positional, x, term),
    or external (defined as having an additional d1 and d2 terms).
    A virtual node does not have parameters of it own, instead it points
    to a 'real' zinc node via its node identifiers and has scale factors
    that create linear combinations of the parameters of the parent node
    """

    def __init__(
        self, node_identifier: int, annotation: list, x: list, d1: list = None, d2: list = None
    ):
        self.node_identifier = node_identifier
        self.annotation = annotation
        assert len(x) == 4, "Scale factor vector must be length 4"
        self.scalefactors = {
            Node.VALUE_LABEL_VALUE: [[node_identifier, *x]],
        }
        if d1 is not None:
            assert len(d1) == len(d2) == 4, "Scale factor vector must be length 4"
            self.scalefactors[Node.VALUE_LABEL_D_DS1] = [[node_identifier, *d1]]
            self.scalefactors[Node.VALUE_LABEL_D_DS2] = [[node_identifier, *d2]]
            self.is_external = True
        else:
            self.is_external = False

    def get_scalefactors(self, label) -> list:
        return self.scalefactors.get(label)

    def get_node_identifier(self):
        return self.node_identifier

    def get_annotation(self):
        return self.annotation

    def copy(self):
        vnode = copy.copy(self)
        vnode.scalefactors = {
            label: [[*sf] for sf in scalefactors]
            for label, scalefactors in self.scalefactors.items()
        }
        return vnode

    def blend(self, new_vnode):
        # Append the positional (x) scalefactors from the incoming node to the current one
        assert not self.is_external and not new_vnode.is_external, (
            "Blending only supported for internal nodes"
        )
        self.scalefactors[Node.VALUE_LABEL_VALUE] = self.scalefactors[
            Node.VALUE_LABEL_VALUE
        ] + [[*sf] for sf in new_vnode.get_scalefactors(Node.VALUE_LABEL_VALUE)]


class Virtual_node_matrix:
    def __init__(self, options: dict):
        hand_elements_along = [
            options["Elements along carpal"],
            options["Elements along metacarpal"],
            options["Elements along proximal phalanx"],
            options["Elements along middle phalanx"],
            options["Elements along distal phalanx"],
        ]
        columns_per_finger = options.get("Number of matrix columns per finger")
        n_internal_rows = options.get("Internal elements across hand")
        ix = 2 * sum(hand_elements_along)
        iy = 6 * columns_per_finger
        iz = 3 * n_internal_rows + 1
        node_matrix = [[[None for k in range(iz)] for j in range(iy)] for i in range(ix)]
        kval_inner = [i + 1 for i in range(n_internal_rows + 1)]
        kval_outer = [1, n_internal_rows + 1]

        self.n_internal_rows = n_internal_rows
        self.matrix_dimensions = [ix, iy, iz]
        self.node_matrix = node_matrix
        self.kval_inner = kval_inner
        self.kval_outer = kval_outer

        # TODO create a phys2log map for x as well?
        self.build_phys_to_log_k_map()

    def get_logical_k_values(self, k_phys):
        ## TODO rename this function
        k_logi = self.phys_to_logic_k_val[k_phys]
        return k_logi

    def build_phys_to_log_k_map(self):
        ## TODO rename this function
        n_internal_rows = self.n_internal_rows
        phys_to_logic_k_val = [
            [0],
            [1],
            *[[3 * i - 1, 3 * i + 1] for i in range(1, n_internal_rows)],
            [3 * n_internal_rows - 1],
            [3 * n_internal_rows],
        ]
        self.phys_to_logic_k_val = phys_to_logic_k_val

    # def create_internal_vnode(self, node_identifier: int, x: list, annotation: dict):
    #     virtual_node = {
    #         Node.VALUE_LABEL_VALUE: [[node_identifier, *x]],
    #         "node_identifier": node_identifier,
    #         "Type": "internal",
    #         "Annotation": annotation,
    #     }
    #     return virtual_node

    # def create_external_vnode(
    #     self, node_identifier: int, x: list, d1: list, d2: list, annotation: dict
    # ):
    #     assert len(x) == len(d1) == len(d2) == 4, "Scale factor vector must be length 4"
    #     virtual_node = {
    #         Node.VALUE_LABEL_VALUE: [[node_identifier, *x]],
    #         Node.VALUE_LABEL_D_DS1: [[node_identifier, *d1]],
    #         Node.VALUE_LABEL_D_DS2: [[node_identifier, *d2]],
    #         "node_identifier": node_identifier,
    #         "Type": "external",
    #         "Annotation": annotation,
    #     }
    #     return virtual_node

    def set_vnode_to_index(self, vnode: Virtual_node, index: list, overwrite=True):
        i, j, k = index
        node_matrix = self.node_matrix
        if overwrite:
            node_matrix[i][j][k] = vnode.copy()
        else:
            if node_matrix[i][j][k] is None:
                node_matrix[i][j][k] = vnode.copy()
            else:
                if not vnode.is_external:
                    existing_vnode = self.get_vnode(index)
                    existing_vnode.blend(vnode)
                    node_matrix[i][j][k] = existing_vnode

    def set_vnode_to_indices(
        self, vnode: Virtual_node, indices: list, overwrite=True, k_allocation_mode="both"
    ):
        ## TODO rename the k_allocation_mode
        for index in indices:
            i, j, phys_k = index
            logic_k_vals = self.get_logical_k_values(phys_k)
            if k_allocation_mode == "lower":
                logic_k_vals = [logic_k_vals[0]]
            elif k_allocation_mode == "upper":
                logic_k_vals = [logic_k_vals[-1]]
            elif k_allocation_mode == "both":
                logic_k_vals = logic_k_vals
            else:
                raise ValueError("Valid modes are upper, lower or both")
            for k in logic_k_vals:
                l_index = [i, j, k]
                self.set_vnode_to_index(vnode, l_index, overwrite)

    def get_matrix(self) -> list:
        return self.node_matrix

    def get_matrix_dimensions(self) -> list:
        return self.matrix_dimensions

    def get_kval(self, inner=True):
        if inner:
            return self.kval_inner
        else:
            return self.kval_outer

    # def blend_vnode(self, existing_node_index: list, new_vnode):
    #     """
    #     Only works for blending internal linear nodes for now

    #     :param self: Description
    #     :param existing_node_index: Description
    #     :type existing_node_index: list
    #     :param new_node: Description
    #     """
    #     vertex = self.get_vnode(existing_node_index)
    #     original_vertex = vertex[Node.VALUE_LABEL_VALUE]
    #     N = len(original_vertex)
    #     updated_vertex = [*original_vertex]
    #     updated_vertex.append([*new_vnode])
    #     for i in [1, 4]:
    #         coeff_sum = 0
    #         for j in range(N + 1):
    #             coeff_sum += updated_vertex[j][i]
    #         # TODO make a decision on whether to keep this check or no.
    #         # assert abs(coeff_sum) == 1, 'Not affine combination'
    #     vertex[Node.VALUE_LABEL_VALUE] = updated_vertex
    #     self.set_vnode_to_index(vertex, existing_node_index)

    def get_vnode(self, index: list) -> Virtual_node:
        i, j, k = index
        node_matrix = self.node_matrix
        vnode = node_matrix[i][j][k]
        return vnode

    def get_node_identifier_at_index(self, index: list) -> int:
        node = self.get_vnode(index)
        node_identifier = node.get_node_identifier()
        return node_identifier

    def print_cube_from_index(self, start_index: list):
        """
        For debugging purposes, prints the instructions for creating the cube element
        starting at start_index.

        :param self: Description
        :param start_index: Description
        :type start_index: list
        """
        start_i, start_j, start_k = start_index
        ranges = [[0, 1], [0, 1], [0, 1]]  # 1
        order = [2, 1, 0]
        reordered_ranges = [ranges[idx] for idx in order]
        indices = []
        # Create the cartersian product
        for prod in product(*reordered_ranges):
            row = [0] * len(ranges)
            # Map the generated values back to their correct positions
            for i, val in enumerate(prod):
                original_axis = order[i]
                row[original_axis] = val
            indices.append(row)
        node_matrix = self.node_matrix
        print("------------------------------")
        for index in indices:
            i, j, k = index
            print(
                [start_i + i, start_j + j, start_k + k],
                node_matrix[start_i + i][start_j + j][start_k + k],
            )
        print("------------------------------")


def generate_internal_node_network(
    node_network: Node_network,
    hand_elements_along: list,
    node_identifier: int = 1,
    options: dict = {},
):
    """
    Docstring for generate_internal_nodes

    :param region: Description
    :type region: Region
    :param fieldcache: Description
    :type fieldcache: Fieldcache
    :param node_identifier: Description
    :type node_identifier: int
    :param options: Description
    :type options: dict
    :return: Description
    :rtype: int
    """
    c_d3 = 0.35
    m_d3 = 0.325
    p_d3 = 0.30
    fingercap_d1 = options["fingercap_d1"]
    finger_d2 = options["finger_d2"]
    finger_d3 = options["finger_d3"]
    hand_dimensions_by_finger = [  # d1, d2, d3
        [  # Finger 1 (little finger)
            [1.0, 0.4, c_d3],  # carpal
            [2.0, 0.4, m_d3],  # metacarpal
            [1.1, 0.4, p_d3],  # proximal phalanx
            [0.7, finger_d2, finger_d3],  # middle phalanx
            [fingercap_d1, finger_d2, finger_d3],  # distal phalanx
        ],
        [  # Finger 2 (ring finger)
            [1.0, 0.4, c_d3],  # carpal
            [2.0, 0.4, m_d3],  # metacarpal
            [1.5, 0.4, p_d3],  # proximal phalanx
            [1.0, finger_d2, finger_d3],  # middle phalanx
            [0.7, finger_d2, finger_d3],  # distal phalanx
        ],
        [  # Finger 3 (middle finger)
            [1.0, 0.4, c_d3],  # carpal
            [2.0, 0.4, m_d3],  # metacarpal
            [1.9, 0.4, p_d3],  # proximal phalanx
            [1.0, finger_d2, finger_d3],  # middle phalanx
            [0.6, finger_d2, finger_d3],  # distal phalanx
        ],
        [  # Finger 4 (index finger)
            [1.0, 0.4, c_d3],  # carpal
            [2.0, 0.4, m_d3],  # metacarpal
            [1.9, 0.4, p_d3],  # proximal phalanx
            [0.8, finger_d2, finger_d3],  # middle phalanx
            [0.6, finger_d2, finger_d3],  # distal phalanx
        ],
        [  # Finger 5 (thumb finger)
            [0.0, 0.0, 0.0],  # carpal
            [1.0, finger_d2, finger_d3],  # metacarpal
            [1.1, finger_d2, finger_d3],  # proximal phalanx
            [0.0, 0.0, 0.0],  # middle phalanx
            [0.8, finger_d2, finger_d3],  # distal phalanx
        ],
    ]
    carpal_spacing = hand_dimensions_by_finger[0][0][1] * 2.0
    thumb_dimensions = hand_dimensions_by_finger[-1]
    thumb_indices = [Bone.METACARPAL, Bone.PROX_PHALANX, Bone.DIST_PHALANX]
    thumb_angle_degrees = options["Thumb angle"]
    angle_radians = math.radians(thumb_angle_degrees)
    splaying_angle_radians = math.radians(options["Splaying angle"])
    index_carpal_node_id = None
    # create nodes up each finger fastest
    x1 = [1.0, 0.0, 0.0]
    x2 = [0.0, 1.0, 0.0]
    x3 = [0.0, 0.0, 1.0]
    for n_finger in range(4):
        finger_dimensions = hand_dimensions_by_finger[n_finger]
        x = [0.0, n_finger * carpal_spacing, 0.0]
        x1 = [1.0, 0.0, 0.0]
        x2 = [0.0, 1.0, 0.0]
        for bone, bone_dimensions in enumerate(finger_dimensions):
            n_elements_along = hand_elements_along[bone]
            bone_dimensions = finger_dimensions[bone]
            if bone == Bone.METACARPAL:
                x1 = rotate_vector_around_vector(
                    x1, x3, (n_finger - 2) * splaying_angle_radians
                )
                x2 = rotate_vector_around_vector(
                    x2, x3, (n_finger - 2) * splaying_angle_radians
                )
            d1 = mult(x1, bone_dimensions[0])
            if bone == Bone.DIST_PHALANX:
                d1 = mult(d1, 1 / (n_elements_along))
            else:
                d1 = mult(d1, 1 / n_elements_along)
            d2 = mult(x2, bone_dimensions[1])
            d3 = mult(x3, bone_dimensions[2])
            if (n_finger == 3) and (bone == Bone.CARPAL):
                index_carpal_node_id = node_identifier
            is_palm = True if bone < 3 else False
            for i_along in range(n_elements_along):
                # TODO only allocate one node per bone.
                if bone == Bone.PROX_PHALANX and i_along == 1:
                    is_palm = False
                    bone_dimensions = finger_dimensions[bone + 1]
                    d2 = mult(x2, bone_dimensions[1])
                    d3 = mult(x3, bone_dimensions[2])
                node_network.set_node_parameters(
                    [x, d1, d2, d3], node_identifier, internal_node=True, palm_node=is_palm
                )
                x = add(x, d1)
                node_identifier += 1
    # Thumb
    x1 = rotate_vector_around_vector(x1, x3, angle_radians)
    x2 = rotate_vector_around_vector(x2, x3, angle_radians)
    # Obtain the starting node position from the metacarpal node
    x, d1, d2, d3 = node_network.get_node_parameters(index_carpal_node_id)
    x = add(x, mult(d2, 1.0))
    x = add(x, mult(x1, 1.5 * magnitude(d1)))
    # x = add(x, mult(d1, 0.15))
    for bone in thumb_indices:
        bone_dimensions = thumb_dimensions[bone]
        n_elements_along = hand_elements_along[bone]
        if bone in [Bone.METACARPAL, Bone.PROX_PHALANX]:
            n_elements_along = max(1, n_elements_along - 1)
        d1 = mult(x1, bone_dimensions[0])
        if bone == Bone.DIST_PHALANX:
            d1 = mult(d1, 1 / (n_elements_along))
        else:
            d1 = mult(d1, 1 / n_elements_along)
        if bone == Bone.METACARPAL:
            x = add(x, mult(d2, -1))
        d2 = mult(x2, bone_dimensions[1])
        d3 = mult(x3, bone_dimensions[2])
        x = x
        if bone == Bone.METACARPAL:
            x = add(x, d2)
        for i_along in range(n_elements_along):
            node_network.set_node_parameters(
                [x, d1, d2, d3], node_identifier, internal_node=True
            )
            x = add(x, d1)
            node_identifier += 1
    return node_network, node_identifier


def generate_internal_node_matrix(
    hand_elements_along,
    node_identifier,
    vnm: Virtual_node_matrix,
    node_network: Node_network,
    options: dict = None,
):
    """
    Docstring for generate_internal_node_matrix

    :param fieldmodule: Description
    :param hand_elements_along: Description
    :param node_identifier: Description
    :param vnm: Description
    :param options: Description
    :return: Description
    :rtype: Any
    """
    # TODO calculate these constants as functions of the node parameters
    bone_c_int = options.get("Scale factor for blending inner nodes of palm")
    bone_c_ext = options.get("Scale factor for blending outer nodes of palm")
    n_internal_rows = options.get("Internal elements across hand")
    n_cols_per_node = options.get("Internal elements along each finger")
    a0 = 1.0
    # TODO change a1 to depend on the i/n value
    a1 = 0.0
    a2 = 0.0
    index_y = 1
    # k_val = [i + 1 for i in range(n_internal_rows + 1)]
    k_val = vnm.get_kval(inner=True)
    # k_val = virtual_node_matrix.get_kval_outer()
    parent_node_id = 1
    columns_per_finger = options.get("Number of matrix columns per finger")
    elements_along_palm = sum(hand_elements_along[Bone.CARPAL : Bone.PROX_PHALANX]) + 1
    elements_along_finger = sum(hand_elements_along[Bone.PROX_PHALANX :]) - 1
    finger_index_x = elements_along_palm + 2  # First P.phalanx element is two-sided
    palm_bones = [Bone.CARPAL, Bone.METACARPAL, Bone.PROX_PHALANX]
    finger_names = options.get("finger names")
    bone_names = options.get("bone names")
    THUMB_INDEX = 4
    # Palm elements
    for finger_index in range(4):
        index_x = 0
        # Carpal, metacarpal and first row of proximal phalanx
        for bone in palm_bones:
            n_elements_along = hand_elements_along[bone]
            if finger_index == 0:
                j_val = [index_y, index_y + 1, index_y + 5]
            elif finger_index == 1:
                j_val = [index_y - 5, index_y - 4, index_y, index_y + 1, index_y + 5]
            elif finger_index == 2:
                j_val = [index_y - 4, index_y, index_y + 1, index_y + 5, index_y + 6]
            else:
                j_val = [index_y - 4, index_y, index_y + 1]
            if bone in [Bone.CARPAL, Bone.METACARPAL]:
                i_val = [index_x + i for i in range(n_elements_along)]
            else:
                i_val = [index_x]
            for i in i_val:
                for j in j_val:
                    for k in k_val:
                        if j in [index_y - 5, index_y + 6]:
                            a0 = 0.5 - bone_c_ext
                        elif j in [index_y - 4, index_y + 5]:
                            a0 = bone_c_int
                        elif j in [index_y]:
                            a0 = 0.5 + bone_c_ext if finger_index == 0 else bone_c_int
                        elif j in [index_y + 1]:
                            a0 = 0.5 + bone_c_ext if finger_index == 3 else bone_c_int
                        a3 = a0 * sampling_coefficient(k, n_internal_rows)
                        bone_name = bone_names[bone]
                        indices = [[i, j, k]]
                        if bone == Bone.CARPAL:
                            annotation = [bone_name] if k == 1 else ""
                            if (j == index_y + 1 and finger_index == 3) or (
                                j == index_y + 6 and finger_index == 2
                            ):
                                indices.append([i + 1, j + 4, k])
                            node = Virtual_node(parent_node_id, annotation, [a0, a1, a2, a3])
                            vnm.set_vnode_to_indices(node, indices, False)
                        elif bone == Bone.METACARPAL:
                            annotation = [bone_name] if k == 1 else ""
                            if (j == index_y + 1 and finger_index == 3) or (
                                j == index_y + 6 and finger_index == 2
                            ):  # Webbing
                                if i == index_x:
                                    indices.append([i + 1, j + 4, k])
                                    indices.append([i + 3, j + 4, k])
                            node = Virtual_node(parent_node_id, annotation, [a0, a1, a2, a3])
                            vnm.set_vnode_to_indices(node, indices, False)
                        elif bone == Bone.PROX_PHALANX:
                            n_elements_along = 1
                            finger_name = finger_names[finger_index]
                            if j > index_y + 1 and finger_index < 3:
                                finger_name = finger_names[finger_index + 1]
                            finger_part = bone_name + " of " + finger_name
                            annotation = [bone_name, finger_name, finger_part]
                            node = Virtual_node(parent_node_id, annotation, [a0, a1, a2, a3])
                            indices.append([i + 2, j, k])
                            vnm.set_vnode_to_indices(node, indices, False)
                            indices
                parent_node_id += 1
            index_x += n_elements_along
        index_y += columns_per_finger
        parent_node_id += elements_along_finger

    a0 = 1.0
    index_y = 1
    parent_node_id = elements_along_palm + 1
    # Finger elements
    for finger_index in range(5):
        if finger_index == THUMB_INDEX:
            finger_bones = [Bone.METACARPAL, Bone.PROX_PHALANX, Bone.DIST_PHALANX]
            index_x = hand_elements_along[0]
        else:
            if hand_elements_along[2] > 1:
                finger_bones = [Bone.PROX_PHALANX, Bone.MID_PHALANX, Bone.DIST_PHALANX]
            else:
                finger_bones = [Bone.MID_PHALANX, Bone.DIST_PHALANX]
            index_x = finger_index_x
        for bone in finger_bones:
            n_elements_along = hand_elements_along[bone]
            if bone in [Bone.METACARPAL, Bone.PROX_PHALANX]:
                n_elements_along = max(1, n_elements_along - 1)
            j_val = [index_y, index_y + 1]
            i_val = [index_x + i for i in range(n_elements_along)]
            for i in i_val:
                for j in j_val:
                    for k in k_val:
                        node_params = node_network.get_node_parameters(parent_node_id)
                        ## TODO get rid of the multiple function calls
                        ## simplify everything into one simple function
                        ellipse = Ellipse(node_params, 1, 1, n_internal_rows, n_cols_per_node)
                        ellipse.create_sampling_angles_uniform()
                        ellipse.sample_scale_factors_along_angles()
                        ellipse.sort_internal_scale_factors()
                        a2_vals, a3_vals = ellipse.get_internal_scale_factors()
                        col_index = 0 if j == index_y else 1
                        a2 = -a2_vals[k][col_index]
                        a3 = a3_vals[k][col_index]
                        indices = [[i, j, k]]
                        finger_name = finger_names[finger_index]
                        bone_name = bone_names[bone]
                        annotation = [bone_name, finger_name, finger_part]
                        finger_part = bone_name + " of " + finger_name
                        if finger_index == THUMB_INDEX:
                            if j == index_y:
                                indices.append([i + 3, index_y - 4, k])
                            if bone == Bone.METACARPAL:
                                annotation = [finger_name]
                                if j == index_y:
                                    if i == index_x:
                                        indices.append([i + 1, index_y - 4, k])
                                else:
                                    if i == index_x:
                                        indices.append([i, index_y - 4, k])
                        node = Virtual_node(parent_node_id, annotation, [a0, a1, a2, a3])
                        vnm.set_vnode_to_indices(node, indices, False)
                parent_node_id += 1
            index_x += n_elements_along
        if finger_index < THUMB_INDEX - 1:
            parent_node_id += elements_along_palm
            index_y += columns_per_finger
        else:
            index_y += 2 * columns_per_finger
    return vnm, node_network, node_identifier


def generate_external_node_matrix(
    hand_elements_along,
    node_identifier,
    vnm: Virtual_node_matrix,
    node_network: Node_network,
    options,
):
    parent_node_id = sum(hand_elements_along) + 1
    elements_along_palm = sum(hand_elements_along[0:2]) + 1
    elements_along_finger = sum(hand_elements_along[2:]) - 1
    n_internal_rows = options.get("Internal elements across hand")
    columns_per_finger = options.get("Number of matrix columns per finger")
    n_finger_internal_cols = options.get("Internal elements along each finger")
    n_palm_internal_cols = 4 * n_finger_internal_cols
    finger_index_x = elements_along_palm + 2  # First P.phalanx element is two-sided
    finger_names = options.get("finger names")
    bone_names = options.get("bone names")
    THUMB_INDEX = 4
    # TODO calculate formulas for these numbers, based on the width of the internal boxes.
    # TODO reformulate how these numbers are calculated.
    a_palm = options.get("Magnitude of major axis for palm ellipses")
    b_palm = options.get("Magnitude of minor axis for palm ellipses")
    a_finger = options.get("Magnitude of major axis for finger ellipses")
    b_finger = options.get("Magnitude of minor axis for finger ellipses")
    palm_ellipses = [None for i in range(sum(hand_elements_along[0:2]) + 1)]
    palm_bones = [Bone.CARPAL, Bone.METACARPAL, Bone.PROX_PHALANX]
    for n in range(elements_along_palm):
        x, d1, d2, d3 = node_network.get_node_parameters(parent_node_id)
        x = add(x, d2)
        ## TODO get rid of the multiple function calls
        ## simplify everything into one simple function
        ellipse = Ellipse(
            [x, d1, d2, d3], a_palm, b_palm, n_internal_rows, n_palm_internal_cols
        )
        ellipse.create_sampling_angles_along_axis()
        ellipse.sample_ellipse_along_angles()
        ellipse.sort_ellipse_samples()
        palm_ellipses[n] = ellipse.get_ellipse_samples()
        parent_node_id += 1
    for n in range(elements_along_palm - 1):
        ellipse_x, ellipse_d1, _ = palm_ellipses[n]
        next_ellipse_x, next_ellipse_d1, _ = palm_ellipses[n + 1]
        for f in range(5):
            for k in range(1, n_internal_rows + 2):
                if ellipse_x[k][f] is not None:
                    ## TODO calculate the d1s more carefully on the last ellipse
                    d1 = sub(next_ellipse_x[k][f], ellipse_x[k][f])
                    ellipse_d1[k][f] = next_ellipse_d1[k][f] = d1
    parent_node_id = 1
    index_y = 1
    d3 = None
    # d1_offset = 0.0
    # TODO formulate k_vals in terms of the number of internal rows
    # TODO K_vals for the elemnents on the sides will probably need 3 elements instead of 2.
    # Palm elements
    for finger_index in range(4):
        index_x = 0
        # Carpal, metacarpal and proximal phalanx
        for bone in palm_bones:
            n_elements_along = hand_elements_along[bone]
            if finger_index == 0:
                j_val = [index_y, index_y + 1]
            else:
                j_val = [index_y + 1]
            if bone in [Bone.CARPAL, Bone.METACARPAL]:
                i_val = [index_x + i for i in range(n_elements_along)]
            else:
                i_val = [index_x]
            for i in i_val:
                ellipse = palm_ellipses[i]
                for j in j_val:
                    # TODO make k_val variable
                    if j == index_y or (finger_index == 3 and j == index_y + 1):
                        k_val = vnm.get_kval(inner=True)
                    else:
                        k_val = vnm.get_kval(inner=False)
                    for k in k_val:
                        # TODO fix bug on ellipse sampling function
                        ellipse_x, ellipse_d1, ellipse_d2 = ellipse
                        a1 = finger_index if j == index_y else finger_index + 1
                        a2 = -1 if j == index_y else 1
                        a3 = -1 if k == 1 else 1
                        x = ellipse_x[k][a1]
                        d1 = ellipse_d1[k][a1]
                        d2 = ellipse_d2[k][a1]
                        # TODO figure out the offsetting? (maybe out of scope)
                        # if bone > 0:
                        # d1 = add(d1, set_magnitude(d1, d1_offset/(c+mc+pp)))
                        indices = [[i, j, k + a3]] if k in [1, n_internal_rows + 1] else []
                        annotation = [""]
                        if bone == Bone.CARPAL:
                            if j == index_y:
                                indices.append([i, j + a2, k])
                            elif finger_index == 3 and j == index_y + 1:
                                indices.append([i, j + a2, k])
                                indices.append([i, j + 4, k + a3])
                            else:
                                indices.append([i, j + 4, k + a3])
                        elif bone == Bone.METACARPAL:
                            if j == index_y:
                                indices.append([i, j + a2, k])
                            elif finger_index == 3 and j == index_y + 1:
                                if i > 1:
                                    indices.append([i, j + a2, k])
                            else:
                                indices.append([i, j + 4, k + a3])
                        elif bone == Bone.PROX_PHALANX:
                            n_elements_along = 1
                            finger_name = (
                                finger_names[finger_index]
                                if j == index_y
                                else finger_names[finger_index + 1]
                            )
                            finger_name = finger_names[3] if finger_index == 3 else finger_name
                            bone_name = bone_names[Bone.PROX_PHALANX]
                            finger_part = bone_name + " of " + finger_name
                            annotation = [bone_name, finger_name, finger_part]
                            if k in [1, n_internal_rows + 1]:
                                indices.append([i + 2, j, k + a3])
                            if j == index_y:
                                indices.append([i, j + a2, k])
                                indices.append([i + 2, j + a2, k])
                            elif finger_index == 3 and j == index_y + 1:
                                indices.append([i, j + a2, k])
                                indices.append([i + 2, j + a2, k])
                            else:
                                indices.append([i, j + 4, k + a3])
                                indices.append([i + 2, j + 4, k + a3])
                        node = Virtual_node(
                            x=[1, 0, 0, 0],
                            d1=[0, 1, 0, 0],
                            d2=[0, 0, 1, 0],
                            node_identifier=node_identifier,
                            annotation=annotation,
                        )
                        vnm.set_vnode_to_indices(node, indices)
                        node_network.set_node_parameters(
                            [x, d1, d2, d3], node_identifier, internal_node=False
                        )
                        node_identifier += 1
                parent_node_id += 1
            index_x += n_elements_along
        index_y += columns_per_finger
        parent_node_id += elements_along_finger
    index_y = 1
    parent_node_id = elements_along_palm + 1
    # Finger elements
    for finger_index in range(5):
        if finger_index == THUMB_INDEX:
            finger_bones = [Bone.METACARPAL, Bone.PROX_PHALANX, Bone.DIST_PHALANX]
            index_x = hand_elements_along[0]
        else:
            if hand_elements_along[2] > 1:
                finger_bones = [Bone.PROX_PHALANX, Bone.MID_PHALANX, Bone.DIST_PHALANX]
            else:
                finger_bones = [Bone.MID_PHALANX, Bone.DIST_PHALANX]
            index_x = finger_index_x
        for bone in finger_bones:
            n_elements_along = hand_elements_along[bone]
            if bone in [Bone.METACARPAL, Bone.PROX_PHALANX]:
                n_elements_along = max(1, n_elements_along - 1)
            j_val = [index_y, index_y + 1]
            i_val = [index_x + i for i in range(n_elements_along)]
            for i in i_val:
                parent_node_params = node_network.get_node_parameters(parent_node_id)
                ## TODO get rid of the multiple function calls
                ## simplify everything into one simple function
                ellipse = Ellipse(
                    parent_node_params,
                    a_finger,
                    b_finger,
                    n_internal_rows,
                    n_finger_internal_cols,
                )
                ellipse.create_sampling_angles_uniform()
                ellipse.sample_ellipse_along_angles()
                ellipse.sort_ellipse_samples()
                ellipse_x, ellipse_d1, ellipse_d2 = ellipse.get_ellipse_samples()
                is_fingertip = bone == Bone.DIST_PHALANX and i == i_val[-1]
                if is_fingertip:
                    cap_d11, cap_d12, cap_d21, cap_d22 = ellipse.get_fingercap_scale_factors()
                a1 = 0
                for j in j_val:
                    if j in [j_val[0], j_val[-1]]:
                        k_val = vnm.get_kval(inner=True)
                    else:
                        k_val = vnm.get_kval(inner=False)
                    for k in k_val:
                        a1 = 0 if j == index_y else 1
                        a2 = -1 if j == index_y else 1
                        a3 = -1 if k == 1 else 1
                        x = ellipse_x[k][a1]
                        d1 = ellipse_d1[k][a1]
                        d2 = ellipse_d2[k][a1]
                        finger_name = finger_names[finger_index]
                        bone_name = bone_names[bone]
                        finger_part = bone_name + " of " + finger_name
                        annotation = [bone_name, finger_name, finger_part]
                        ## TODO replace this check
                        indices = [[i, j, k + a3]] if k in [1, n_internal_rows + 1] else []
                        indices.append([i, j + a2, k])
                        node = Virtual_node(
                            node_identifier=node_identifier,
                            annotation=annotation,
                            x=[1, 0, 0, 0],
                            d1=[0, 1, 0, 0],
                            d2=[0, 0, 1, 0],
                        )
                        vnm.set_vnode_to_indices(node, indices)
                        node_network.set_node_parameters(
                            [x, d1, d2, d3], node_identifier, internal_node=False
                        )
                        if is_fingertip:
                            # The last ring is the base of the fingercap, made of front-facing
                            # virtual nodes at i + 1. The pre-cap faces shared with it
                            # (local nodes 2-4-6-8) use the same d2 on their outer edge, so
                            # both faces have the same curvature. d21 holds the upper and
                            # lower halves, for the elements above and below the node.
                            # TODO move the last row of inner nodes of the distal phalanx
                            # slightly behind the last ring of external nodes. They are
                            # currently coplanar, giving a zero jacobian at the 4 inner
                            # corners of each fingercap element. Either move the internal
                            # nodes backward or displace the external nodes forward.
                            d11 = cap_d11[k][a1]
                            d12 = cap_d12[k][a1]
                            d21 = cap_d21[k][a1]
                            d22 = cap_d22[k][a1]
                            cap_annotation = [finger_name]
                            if k in [1, n_internal_rows + 1]:
                                node = Virtual_node(
                                    node_identifier=node_identifier,
                                    annotation=annotation,
                                    x=[1, 0, 0, 0],
                                    d1=[0, 1, 0, 0],
                                    d2=[0, a3 * d11, a3 * d12, 0],
                                )
                                vnm.set_vnode_to_indices(node, [[i, j, k + a3]])
                            for k_allocation_mode, d21_half in zip(["upper", "lower"], d21):
                                node = Virtual_node(
                                    node_identifier=node_identifier,
                                    annotation=annotation,
                                    x=[1, 0, 0, 0],
                                    d1=[0, 1, 0, 0],
                                    d2=[0, -a2 * d21_half, -a2 * d22, 0],
                                )
                                vnm.set_vnode_to_indices(
                                    node, [[i, j + a2, k]], k_allocation_mode=k_allocation_mode
                                )
                                node = Virtual_node(
                                    node_identifier=node_identifier,
                                    annotation=cap_annotation,
                                    x=[1, 0, 0, 0],
                                    d1=[0, d11, d12, 0],
                                    d2=[0, d21_half, d22, 0],
                                )
                                vnm.set_vnode_to_indices(
                                    node, [[i + 1, j, k]], k_allocation_mode=k_allocation_mode
                                )
                        node_identifier += 1
                        a1 += 1
                parent_node_id += 1
            index_x += n_elements_along
        if finger_index < THUMB_INDEX - 1:
            parent_node_id += elements_along_palm
            index_y += columns_per_finger
        else:
            index_y += 2 * columns_per_finger
    # Palm-connection
    # TODO rewrite this whole section, it will not survive the change from 4- to 6-around
    # j_val = [3, 8, 13]
    # i = c + mc + 2  # First row of PP elements
    # bone_name = bone_names[2]
    # finger_index = 0
    # for j in j_val:
    #     for k in k_val:
    #         a3 = -1 if k == 1 else 1
    #         c1 = 0.25 * a3
    #         finger_name = finger_names[finger_index]
    #         finger_part = bone_name + ' of ' + finger_name
    #         annotation = [bone_name, finger_name, finger_part]
    #         index = [i, j - 1, k + a3]
    #         node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #         node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, -c1, 0],
    #             d2 = [0, 0, 1, 0],
    #             node_identifier = node_id, annotation = annotation
    #         )
    #         virtual_node_matrix.set_virtual_node_to_index(node, [i, j - 1, k + a3])
    #         node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, -c1, 0],
    #             d2 = [0, a3, 0, 0],
    #             node_identifier = node_id, annotation = annotation
    #         )
    #         virtual_node_matrix.set_virtual_node_to_index(node, [i, j, k])

    #         finger_name = finger_names[finger_index + 1]
    #         finger_part = bone_name + ' of ' + finger_name
    #         annotation = [bone_name, finger_name, finger_part]
    #         node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, c1, 0],
    #             d2 = [0, -a3, 0, 0],
    #             node_identifier = node_id, annotation = annotation
    #         )
    #         virtual_node_matrix.set_virtual_node_to_index(node, [i, j + 2, k])

    #         node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, c1, 0],
    #             d2 = [0, 0, 1, 0],
    #             node_identifier = node_id, annotation = annotation
    #         )
    #         virtual_node_matrix.set_virtual_node_to_index(node, [i, j + 3, k + a3])
    #     finger_index += 1
    """
    """
    # The Thumb-palm connection elements
    # TODO perish.
    # TODO the day has come
    # TODO when I have to figure this out.
    # It is against every fiber of my being that I am forced to set these manually
    # index_y = 26
    # thumb_angle_degrees = options["Thumb angle"]
    # thumb_c = thumb_angle_degrees / 90
    # # Back of the thumb
    # # TODO reformulate k_val, and make it consistent with the values used at the palm
    # k_val = [0, 3]
    # for k in k_val:
    #     a1 = -1 if k == 3 else 1
    #     a3 = -1 if k == 3 else 1
    #     index = [0, index_y - 9, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 0.5, -a1 * 0.5, 0],
    #             d2 = [0, 0, 1, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [0, index_y - 5, k + a3])
    #     index = [1, index_y + 1, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, a1 * thumb_c, 0],
    #             d2 = [0, 0, 1, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [0, index_y - 4, k + a3])

    # # Bottom carpal-metacarpal connection
    # for k in k_val:
    #     a1 = -1 if k == 3 else 1
    #     index = [1, index_y + 1, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 0, a1, 0],
    #             d2 = [0, -a1, -thumb_c, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )

    #     virtual_node_matrix.set_virtual_node_to_index(node, [1, index_y - 4, k])

    #     index = [1, index_y, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 0, a1, 0],
    #             d2 = [0, -a1 * thumb_c, 0, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [2, index_y - 4, k])

    #     index = [0, index_y - 9, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, 0, 0],
    #             d2 = [0, -a1 * 0.5, 0.5, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [1, index_y - 5, k])

    #     index = [1, index_y - 9, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, 0, 0],
    #             d2 = [0, 0, 0.25, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [2, index_y - 5, k])
    # # Intermediate metacarpal connections
    # for k in k_val:
    #     a1 = 2 * thumb_c
    #     a3 = -1 if k == 3 else 1
    #     index = [1, index_y, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, 0, 0],
    #             d2 = [0, -a3 * thumb_c, 0, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [4, index_y - 4, k])

    #     index = [1, index_y - 9, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, 0, 0],
    #             d2 = [0, 0, 0.25, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [4, index_y - 5, k])

    #     index = [2, index_y, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, 0, 0],
    #             d2 = [0, -a1 * a3, 0.5, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [5, index_y - 4, k])

    #     index = [2, index_y - 9, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, 1, 0, 0],
    #             d2 = [0, a1 * a3, 0.5, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [5, index_y - 5, k])

    # # Front of the webbing
    # for k in k_val:
    #     a1 = 2 * thumb_c
    #     a3 = -1 if k == 3 else 1
    #     index = [2, index_y - 9, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, -a1, -a3 * 0.5, 0],
    #             d2 = [0, 0, 1, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [6, index_y - 5, k + a3])

    #     index = [2, index_y, k]
    #     node_id = virtual_node_matrix.get_node_identifier_at_index(index)
    #     node = virtual_node_matrix.create_external_vnode(
    #             x  = [1, 0, 0, 0],
    #             d1 = [0, a1, -a3 * 0.5, 0],
    #             d2 = [0, 0, -1, 0],
    #             node_identifier = node_id, annotation = annotation
    #     )
    #     virtual_node_matrix.set_virtual_node_to_index(node, [6, index_y - 4, k + a3])

    return vnm, node_network, node_identifier


def create_cube_element(
    fieldmodule: Fieldmodule,
    element_identifier: int,
    vnm: Virtual_node_matrix,
    node_indices: list,
) -> list:
    ## TODO add tricubic configuration
    """
    A cube element is made of 8 corners, shown here in the 'canonical' orientation
      8----6
     /|   /|
    7----5 |
    | 4 -|-2
    |/   |/
    3----1
    with an axis given by
        z  x
        | /
     y--+
    There are six possible orientations, everyone associated with one face of the cube.
    All six possible orientations can be mapped back into the canonical orientation
    by the following transformation
    1) (x, y, z) (unchanged)
    2) (x, z, -y)
    3) (x, -y, -z)
    4) (x, -x, y)
    5) (-z, x, -y)
    6) (z, x, -y)
    Every possible orientation has a different ordering of the corners.
    for example, in orientation 2), the corners are ordered in the following way
      6----2
     /|   /|
    5----1 |
    | 8 -|-4
    |/   |/
    7----3
    which gives us an axis
           x
          /
    z<---+
         |
         y
    The first part of the function finds the correct orientation of the cube,
    so that the bicubic face of the element (if it exists) is always made up of
    corners 5, 6, 7 and 8. This is less important in the case of trilinear elements.
    Immediately after, the eight corner of the cube are reorganized according to this
    transformation and the value scale factors are collected from each virtual node.
    A virtual node is comprised of lists of scale factors in the format
    [parent_node_id, value_scale_factor, d1_scale_factor, d2_scale_factor, d3_scale_factor].
    These are mapped into a list of expression terms and a dictionary which
    keeps track of the mapping between the real value and the element id of the scale factor.
    If at least 1 node is tagged as external, the function also collects expression terms
    for the d1 and d2 directions of the corner.
    The d1 and d2 direction of the 4 linear nodes (1, 2, 3, 4)
    are given by the following expressions,
    where v1, d11 and d21 are the position, d1 and d2 directions of node 1
    d11 = v2 - v1,     d12 = v3 - v1
    d21 = v2 - v1,     d22 = v4 - v2
    d31 = v4 - v3,     d32 = v3 - v1
    d41 = v4 - v3,     d42 = v4 - v2

    :param fieldmodule: Description
    :param element_identifier: Description
    :param node_matrix: Description
    TODO fix docstring
    :param ix: Description
    :param iy: Description
    :param iz: Description
    :return: List containing a result code and a handle to the element
        (None if the element construction failed)
    :rtype: list
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    # Criteria to identify the z-axis of the element
    ix, iy, iz = node_indices
    node_1 = vnm.get_vnode([ix, iy, iz])
    node_2 = vnm.get_vnode([ix + 1, iy, iz])
    node_3 = vnm.get_vnode([ix, iy + 1, iz])
    if node_1 is None or node_2 is None or node_3 is None:
        return -2, None
    # 6 possible cases, one for each face of a cube
    if not node_1.is_external and not node_2.is_external and not node_3.is_external:
        ranges = [[0, 1], [0, 1], [0, 1]]  # 1
        order = [2, 1, 0]
    elif not node_1.is_external and not node_2.is_external and node_3.is_external:
        ranges = [[0, 1], [0, 1], [1, 0]]  # 2
        order = [1, 2, 0]
    elif node_1.is_external and node_2.is_external and node_3.is_external:
        ranges = [[0, 1], [1, 0], [1, 0]]  # 3
        order = [2, 1, 0]
    elif node_1.is_external and node_2.is_external and not node_3.is_external:
        ranges = [[0, 1], [1, 0], [0, 1]]  # 4
        order = [1, 2, 0]
    elif node_1.is_external and not node_2.is_external and node_3.is_external:
        ranges = [[1, 0], [0, 1], [1, 0]]  # 5
        order = [0, 2, 1]
    else:
        ranges = [[0, 1], [0, 1], [0, 1]]  # 6
        order = [0, 2, 1]
    # Obtained the ordered list of indices to parse through
    reordered_ranges = [ranges[idx] for idx in order]
    indices = []
    # Create the cartersian product
    for prod in product(*reordered_ranges):
        row = [0] * len(ranges)
        # Map the generated values back to their correct positions
        for i, val in enumerate(prod):
            original_axis = order[i]
            row[original_axis] = val
        indices.append(row)
    is_bicubic = False
    local_node = 0
    local_node_ids = {}
    value_expression_terms = {}
    global_to_local_scale_factor_ids = {}
    d1_expression_terms = {}
    d2_expression_terms = {}
    negative_value_ets = {}
    readable_negative_ets = {}
    n_local_nodes = 0
    n_scale_factors = 0
    value_labels = [
        0,
        Node.VALUE_LABEL_VALUE,
        Node.VALUE_LABEL_D_DS1,
        Node.VALUE_LABEL_D_DS2,
        Node.VALUE_LABEL_D_DS3,
    ]
    value_label_names = [0, "v", "d1", "d2", "d3"]
    local_node = 0
    for index in indices:
        et = []
        neg_et = []
        ret = []
        i, j, k = index
        local_node += 1
        # TODO change the function to instead get the node from the vnm object and keep hints
        vnode = vnm.get_vnode([ix + i, iy + j, iz + k])
        if vnode is None:
            return -2, None
        if vnode.is_external:
            is_bicubic = True
        vnode_scalefactors = vnode.get_scalefactors(Node.VALUE_LABEL_VALUE)
        # Value expression terms
        for global_node in vnode_scalefactors:
            for factor in range(len(global_node)):
                if factor == 0:
                    # zero factor is always the node_id
                    global_node_id = global_node[factor]
                    if global_node_id not in local_node_ids:
                        n_local_nodes += 1
                        local_node_ids[global_node_id] = n_local_nodes
                else:
                    global_scale_factor = global_node[factor]
                    if global_scale_factor == 0:
                        continue
                    if global_scale_factor not in global_to_local_scale_factor_ids:
                        n_scale_factors += 1
                        global_to_local_scale_factor_ids[global_scale_factor] = n_scale_factors
                    local_scale_factor_id = [
                        global_to_local_scale_factor_ids[global_scale_factor]
                    ]
                    et.append(
                        (
                            local_node_ids[global_node_id],
                            value_labels[factor],
                            local_scale_factor_id,
                        )
                    )
            value_expression_terms[local_node] = et
    local_to_global_scale_factor_ids = {
        value: key for key, value in global_to_local_scale_factor_ids.items()
    }
    if is_bicubic:
        # Create 'negative' value expression terms
        for local_node in [1, 2, 3]:
            neg_et = []
            red_et = []
            et = value_expression_terms[local_node]
            for term in et:
                l_node_id = term[0]
                label = term[1]
                scale_factor_id = term[2][0]
                global_scale_factor = -local_to_global_scale_factor_ids[scale_factor_id]
                if global_scale_factor not in global_to_local_scale_factor_ids:
                    n_scale_factors += 1
                    global_to_local_scale_factor_ids[global_scale_factor] = n_scale_factors
                neg_et.append(
                    (l_node_id, label, global_to_local_scale_factor_ids[global_scale_factor])
                )
            negative_value_ets[local_node] = neg_et
            readable_negative_ets[local_node] = red_et
        # Create d1 expression terms
        # d1 and d2 terms for the linear part follow a simple formula
        # d1 and d2 terms for the cubic part are calculated from the virtual node information
        d1_expression_terms[1] = value_expression_terms[2] + negative_value_ets[1]
        d1_expression_terms[2] = value_expression_terms[2] + negative_value_ets[1]
        d1_expression_terms[3] = value_expression_terms[4] + negative_value_ets[3]
        d1_expression_terms[4] = value_expression_terms[4] + negative_value_ets[3]
        label = Node.VALUE_LABEL_D_DS1
        for local_node in range(5, 9):
            et = []
            ret = []
            i, j, k = indices[local_node - 1]
            vnode = vnm.get_vnode([ix + i, iy + j, iz + k])
            vnode_scalefactors = vnode.get_scalefactors(label)
            if vnode_scalefactors is None:
                return -1, None
            for global_node in vnode_scalefactors:
                for factor in range(len(global_node)):
                    if factor == 0:
                        # Check for node_ids
                        global_node_id = global_node[factor]
                        if global_node_id not in local_node_ids:
                            n_local_nodes += 1
                            local_node_ids[global_node_id] = n_local_nodes
                    else:
                        # Check for scale_factor_ids
                        scale_factor = global_node[factor]
                        if scale_factor == 0:
                            continue
                        if scale_factor not in global_to_local_scale_factor_ids:
                            n_scale_factors += 1
                            global_to_local_scale_factor_ids[scale_factor] = n_scale_factors
                        et.append(
                            (
                                local_node_ids[global_node_id],
                                value_labels[factor],
                                [global_to_local_scale_factor_ids[scale_factor]],
                            )
                        )
                        ret.append(
                            (
                                str(global_node_id).zfill(2),
                                value_label_names[factor],
                                str(scale_factor).zfill(4),
                            )
                        )
                d1_expression_terms[local_node] = et
        # create d2 expression terms
        d2_expression_terms[1] = value_expression_terms[3] + negative_value_ets[1]
        d2_expression_terms[2] = value_expression_terms[4] + negative_value_ets[2]
        d2_expression_terms[3] = value_expression_terms[3] + negative_value_ets[1]
        d2_expression_terms[4] = value_expression_terms[4] + negative_value_ets[2]
        label = Node.VALUE_LABEL_D_DS2
        for local_node in range(5, 9):
            et = []
            ret = []
            i, j, k = indices[local_node - 1]
            vnode = vnm.get_vnode([ix + i, iy + j, iz + k])
            vnode_scalefactors = vnode.get_scalefactors(label)
            if vnode_scalefactors is None:
                return -1, None
            for global_node in vnode_scalefactors:
                for factor in range(len(global_node)):
                    if factor == 0:
                        # Check for node_ids
                        global_node_id = global_node[factor]
                        if global_node_id not in local_node_ids:
                            n_local_nodes += 1
                            local_node_ids[global_node_id] = n_local_nodes
                    else:
                        # Check for scale_factor_ids
                        scale_factor = global_node[factor]
                        if scale_factor == 0:
                            continue
                        if scale_factor not in global_to_local_scale_factor_ids:
                            n_scale_factors += 1
                            global_to_local_scale_factor_ids[scale_factor] = n_scale_factors
                        et.append(
                            (
                                local_node_ids[global_node_id],
                                value_labels[factor],
                                [global_to_local_scale_factor_ids[scale_factor]],
                            )
                        )
                        ret.append(
                            (
                                str(global_node_id).zfill(2),
                                value_label_names[factor],
                                str(scale_factor).zfill(4),
                            )
                        )
                d2_expression_terms[local_node] = et
    # Create and remap eft
    if is_bicubic:
        # Bicubic linear element (skin)
        bicubic_linear_basis = fieldmodule.createElementbasis(
            3, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY
        )
        bicubic_linear_basis.setFunctionType(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        eft = mesh3d.createElementfieldtemplate(bicubic_linear_basis)
        setEftScaleFactorIds(eft, [], [], n_scale_factors)
        if n_local_nodes > 8:
            local_node_indexes = [1, 2, 3, 4] + [n_local_nodes - 3 + i for i in range(4)]
            remapEftLocalNodes(eft, n_local_nodes, local_node_indexes)
            # Re-numbering the ets dictionaries the new local node indexes
            for ets in [value_expression_terms, d1_expression_terms, d2_expression_terms]:
                temp_dict = {i: ets.pop(i) for i in range(5, 9)}
                for i in range(4):
                    ets[n_local_nodes - 3 + i] = temp_dict.get(5 + i)
        for local_node in value_expression_terms:
            remapEftNodeValueLabelWithNodes(
                eft, local_node, Node.VALUE_LABEL_VALUE, value_expression_terms[local_node]
            )
            remapEftNodeValueLabelWithNodes(
                eft, local_node, Node.VALUE_LABEL_D_DS1, d1_expression_terms[local_node]
            )
            remapEftNodeValueLabelWithNodes(
                eft, local_node, Node.VALUE_LABEL_D_DS2, d2_expression_terms[local_node]
            )
        if n_local_nodes <= 8:
            remapEftLocalNodes(eft, n_local_nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        # Create element template
        etemplate = mesh3d.createElementtemplate()
        etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        result = etemplate.defineField(coordinates, -1, eft)
        if result != RESULT_OK:
            return result, None
    else:
        # Trilinear element (bone)
        trilinear_basis = fieldmodule.createElementbasis(
            3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE
        )
        eft = mesh3d.createElementfieldtemplate(trilinear_basis)
        setEftScaleFactorIds(eft, [], [], n_scale_factors)
        for local_node in value_expression_terms:
            remapEftNodeValueLabelWithNodes(
                eft, local_node, Node.VALUE_LABEL_VALUE, value_expression_terms[local_node]
            )
        remapEftLocalNodes(eft, n_local_nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        # Create element template
        etemplate = mesh3d.createElementtemplate()
        etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        result = etemplate.defineField(coordinates, -1, eft)
        if result != RESULT_OK:
            return result, None
    # Create element
    element = mesh3d.createElement(element_identifier, etemplate)
    node_ids = list(local_node_ids.keys())
    element.setNodesByIdentifier(eft, node_ids)
    scale_factors = list(global_to_local_scale_factor_ids.keys())
    element.setScaleFactors(eft, scale_factors)
    return result, element


def sampling_coefficient(k, n):
    """
    Samples ihe [-1, 1] interval in n segments.

    :param k: Integer from 1 to n + 1
    :param n: Total number of segments
    :return: Real number between [-1, 1]
    """
    return (2 * k - n - 2) / n


def setNodeFieldParameters(
    field, fieldcache, x, d1=None, d2=None, d3=None, d12=None, d13=None
):
    """
    Assign node field parameters x, d1, d2, d3 of field.
    :param field: Field parameters to assign.
    :param fieldcache: Fieldcache with node set.
    :param x: Parameters to set for Node.VALUE_LABEL_VALUE.
    :param d1: Parameters to set for Node.VALUE_LABEL_D_DS1.
    :param d2: Parameters to set for Node.VALUE_LABEL_D_DS2.
    :param d3: Parameters to set for Node.VALUE_LABEL_D_DS3.
    :param d12: Optional parameters to set for Node.VALUE_LABEL_D2_DS1DS2.
    :param d13: Optional parameters to set for Node.VALUE_LABEL_D2_DS1DS3.
    :return:
    """
    field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
    if d1:
        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, d1)
    if d2:
        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, d2)
    if d3:
        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)
    if d12:
        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, d12)
    if d13:
        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, d13)
