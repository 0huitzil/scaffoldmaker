"""
Generates a lung scaffold with common hilum but open fissures with lobes built from 1/6 ellipsoid segments.
"""
from cmlibs.maths.vectorops import magnitude
from cmlibs.utils.zinc.field import find_or_create_field_coordinates
from cmlibs.zinc.element import Element
from cmlibs.zinc.field import Field

from scaffoldmaker.annotation.annotationgroup import (
    AnnotationGroup, findAnnotationGroupByName, findOrCreateAnnotationGroupForTerm, getAnnotationGroupForTerm)
from scaffoldmaker.annotation.lung_terms import get_lung_term
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.utils.geometry import getEllipsePointAtTrueAngle
from scaffoldmaker.utils.interpolation import getNearestLocationOnCurve
from scaffoldmaker.utils.meshrefinement import MeshRefinement
from scaffoldmaker.utils.ellipsoidmesh import EllipsoidMesh, EllipsoidSurfaceD3Mode
from scaffoldmaker.utils.zinc_utils import get_mesh_first_element_with_node, translate_nodeset_coordinates

import copy
import math


class MeshType_3d_lung4(Scaffold_base):
    """
    Generates a lung scaffold with common hilum but open fissures with lobes built from 1/6 ellipsoid segments.
    """

    @classmethod
    def getName(cls):
        return "3D Lung 4"

    @classmethod
    def getParameterSetNames(cls):
        return [
            "Default",
            "Human 1 Coarse",
            "Human 1 Medium",
            "Human 1 Fine",
            "Ellipsoid"
        ]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {}
        useParameterSetName = "Human 1 Coarse" if (parameterSetName == "Default") else parameterSetName
        options["Left lung"] = True
        options["Right lung"] = True
        options["Ellipsoid height"] = 1.0
        options["Ellipsoid dorsal-ventral size"] = 0.8
        options["Ellipsoid medial-lateral size"] = 0.4
        options["Left-right lung spacing"] = 0.5
        options["Lower lobe extension"] = 0.3
        options["Refine"] = False
        options["Refine number of elements"] = 4

        if "Medium" in useParameterSetName:
            options["Number of elements lateral"] = 4
            options["Number of elements lower lobe extension"] = 3
            options["Number of elements oblique"] = 8
            options["Number of transition elements"] = 1
        elif "Fine" in useParameterSetName:
            options["Number of elements lateral"] = 6
            options["Number of elements lower lobe extension"] = 4
            options["Number of elements oblique"] = 12
            options["Number of transition elements"] = 1
        else:
            options["Number of elements lateral"] = 4
            options["Number of elements lower lobe extension"] = 2
            options["Number of elements oblique"] = 6
            options["Number of transition elements"] = 1

        if "Human" in useParameterSetName:
            options["Lower lobe base concavity"] = 0.1
            options["Ventral edge sharpness factor"] = 0.9
            options["Cardiac curvature"] = 2.0
        else:
            options["Lower lobe base concavity"] = 0.0
            options["Ventral edge sharpness factor"] = 0.0
            options["Cardiac curvature"] = 0.0

        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Left lung",
            "Right lung",
            "Number of elements lateral",
            "Number of elements lower lobe extension",
            "Number of elements oblique",
            "Number of transition elements",
            "Ellipsoid height",
            "Ellipsoid dorsal-ventral size",
            "Ellipsoid medial-lateral size",
            "Left-right lung spacing",
            "Lower lobe base concavity",
            "Lower lobe extension",
            "Ventral edge sharpness factor",
            "Cardiac curvature",
            "Refine",
            "Refine number of elements"
        ]

    @classmethod
    def checkOptions(cls, options):
        dependent_changes = False

        max_transition_count = None
        for key in [
            "Number of elements lateral",
            "Number of elements oblique"
        ]:
            if options[key] < 4:
                options[key] = 4
            elif options[key] % 2:
                options[key] += 1
            transition_count = (options[key] // 2) - 1
            if (max_transition_count is None) or (transition_count < max_transition_count):
                max_transition_count = transition_count

        if options["Number of transition elements"] < 1:
            options["Number of transition elements"] = 1
        elif options["Number of transition elements"] > max_transition_count:
            options["Number of transition elements"] = max_transition_count
            dependent_changes = True

        for key, default in [
            ("Ellipsoid height", 1.0),
            ("Ellipsoid dorsal-ventral size", 0.8),
            ("Ellipsoid medial-lateral size", 0.4)
        ]:
            if options[key] <= 0.0:
                options[key] = default
        for key in [
            "Lower lobe base concavity",
            "Lower lobe extension",
            "Cardiac curvature"
        ]:
            if options[key] < 0.0:
                options[key] = 0.0
        if options["Lower lobe extension"] == 0.0:
            options["Number of elements lower lobe extension"] = 0
            dependent_changes = True
        elif options["Number of elements lower lobe extension"] < 1:
            options["Number of elements lower lobe extension"] = 1
            dependent_changes = True
        depth = options["Ellipsoid dorsal-ventral size"]
        height = options["Ellipsoid height"]
        max_extension = 0.99 * magnitude(getEllipsePointAtTrueAngle(depth / 2.0, height / 2.0, math.pi / 3.0))
        if options["Lower lobe extension"] > max_extension:
            options["Lower lobe extension"] = max_extension

        if options["Left-right lung spacing"] < 0.0:
            options["Left-right lung spacing"] = 0.0

        for dimension in [
            "Ventral edge sharpness factor"
        ]:
            if options[dimension] < 0.0:
                options[dimension] = 0.0
            elif options[dimension] > 1.0:
                options[dimension] = 1.0

        if options['Refine number of elements'] < 1:
            options['Refine number of elements'] = 1

        return dependent_changes

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the base tricubic Hermite mesh. See also generateMesh().
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: list of AnnotationGroup, None
        """
        has_left_lung = options["Left lung"]
        has_right_lung = options["Right lung"]

        elements_count_lateral = options["Number of elements lateral"]
        elements_count_lower_extension = options["Number of elements lower lobe extension"]
        elements_count_oblique = options["Number of elements oblique"]
        elements_count_transition = options["Number of transition elements"]
        lung_spacing = options["Left-right lung spacing"] * 0.5
        lower_lobe_extension = options["Lower lobe extension"]
        lower_lobe_base_concavity = options["Lower lobe base concavity"]
        ventral_sharpness_factor = options["Ventral edge sharpness factor"]
        cardiac_curvature = options["Cardiac curvature"]
        ellipsoid_ml_size = options["Ellipsoid medial-lateral size"]
        ellipsoid_dv_size = options["Ellipsoid dorsal-ventral size"]
        ellipsoid_height = options["Ellipsoid height"]

        fieldmodule = region.getFieldmodule()
        fieldcache = fieldmodule.createFieldcache()
        coordinates = find_or_create_field_coordinates(fieldmodule)
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        mesh = fieldmodule.findMeshByDimension(3)

        # annotation groups & nodeset groups
        lung_group = AnnotationGroup(region, get_lung_term("lung"))

        if has_left_lung:
            left_lung_group = AnnotationGroup(region, get_lung_term("left lung"))
            lower_left_lung_group = AnnotationGroup(region, get_lung_term("lower lobe of left lung"))
            upper_left_lung_group = AnnotationGroup(region, get_lung_term("upper lobe of left lung"))
            left_anterior_lung_group = AnnotationGroup(region, ("anterior left lung", ""))
            left_lateral_lung_group = AnnotationGroup(region, ("lateral left lung", ""))
            left_medial_lung_group = AnnotationGroup(region, ("medial left lung", ""))
            left_annotation_groups = [left_lung_group, lower_left_lung_group, upper_left_lung_group,
                                      left_anterior_lung_group, left_lateral_lung_group, left_medial_lung_group]
        else:
            left_lung_group = lower_left_lung_group = upper_left_lung_group = None
            left_lateral_lung_group = left_medial_lung_group = left_anterior_lung_group = None
            left_annotation_groups = []

        if has_right_lung:
            right_lung_group = AnnotationGroup(region, get_lung_term("right lung"))
            lower_right_lung_group = AnnotationGroup(region, get_lung_term("lower lobe of right lung"))
            middle_right_lung_group = AnnotationGroup(region, get_lung_term("middle lobe of right lung"))
            upper_right_lung_group = AnnotationGroup(region, get_lung_term("upper lobe of right lung"))
            right_anterior_lung_group = AnnotationGroup(region, ("anterior right lung", ""))
            right_lateral_lung_group = AnnotationGroup(region, ("lateral right lung", ""))
            right_medial_lung_group = AnnotationGroup(region, ("medial right lung", ""))
            right_annotation_groups = [
                right_lung_group, lower_right_lung_group, middle_right_lung_group, upper_right_lung_group,
                right_anterior_lung_group, right_lateral_lung_group, right_medial_lung_group]
        else:
            right_lung_group = lower_right_lung_group = middle_right_lung_group = upper_right_lung_group = None
            right_anterior_lung_group = right_lateral_lung_group = right_medial_lung_group = None
            right_annotation_groups = []

        box_group = AnnotationGroup(region, ("box", ""))
        transition_group = AnnotationGroup(region, ("transition", ""))

        annotation_groups = [lung_group] + left_annotation_groups + right_annotation_groups + \
                            [box_group, transition_group]

        half_ml_size = ellipsoid_ml_size * 0.5
        half_dv_size = ellipsoid_dv_size * 0.5
        half_height = ellipsoid_height * 0.5

        pi__3 = math.pi / 3.0
        sin_pi__3 = math.sin(pi__3)

        left_lung, right_lung = 0, 1
        lungs = [lung for show, lung in [(has_left_lung, left_lung), (has_right_lung, right_lung)] if show]
        node_identifier, element_identifier = 1, 1

        marker_name_element_xi = []

        # need to build both left and right lungs then delete any not used so element, face, line, node numbers
        # are the same if only left or right
        for lung in [left_lung, right_lung]:

            if lung == left_lung:
                if has_left_lung:
                    lower_octant_group_lists = []
                    upper_octant_group_lists = []
                    for octant in range(8):
                        octant_group_list = [group.getGroup() for group in
                                             [lung_group, left_lung_group, lower_left_lung_group] +
                                             [left_medial_lung_group if (octant & 1) else left_lateral_lung_group]]
                        lower_octant_group_lists.append(octant_group_list)
                        octant_group_list = [group.getGroup() for group in
                                             [lung_group, left_lung_group, upper_left_lung_group] +
                                             [left_medial_lung_group if (octant & 1) else left_lateral_lung_group]]
                        if octant & 2:
                            octant_group_list.append(left_anterior_lung_group.getGroup())
                        upper_octant_group_lists.append(octant_group_list)
                else:
                    lower_octant_group_lists = upper_octant_group_lists = None
                middle_octant_group_lists = None
            else:
                if has_right_lung:
                    lower_octant_group_lists = []
                    middle_octant_group_lists = []
                    upper_octant_group_lists = []
                    for octant in range(8):
                        octant_group_list = [group.getGroup() for group in
                                             [lung_group, right_lung_group, lower_right_lung_group] +
                                             [right_lateral_lung_group if (octant & 1) else right_medial_lung_group]]
                        lower_octant_group_lists.append(octant_group_list)
                        octant_group_list = [group.getGroup() for group in
                                             [lung_group, right_lung_group, middle_right_lung_group] +
                                             [right_lateral_lung_group if (octant & 1) else right_medial_lung_group]]
                        if octant & 2:
                            octant_group_list.append(right_anterior_lung_group.getGroup())
                        middle_octant_group_lists.append(octant_group_list)
                        octant_group_list = [group.getGroup() for group in
                                             [lung_group, right_lung_group, upper_right_lung_group] +
                                             [right_lateral_lung_group if (octant & 1) else right_medial_lung_group]]
                        if octant & 2:
                            octant_group_list.append(right_anterior_lung_group.getGroup())
                        upper_octant_group_lists.append(octant_group_list)
                else:
                    lower_octant_group_lists = middle_octant_group_lists = upper_octant_group_lists = None

            element_counts = [elements_count_lateral, elements_count_oblique, elements_count_oblique]
            lower_ellipsoid = EllipsoidMesh(
                half_ml_size, half_dv_size, half_height, element_counts, elements_count_transition)
            lower_ellipsoid.set_surface_d3_mode(EllipsoidSurfaceD3Mode.SURFACE_NORMAL_PLANE_PROJECTION)
            lower_ellipsoid.set_box_transition_groups(box_group.getGroup(), transition_group.getGroup())
            upper_ellipsoid = EllipsoidMesh(
                half_ml_size, half_dv_size, half_height, element_counts, elements_count_transition)
            upper_ellipsoid.set_surface_d3_mode(EllipsoidSurfaceD3Mode.SURFACE_NORMAL_PLANE_PROJECTION)
            upper_ellipsoid.set_box_transition_groups(box_group.getGroup(), transition_group.getGroup())
            if lung == right_lung:
                middle_ellipsoid = EllipsoidMesh(
                    half_ml_size, half_dv_size, half_height, element_counts, elements_count_transition)
                middle_ellipsoid.set_surface_d3_mode(EllipsoidSurfaceD3Mode.SURFACE_NORMAL_PLANE_PROJECTION)
                middle_ellipsoid.set_box_transition_groups(box_group.getGroup(), transition_group.getGroup())
            else:
                middle_ellipsoid = upper_ellipsoid
            half_counts = [count // 2 for count in element_counts]
            octant1 = middle_ellipsoid.build_octant(half_counts, -pi__3, 0.0)
            middle_ellipsoid.merge_octant(octant1, quadrant=3)
            if lung == right_lung:
                middle_ellipsoid.copy_to_negative_axis1()

            # save hilum coordinates for all other lobes
            hilum_x = []
            ox = octant1.get_parameters()
            box_count1 = octant1.get_box_counts()[0]
            for n1 in range(element_counts[0] + 1):
                mirror_x = n1 < half_counts[0]
                o1 = abs(n1 - half_counts[0])
                parameters = ox[0][0][o1]
                obox = o1 <= box_count1
                parameters = [
                    copy.copy(parameters[0]),
                    copy.copy(parameters[3 if obox else 2]),
                    [-d for d in parameters[2 if obox else 1]],
                    copy.copy(parameters[1 if obox else 3])
                ]
                if mirror_x:
                    for i in range(3):
                        parameters[i][0] = -parameters[i][0]
                hilum_x.append(parameters)

            octant2 = upper_ellipsoid.build_octant(half_counts, 0.0, pi__3)
            upper_ellipsoid.merge_octant(octant2, quadrant=0)
            octant3 = upper_ellipsoid.build_octant(half_counts, pi__3, 2.0 * pi__3)
            upper_ellipsoid.merge_octant(octant3, quadrant=1)
            upper_ellipsoid.copy_to_negative_axis1()

            octant4 = lower_ellipsoid.build_octant(half_counts, 2.0 * pi__3, math.pi,
                                                   lower_lobe_extension, elements_count_lower_extension)
            # merge into separate lower ellipsoid to have space for extension elements
            lower_ellipsoid_mesh = EllipsoidMesh(
                half_ml_size, half_dv_size, half_height,
                [element_counts[0], element_counts[1], element_counts[2] + 2 * elements_count_lower_extension],
                elements_count_transition)
            lower_ellipsoid_mesh.set_surface_d3_mode(EllipsoidSurfaceD3Mode.SURFACE_NORMAL_PLANE_PROJECTION)
            lower_ellipsoid_mesh.set_box_transition_groups(box_group.getGroup(), transition_group.getGroup())
            lower_ellipsoid_mesh.merge_octant(octant4, quadrant=1)
            lower_ellipsoid_mesh.copy_to_negative_axis1()

            node_layout_manager = lower_ellipsoid.get_node_layout_manager()
            node_layout_permuted = node_layout_manager.getNodeLayoutRegularPermuted(d3Defined=True)
            for n1 in range(element_counts[0] + 1):
                lower_ellipsoid_mesh.set_node_parameters(
                    n1, half_counts[1], element_counts[2] + 2 * elements_count_lower_extension - half_counts[2],
                    hilum_x[n1], node_layout=node_layout_permuted)
            lower_ellipsoid_mesh.set_octant_group_lists(lower_octant_group_lists)
            node_identifier, element_identifier = lower_ellipsoid_mesh.generate_mesh(
                fieldmodule, coordinates, node_identifier, element_identifier)

            for ellipsoid in [middle_ellipsoid, upper_ellipsoid] if (lung == right_lung) else [upper_ellipsoid]:
                node_layout_manager = ellipsoid.get_node_layout_manager()
                node_layout_6way = node_layout_manager.getNodeLayout6Way12(d3Defined=True)
                for n1 in range(element_counts[0] + 1):
                    nid = (lower_ellipsoid_mesh.get_node_identifier(
                        n1, half_counts[1], element_counts[2] + 2 * elements_count_lower_extension - half_counts[2])
                           if (((lung == left_lung) and (n1 >= half_counts[0])) or
                               ((lung == right_lung) and (n1 <= half_counts[0]))) else None)
                    ellipsoid.set_node_parameters(n1, half_counts[1], half_counts[2], hilum_x[n1],
                                                        nid, node_layout=node_layout_6way)
                ellipsoid.set_octant_group_lists(
                    middle_octant_group_lists if ((ellipsoid == middle_ellipsoid) and
                                                  (ellipsoid != upper_ellipsoid)) else upper_octant_group_lists)
                node_identifier, element_identifier = ellipsoid.generate_mesh(
                    fieldmodule, coordinates, node_identifier, element_identifier)

            # find elements for marker points
            if ((lung == left_lung) and has_left_lung) or ((lung == right_lung) and has_right_lung):
                nid = upper_ellipsoid.get_node_identifier(
                    elements_count_lateral // 2, 0, element_counts[2])
                group = upper_left_lung_group if (lung == left_lung) else upper_right_lung_group
                if nid and group:
                    marker_name_element_xi.append((
                        "apex of left lung" if (lung == left_lung) else "apex of right lung",
                        get_mesh_first_element_with_node(
                            group.getMeshGroup(mesh), coordinates, nodes.findNodeByIdentifier(nid)),
                        [1.0, 1.0, 1.0]))
                nid = lower_ellipsoid_mesh.get_node_identifier(
                    elements_count_lateral // 2, 0, elements_count_oblique // 2 + elements_count_lower_extension)
                group = lower_left_lung_group if (lung == left_lung) else lower_right_lung_group
                if nid and group:
                    marker_name_element_xi.append((
                        "dorsal base of left lung" if (lung == left_lung) else "dorsal base of right lung",
                        get_mesh_first_element_with_node(
                            group.getMeshGroup(mesh), coordinates, nodes.findNodeByIdentifier(nid)),
                    [1.0, 0.0, 1.0]))
                if lung == right_lung:
                    nid = middle_ellipsoid.get_node_identifier(
                        elements_count_lateral, elements_count_oblique // 2, elements_count_oblique // 2)
                    group = middle_right_lung_group
                    if nid and group:
                        marker_name_element_xi.append((
                            "laterodorsal tip of middle lobe of right lung",
                            get_mesh_first_element_with_node(
                                group.getMeshGroup(mesh), coordinates, nodes.findNodeByIdentifier(nid)),
                            [0.0, 1.0, 1.0]))
                # need to find element xi where medial base edge crosses 0.0
                group = lower_left_lung_group if (lung == left_lung) else lower_right_lung_group
                if group:
                    n1 = elements_count_lateral if (lung == left_lung) else 0
                    n3 = elements_count_oblique // 2 + elements_count_lower_extension
                    last_n2 = None
                    last_nid = None
                    last_nx = None
                    last_nd1 = None
                    for n2 in range(0, elements_count_oblique // 2 + 1):
                        nid = lower_ellipsoid_mesh.get_node_identifier(n1, n2, n3)
                        if nid:
                            nx, nd1 = lower_ellipsoid_mesh.get_node_parameters(n1, n2, n3)[:2]
                            if last_nid:
                                if (last_nx[1] <= 0.0) and (nx[1] >= 0.0):
                                    element = get_mesh_first_element_with_node(
                                        group.getMeshGroup(mesh), coordinates, nodes.findNodeByIdentifier(
                                            nid if ((lung == left_lung) or (last_n2 == 0)) else last_nid))
                                    if element.isValid():
                                        # xi1 goes counter-clockwise from above so different for left and right
                                        if lung == left_lung:
                                            curve_location, x = getNearestLocationOnCurve(
                                                [last_nx[1:2], nx[1:2]], [last_nd1[1:2], nd1[1:2]], [0.0],
                                                startLocation=(0, -last_nx[1] / (nx[1] - last_nx[1])))
                                        else:
                                            curve_location, x = getNearestLocationOnCurve(
                                                [nx[1:2], last_nx[1:2]], [nd1[1:2], last_nd1[1:2]], [0.0],
                                                startLocation=(0, nx[1] / (nx[1] - last_nx[1])))
                                        marker_name_element_xi.append((
                                            "medial base of left lung" if (lung == left_lung) else
                                            "medial base of right lung",
                                            element, [curve_location[1], 0.0, 1.0]))
                                    break
                            last_n2 = n2
                            last_nid = nid
                            last_nx = nx
                            last_nd1 = nd1
                ventral_ellipsoid = upper_ellipsoid if (lung == left_lung) else middle_ellipsoid
                nid = ventral_ellipsoid.get_node_identifier(
                    elements_count_lateral // 2, elements_count_oblique // 2, 0)
                group = upper_left_lung_group if (lung == left_lung) else middle_right_lung_group
                if nid and group:
                    marker_name_element_xi.append((
                        "ventral base of left lung" if (lung == left_lung) else "ventral base of right lung",
                        get_mesh_first_element_with_node(
                            group.getMeshGroup(mesh), coordinates, nodes.findNodeByIdentifier(nid)),
                        [0.0, 0.0, 1.0]))
            else:
                # need to make sure marker node identifiers are consistent if left/right not used
                for i in range(4 if (lung == left_lung) else 5):
                    marker_name_element_xi.append(("dummy", None, None))

        for lung in lungs:
            is_left = lung == left_lung
            lung_nodeset = (left_lung_group if is_left else right_lung_group).getNodesetGroup(nodes)

            if (lower_lobe_base_concavity > 0.0) and (lower_lobe_extension > 0.0):
                lower_lobe_group = lower_left_lung_group if (lung == left_lung) else lower_right_lung_group
                form_lower_lobe_base_concavity(
                    lower_lobe_base_concavity, lower_lobe_extension, half_ml_size, half_dv_size, half_height,
                    fieldmodule, coordinates, lower_lobe_group.getNodesetGroup(nodes))

            if ventral_sharpness_factor != 0.0:
                taper_lung_edge(ventral_sharpness_factor, fieldmodule, coordinates, lung_nodeset, half_dv_size)

            if cardiac_curvature > 0.0:
                curve_cardiac_anterior(-cardiac_curvature if is_left else cardiac_curvature,
                                       half_ml_size, half_dv_size, half_height, fieldmodule, coordinates, lung_nodeset)

            translate_nodeset_coordinates(lung_nodeset, coordinates,
                                          [-lung_spacing if is_left else lung_spacing, 0.0, 0.0])

        # marker points; make after regular nodes so higher node numbers
        lung_nodeset = lung_group.getNodesetGroup(nodes)
        for marker_name, element, xi in marker_name_element_xi:
            if element:  # skip dummy markers for missing left/right, but increment node_identifier always
                annotation_group = findOrCreateAnnotationGroupForTerm(
                    annotation_groups, region, get_lung_term(marker_name), isMarker=True)
                marker_node = annotation_group.createMarkerNode(node_identifier, element=element, xi=xi)
                if 'left' in marker_name:
                    left_lung_group.getNodesetGroup(nodes).addNode(marker_node)
                else:
                    right_lung_group.getNodesetGroup(nodes).addNode(marker_node)
                lung_nodeset.addNode(marker_node)
            node_identifier += 1

        return annotation_groups, None

    @classmethod
    def refineMesh(cls, meshRefinement, options):
        """
        Refine source mesh into separate region, with change of basis.
        :param meshRefinement: MeshRefinement, which knows source and target region.
        :param options: Dict containing options. See getDefaultOptions().
        """
        assert isinstance(meshRefinement, MeshRefinement)
        refine_elements_count = options['Refine number of elements']
        meshRefinement.refineAllElementsCubeStandard3d(
            refine_elements_count, refine_elements_count, refine_elements_count)

    @classmethod
    def defineFaceAnnotations(cls, region, options, annotation_groups):
        """
        Add face annotation groups from the highest dimension mesh.
        Must have defined faces and added subelements for highest dimension groups.
        :param region: Zinc region containing model.
        :param options: Dict containing options. See getDefaultOptions().
        :param annotation_groups: List of annotation groups for top-level elements.
        New face annotation groups are appended to this list.
        """
        fm = region.getFieldmodule()
        mesh1d = fm.findMeshByDimension(1)
        mesh2d = fm.findMeshByDimension(2)

        has_left_lung = options["Left lung"]
        has_right_lung = options["Right lung"]

        if (not has_left_lung) or (not has_right_lung):
            # destroy elements, faces, lines and nodes now to ensure persistent identifiers when only one side
            # these are any objects not in the "lung" group:
            not_lung = fm.createFieldNot(getAnnotationGroupForTerm(annotation_groups, get_lung_term("lung")).getGroup())
            for mesh in [fm.findMeshByDimension(3), mesh2d, mesh1d]:
                mesh.destroyElementsConditional(not_lung)
            nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            nodes.destroyNodesConditional(not_lung)
            del nodes

        is_exterior = fm.createFieldIsExterior()
        is_face_xi1_0 = fm.createFieldIsOnFace(Element.FACE_TYPE_XI1_0)
        is_face_xi1_1 = fm.createFieldIsOnFace(Element.FACE_TYPE_XI1_1)
        is_face_xi2_0 = fm.createFieldIsOnFace(Element.FACE_TYPE_XI2_0)
        is_face_xi2_1 = fm.createFieldIsOnFace(Element.FACE_TYPE_XI2_1)
        is_face_xi3_0 = fm.createFieldIsOnFace(Element.FACE_TYPE_XI3_0)
        is_face_xi3_1 = fm.createFieldIsOnFace(Element.FACE_TYPE_XI3_1)
        is_exterior_face_xi3_1 = fm.createFieldAnd(is_exterior, is_face_xi3_1)

        box_group = findAnnotationGroupByName(annotation_groups, "box")
        is_box = box_group.getGroup()
        transition_group = findAnnotationGroupByName(annotation_groups, "transition")
        is_trans = transition_group.getGroup()
        # this works correctly for faces, but gets extra layers for lines on base and fissures which are exterior:
        is_on_ellipsoid = fm.createFieldAnd(fm.createFieldAnd(is_exterior_face_xi3_1, is_trans),
                                            fm.createFieldNot(is_box))

        is_face_xi1_0_or_xi1_1_or_xi2_0 = fm.createFieldOr(
            fm.createFieldOr(is_face_xi1_0, is_face_xi1_1), is_face_xi2_0)
        is_face_xi1_0_or_xi1_1_or_xi2_1 = fm.createFieldOr(
            fm.createFieldOr(is_face_xi1_0, is_face_xi1_1), is_face_xi2_1)
        is_face_box30_trans20 = fm.createFieldOr(
            fm.createFieldAnd(is_box, is_face_xi3_0), fm.createFieldAnd(is_trans, is_face_xi2_0))
        is_face_box31_trans21 = fm.createFieldOr(
            fm.createFieldAnd(is_box, is_face_xi3_1), fm.createFieldAnd(is_trans, is_face_xi2_1))

        face_term_conditionals_map = {}

        if has_left_lung:
            is_lower_left = getAnnotationGroupForTerm(annotation_groups, get_lung_term("lower lobe of left lung")).getGroup()
            is_upper_left = getAnnotationGroupForTerm(annotation_groups, get_lung_term("upper lobe of left lung")).getGroup()
            is_anterior_left = findAnnotationGroupByName(annotation_groups, "anterior left lung").getGroup()
            is_lateral_left = findAnnotationGroupByName(annotation_groups, "lateral left lung").getGroup()
            is_medial_left = findAnnotationGroupByName(annotation_groups, "medial left lung").getGroup()

            left_face_term_conditionals_map = {
                "base of lower lobe of left lung surface": (is_lower_left, is_exterior, is_face_box30_trans20),
                "base of upper lobe of left lung surface": (is_upper_left, is_exterior, is_face_xi1_0_or_xi1_1_or_xi2_0,
                                                            is_anterior_left),
                "lateral surface of left lung": (is_lateral_left, is_on_ellipsoid),
                "lateral surface of lower lobe of left lung": (is_lower_left, is_on_ellipsoid, is_lateral_left),
                "lateral surface of upper lobe of left lung": (is_upper_left, is_on_ellipsoid, is_lateral_left),
                "lower lobe of left lung surface": (is_lower_left, is_exterior),
                "upper lobe of left lung surface": (is_upper_left, is_exterior),
                "medial surface of left lung": (is_medial_left, is_on_ellipsoid),
                "medial surface of lower lobe of left lung": (is_lower_left, is_on_ellipsoid, is_medial_left),
                "medial surface of upper lobe of left lung": (is_upper_left, is_on_ellipsoid, is_medial_left),
                "oblique fissure of lower lobe of left lung": (is_lower_left, is_exterior,
                                                               is_face_xi1_0_or_xi1_1_or_xi2_1),
                "oblique fissure of upper lobe of left lung": (is_upper_left, is_exterior,
                                                               fm.createFieldOr(is_face_xi1_0_or_xi1_1_or_xi2_0,
                                                                                is_face_box30_trans20)),
            }
            face_term_conditionals_map.update(left_face_term_conditionals_map)
        else:
            is_lower_left = is_upper_left = is_lateral_left = is_medial_left = None

        if has_right_lung:
            is_lower_right = getAnnotationGroupForTerm(annotation_groups, get_lung_term("lower lobe of right lung")).getGroup()
            is_middle_right = getAnnotationGroupForTerm(annotation_groups, get_lung_term("middle lobe of right lung")).getGroup()
            is_upper_right = getAnnotationGroupForTerm(annotation_groups, get_lung_term("upper lobe of right lung")).getGroup()
            is_anterior_right = findAnnotationGroupByName(annotation_groups, "anterior right lung").getGroup()
            is_lateral_right = findAnnotationGroupByName(annotation_groups, "lateral right lung").getGroup()
            is_medial_right = findAnnotationGroupByName(annotation_groups, "medial right lung").getGroup()

            right_face_term_conditionals_map = {
                "base of lower lobe of right lung surface": (is_lower_right, is_exterior, is_face_box30_trans20),
                "base of middle lobe of right lung surface": (is_middle_right, is_exterior, is_face_xi1_0_or_xi1_1_or_xi2_0),
                "horizontal fissure of middle lobe of right lung": (is_middle_right, is_exterior, is_face_box31_trans21),
                "horizontal fissure of upper lobe of right lung": (is_upper_right, is_exterior, is_face_box30_trans20, is_anterior_right),
                "lateral surface of lower lobe of right lung": (is_lower_right, is_on_ellipsoid, is_lateral_right),
                "lateral surface of middle lobe of right lung": (is_middle_right, is_on_ellipsoid, is_lateral_right),
                "lateral surface of right lung": (is_lateral_right, is_on_ellipsoid),
                "lateral surface of upper lobe of right lung": (is_upper_right, is_on_ellipsoid, is_lateral_right),
                "lower lobe of right lung surface": (is_lower_right, is_exterior),
                "middle lobe of right lung surface": (is_middle_right, is_exterior),
                "upper lobe of right lung surface": (is_upper_right, is_exterior),
                "medial surface of lower lobe of right lung": (is_lower_right, is_on_ellipsoid, is_medial_right),
                "medial surface of middle lobe of right lung": (is_middle_right, is_on_ellipsoid, is_medial_right),
                "medial surface of right lung": (is_medial_right, is_on_ellipsoid),
                "medial surface of upper lobe of right lung": (is_upper_right, is_on_ellipsoid, is_medial_right),
                "oblique fissure of lower lobe of right lung": (is_lower_right, is_exterior, is_face_xi1_0_or_xi1_1_or_xi2_1),
                "oblique fissure of middle lobe of right lung": (is_middle_right, is_exterior, is_face_xi1_0_or_xi1_1_or_xi2_0),
                "oblique fissure of upper lobe of right lung": (is_upper_right, is_exterior, fm.createFieldAnd(is_face_box30_trans20, fm.createFieldNot(is_anterior_right))),
            }
            face_term_conditionals_map.update(right_face_term_conditionals_map)
        else:
            is_lower_right = is_middle_right = is_upper_right = is_lateral_right = is_medial_right = None

        is_face_conditional = {}
        for face_term, conditionals in face_term_conditionals_map.items():
            annotation_group = findOrCreateAnnotationGroupForTerm(annotation_groups, region, get_lung_term(face_term))
            group = annotation_group.getGroup()
            conditional = conditionals[0]
            for add_conditional in conditionals[1:]:
                conditional = fm.createFieldAnd(conditional, add_conditional)
            annotation_group.getMeshGroup(mesh2d).addElementsConditional(conditional)
            annotation_group.addSubelements()
            is_face_conditional[face_term] = group

        line_term_conditionals_map = {}

        if has_left_lung:
            is_left_lateral_surface = is_face_conditional["lateral surface of left lung"]
            is_left_medial_surface = is_face_conditional["medial surface of left lung"]
            left_line_term_conditionals_map = {
                "antero-posterior edge of upper lobe of left lung":
                    (is_upper_left, is_left_lateral_surface, is_left_medial_surface),
                "base edge of oblique fissure of lower lobe of left lung": (
                    is_face_conditional["base of lower lobe of left lung surface"], is_face_conditional["oblique fissure of lower lobe of left lung"]),
                "lateral edge of base of lower lobe of left lung": (
                    is_face_conditional["base of lower lobe of left lung surface"], is_left_lateral_surface),
                "lateral edge of base of upper lobe of left lung": (
                    is_face_conditional["base of upper lobe of left lung surface"], is_left_lateral_surface),
                "lateral edge of oblique fissure of lower lobe of left lung": (
                    is_face_conditional["oblique fissure of lower lobe of left lung"], is_left_lateral_surface),
                "lateral edge of oblique fissure of upper lobe of left lung": (
                    is_face_conditional["oblique fissure of upper lobe of left lung"], is_left_lateral_surface),
                "medial edge of base of lower lobe of left lung": (
                    is_face_conditional["base of lower lobe of left lung surface"], is_left_medial_surface),
                "medial edge of base of upper lobe of left lung": (
                    is_face_conditional["base of upper lobe of left lung surface"], is_left_medial_surface),
                "medial edge of oblique fissure of lower lobe of left lung": (
                    is_face_conditional["oblique fissure of lower lobe of left lung"], is_left_medial_surface),
                "medial edge of oblique fissure of upper lobe of left lung": (
                    is_face_conditional["oblique fissure of upper lobe of left lung"], is_left_medial_surface),
                "posterior edge of lower lobe of left lung":
                    (is_lower_left, is_left_lateral_surface, is_left_medial_surface),
            }
            line_term_conditionals_map.update(left_line_term_conditionals_map)

        if has_right_lung:
            is_right_lateral_surface = is_face_conditional["lateral surface of right lung"]
            is_right_medial_surface = is_face_conditional["medial surface of right lung"]
            right_line_term_conditionals_map = {
                "anterior edge of middle lobe of right lung":
                    (is_middle_right, is_right_lateral_surface, is_right_medial_surface),
                "antero-posterior edge of upper lobe of right lung":
                    (is_upper_right, is_right_lateral_surface, is_right_medial_surface),
                "base edge of oblique fissure of lower lobe of right lung": (
                    is_face_conditional["base of lower lobe of right lung surface"], is_face_conditional["oblique fissure of lower lobe of right lung"]),
                "lateral edge of base of lower lobe of right lung": (
                    is_face_conditional["base of lower lobe of right lung surface"], is_right_lateral_surface),
                "lateral edge of base of middle lobe of right lung": (
                    is_face_conditional["base of middle lobe of right lung surface"], is_right_lateral_surface),
                "lateral edge of horizontal fissure of middle lobe of right lung": (
                    is_face_conditional["horizontal fissure of middle lobe of right lung"], is_right_lateral_surface),
                "lateral edge of horizontal fissure of upper lobe of right lung": (
                    is_face_conditional["horizontal fissure of upper lobe of right lung"], is_right_lateral_surface),
                "lateral edge of oblique fissure of lower lobe of right lung": (
                    is_face_conditional["oblique fissure of lower lobe of right lung"], is_right_lateral_surface),
                "lateral edge of oblique fissure of middle lobe of right lung": (
                    is_face_conditional["oblique fissure of middle lobe of right lung"], is_right_lateral_surface),
                "lateral edge of oblique fissure of upper lobe of right lung": (
                    is_face_conditional["oblique fissure of upper lobe of right lung"], is_right_lateral_surface),
                "medial edge of base of lower lobe of right lung": (
                    is_face_conditional["base of lower lobe of right lung surface"], is_right_medial_surface),
                "medial edge of base of middle lobe of right lung": (
                    is_face_conditional["base of middle lobe of right lung surface"], is_right_medial_surface),
                "medial edge of horizontal fissure of middle lobe of right lung": (
                    is_face_conditional["horizontal fissure of middle lobe of right lung"], is_right_medial_surface),
                "medial edge of horizontal fissure of upper lobe of right lung": (
                    is_face_conditional["horizontal fissure of upper lobe of right lung"], is_right_medial_surface),
                "medial edge of oblique fissure of lower lobe of right lung": (
                    is_face_conditional["oblique fissure of lower lobe of right lung"], is_right_medial_surface),
                "medial edge of oblique fissure of middle lobe of right lung": (
                    is_face_conditional["oblique fissure of middle lobe of right lung"], is_right_medial_surface),
                "medial edge of oblique fissure of upper lobe of right lung": (
                    is_face_conditional["oblique fissure of upper lobe of right lung"], is_right_medial_surface),
                "posterior edge of lower lobe of right lung":
                    (is_lower_right, is_right_lateral_surface, is_right_medial_surface)
            }
            line_term_conditionals_map.update(right_line_term_conditionals_map)

        for line_term, conditionals in line_term_conditionals_map.items():
            annotation_group = findOrCreateAnnotationGroupForTerm(annotation_groups, region, (line_term, ""))
            conditional = conditionals[0]
            for add_conditional in conditionals[1:]:
                conditional = fm.createFieldAnd(conditional, add_conditional)
            annotation_group.getMeshGroup(mesh1d).addElementsConditional(conditional)

        tweak_middle_lobe_tip_edges = True
        if has_right_lung and tweak_middle_lobe_tip_edges:
            # tweaks edges of base of middle lobe of right lung to include anterior edge of middle lobe
            # as these edges of the base data can go sharply up the anterior edge
            # but this is a difficult and undesirable fit for the middle lobe shape
            is_anterior_edge = fm.findFieldByName("anterior edge of middle lobe of right lung").castGroup()
            for base_edge_group_name in [
                "lateral edge of base of middle lobe of right lung",
                "medial edge of base of middle lobe of right lung"
            ]:
                base_edge_group = fm.findFieldByName(base_edge_group_name).castGroup()
                base_edge_group.getMeshGroup(mesh1d).addElementsConditional(is_anterior_edge)

        # remove temporary annotation groups
        for group_name in [
            "anterior left lung",
            "anterior right lung",
            "box",
            "lateral left lung",
            "lateral right lung",
            "medial left lung",
            "medial right lung",
            "transition"
        ]:
            annotation_group = findAnnotationGroupByName(annotation_groups, group_name)
            if annotation_group:
                annotation_groups.remove(annotation_group)


def curve_cardiac_anterior(curvature, half_ml_size, half_dv_size, half_height, fieldmodule, coordinates, nodeset):
    """
    Transform coordinates by bending with curvature about a centre point the radius in
    x direction from stationaryPointXY.
    :param curvature: 1/radius. Must be non-zero.
    :param half_ml_size: Half medial-lateral ellipsoid size.
    :param half_dv_size: Half dorsal-ventral ellipsoid size.
    :param half_height: Half ellipsoid height.
    :param fieldmodule: Field module being worked with.
    :param coordinates: The coordinate field, initially circular in y-z plane.
    :param nodeset: Zinc Nodeset containing nodes to transform.
    """
    radius_of_curvature = 1.0 / curvature
    x_offset = fieldmodule.createFieldConstant(radius_of_curvature)

    pi__3 = math.pi / 3.0
    cos_pi__3 = math.cos(pi__3)
    sin_pi__3 = math.sin(pi__3)
    value_small = fieldmodule.createFieldConstant(1.0E-10)
    value05 = fieldmodule.createFieldConstant(0.5)
    x = fieldmodule.createFieldComponent(coordinates, 1)
    y = fieldmodule.createFieldComponent(coordinates, 2)
    z = fieldmodule.createFieldComponent(coordinates, 3)
    yz = fieldmodule.createFieldConcatenate([y, z])
    r = fieldmodule.createFieldMagnitude(yz)
    # angle around hilum, not changing
    theta = fieldmodule.createFieldAtan2(z, y)
    cos_theta = fieldmodule.createFieldCos(theta)
    sin_theta = fieldmodule.createFieldSin(theta)
    # obliqueness factor as want maximum curvature at anterior oblique fissure, none opposite
    tip_yz = getEllipsePointAtTrueAngle(half_dv_size, half_height, -pi__3)
    double_max_o = fieldmodule.createFieldConstant(2.0 * magnitude(tip_yz))
    oblique = fieldmodule.createFieldConstant([cos_pi__3, -sin_pi__3])
    o = value05 + fieldmodule.createFieldDotProduct(yz, oblique) / double_max_o
    phi_o = o * o  # so curvature mostly at anterior edge
    # radial curvature
    alpha = phi_o * r / x_offset
    cos_alpha = fieldmodule.createFieldCos(alpha)
    sin_alpha = fieldmodule.createFieldSin(alpha)

    mod_x_offset = x_offset / phi_o
    mod_offset_x = x + mod_x_offset
    new_x = mod_offset_x * cos_alpha - mod_x_offset

    new_r = mod_offset_x * sin_alpha
    new_y = new_r * cos_theta
    new_z = new_r * sin_theta
    new_coordinates = fieldmodule.createFieldIf(fieldmodule.createFieldLessThan(o, value_small), coordinates,
                                                fieldmodule.createFieldConcatenate([new_x, new_y, new_z]))
    fieldassignment = coordinates.createFieldassignment(new_coordinates)
    fieldassignment.setNodeset(nodeset)
    fieldassignment.assign()


def form_lower_lobe_base_concavity(lower_lobe_base_concavity, lower_lobe_extension,
                                   half_ml_size, half_dv_size, half_height,
                                   fieldmodule, coordinates, nodeset):
    """
    Reshape the base of the lower lobe to be concave, tapering off to the oblique fissure.
    :param lower_lobe_base_concavity: Lower lobe base concavity distance. Note this is reduced by the taper.
    :param lower_lobe_extension: Distance along oblique fissure below origin that lower lobe extends.
    :param half_ml_size: Half medial-lateral ellipsoid size.
    :param half_dv_size: Half dorsal-ventral ellipsoid size.
    :param half_height: Half ellipsoid height.
    :param fieldmodule: Fieldmodule owning fields to modify.
    :param coordinates: Coordinate field to modify.
    :param nodeset: Nodeset to modify.
    """
    pi__3 = math.pi / 3.0
    cos_pi__3 = math.cos(pi__3)
    sin_pi__3 = math.sin(pi__3)
    value0 = fieldmodule.createFieldConstant(0.0)
    value1 = fieldmodule.createFieldConstant(1.0)
    value2 = fieldmodule.createFieldConstant(2.0)
    value3 = fieldmodule.createFieldConstant(3.0)
    x = fieldmodule.createFieldComponent(coordinates, 1)
    y = fieldmodule.createFieldComponent(coordinates, 2)
    z = fieldmodule.createFieldComponent(coordinates, 3)
    xx = x * x
    yy = y * y
    minus_c = fieldmodule.createFieldConstant(-half_height)
    nz = z / minus_c
    zfact = fieldmodule.createFieldSqrt(value1 - nz * nz)
    a = fieldmodule.createFieldConstant(half_ml_size) * zfact
    b = fieldmodule.createFieldConstant(half_dv_size) * zfact
    aa = a * a
    bb = b * b
    r = fieldmodule.createFieldSqrt((xx / aa) + (yy / bb))
    rr = r * r
    phi_r = value1 - rr
    z_ext = -lower_lobe_extension * sin_pi__3
    e = z / fieldmodule.createFieldConstant(z_ext)
    ee = e * e
    eee = ee * e
    phi_e = fieldmodule.createFieldIf(fieldmodule.createFieldLessThan(e, value0), value0,
                                      value3 * ee - value2 * eee)
    # taper concavity to zero at oblique fissure
    y_ext = fieldmodule.createFieldConstant(lower_lobe_extension * cos_pi__3)
    y_max = b + y_ext
    d = value1 + (y - e * y_ext) / y_max
    dd = d * d
    phi_y = value1 - dd
    phi = phi_e * phi_r * phi_y
    delta_y = phi * fieldmodule.createFieldConstant(-lower_lobe_base_concavity * cos_pi__3 / sin_pi__3)
    delta_z = phi * fieldmodule.createFieldConstant(lower_lobe_base_concavity)
    # get span a at displaced z, to calculate delta_x
    displ_nz = (z + delta_z) / minus_c
    displ_zfact = fieldmodule.createFieldSqrt(value1 - displ_nz * displ_nz)
    displ_a = fieldmodule.createFieldConstant(half_ml_size) * displ_zfact
    delta_x = (x * displ_a / a - x)
    displacement = fieldmodule.createFieldConcatenate([delta_x, delta_y, delta_z])
    new_coordinates = coordinates + displacement
    fieldassignment = coordinates.createFieldassignment(new_coordinates)
    fieldassignment.setNodeset(nodeset)
    fieldassignment.assign()


def taper_lung_edge(sharpeningFactor, fieldmodule, coordinates, nodeset, halfValue, isBase=False):
    """
    Apply a tapering transformation to the lung geometry to sharpen the anterior edge or the base.
    If isBase is False, it sharpens the anterior edge (along the y-axis).
    If isBase is True, it sharpens the base (along the z-axis), but only for nodes below a certain height.
    :param sharpeningFactor: A value between 0 and 1, where 1 represents the maximum sharpness.
    :param fieldmodule: Field module being worked with.
    :param coordinates: The coordinate field. The anterior edge is towards the +y-axis, and the base is towards the
    -z-axis.
    :param nodeset: Zinc NodesetGroup containing nodes to transform.
    :param halfValue: Half value of lung breadth/height depending on isBase.
    :param isBase: False if transforming the anterior edge, True if transforming the base of the lung.
    """
    x = fieldmodule.createFieldComponent(coordinates, 1)
    y = fieldmodule.createFieldComponent(coordinates, 2)
    z = fieldmodule.createFieldComponent(coordinates, 3)

    coord_value = z if isBase else y
    start_value = 0.5 * halfValue if isBase else -0.5 * halfValue
    end_value = -1.1 * halfValue if isBase else 1.1 * halfValue

    start_value_field = fieldmodule.createFieldConstant(start_value)
    end_value_field = fieldmodule.createFieldConstant(end_value)

    xi = (coord_value - start_value_field) / fieldmodule.createFieldConstant(end_value - start_value)
    xi__2 = xi * xi
    one = fieldmodule.createFieldConstant(1.0)
    x_scale = one - fieldmodule.createFieldConstant(sharpeningFactor) * xi__2
    if isBase:
        new_x = fieldmodule.createFieldIf(fieldmodule.createFieldLessThan(coord_value, start_value_field), x * x_scale, x)
    else:
        new_x = fieldmodule.createFieldIf(fieldmodule.createFieldGreaterThan(coord_value, start_value_field), x * x_scale, x)
    new_coordinates = fieldmodule.createFieldConcatenate([new_x, y, z])

    fieldassignment = coordinates.createFieldassignment(new_coordinates)
    fieldassignment.setNodeset(nodeset)
    fieldassignment.assign()
