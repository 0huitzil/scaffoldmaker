import math
from itertools import product

from cmlibs.maths.vectorops import (
    add,
    magnitude,
    mult,
    rotate_vector_around_vector,
    set_magnitude,
)
from cmlibs.utils.zinc.field import find_or_create_field_coordinates
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.fieldmodule import Fieldmodule
from cmlibs.zinc.node import Node
from cmlibs.zinc.region import Region
from cmlibs.zinc.result import RESULT_OK

from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, findOrCreateAnnotationGroupForTerm
from scaffoldmaker.annotation.hand_terms import get_hand_term
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.utils.eft_utils import (
    remapEftLocalNodes,
    remapEftNodeValueLabelWithNodes,
    setEftScaleFactorIds,
)
from scaffoldmaker.utils.geometry import sampleEllipsePoints


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
        }
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return ["Thumb angle"]

    @classmethod
    def checkOptions(cls, options):
        dependent_changes = False
        for key, angleRange in {
            "Thumb angle": (10.0, 80.0),
        }.items():
            if options[key] < angleRange[0]:
                options[key] = angleRange[0]
            elif options[key] > angleRange[1]:
                options[key] = angleRange[1]
        return dependent_changes

    @classmethod
    def generateBaseMesh(cls, region: Region, options):
        """
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        return: list of AnnotationGroup,
        """
        #################
        # Options
        #################
        thumb_angle_degrees = options["Thumb angle"]
        #################
        # Setup zinc
        #################
        fieldmodule = region.getFieldmodule()
        coordinates = find_or_create_field_coordinates(fieldmodule)
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        fieldcache = fieldmodule.createFieldcache()
        nodetemplate = get_simple_nodetemplate(fieldmodule)
        #################
        # Create nodes
        #################
        node_identifier = 1
        # Fingers 1 - 4 (finger 5, thumb, is added below)
        finger_dimensions = [  # d1, d2, d3, d12
            [1.0, 0.4, 0.3, 0.4],  # carpal
            [2.5, 0.4, 0.3, 0.2],  # metacarpal
            [1.5, 0.3, 0.3, 0.2],  # proximal phalanx
            [1.5, 0.2, 0.3, 0.2],  # middle phalanx
            [1.0, 0.2, 0.25, 0.2],  # distal phalanx
        ]
        carpal_spacing = finger_dimensions[0][1] * 2.0
        index_carpal_node_id = None
        # create nodes up each finger fastest
        x1 = [1.0, 0.0, 0.0]
        x2 = [0.0, 1.0, 0.0]
        x3 = [0.0, 0.0, 1.0]
        for j in range(4):
            x = [0.0, j * carpal_spacing, 0.0]
            for i, bone_dimensions in enumerate(finger_dimensions):
                node = nodes.createNode(node_identifier, nodetemplate)
                fieldcache.setNode(node)
                bone_dimensions = finger_dimensions[i]
                d1 = mult(x1, bone_dimensions[0])
                d2 = mult(x2, bone_dimensions[1])
                d3 = mult(x3, bone_dimensions[2])
                d12 = mult(x2, bone_dimensions[3])
                if i == 2:
                    x = add(x, mult(d2, -1.0 + (2.0 / 3.0) * j))
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
                x = add(x, d1)
                if (j == 3) and (i == 1):
                    index_carpal_node_id = node_identifier
                node_identifier += 1
        thumb_dimensions = [  # d1, d2, d3, d12
            [1.0, 0.3, 0.3, 0.2],  # metacarpal
            [1.0, 0.3, 0.25, 0.2],  # proximal phalanx
            [1.0, 0.3, 0.25, 0.2],  # distal phalanx
        ]
        node_identifier = create_thumb_nodes(
            fieldmodule, node_identifier, thumb_dimensions, index_carpal_node_id, thumb_angle_degrees
        )
        #################
        # Create virtual node matrix
        #################
        # Get scale factor matrix
        c, mc, pp, mp, dp = [1, 2, 1, 1, 2]

        virtual_node_matrix = [
            [[None for k in range(4)] for j in range(30)] for i in range(2 * (c + mc + pp + mp + dp))]
        virtual_node_matrix, node_identifier = generate_internal_node_matrix(
            fieldmodule, [c, mc, pp, mp, dp], node_identifier, virtual_node_matrix, options
        )
        virtual_node_matrix, node_identifier = generate_external_node_matrix(
            fieldmodule, [c, mc, pp, mp, dp], node_identifier, virtual_node_matrix, options
        )
        # Setup annotation groups
        hand_group = AnnotationGroup(region, get_hand_term("hand"))
        carpal_group = AnnotationGroup(region, get_hand_term("carpal"))
        metacarpal_group = AnnotationGroup(region, get_hand_term("metacarpal"))
        phalanx_prox_group = AnnotationGroup(region, get_hand_term("proximal phalanx"))
        phalanx_mid_group = AnnotationGroup(region, get_hand_term("middle phalanx"))
        phalanx_dist_group = AnnotationGroup(region, get_hand_term("distal phalanx"))
        finger_thumb_group = AnnotationGroup(region, get_hand_term("thumb"))
        finger_index_group = AnnotationGroup(region, get_hand_term("index finger"))
        finger_middle_group = AnnotationGroup(region, get_hand_term("middle finger"))
        finger_ring_group = AnnotationGroup(region, get_hand_term("ring finger"))
        finger_little_group = AnnotationGroup(region, get_hand_term("little finger"))
        mesh3d = fieldmodule.findMeshByDimension(3)
        annotation_groups = [
            hand_group,
            carpal_group,
            metacarpal_group,
            phalanx_prox_group,
            phalanx_mid_group,
            phalanx_dist_group,
            finger_thumb_group,
            finger_index_group,
            finger_middle_group,
            finger_ring_group,
            finger_little_group,
        ]
        # Let it rip
        z_len = len(virtual_node_matrix[0][0]) - 1
        y_len = len(virtual_node_matrix[0]) - 1
        x_len = len(virtual_node_matrix) - 1
        element_identifier = 1
        for k in range(z_len):
            for j in range(y_len):
                for i in range(x_len):
                    result, element = create_cube_element(
                        fieldmodule, element_identifier, virtual_node_matrix, i, j, k
                    )
                    if result == RESULT_OK:
                        element_annotations = [hand_group]
                        # find annotation groups
                        try:
                            node_annotations = virtual_node_matrix[i][j][k]["Annotation"]
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
        Add orientation anterior 1-D annotation group.
        :param region: Zinc region containing model.
        :param options: Dict containing options. See getDefaultOptions().
        :param annotationGroups: List of annotation groups for elements created in generateBaseMesh().
        New face/line annotation groups are appended to this list.
        """
        fieldmodule = region.getFieldmodule()
        mesh2d = fieldmodule.findMeshByDimension(2)
        mesh1d = fieldmodule.findMeshByDimension(1)
        is_exterior = fieldmodule.createFieldIsExterior()
        is_face_xi3_0 = fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0)

        skin_group = findOrCreateAnnotationGroupForTerm(annotationGroups, region, get_hand_term('skin of hand'))
        is_skin = fieldmodule.createFieldAnd(is_exterior, fieldmodule.createFieldNot(is_face_xi3_0))
        skin_group.getMeshGroup(mesh2d).addElementsConditional(is_skin)

        finger_names = ["little finger", "ring finger", "middle finger", "index finger", "thumb"]

        for finger_name in finger_names:
            finger_group = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_hand_term(finger_name))
            finger_skin_group = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_hand_term('skin of ' + finger_name))
            finger_skin_group.getMeshGroup(mesh2d).addElementsConditional(
            fieldmodule.createFieldAnd(finger_group.getGroup(), is_exterior)
            )



def generate_internal_node_matrix(fieldmodule, hand_elements_along, node_identifier, virtual_node_matrix, options=None):
    """
    Docstring for generate_internal_node_matrix

    :param fieldmodule: Description
    :param hand_elements_along: Description
    :param node_identifier: Description
    :param virtual_node_matrix: Description
    :param options: Description
    :return: Description
    :rtype: Any
    """
    c, mc, pp, mp, dp = hand_elements_along
    bone_w = 1
    bone_h = 1
    a0 = 1
    parent_node_id = 0
    y = 1
    k_val = [1, 2]
    # Finger
    for f in range(4):
        # Carpal
        parent_node_id += 1
        x = 0
        y_val = [y, y + 1] if f == 0 else [y + 1]
        # y_val = [y, y+1]
        x_val = [(x + i, i / c) for i in range(c)]
        for k in k_val:
            for j in y_val:
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a3 = -bone_h if k == 1 else bone_h
                    annotation = "carpal" if k == 1 else ""
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        "Type": "internal",
                        "Annotation": [annotation],
                    }
                    if j == y:
                        if virtual_node_matrix[i][j][k] is None:
                            virtual_node_matrix[i][j][k] = node
                    elif j == y + 1:
                        virtual_node_matrix[i][j][k] = node
                        virtual_node_matrix[i][y + 5][k] = node
                    node_identifier = add_node(
                        fieldmodule, parent_node_id, node_identifier, [a0, a1, a2, a3])
        x = c
        # Metacarpal
        parent_node_id += 1
        y_val = [y, y + 1, y + 5] if y != 16 else [y, y + 1]
        x_val = [(x + i, i / mc) for i in range(mc)]
        x_val = [(x, 0), (x + 1, 0.35)]
        for k in k_val:
            for j in y_val:
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a3 = -bone_h if k == 1 else bone_h
                    annotation = "metacarpal" if k == 1 else ""
                    if virtual_node_matrix[i][j][k] is None:
                        virtual_node_matrix[i][j][k] = {
                            Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                            "Type": "internal",
                            "Annotation": [annotation],
                        }
        x += mc
        # PP
        parent_node_id += 1
        finger_names = ["little finger", "ring finger", "middle finger", "index finger", "thumb"]
        y_val = [y, y + 1, y + 5] if y != 16 else [y, y + 1]
        x_val = [(x + i, (-0.25) * (i+1) / pp) for i in range(pp)]
        for k in k_val:
            for j in y_val:
                finger_name = finger_names[f] if (j != y+5) else finger_names[f + 1]
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a2 = a2 if j not in [1, 17] else a2 / 2
                    a3 = -bone_h if k == 1 else bone_h
                    annotation = finger_name if k == 1 else ""
                    if virtual_node_matrix[i][j][k] is None:
                        virtual_node_matrix[i][j][k] = {
                            Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                            "Type": "internal",
                            "Annotation": [finger_name],
                        }
                    if virtual_node_matrix[i + 2][j][k] is None:
                        virtual_node_matrix[i + 2][j][k] = {
                            Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                            "Type": "internal",
                            "Annotation": [finger_name],
                        }
        x += pp + 2
        # MP
        finger_name = finger_names[f]
        parent_node_id += 1
        y_val = [y, y + 1]
        x_val = [(x + i, i / mp) for i in range(mp)]
        for k in k_val:
            for j in y_val:
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a3 = -bone_h if k == 1 else bone_h
                    if virtual_node_matrix[i][j][k] is None:
                        virtual_node_matrix[i][j][k] = {
                            Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                            "Type": "internal",
                            "Annotation": ["middle phalanx", finger_name],
                        }
        x += mp
        # DP
        parent_node_id += 1
        y_val = [y, y + 1]
        x_val = [(x + i, 0.75 * i) for i in range(dp)]
        for k in k_val:
            for j in y_val:
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a3 = -bone_h if k == 1 else bone_h
                    if virtual_node_matrix[i][j][k] is None:
                        virtual_node_matrix[i][j][k] = {
                            Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                            "Type": "internal",
                            "Annotation": ["distal phalanx", finger_name],
                        }
        x += dp
        y += 5
    # Thumb
    mc = 1
    y += 5
    x = c
    parent_node_id += 1
    y_val = [y, y + 1]
    x_val = [(x + i, i / mc) for i in range(mc)]
    finger_name = finger_names[-1]
    for k in k_val:
        for j in y_val:
            for i in x_val:
                a1 = i[1]
                i = i[0]
                a2 = bone_w if j == y + 1 else -bone_w
                a3 = -bone_h if k == 1 else bone_h
                if virtual_node_matrix[i][j][k] is None:
                    virtual_node_matrix[i][j][k] = {
                        Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                        "Type": "internal",
                        "Annotation": ["metacarpal", finger_name],
                    }
                if i == 1 and j == y + 1:
                    virtual_node_matrix[0][y - 4][k] = {
                        Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                        "Type": "internal",
                        "Annotation": ["metacarpal", finger_name],
                    }
    x += mc
    # PP
    parent_node_id += 1
    y_val = [y, y + 1]
    x_val = [(x + i, i / pp) for i in range(pp)]
    x_val = [(x, -0.2)]
    for k in k_val:
        for j in y_val:
            for i in x_val:
                a1 = i[1]
                i = i[0]
                a2 = -bone_w if j == y else bone_w
                a3 = -bone_h if k == 1 else bone_h
                if virtual_node_matrix[i][j][k] is None:
                    virtual_node_matrix[i][j][k] = {
                        Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                        "Type": "internal",
                        "Annotation": ["proximal phalanx", finger_name],
                    }
    x += pp
    # DP
    parent_node_id += 1
    y_val = [y, y + 1]
    x_val = [(x + i, 0.75 * i) for i in range(dp)]
    for k in k_val:
        for j in y_val:
            for i in x_val:
                a1 = i[1]
                i = i[0]
                a2 = -bone_w if j == y else bone_w
                a3 = -bone_h if k == 1 else bone_h
                if virtual_node_matrix[i][j][k] is None:
                    virtual_node_matrix[i][j][k] = {
                        Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                        "Type": "internal",
                        "Annotation": ["distal phalanx", finger_name],
                    }
    # Intermediate metacarpal connections
    for k in k_val:
        # Bottom connection
        virtual_node_matrix[1][y - 5][k] = virtual_node_matrix[0][y - 9][k]
        virtual_node_matrix[1][y - 4][k] = virtual_node_matrix[1][y + 1][k]
        virtual_node_matrix[2][y - 5][k] = virtual_node_matrix[1][y - 9][k]
        virtual_node_matrix[2][y - 4][k] = virtual_node_matrix[1][y][k]
        # Intermediate connection
        virtual_node_matrix[4][y - 5][k] = virtual_node_matrix[1][y - 9][k]
        virtual_node_matrix[4][y - 4][k] = virtual_node_matrix[1][y][k]
        virtual_node_matrix[5][y - 5][k] = virtual_node_matrix[2][y - 9][k]
        virtual_node_matrix[5][y - 4][k] = virtual_node_matrix[2][y][k]

    return virtual_node_matrix, node_identifier


def generate_external_node_matrix(fieldmodule, number_elements, node_identifier, virtual_node_matrix, options):
    c, mc, pp, mp, dp = number_elements
    # Finger elements are created in reverse order, from 5 (little) to 1 (thumb)
    # Palm skin nodes are sampled from an ellipse surrounding the internal nodes
    x, d1, d2, d3 = getNodeFieldParameters(fieldmodule, "coordinates", 6)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    fieldcache = fieldmodule.createFieldcache()
    finger_names = ["little finger", "ring finger", "middle finger", "index finger", "thumb"]
    center = add(x, d2)  # Ellipse center
    a = 5.5 * magnitude(d2)
    b = 2.5 * magnitude(d3)
    major_axis = mult([0, -1, 0], a)
    minor_axis = mult([0, 0, 1], b)
    ellipse = sampleEllipsePoints(center, major_axis, minor_axis, math.pi / 2, 2 * math.pi + math.pi / 2, 10)
    upper_ellipse_x = [ellipse[0][i] for i in [8, 9, 0, 1, 2]]
    upper_ellipse_d2 = [ellipse[1][i] for i in [8, 9, 0, 1, 2]]
    lower_ellipse_x = [ellipse[0][i] for i in [7, 6, 5, 4, 3]]
    lower_ellipse_d2 = [ellipse[1][i] for i in [7, 6, 5, 4, 3]]
    parent_node = 1
    node_id = 0
    y = 1
    jj = 0
    n = c
    # Carpal
    for f in range(4):
        node_id += 1
        x = 0
        y_val = [y, y + 1] if f == 0 else [y + 1]
        x_val = [x + i for i in range(n)]
        for j in y_val:
            for k in [1, 2]:
                ellipse_x = upper_ellipse_x if k == 2 else lower_ellipse_x
                ellipse_d2 = upper_ellipse_d2 if k == 2 else lower_ellipse_d2
                d1 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)[1]
                d1 = mult(d1, 1 / n)
                a3 = -1 if k == 1 else 1
                a2 = -1 if j == 1 else 1
                for i in x_val:
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": [""],
                    }
                    if j == 1:
                        virtual_node_matrix[i][j + a2][k] = node
                        virtual_node_matrix[i][j][k + a3] = node
                    elif j == 17:
                        virtual_node_matrix[i][j + a2][k] = node
                        virtual_node_matrix[i][j][k + a3] = node
                        virtual_node_matrix[i][j + 4][k + a3] = node
                    else:
                        virtual_node_matrix[i][j][k + a3] = node
                        virtual_node_matrix[i][j + 4][k + a3] = node
                    node_loc = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                    node_loc = add(node_loc, d1)
                    ellipse_x[jj] = node_loc
            jj += 1
        y += 5
        parent_node += 5
    # Metacarpal
    parent_node = 2
    jj = 0
    n = mc
    x += c
    y = 1
    for f in range(4):
        y_val = [y, y + 1] if f == 0 else [y + 1]
        x_val = [x + i for i in range(n)]
        z_val = [1, 2]
        for j in y_val:
            for k in z_val:
                ellipse_x = upper_ellipse_x if k == 2 else lower_ellipse_x
                ellipse_d2 = upper_ellipse_d2 if k == 2 else lower_ellipse_d2
                d1 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)[1]
                d1 = mult(d1, 1.0 / n)
                a3 = -1 if k == 1 else 1
                a2 = -1 if j == 1 else 1
                for i in x_val:
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": [""],
                    }
                    if j == 1:
                        virtual_node_matrix[i][j + a2][k] = node
                        virtual_node_matrix[i][j][k + a3] = node
                    elif j == 17:
                        if i > 1:
                            virtual_node_matrix[i][j + a2][k] = node
                        virtual_node_matrix[i][j][k + a3] = node
                    else:
                        virtual_node_matrix[i][j][k + a3] = node
                        virtual_node_matrix[i][j + 4][k + a3] = node
                    node_loc = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                    node_loc = add(node_loc, d1)
                    ellipse_x[jj] = node_loc
            jj += 1
        y += 5
        parent_node += 5
    # PP
    parent_node = 3
    jj = 0
    n = pp
    x += mc
    y = 1
    for f in range(4):
        y_val = [y, y + 1] if f == 0 else [y + 1]
        x_val = [x + i for i in range(n)]
        z_val = [1, 2]
        for j in y_val:
            for k in z_val:
                ellipse_x = upper_ellipse_x if k == 2 else lower_ellipse_x
                ellipse_d2 = upper_ellipse_d2 if k == 2 else lower_ellipse_d2
                d1 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)[1]
                d1 = mult(d1, 1.0 / n)
                a3 = -1 if k == 1 else 1
                a2 = -1 if j == 1 else 1
                finger_name = finger_names[f] if j == y else finger_names[f + 1]
                for i in x_val:
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": ['proximal phalanx', finger_name],
                    }
                    if j == 1:
                        virtual_node_matrix[i][j + a2][k] = node
                        virtual_node_matrix[i][j][k + a3] = node
                        virtual_node_matrix[i + 2][j + a2][k] = node
                        virtual_node_matrix[i + 2][j][k + a3] = node
                    elif j == 17:
                        if i > 1:
                            virtual_node_matrix[i][j + a2][k] = node
                            virtual_node_matrix[i + 2][j + a2][k] = node
                        virtual_node_matrix[i][j][k + a3] = node
                        virtual_node_matrix[i + 2][j][k + a3] = node
                    else:
                        virtual_node_matrix[i][j][k + a3] = node
                        virtual_node_matrix[i][j + 4][k + a3] = node
                        virtual_node_matrix[i + 2][j][k + a3] = node
                        virtual_node_matrix[i + 2][j + 4][k + a3] = node
                    node_loc = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                    node_loc = add(node_loc, d1)
                    ellipse_x[jj] = node_loc
            jj += 1
        y += 5
        # An update to the d2 of the parent node to make sure the internal and external elements align
        x0, d1, d2, d3 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)
        d2 = [0, abs(node_loc[1] - x0[1]), 0]
        node = nodes.findNodeByIdentifier(parent_node)
        fieldcache.setNode(node)
        setNodeFieldParameters(coordinates, fieldcache, x0, d1, d2, d3, d12=None, d13=None)
        parent_node += 5
    # Finger-palm connections
    y_val = [3, 8, 13]
    z_val = [1, 2]
    i += 2  # Last row of MC elements
    ii = 0
    for j in y_val:
        for k in z_val:
            a3 = -1 if k == 1 else 1
            finger_name = finger_names[ii]
            node_id = virtual_node_matrix[i][j - 1][k + a3][Node.VALUE_LABEL_VALUE][0][0]
            virtual_node_matrix[i][j - 1][k + a3] = {
                Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
                Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, -0.5 * a3, 0]],
                Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
                "Type": "external",
                "Annotation": [finger_name]
            }
            virtual_node_matrix[i][j][k] = {
                Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
                Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, -0.5 * a3, 0]],
                Node.VALUE_LABEL_D_DS2: [[node_id, 0, a3, 0, 0]],
                "Type": "external",
                "Annotation": [finger_name]
            }
            finger_name = finger_names[ii+1]
            virtual_node_matrix[i][j + 2][k] = {
                Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
                Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0.5 * a3, 0]],
                Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a3, 0, 0]],
                "Type": "external",
                "Annotation": [finger_name]
            }
            virtual_node_matrix[i][j + 3][k + a3] = {
                Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
                Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0.5 * a3, 0]],
                Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
                "Type": "external",
                "Annotation": [finger_name]
            }
        ii += 1
    # Fingers 5 to 2
    parent_node = 3
    y = 1
    n = pp
    jj = 0
    a = 2
    b = 2
    z_val = [1, 2]
    for f in range(4):
        # MP
        parent_node += 1
        x = c + mc + pp + 2
        n = mp
        jj = 0
        y_val = [y, y + 1]
        x_val = [x + i for i in range(n)]
        # Estimate ellipse
        center, d1, d2, d3 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)
        major_axis = mult(d2, -a)
        minor_axis = mult(d3, b)
        ellipse_x = []
        ellipse_d2 = []
        for ii in range(4):
            ellipse = sampleEllipsePoints(
                center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4, (1 + 2 * ii) * math.pi / 4, 1
            )
            ellipse_x.append(ellipse[0][0])
            ellipse_d2.append(ellipse[1][0])
        ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
        ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
        jj = 0
        for j in y_val:
            for k in z_val:
                d1 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)[1]
                d1 = mult(d1, 1 / n)
                a2 = -1 if j == y else 1
                a3 = -1 if k == 1 else 1
                finger_name = finger_names[f]
                for i in x_val:
                    node_loc = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": ['middle phalanx', finger_name],
                    }
                    virtual_node_matrix[i][j + a2][k] = node
                    virtual_node_matrix[i][j][k + a3] = node
                    node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                    node_loc = add(node_loc, d1)
                    ellipse_x[jj] = node_loc
                jj += 1
        # DP
        parent_node += 1
        x = c + mc + pp + mp + 2
        n = dp
        jj = 0
        y_val = [y, y + 1]
        x_val = [x + i for i in range(n)]
        # Estimate ellipse
        center, d1, d2, d3 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)
        major_axis = mult(d2, -a)
        minor_axis = mult(d3, b)
        ellipse_x = []
        ellipse_d2 = []
        for ii in range(4):
            ellipse = sampleEllipsePoints(
                center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4, (1 + 2 * ii) * math.pi / 4, 1
            )
            ellipse_x.append(ellipse[0][0])
            ellipse_d2.append(ellipse[1][0])
        ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
        ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
        jj = 0
        for j in y_val:
            for k in z_val:
                _, d1, d2, d3 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)
                d1 = mult(d1, 1 / (n - 1))
                a1 = 0.5
                a2 = -1 if j == y else 1
                a3 = -1 if k == 1 else 1
                finger_name = finger_names[f]
                for i in x_val:
                    node_loc = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": [finger_name],
                    }
                    virtual_node_matrix[i][j + a2][k] = node
                    virtual_node_matrix[i][j][k + a3] = node
                    if i == x_val[-1]:  # Special element to cap the fingers
                        virtual_node_matrix[i][j][k + a3] = {
                            Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                            Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                            Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, -a1 * a2 * a3, 1, 0]],
                            "Type": "external",
                            "Annotation": [finger_name],
                        }
                        virtual_node_matrix[i][j + a2][k] = {
                            Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                            Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                            Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, a1 * a2 * a3, 1, 0]],
                            "Type": "external",
                            "Annotation": [finger_name],
                        }
                        virtual_node_matrix[i + 1][j][k] = {
                            Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                            Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, a1 * -a2, a3, 0]],
                            Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, a1 * a3, a2, 0]],
                            "Type": "external",
                            "Annotation": [finger_name],
                        }
                    node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                    node_loc = add(node_loc, d1)
                    ellipse_x[jj] = node_loc
                jj += 1
        y += 5
        parent_node += 3
    # Thumb MC
    mc = 1
    parent_node = 21
    jj = 0
    n = mc
    x = c
    y += 5
    y_val = [y, y + 1]
    x_val = [x + i for i in range(n)]
    z_val = [1, 2]
    # Estimate ellipse
    center, d1, d2, d3 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)
    major_axis = mult(d2, -a)
    minor_axis = mult(d3, b)
    ellipse_x = []
    ellipse_d2 = []
    for ii in range(4):
        ellipse = sampleEllipsePoints(
            center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4, (1 + 2 * ii) * math.pi / 4, 1
        )
        ellipse_x.append(ellipse[0][0])
        ellipse_d2.append(ellipse[1][0])
    ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
    ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
    finger_name = finger_names[-1]
    jj = 0
    for j in y_val:
        for k in z_val:
            d1 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)[1]
            d1 = mult(d1, 1 / (n))
            a2 = -1 if j == y else 1
            a3 = -1 if k == 1 else 1
            for i in x_val:
                node_loc = ellipse_x[jj]
                d2 = ellipse_d2[jj]
                node = {
                    Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                    Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                    Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                    "Type": "external",
                    "Annotation": [finger_name],
                }
                virtual_node_matrix[i][j][k + a3] = node
                if j == y + 1:
                    virtual_node_matrix[i][j + a2][k] = node
                node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                node_loc = add(node_loc, d1)
                ellipse_x[jj] = node_loc
            jj += 1

    # Thumb PP
    parent_node += 1
    # y += 1
    x = c + mc
    n = pp
    y_val = [y, y + 1]
    x_val = [x + i for i in range(n)]
    # Estimate ellipse
    center, d1, d2, d3 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)
    major_axis = mult(d2, -a)
    minor_axis = mult(d3, b)
    ellipse_x = []
    ellipse_d2 = []
    for ii in range(4):
        ellipse = sampleEllipsePoints(
            center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4, (1 + 2 * ii) * math.pi / 4, 1
        )
        ellipse_x.append(ellipse[0][0])
        ellipse_d2.append(ellipse[1][0])
    ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
    ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
    jj = 0
    for j in y_val:
        for k in z_val:
            d1 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)[1]
            d1 = mult(d1, 1 / n)
            a2 = -1 if j == y else 1
            a3 = -1 if k == 1 else 1
            for i in x_val:
                node_loc = ellipse_x[jj]
                d2 = ellipse_d2[jj]
                node = {
                    Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                    Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                    Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                    "Type": "external",
                    "Annotation": [finger_name],
                }
                virtual_node_matrix[i][j + a2][k] = node
                virtual_node_matrix[i][j][k + a3] = node
                node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                node_loc = add(node_loc, d1)
                ellipse_x[jj] = node_loc
            jj += 1
    # DP
    parent_node += 1
    x = c + mc + pp
    n = dp
    y_val = [y, y + 1]
    x_val = [x + i for i in range(n)]
    # Estimate ellipse
    center, d1, d2, d3 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)
    major_axis = mult(d2, -a)
    minor_axis = mult(d3, b)
    ellipse_x = []
    ellipse_d2 = []
    for ii in range(4):
        ellipse = sampleEllipsePoints(
            center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4, (1 + 2 * ii) * math.pi / 4, 1
        )
        ellipse_x.append(ellipse[0][0])
        ellipse_d2.append(ellipse[1][0])
    ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
    ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
    jj = 0
    for j in y_val:
        for k in z_val:
            d1 = getNodeFieldParameters(fieldmodule, "coordinates", parent_node)[1]
            d1 = mult(d1, 1 / (n - 1))
            a1 = 0.5
            a2 = -1 if j == y else 1
            a3 = -1 if k == 1 else 1
            for i in x_val:
                node_loc = ellipse_x[jj]
                d2 = ellipse_d2[jj]
                node = {
                    Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                    Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                    Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                    "Type": "external",
                    "Annotation": [finger_name],
                }
                virtual_node_matrix[i][j + a2][k] = node
                virtual_node_matrix[i][j][k + a3] = node
                if i == x_val[-1]:
                    virtual_node_matrix[i][j][k + a3] = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, -a1 * a2 * a3, 1, 0]],
                        "Type": "external",
                        "Annotation": [finger_name],
                    }
                    virtual_node_matrix[i][j + a2][k] = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, a1 * a2 * a3, 1, 0]],
                        "Type": "external",
                        "Annotation": [finger_name],
                    }
                    virtual_node_matrix[i + 1][j][k] = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, a1 * -a2, a3, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, a1 * a3, a2, 0]],
                        "Type": "external",
                        "Annotation": [finger_name],
                    }
                node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                node_loc = add(node_loc, d1)
                ellipse_x[jj] = node_loc
            jj += 1

    # The Thumb-palm connection elements
    # It is against every fiber of my being that I am forced to set these manually
    thumb_angle_degrees = options["Thumb angle"]
    thumb_c = thumb_angle_degrees / 90
    # Back of the thumb
    z_val = [0, 3]
    for k in z_val:
        a1 = -1 if k == 3 else 1
        a3 = -1 if k == 3 else 1
        node_id = virtual_node_matrix[0][y - 9][k][Node.VALUE_LABEL_VALUE][0][0]  # Get id from existing node
        virtual_node_matrix[0][y - 5][k + a3] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 0.5, -a1 * 0.5, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
            "Type": "external",
        }
        node_id = virtual_node_matrix[1][y + 1][k][Node.VALUE_LABEL_VALUE][0][0]  # Get id from existing node
        virtual_node_matrix[0][y - 4][k + a3] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, a1 * thumb_c, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
            "Type": "external",
        }

    # Bottom carpal-metacarpal connection
    z_val = [0, 3]
    for k in z_val:
        a1 = -1 if k == 3 else 1
        a2
        node_id = virtual_node_matrix[1][y + 1][k][Node.VALUE_LABEL_VALUE][0][0]  # Get id from existing node
        virtual_node_matrix[1][y - 4][k] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 0, a1, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a1, -thumb_c, 0]],
            "Type": "external",
        }
        node_id = virtual_node_matrix[1][y][k][Node.VALUE_LABEL_VALUE][0][0]
        virtual_node_matrix[2][y - 4][k] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 0, a1, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a1 * thumb_c, 0, 0]],
            "Type": "external",
        }

        node_id = virtual_node_matrix[0][y - 9][k][Node.VALUE_LABEL_VALUE][0][0]
        virtual_node_matrix[1][y - 5][k] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a1 * 0.5, 0.5, 0]],
            "Type": "external",
        }
        node_id = virtual_node_matrix[1][y - 9][k][Node.VALUE_LABEL_VALUE][0][0]
        virtual_node_matrix[2][y - 5][k] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 0.25, 0]],
            "Type": "external",
        }
    # Intermediate metacarpal connections
    z_val = [0, 3]
    for k in z_val:
        a1 = 2 * thumb_c
        a3 = -1 if k == 3 else 1

        node_id = virtual_node_matrix[1][y][k][Node.VALUE_LABEL_VALUE][0][0]
        virtual_node_matrix[4][y - 4][k] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a3 * thumb_c, 0, 0]],
            "Type": "external",
        }

        node_id = virtual_node_matrix[1][y - 9][k][Node.VALUE_LABEL_VALUE][0][0]
        virtual_node_matrix[4][y - 5][k] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 0.25, 0]],
            "Type": "external",
        }
        node_id = virtual_node_matrix[2][y][k][Node.VALUE_LABEL_VALUE][0][0]
        virtual_node_matrix[5][y - 4][k] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a1 * a3, 0.5, 0]],
            "Type": "external",
        }

        node_id = virtual_node_matrix[2][y - 9][k][Node.VALUE_LABEL_VALUE][0][0]
        virtual_node_matrix[5][y - 5][k] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, a1 * a3, 0.5, 0]],
            "Type": "external",
        }

    # Front of the webbing
    z_val = [0, 3]
    for k in z_val:
        a1 = 2 * thumb_c
        # a1 = -1 if k == 3 else 1
        a3 = -1 if k == 3 else 1
        node_id = virtual_node_matrix[2][y - 9][k][Node.VALUE_LABEL_VALUE][0][0]  # Get id from existing node
        virtual_node_matrix[6][y - 5][k + a3] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, -a1, -a3 * 0.5, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
            "Type": "external",
        }
        node_id = virtual_node_matrix[2][y][k][Node.VALUE_LABEL_VALUE][0][0]  # Get id from existing node
        virtual_node_matrix[6][y - 4][k + a3] = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, a1, -a3 * 0.5, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, -1, 0]],
            "Type": "external",
        }

    return virtual_node_matrix, node_identifier


def create_cube_element(fieldmodule, element_identifier, node_matrix, x, y, z):
    """
    Docstring for create_cube_element
    First part of this function finds the orientation of the cube element
                   z     x
                   |    /
                   |   /
                   |  /
                   | /
        y ---------+
    which is represented by by corresponding transformation
    1) (x, y, z) (standard, unchanged)
    2) (x, z, -y)
    3) (x, -y, -z)
    4) (x, -x, y)
    5) (-z, x, -y)
    6) (z, x, -y)
    Immediately after, the eight corner of the cube are reorganized according to this
    transformation and the value scale factors are collected from each virtual node.
    If are of the virtual nodes are external, the scale factors for d1, d2 are either
    collected from the virtual node or calculated (for the linear direction)
    :param fieldmodule: Description
    :param element_identifier: Description
    :param node_matrix: Description
    :param x: Description
    :param y: Description
    :param z: Description
    :return: Description
    :rtype: Any
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    # Criteria to identify the z-axis of the element
    node_1 = node_matrix[x][y][z]
    node_2 = node_matrix[x + 1][y][z]
    node_3 = node_matrix[x][y + 1][z]
    if node_1 is None or node_2 is None or node_3 is None:
        return -2, None
    n1_ext = True if node_1["Type"] == "external" else False
    n2_ext = True if node_2["Type"] == "external" else False
    n3_ext = True if node_3["Type"] == "external" else False
    # 6 possible cases, one for each face of a cube
    if not n1_ext and not n2_ext and not n3_ext:
        ranges = [[0, 1], [0, 1], [0, 1]]
        order = [2, 1, 0]
    elif not n1_ext and not n2_ext and n3_ext:
        ranges = [[0, 1], [0, 1], [1, 0]]
        order = [1, 2, 0]
    elif n1_ext and n2_ext and n3_ext:
        ranges = [[0, 1], [1, 0], [1, 0]]
        order = [2, 1, 0]
    elif n1_ext and n2_ext and not n3_ext:
        ranges = [[0, 1], [1, 0], [0, 1]]
        order = [1, 2, 0]
    elif n1_ext and not n2_ext and n3_ext:
        ranges = [[1, 0], [0, 1], [1, 0]]
        order = [0, 2, 1]
    else:
        ranges = [[0, 1], [0, 1], [1, 0]]
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
    global_to_local_scale_factor_ids = {}
    local_node = 0
    local_node_ids = {}
    value_expression_terms = {}
    d1_expression_terms = {}
    d2_expression_terms = {}
    negative_value_ets = {}
    readable_expression_terms = {}
    readable_negative_ets = {}
    n_local_nodes = 0
    n_scale_factors = 0
    value_labels = [0, Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                    Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3]
    value_label_names = [0, "v", "d1", "d2", "d3"]
    local_node = 0
    for index in indices:
        et = []
        neg_et = []
        ret = []
        i, j, k = index
        local_node += 1
        virtual_node = node_matrix[x + i][y + j][z + k]
        if virtual_node is None:
            return -2, None
        if virtual_node["Type"] == "external":
            is_bicubic = True
        virtual_node = virtual_node[Node.VALUE_LABEL_VALUE]
        # Value expression terms
        for global_node in virtual_node:
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
                    local_scale_factor_id = [global_to_local_scale_factor_ids[global_scale_factor]]
                    et.append((local_node_ids[global_node_id], value_labels[factor], local_scale_factor_id))
            value_expression_terms[local_node] = et
    local_to_global_scale_factor_ids = {value: key for key, value in global_to_local_scale_factor_ids.items()}
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
                neg_et.append((l_node_id, label, global_to_local_scale_factor_ids[global_scale_factor]))
            negative_value_ets[local_node] = neg_et
            readable_negative_ets[local_node] = red_et
        # Create d1 expression terms
        # d1 and d2 terms for the linear part follow a simple formula
        # d1 and d2 terms for the cubic part are calculated directly from the virtual node information
        d1_expression_terms[1] = value_expression_terms[2] + negative_value_ets[1]
        d1_expression_terms[2] = value_expression_terms[2] + negative_value_ets[1]
        d1_expression_terms[3] = value_expression_terms[4] + negative_value_ets[3]
        d1_expression_terms[4] = value_expression_terms[4] + negative_value_ets[3]
        label = Node.VALUE_LABEL_D_DS1
        for local_node in range(5, 9):
            et = []
            ret = []
            i, j, k = indices[local_node - 1]
            virtual_node = node_matrix[x + i][y + j][z + k]
            virtual_node = virtual_node[label]
            for global_node in virtual_node:
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
                            (str(global_node_id).zfill(2), value_label_names[factor], str(scale_factor).zfill(4))
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
            virtual_node = node_matrix[x + i][y + j][z + k]
            virtual_node = virtual_node[label]
            for global_node in virtual_node:
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
                            (str(global_node_id).zfill(2), value_label_names[factor], str(scale_factor).zfill(4))
                        )
                d2_expression_terms[local_node] = et
    # Create and remap eft
    if is_bicubic:
        # Bicubic linear element (skin)
        bicubic_linear_basis = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY)
        bicubic_linear_basis.setFunctionType(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        eft = mesh3d.createElementfieldtemplate(bicubic_linear_basis)
        setEftScaleFactorIds(eft, [], [], n_scale_factors)
        for local_node in value_expression_terms:
            remapEftNodeValueLabelWithNodes(eft, local_node, Node.VALUE_LABEL_VALUE, value_expression_terms[local_node])
            remapEftNodeValueLabelWithNodes(eft, local_node, Node.VALUE_LABEL_D_DS1, d1_expression_terms[local_node])
            remapEftNodeValueLabelWithNodes(eft, local_node, Node.VALUE_LABEL_D_DS2, d2_expression_terms[local_node])
        remapEftLocalNodes(eft, n_local_nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        # Create element template
        etemplate = mesh3d.createElementtemplate()
        etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        result = etemplate.defineField(coordinates, -1, eft)
        if result != RESULT_OK:
            return result, None
    else:
        # Trilinear element (bone)
        trilinear_basis = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        eft = mesh3d.createElementfieldtemplate(trilinear_basis)
        setEftScaleFactorIds(eft, [], [], n_scale_factors)
        for local_node in value_expression_terms:
            remapEftNodeValueLabelWithNodes(eft, local_node, Node.VALUE_LABEL_VALUE, value_expression_terms[local_node])
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
    # element_identifier += 1
    return result, element


def add_skin_node_new(fieldmodule: Fieldmodule, node_identifier: int, node_location: list, d1, d2, d3=None):
    # Zinc setup
    coordinates = find_or_create_field_coordinates(fieldmodule)
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = get_simple_nodetemplate(fieldmodule)
    fieldcache = fieldmodule.createFieldcache()
    # Create skin node
    skin_node = nodes.createNode(node_identifier, nodetemplate)
    fieldcache.setNode(skin_node)
    d1 = [0, 0, 0] if d1 is None else d1
    d2 = [0, 0, 0] if d2 is None else d2
    d3 = [0, 0, 0] if d3 is None else d3
    setNodeFieldParameters(coordinates, fieldcache, node_location, d1, d2, d3)
    node_identifier += 1
    return node_identifier


def add_node(fieldmodule: Fieldmodule, bone_node_id: int, node_identifier: int, scale_factors: list):
    # Zinc setup
    coordinates = find_or_create_field_coordinates(fieldmodule)
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = get_simple_nodetemplate(fieldmodule)
    fieldcache = fieldmodule.createFieldcache()
    # Get position from node
    bone_node = nodes.findNodeByIdentifier(bone_node_id)
    fieldcache.setNode(bone_node)
    a0, a1, a2, a3 = scale_factors
    value_labels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3]
    node_params = []
    for label in value_labels:
        node_params.append(coordinates.getNodeParameters(fieldcache, -1, label, 1, 3)[1])
    x0, e1, e2, e3 = node_params
    # Calculating skin node parameters
    x = [0, 0, 0]
    for i in range(4):
        x = add(x, mult(node_params[i], scale_factors[i]))
    # Create skin node
    skin_node = nodes.createNode(node_identifier, nodetemplate)
    fieldcache.setNode(skin_node)
    d1 = node_params[1]
    d2 = [0, 0, 0]
    d3 = [0, 0, 0]
    setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
    node_identifier += 1
    return node_identifier


def add_skin_node(
    fieldmodule: Fieldmodule, bone_node_id: int, node_identifier: int, scale_factors: list, directions: list
):
    # Zinc setup
    coordinates = find_or_create_field_coordinates(fieldmodule)
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = get_simple_nodetemplate(fieldmodule)
    fieldcache = fieldmodule.createFieldcache()
    # Get position from node
    bone_node = nodes.findNodeByIdentifier(bone_node_id)
    fieldcache.setNode(bone_node)
    a0, a1, a2, a3 = scale_factors
    value_labels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3]
    node_params = []
    for label in value_labels:
        node_params.append(coordinates.getNodeParameters(fieldcache, -1, label, 1, 3)[1])
    x0, e1, e2, e3 = node_params
    # Calculating skin node parameters
    x = [0, 0, 0]
    for i in range(4):
        x = add(x, mult(node_params[i], scale_factors[i]))
    # Create skin node
    skin_node = nodes.createNode(node_identifier, nodetemplate)
    fieldcache.setNode(skin_node)
    d1, d2 = directions
    # d2 = set_magnitude(d2, magnitude(mult(node_params[2], scale_factors[2])))
    # d1 = set_magnitude(d1, magnitude(mult(node_params[1], 1-scale_factors[1])))
    d2 = set_magnitude(d2, magnitude(node_params[2])) if magnitude(d2) != 0 else d2
    # d1 = set_magnitude(d1, magnitude(node_params[1])) if magnitude(d1) != 0 else d2
    d3 = [0, 0, 0]
    setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
    node_identifier += 1
    return node_identifier


def create_thumb_nodes(fieldmodule, node_identifier, finger_dimensions, metacarpal_node_id, angle_degrees=45):
    """
    Docstring for create_thumb_nodes

    :param fieldmodule: Description
    :param node_identifier: Description
    :param finger_dimensions: Description
    :param carpal_node: Description
    :return: Description
    :rtype: Any
    """
    # Zinc setip
    coordinates = find_or_create_field_coordinates(fieldmodule)
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = get_simple_nodetemplate(fieldmodule)
    fieldcache = fieldmodule.createFieldcache()
    # Set basic directions
    x1 = [1, 0, 0]
    x2 = [0, 1, 0]
    x3 = [0, 0, 1]
    # Thumb flexion angle
    angle_radians = math.radians(angle_degrees)
    x1 = rotate_vector_around_vector(x1, x3, angle_radians)
    x2 = rotate_vector_around_vector(x2, x3, angle_radians)
    # Obtaining the starting node position from the metacarpal node
    metacarpal_node = nodes.findNodeByIdentifier(metacarpal_node_id)
    fieldcache.setNode(metacarpal_node)
    node_location = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)[1]
    d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)[1]
    d2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)[1]
    node_location = add(node_location, d2)
    node_location = add(node_location, d2)
    node_location = add(node_location, d2)
    # node_location = add(node_location, mult(d1, 0.1))

    # Manually override the width of the thumb node?
    # finger_dimensions[0][1] = magnitude(d1)/6
    """
    Finger bone dimensions have the format
    finger_dimensions = [metacarpal, p_phalax, m_phalanx, d_phalanx]
    and each bone has four corresponding dimensions
    [length, box_width, height, bone_width]
    """
    finger_node_identifier = node_identifier
    for i in range(3):
        node = nodes.createNode(finger_node_identifier, nodetemplate)
        fieldcache.setNode(node)
        bone_dimensions = finger_dimensions[i]
        d1 = mult(x1, bone_dimensions[0])
        d2 = mult(x2, bone_dimensions[1])
        d3 = mult(x3, bone_dimensions[2])
        d12 = mult(x2, bone_dimensions[3])
        x = node_location
        if i == 0:
            x = add(x, d2)
        #  x = add(x, d2)
        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
        x = add(x, d1)
        node_location = x
        #
        finger_node_identifier += 1
    node_identifier += 1
    return finger_node_identifier


def create_finger_nodes_new(fieldmodule, node_identifier, finger_dimensions, starting_location, finger_number=1):
    """
    Docstring for create_finger_nodes
    :param fieldmodule: Description
    :param node_identifier: Description
    :param finger_dimensions: Description
    :param carpal_node: Description
    :return: Description
    :rtype: Any
    """
    coordinates = find_or_create_field_coordinates(fieldmodule)
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = get_simple_nodetemplate(fieldmodule)
    fieldcache = fieldmodule.createFieldcache()
    # Set basic directions
    x1 = [1, 0, 0]
    x2 = [0, 1, 0]
    x3 = [0, 0, 1]
    """
    Finger bone dimensions have the format
    finger_dimensions = [metacarpal, p_phalax, m_phalanx, d_phalanx]
    and each bone has four corresponding dimensions
    [length, box_width, height, bone_width]
    """
    node_location = starting_location
    finger_node_identifier = node_identifier
    for i in range(len(finger_dimensions)):
        node = nodes.createNode(finger_node_identifier, nodetemplate)
        fieldcache.setNode(node)
        bone_dimensions = finger_dimensions[i]
        d1 = mult(x1, bone_dimensions[0])
        d2 = mult(x2, bone_dimensions[1])
        d3 = mult(x3, bone_dimensions[2])
        d12 = mult(x2, bone_dimensions[3])
        x = node_location
        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
        x = add(x, d1)
        node_location = x
        #
        finger_node_identifier += 1
    return finger_node_identifier


def create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_node):
    """
    Docstring for create_finger_nodes

    :param fieldmodule: Description
    :param node_identifier: Description
    :param finger_dimensions: Description
    :param carpal_node: Description
    :return: Description
    :rtype: Any
    """
    coordinates = find_or_create_field_coordinates(fieldmodule)
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = get_simple_nodetemplate(fieldmodule)
    fieldcache = fieldmodule.createFieldcache()
    # Set basic directions
    x1 = [1, 0, 0]
    x2 = [0, 1, 0]
    x3 = [0, 0, 1]
    """
    Finger bone dimensions have the format
    finger_dimensions = [metacarpal, p_phalax, m_phalanx, d_phalanx]
    and each bone has four corresponding dimensions
    [length, box_width, height, bone_width]
    """
    node_location = carpal_node
    finger_node_identifier = node_identifier
    for i in range(5):
        node = nodes.createNode(finger_node_identifier, nodetemplate)
        fieldcache.setNode(node)
        bone_dimensions = finger_dimensions[i]
        d1 = mult(x1, bone_dimensions[0])
        d2 = mult(x2, bone_dimensions[1])
        d3 = mult(x3, bone_dimensions[2])
        d12 = mult(x2, bone_dimensions[3])
        x = node_location
        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
        x = add(x, d1)
        node_location = x
        #
        finger_node_identifier += 4
    node_identifier += 1
    return node_identifier


def get_simple_nodetemplate(fieldmodule):
    """ """
    coordinates = find_or_create_field_coordinates(fieldmodule)
    value_labels = [
        Node.VALUE_LABEL_VALUE,
        Node.VALUE_LABEL_D_DS1,
        Node.VALUE_LABEL_D_DS2,
        Node.VALUE_LABEL_D2_DS1DS2,
        Node.VALUE_LABEL_D_DS3,
        Node.VALUE_LABEL_D2_DS1DS3,
    ]
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = nodes.createNodetemplate()
    nodetemplate.defineField(coordinates)
    for value_label in value_labels[1:]:
        nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)
    return nodetemplate


def setNodeFieldParameters(field, fieldcache, x, d1, d2, d3, d12=None, d13=None):
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
    field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, d1)
    field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, d2)
    field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)
    if d12:
        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, d12)
    if d13:
        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, d13)


def getNodeFieldParameters(fieldmodule, field_name, node_identifier, d12=None, d13=None):
    """
    Retrieve node field parameters x, d1, d2, d3 of field.
    :param field: Field parameters to assign.
    :param x: Parameters to set for Node.VALUE_LABEL_VALUE.
    :param d1: Parameters to set for Node.VALUE_LABEL_D_DS1.
    :param d2: Parameters to set for Node.VALUE_LABEL_D_DS2.
    :param d3: Parameters to set for Node.VALUE_LABEL_D_DS3.
    :param d12: Optional parameters to set for Node.VALUE_LABEL_D2_DS1DS2.
    :param d13: Optional parameters to set for Node.VALUE_LABEL_D2_DS1DS3.
    :return:
    """
    fieldcache = fieldmodule.createFieldcache()
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    node = nodes.findNodeByIdentifier(node_identifier)
    fieldcache.setNode(node)
    field = find_or_create_field_coordinates(fieldmodule, field_name)
    params = []
    params.append(field.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)[1])
    params.append(field.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)[1])
    params.append(field.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)[1])
    params.append(field.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)[1])
    if d12:
        params.append(field.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)[1])
    if d13:
        params.append(field.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)[1])
    return params
