import math
from itertools import product

from cmlibs.maths.vectorops import add, mult, rotate_vector_around_vector
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
        fieldmodule = region.getFieldmodule()
        fieldcache = fieldmodule.createFieldcache()

        node_identifier = 1
        hand_elements_along = [1, 2, 1, 1, 2] # carpal, metacarpal, prox, middle, dist phalanx
        node_bag = Node_bag(region, fieldcache)
        ix = 2 * sum(hand_elements_along)
        iy = 6 * 5
        iz = 4
        virtual_node_matrix = Virtual_node_matrix([ix, iy, iz])

        node_bag, node_identifier = generate_internal_nodes(
            node_bag, hand_elements_along, node_identifier, options = options
        )

        virtual_node_matrix, node_bag, node_identifier = generate_internal_node_matrix(
            hand_elements_along, node_identifier, virtual_node_matrix,
            node_bag=node_bag, options=options
        )

        virtual_node_matrix, node_bag, node_identifier = generate_external_node_matrix(
            hand_elements_along, node_identifier, virtual_node_matrix,
            node_bag=node_bag, options=options
        )

        finger_names = ["little finger", "ring finger",
                        "middle finger", "index finger", "thumb"]
        finger_bone_names = ['proximal phalanx', 'middle phalanx', 'distal phalanx']
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
                finger_bone_group = AnnotationGroup(region, get_hand_term(
                    bone + ' of ' +  finger
                ))
                annotation_groups.append(finger_bone_group)

        element_identifier = 1
        node_bag.set_zinc_nodes()
        matrix = virtual_node_matrix.get_matrix()
        for k in range(iz - 1):
            for j in range(iy - 1):
                for i in range(ix - 1):
                    result, element = create_cube_element(
                        fieldmodule, element_identifier, matrix, [i, j, k],
                    )
                    if result == RESULT_OK:
                        element_annotations = [hand_group]
                        # find annotation groups
                        try:
                            node_annotations = matrix[i][j][k]["Annotation"]
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
        :param annotationGroups: List of annotation groups for elements
            created in generateBaseMesh().
            New face/line annotation groups are appended to this list.
        """
        fieldmodule = region.getFieldmodule()
        mesh2d = fieldmodule.findMeshByDimension(2)
        is_exterior = fieldmodule.createFieldIsExterior()
        is_face_xi3_0 = fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0)

        skin_group = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_hand_term('skin of hand'))
        is_skin = fieldmodule.createFieldAnd(
            is_exterior, fieldmodule.createFieldNot(is_face_xi3_0)
        )
        skin_group.getMeshGroup(mesh2d).addElementsConditional(is_skin)

        finger_names = ["little finger", "ring finger",
                        "middle finger", "index finger", "thumb"]
        finger_bone_names = ['proximal phalanx', 'middle phalanx', 'distal phalanx']
        for finger_name in finger_names:
            finger_group = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_hand_term(finger_name))
            finger_skin_group = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_hand_term('skin of ' + finger_name))
            finger_skin_group.getMeshGroup(mesh2d).addElementsConditional(
            fieldmodule.createFieldAnd(finger_group.getGroup(), is_exterior)
            )
            for bone_name in finger_bone_names:
                finger_bone_group = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_hand_term(bone_name + ' of ' +  finger_name)
                )
                finger_bone_skin_group = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_hand_term(
                    'skin of ' + bone_name + ' of ' +  finger_name)
                )
                finger_bone_skin_group.getMeshGroup(mesh2d).addElementsConditional(
                fieldmodule.createFieldAnd(finger_bone_group.getGroup(), is_exterior)
                )

class Node_bag():

    def __init__(self, region: Region, fieldcache: Fieldcache):
        self.fieldmodule = region.getFieldmodule()
        self.fieldcache = fieldcache
        self.coordinates = find_or_create_field_coordinates(self.fieldmodule, 'coordinates')
        self.nodes = self.fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.node_identifier = 1
        self.nodes_dict = {}
        value_labels = [
            Node.VALUE_LABEL_VALUE,
            Node.VALUE_LABEL_D_DS1,
            Node.VALUE_LABEL_D_DS2,
        ]
        nodes = self.nodes
        coordinates = self.coordinates
        external_nodetemplate = nodes.createNodetemplate()
        external_nodetemplate.defineField(coordinates)
        for value_label in value_labels[1:]:
            external_nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)
        self.external_nodetemplate = external_nodetemplate

        value_labels = [
            Node.VALUE_LABEL_VALUE,
            Node.VALUE_LABEL_D_DS1,
            Node.VALUE_LABEL_D_DS2,
            Node.VALUE_LABEL_D_DS3,
        ]
        internal_nodetemplate = nodes.createNodetemplate()
        internal_nodetemplate.defineField(coordinates)
        for value_label in value_labels[1:]:
            internal_nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)
        self.internal_nodetemplate = internal_nodetemplate

    def set_node_parameters(self, node_coordinates: list, node_identifier:int = 0,
                 internal_node=True) -> None :
        nodes_dict = self.nodes_dict
        assert len(node_coordinates) == 4
        x, d1, d2, d3 = node_coordinates
        nodes_dict[node_identifier] = {
            'parameters': [x, d1, d2, d3],
            'internal': internal_node
        }

    def set_zinc_nodes(self):
        node_dict = self.nodes_dict
        coordinates = self.coordinates
        fieldcache = self.fieldcache
        for node_identifier, node_params in node_dict.items():
            is_internal = node_params.get('internal')
            node_coordinates = node_params.get('parameters')
            if is_internal:
                x, d1, d2, d3 = node_coordinates
                nodetemplate = self.internal_nodetemplate
            else:
                x, d1, d2, d3 = node_coordinates
                d3 = None
                nodetemplate = self.external_nodetemplate
            node = self.nodes.createNode(node_identifier, nodetemplate)
            fieldcache.setNode(node)
            setNodeFieldParameters(coordinates, fieldcache, x, d1 = d1, d2 = d2, d3 = d3)

    def get_node_parameters(self, node_identifier):
        nodes_dict = self.nodes_dict
        node_dict = nodes_dict[node_identifier].get('parameters')
        return node_dict

    def get_node_identifier(self)-> int:
        return self.node_identifier

class Virtual_node_matrix():

    def __init__(self, matrix_dimensions: list):
        ix, iy, iz = matrix_dimensions
        self.node_matrix = [[[None for k in range(iz)] for j in range(iy)] for i in range(ix)]

    def set_virtual_node_to_index(self, node: dict, index: list, replace=True):
        i, j, k = index
        node_matrix = self.node_matrix
        if replace:
            node_matrix[i][j][k] = node
        else:
            if node_matrix[i][j][k] is None:
                node_matrix[i][j][k] = node

    def set_virtual_node_to_indices(self, node:dict, indices: list, replace=True):
        for index in indices:
            self.set_virtual_node_to_index(node, index, replace)

    def get_matrix(self) -> list:
        return self.node_matrix

    def get_virtual_node(self, index: list) -> dict:
        i, j, k = index
        node_matrix = self.node_matrix
        virtual_node = node_matrix[i][j][k]
        return virtual_node

    def print_cube_from_index(self, start_index: list):
        start_i, start_j, start_k = start_index
        ranges = [[0, 1], [0, 1], [0, 1]] # 1
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
        for index in indices:
            i, j, k = index
            print([start_i + i, start_j + j, start_k + k],
                  node_matrix[start_i + i][start_j + j][start_k + k])



def generate_internal_nodes(node_bag: Node_bag, hand_elements_along: list,
                            node_identifier: int = 1, options: dict = {}):
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
    
    c, mc, pp, mp, dp = hand_elements_along
    finger_dimensions = [  # d1, d2, d3
        [1.0, 0.4, 0.2],  # carpal
        [2.5, 0.4, 0.2],  # metacarpal
        [1.5, 0.3, 0.2],  # proximal phalanx
        [1.5, 0.2, 0.2],  # middle phalanx
        [1.0, 0.2, 0.2],  # distal phalanx
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
            bone_dimensions = finger_dimensions[i]
            d1 = mult(x1, bone_dimensions[0])
            d2 = mult(x2, bone_dimensions[1])
            d3 = mult(x3, bone_dimensions[2])
            if i == 2:
                x = add(x, mult(d2, -1.0 + (2.0 / 3.0) * j))
            node_bag.set_node_parameters([x, d1, d2, d3], node_identifier, internal_node=True)
            x = add(x, d1)
            if (j == 3) and (i == 1):
                index_carpal_node_id = node_identifier
            node_identifier += 1
    thumb_dimensions = [  # d1, d2, d3, d12
            [1.0, 0.25, 0.2, 0.2],  # metacarpal
            [1.0, 0.25, 0.2, 0.2],  # proximal phalanx
            [1.0, 0.2, 0.2, 0.2],  # distal phalanx
        ]
    thumb_angle_degrees = options["Thumb angle"]
    # Thumb flexion angle
    angle_radians = math.radians(thumb_angle_degrees)
    x1 = rotate_vector_around_vector(x1, x3, angle_radians)
    x2 = rotate_vector_around_vector(x2, x3, angle_radians)
    # Obtain the starting node position from the metacarpal node
    x, d1, d2, d3 = node_bag.get_node_parameters(index_carpal_node_id)
    x = add(x, d2)
    x = add(x, d2)
    x = add(x, d2)
    x = add(x, mult(d1, 0.25))
    for i in range(3):
        bone_dimensions = thumb_dimensions[i]
        d1 = mult(x1, bone_dimensions[0])
        d2 = mult(x2, bone_dimensions[1])
        d3 = mult(x3, bone_dimensions[2])
        x = x
        if i == 0:
            x = add(x, d2)
        node_bag.set_node_parameters([x, d1, d2, d3], node_identifier, internal_node=True)
        x = add(x, d1)
        node_identifier += 1
    return node_bag, node_identifier


def generate_internal_node_matrix(hand_elements_along, node_identifier,
                                  virtual_node_matrix: Virtual_node_matrix,
                                  node_bag: Node_bag, options=None):
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
    index_y = 1
    d1_offset = 0.25
    k_val = [1, 2]
    columns_per_finger = 5
    finger_names = ["little finger", "ring finger", "middle finger", "index finger", "thumb"]
    bone_names = ['carpal', 'metacarpal', 'proximal phalanx',
                         'middle phalanx', 'distal phalanx']
    # Finger
    for f in range(4):
        # Carpal
        parent_node_id += 1
        index_x = 0
        j_val = [index_y, index_y + 1] if f == 0 else [index_y + 1]
        i_val = [(index_x + i, i / c) for i in range(c)]
        for k in k_val:
            for j in j_val:
                for i in i_val:
                    indices = []
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == index_y else bone_w
                    a3 = -bone_h if k == 1 else bone_h
                    annotation = bone_names[0] if k == 1 else ""
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        "Type": "internal",
                        "Annotation": [annotation],
                    }
                    indices.append([i, j, k])
                    if j == index_y + 1:
                        if f == 3: # Webbing connection
                            indices.append([i + 1, j + 4, k])
                        else:
                            indices.append([i, j + 4, k])
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)
                    x, d1, d2, d3 = node_bag.get_node_parameters(parent_node_id)
                    x = add(x, mult(d2, a2))
                    x = add(x, mult(d3, a3))
                    node_bag.set_node_parameters(
                        [x, d1, None, None], node_identifier, internal_node=False
                    )
                    node_identifier += 1
        index_x = c
        # Metacarpal
        parent_node_id += 1
        j_val = [index_y, index_y + 1, index_y + 5]
        i_val = [(index_x + i, i / mc) for i in range(mc)]
        i_val = [(index_x, 0), (index_x + 1, d1_offset)]
        for k in k_val:
            for j in j_val:
                for i in i_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == index_y else bone_w
                    a3 = -bone_h if k == 1 else bone_h
                    annotation = bone_names[1] if k == 1 else ""
                    node = {
                        Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                        "Type": "internal",
                        "Annotation": [annotation],
                    }
                    indices = [[i, j, k]]
                    if f == 3 and j == index_y + 5: # Webbing connection
                        if i == index_x:
                            indices = [[i + 1, j, k],[i + 3, j, k]]
                        else:
                            indices = [[i + 3, j, k]]
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)
        index_x += mc
        # PP
        parent_node_id += 1
        j_val = [index_y, index_y + 1, index_y + 5] if f != 4 else [index_y, index_y + 1]
        i_val = [(index_x + i, (-d1_offset) * (i+1) / pp) for i in range(pp)]
        for k in k_val:
            for j in j_val:
                finger_name = finger_names[f] if (j != index_y+5) else finger_names[f + 1]
                for i in i_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == index_y else bone_w
                    a2 = a2 if j not in [1, 17] else a2 / 2
                    a3 = -bone_h if k == 1 else bone_h
                    bone_name = bone_names[2]
                    finger_part = bone_name + ' of ' + finger_name
                    annotation = [bone_name, finger_name, finger_part]
                    node = {
                        Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                        "Type": "internal",
                        "Annotation": annotation,
                    }
                    indices = [[i, j, k], [i + 2, j, k]]
                    if j == index_y + 1:
                        indices.append([i, j + 4, k])
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)
        index_x += pp + 2
        # MP
        finger_name = finger_names[f]
        parent_node_id += 1
        j_val = [index_y, index_y + 1]
        i_val = [(index_x + i, i / mp) for i in range(mp)]
        for k in k_val:
            for j in j_val:
                for i in i_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == index_y else bone_w
                    a3 = -bone_h if k == 1 else bone_h
                    bone_name = bone_names[3]
                    finger_part = bone_name + ' of ' + finger_name
                    annotation = [bone_name, finger_name, finger_part]
                    node = {
                        Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                        "Type": "internal",
                        "Annotation": annotation,
                    }
                    indices = [[i, j, k]]
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)
        index_x += mp
        # DP
        parent_node_id += 1
        j_val = [index_y, index_y + 1]
        i_val = [(index_x + i, (1 - d1_offset) * i) for i in range(dp)]
        for k in k_val:
            for j in j_val:
                for i in i_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == index_y else bone_w
                    a3 = -bone_h if k == 1 else bone_h
                    bone_name = bone_names[4]
                    finger_part = bone_name + ' of ' + finger_name
                    annotation = [bone_name, finger_name, finger_part]
                    node = {
                        Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                        "Type": "internal",
                        "Annotation": annotation,
                    }
                    indices = [[i, j, k]]
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)
        index_x += dp
        index_y += columns_per_finger
    # Thumb
    # Metacarpal
    mc = 1
    index_y += columns_per_finger
    index_x = c
    parent_node_id += 1
    j_val = [index_y, index_y + 1]
    i_val = [(index_x + i, i / mc) for i in range(mc)]
    finger_name = finger_names[-1]
    for k in k_val:
        for j in j_val:
            for i in i_val:
                a1 = i[1]
                i = i[0]
                a2 = bone_w if j == index_y + 1 else -bone_w
                a3 = -bone_h if k == 1 else bone_h
                node = {
                    Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                    "Type": "internal",
                    "Annotation": [finger_name],
                }
                indices = [[i, j, k]]
                if j == index_y:
                    indices.append([i + 3, index_y - 4, k])
                    if i == index_x:
                        indices.append([i + 1, index_y - 4, k])
                if i == index_x and j == index_y + 1:
                    indices.append([i, index_y - 4, k])
                virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)

    index_x += mc
    # Proximal phalanx
    parent_node_id += 1
    j_val = [index_y, index_y + 1]
    i_val = [(index_x + i, i / pp) for i in range(pp)]
    i_val = [(index_x, -d1_offset)]
    for k in k_val:
        for j in j_val:
            for i in i_val:
                a1 = i[1]
                i = i[0]
                a2 = -bone_w if j == index_y else bone_w
                a3 = -bone_h if k == 1 else bone_h
                bone_name = bone_names[2]
                finger_part = bone_name + ' of ' + finger_name
                annotation = [bone_name, finger_name, finger_part]
                node = {
                    Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                    "Type": "internal",
                    "Annotation": annotation,
                }
                indices = [[i, j, k]]
                if j == index_y:
                    indices.append([i + 3, index_y - 4, k])
                virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)
    index_x += pp
    # Distal phalanx
    parent_node_id += 1
    j_val = [index_y, index_y + 1]
    i_val = [(index_x + i, (1 - d1_offset) * i) for i in range(dp)]
    for k in k_val:
        for j in j_val:
            for i in i_val:
                a1 = i[1]
                i = i[0]
                a2 = -bone_w if j == index_y else bone_w
                a3 = -bone_h if k == 1 else bone_h
                bone_name = bone_names[4]
                finger_part = bone_name + ' of ' + finger_name
                annotation = [bone_name, finger_name, finger_part]
                node = {
                    Node.VALUE_LABEL_VALUE: [[parent_node_id, a0, a1, a2, a3]],
                    "Type": "internal",
                    "Annotation": annotation,
                }
                indices = [[i, j, k]]
                virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)

    return virtual_node_matrix, node_bag, node_identifier


def generate_external_node_matrix(number_elements_along,
                                  node_identifier, virtual_node_matrix: Virtual_node_matrix,
                                  node_bag: Node_bag, options):
    c, mc, pp, mp, dp = number_elements_along
    parent_node = 6
    x, d1, d2, d3 = node_bag.get_node_parameters(parent_node)
    finger_names = ["little finger", "ring finger", "middle finger", "index finger", "thumb"]
    bone_names = ['carpal', 'metacarpal', 'proximal phalanx',
                  'middle phalanx', 'distal phalanx']
    center = add(x, d2)
    a = 5.5
    b = 3.5
    major_axis = mult(d2, -a)
    minor_axis = mult(d3, b)
    ellipse = sampleEllipsePoints(center, major_axis,
                                  minor_axis, math.pi / 2, 2 * math.pi + math.pi / 2, 10)
    upper_ellipse_x = [ellipse[0][i] for i in [8, 9, 0, 1, 2]]
    upper_ellipse_d2 = [ellipse[1][i] for i in [8, 9, 0, 1, 2]]
    lower_ellipse_x = [ellipse[0][i] for i in [7, 6, 5, 4, 3]]
    lower_ellipse_d2 = [ellipse[1][i] for i in [7, 6, 5, 4, 3]]
    parent_node = 1
    node_id = 0
    index_y = 1
    jj = 0
    n = c
    d3 = None
    k_val = [1, 2]
    # Carpal
    for f in range(4):
        node_id += 1
        index_x = 0
        j_val = [index_y, index_y + 1] if f == 0 else [index_y + 1]
        i_val = [index_x + i for i in range(n)]
        for j in j_val:
            for k in k_val:
                ellipse_x = upper_ellipse_x if k == 2 else lower_ellipse_x
                ellipse_d2 = upper_ellipse_d2 if k == 2 else lower_ellipse_d2
                d1 = node_bag.get_node_parameters(parent_node)[1]
                d1 = mult(d1, 1 / n)
                a3 = -1 if k == 1 else 1
                a2 = -1 if j == 1 else 1
                for i in i_val:
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": [""],
                    }
                    indices = [[i, j, k + a3]]
                    if j == 1:
                        indices.append([i, j + a2, k])
                    elif j == 17:
                        indices.append([i, j + a2, k])
                        indices.append([i, j + 4, k + a3])
                    else:
                        indices.append([i, j + 4, k + a3])
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                    x = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node_bag.set_node_parameters(
                        [x, d1, d2, d3], node_identifier, internal_node=False
                    )
                    node_identifier += 1
                    x = add(x, d1)
                    ellipse_x[jj] = x
            jj += 1
        index_y += 5
        parent_node += 5
    # Metacarpal
    parent_node = 2
    jj = 0
    n = mc
    index_x += c
    index_y = 1
    for f in range(4):
        j_val = [index_y, index_y + 1] if f == 0 else [index_y + 1]
        i_val = [index_x + i for i in range(n)]
        for j in j_val:
            for k in k_val:
                ellipse_x = upper_ellipse_x if k == 2 else lower_ellipse_x
                ellipse_d2 = upper_ellipse_d2 if k == 2 else lower_ellipse_d2
                d1 = node_bag.get_node_parameters(parent_node)[1]
                d1 = mult(d1, 1.0 / n)
                a3 = -1 if k == 1 else 1
                a2 = -1 if j == 1 else 1
                for i in i_val:
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": [""],
                    }
                    indices = [[i, j, k + a3]]
                    if j == 1:
                        indices.append([i, j + a2, k])
                    elif j == 17:
                        if i > 1:
                            indices.append([i, j + a2, k])
                        indices.append([i, j, k + a3])
                    else:
                        indices.append([i, j, k + a3])
                        indices.append([i, j + 4, k + a3])
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                    x = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node_bag.set_node_parameters(
                        [x, d1, d2, d3], node_identifier, internal_node=False
                    )
                    node_identifier += 1
                    x = add(x, d1)
                    ellipse_x[jj] = x
            jj += 1
        index_y += 5
        parent_node += 5
    # PP
    parent_node = 3
    jj = 0
    n = pp
    index_x += mc
    index_y = 1
    for f in range(4):
        j_val = [index_y, index_y + 1] if f == 0 else [index_y + 1]
        i_val = [index_x + i for i in range(n)]
        for j in j_val:
            for k in k_val:
                ellipse_x = upper_ellipse_x if k == 2 else lower_ellipse_x
                ellipse_d2 = upper_ellipse_d2 if k == 2 else lower_ellipse_d2
                d1 = node_bag.get_node_parameters(parent_node)[1]
                d1 = mult(d1, 1.0 / n)
                a3 = -1 if k == 1 else 1
                a2 = -1 if j == 1 else 1
                finger_name = finger_names[f] if j == index_y else finger_names[f + 1]
                bone_name = bone_names[2]
                annotation = [bone_name, finger_name]
                for i in i_val:
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": annotation,
                    }
                    indices = [[i, j, k + a3],[i + 2, j, k + a3] ]
                    if j == 1:
                        indices.append([i, j + a2, k])
                        indices.append([i + 2, j + a2, k])
                    elif j == 17:
                        if i > 1:
                            indices.append([i, j + a2, k])
                            indices.append([i + 2, j + a2, k])
                    else:
                        indices.append([i, j + 4, k + a3])
                        indices.append([i + 2, j + 4, k + a3])
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                    x = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node_bag.set_node_parameters(
                        [x, d1, d2, d3], node_identifier, internal_node=False
                    )
                    node_identifier += 1
                    x = add(x, d1)
                    ellipse_x[jj] = x
            jj += 1
        index_y += 5
        # An update to the d2 of the parent node to make sure
        # the internal and external elements align
        x0, d1, d2, d3 = node_bag.get_node_parameters(parent_node)
        d2 = [0, abs(x[1] - x0[1]), 0]
        node_bag.set_node_parameters([x0, d1, d2, d3], parent_node, internal_node=True)
        parent_node += 5
    # Finger-palm connections
    j_val = [3, 8, 13]
    i += 2  # Last row of MC elements
    ii = 0
    for j in j_val:
        for k in k_val:
            a3 = -1 if k == 1 else 1
            finger_name = finger_names[ii]
            finger_part = bone_name + ' of ' + finger_name
            annotation = [bone_name, finger_name, finger_part]
            node_id = virtual_node_matrix.get_virtual_node([i, j - 1, k + a3])
            node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
            node = {
                Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
                Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, -0.5 * a3, 0]],
                Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
                "Type": "external",
                "Annotation": annotation
            }
            virtual_node_matrix.set_virtual_node_to_index(node, [i, j - 1, k + a3])
            node = {
                Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
                Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, -0.5 * a3, 0]],
                Node.VALUE_LABEL_D_DS2: [[node_id, 0, a3, 0, 0]],
                "Type": "external",
                "Annotation": annotation
            }
            virtual_node_matrix.set_virtual_node_to_index(node, [i, j, k])
            finger_name = finger_names[ii+1]
            finger_part = bone_name + ' of ' + finger_name
            annotation = [bone_name, finger_name, finger_part]
            node = {
                Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
                Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0.5 * a3, 0]],
                Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a3, 0, 0]],
                "Type": "external",
                "Annotation": annotation
            }
            virtual_node_matrix.set_virtual_node_to_index(node, [i, j + 2, k])

            node = {
                Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
                Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0.5 * a3, 0]],
                Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
                "Type": "external",
                "Annotation": annotation
            }
            virtual_node_matrix.set_virtual_node_to_index(node, [i, j + 3, k + a3])
        ii += 1
    # Fingers 5 to 2
    parent_node = 3
    index_y = 1
    n = pp
    jj = 0
    a = 2
    b = 2.75
    for f in range(4):
        # MP
        parent_node += 1
        index_x = c + mc + pp + 2
        n = mp
        jj = 0
        j_val = [index_y, index_y + 1]
        i_val = [index_x + i for i in range(n)]
        # Estimate ellipse
        center, d1, d2, d3 = node_bag.get_node_parameters(parent_node)
        major_axis = mult(d2, -a)
        minor_axis = mult(d3, b)
        ellipse_x = []
        ellipse_d2 = []
        for ii in range(4):
            ellipse = sampleEllipsePoints(
                center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4,
                (1 + 2 * ii) * math.pi / 4, 1
            )
            ellipse_x.append(ellipse[0][0])
            ellipse_d2.append(ellipse[1][0])
        ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
        ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
        jj = 0
        for j in j_val:
            for k in k_val:
                d1 = node_bag.get_node_parameters(parent_node)[1]
                d1 = mult(d1, 1 / n)
                a2 = -1 if j == index_y else 1
                a3 = -1 if k == 1 else 1
                finger_name = finger_names[f]
                for i in i_val:
                    x = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    bone_name = bone_names[3]
                    finger_part = bone_name + ' of ' + finger_name
                    annotation = [bone_name, finger_name, finger_part]
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": annotation,
                    }
                    indices = [[i, j + a2, k], [i, j, k + a3]]
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                    node_bag.set_node_parameters(
                        [x, d1, d2, d3], node_identifier, internal_node=False
                    )
                    node_identifier += 1
                    x = add(x, d1)
                    ellipse_x[jj] = x
                jj += 1
        # DP
        parent_node += 1
        index_x = c + mc + pp + mp + 2
        n = dp
        jj = 0
        j_val = [index_y, index_y + 1]
        i_val = [index_x + i for i in range(n)]
        # Estimate ellipse
        center, d1, d2, d3 = node_bag.get_node_parameters(parent_node)
        major_axis = mult(d2, -a)
        minor_axis = mult(d3, b)
        ellipse_x = []
        ellipse_d2 = []
        for ii in range(4):
            ellipse = sampleEllipsePoints(
                center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4,
                (1 + 2 * ii) * math.pi / 4, 1
            )
            ellipse_x.append(ellipse[0][0])
            ellipse_d2.append(ellipse[1][0])
        ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
        ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
        jj = 0
        for j in j_val:
            for k in k_val:
                _, d1, d2, d3 = node_bag.get_node_parameters(parent_node)
                d1 = mult(d1, 1 / (n - 1))
                a1 = 0.5
                a2 = -1 if j == index_y else 1
                a3 = -1 if k == 1 else 1
                finger_name = finger_names[f]
                for i in i_val:
                    x = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    bone_name = bone_names[4]
                    finger_part = bone_name + ' of ' + finger_name
                    annotation = [bone_name, finger_name, finger_part]
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                        "Type": "external",
                        "Annotation": annotation,
                    }
                    indices = [[i, j + a2, k], [i, j, k + a3]]
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                    if i == i_val[-1]:  # Special element to cap the fingers
                        c1 = a1 * a2 * a3
                        c2 = a1 * a2
                        c3 = a1 * a3
                        node = {
                            Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                            Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                            Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, -c1, 1, 0]],
                            "Type": "external",
                            "Annotation": [finger_name],
                        }
                        virtual_node_matrix.set_virtual_node_to_index(node, [i, j, k + a3])
                        node = {
                            Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                            Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                            Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, c1, 1, 0]],
                            "Type": "external",
                            "Annotation": [finger_name],
                        }
                        virtual_node_matrix.set_virtual_node_to_index(node, [i, j + a2, k])
                        node = {
                            Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                            Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, -c2, a3, 0]],
                            Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, c3, a2, 0]],
                            "Type": "external",
                            "Annotation": [finger_name],
                        }
                        virtual_node_matrix.set_virtual_node_to_index(node, [i + 1, j, k])
                    node_bag.set_node_parameters(
                        [x, d1, d2, d3], node_identifier, internal_node=False
                    )
                    node_identifier += 1
                    x = add(x, d1)
                    ellipse_x[jj] = x
                jj += 1
        index_y += 5
        parent_node += 3
    # Thumb MC
    mc = 1
    parent_node = 21
    jj = 0
    n = mc
    index_x = c
    index_y += 5
    j_val = [index_y, index_y + 1]
    i_val = [index_x + i for i in range(n)]
    # Estimate ellipse
    center, d1, d2, d3 = node_bag.get_node_parameters(parent_node)
    major_axis = mult(d2, -a)
    minor_axis = mult(d3, b)
    ellipse_x = []
    ellipse_d2 = []
    for ii in range(4):
        ellipse = sampleEllipsePoints(
            center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4,
            (1 + 2 * ii) * math.pi / 4, 1
        )
        ellipse_x.append(ellipse[0][0])
        ellipse_d2.append(ellipse[1][0])
    ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
    ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
    finger_name = finger_names[-1]
    jj = 0
    for j in j_val:
        for k in k_val:
            d1 = node_bag.get_node_parameters(parent_node)[1]
            d1 = mult(d1, 1 / (n))
            a2 = -1 if j == index_y else 1
            a3 = -1 if k == 1 else 1
            for i in i_val:
                x = ellipse_x[jj]
                d2 = ellipse_d2[jj]
                node = {
                    Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                    Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                    Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                    "Type": "external",
                    "Annotation": [finger_name],
                }
                indices = [[i, j, k + a3]]
                if j == index_y + 1:
                    indices.append([i, j + a2, k])
                virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                node_bag.set_node_parameters(
                    [x, d1, d2, d3], node_identifier, internal_node=False
                )
                node_identifier += 1
                x = add(x, d1)
                ellipse_x[jj] = x
            jj += 1

    # Thumb PP
    parent_node += 1
    index_x = c + mc
    n = pp
    j_val = [index_y, index_y + 1]
    i_val = [index_x + i for i in range(n)]
    # Estimate ellipse
    center, d1, d2, d3 = node_bag.get_node_parameters(parent_node)
    major_axis = mult(d2, -a)
    minor_axis = mult(d3, b)
    ellipse_x = []
    ellipse_d2 = []
    for ii in range(4):
        ellipse = sampleEllipsePoints(
            center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4,
            (1 + 2 * ii) * math.pi / 4, 1
        )
        ellipse_x.append(ellipse[0][0])
        ellipse_d2.append(ellipse[1][0])
    ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
    ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
    jj = 0
    for j in j_val:
        for k in k_val:
            d1 = node_bag.get_node_parameters(parent_node)[1]
            d1 = mult(d1, 1 / n)
            a2 = -1 if j == index_y else 1
            a3 = -1 if k == 1 else 1
            for i in i_val:
                x = ellipse_x[jj]
                d2 = ellipse_d2[jj]
                bone_name = bone_names[2]
                finger_part = bone_name + ' of ' + finger_name
                annotation = [bone_name, finger_name, finger_part]
                node = {
                    Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                    Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                    Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                    "Type": "external",
                    "Annotation": annotation,
                }
                indices = [[i, j + a2, k], [i, j, k + a3]]
                virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                node_bag.set_node_parameters(
                    [x, d1, d2, d3], node_identifier, internal_node=False
                    )
                node_identifier += 1
                x = add(x, d1)
                ellipse_x[jj] = x
            jj += 1
    # DP
    parent_node += 1
    index_x = c + mc + pp
    n = dp
    j_val = [index_y, index_y + 1]
    i_val = [index_x + i for i in range(n)]
    # Estimate ellipse
    center, d1, d2, d3 = node_bag.get_node_parameters(parent_node)
    major_axis = mult(d2, -a)
    minor_axis = mult(d3, b)
    ellipse_x = []
    ellipse_d2 = []
    for ii in range(4):
        ellipse = sampleEllipsePoints(
            center, major_axis, minor_axis, (-1 + 2 * ii) * math.pi / 4,
            (1 + 2 * ii) * math.pi / 4, 1
        )
        ellipse_x.append(ellipse[0][0])
        ellipse_d2.append(ellipse[1][0])
    ellipse_x = [ellipse_x[i] for i in [0, 1, 3, 2]]
    ellipse_d2 = [ellipse_d2[i] for i in [0, 1, 3, 2]]
    jj = 0
    for j in j_val:
        for k in k_val:
            d1 = node_bag.get_node_parameters(parent_node)[1]
            d1 = mult(d1, 1 / (n - 1))
            a1 = 0.5
            a2 = -1 if j == index_y else 1
            a3 = -1 if k == 1 else 1
            for i in i_val:
                x = ellipse_x[jj]
                d2 = ellipse_d2[jj]
                bone_name = bone_names[4]
                finger_part = bone_name + ' of ' + finger_name
                annotation = [bone_name, finger_name, finger_part]
                node = {
                    Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                    Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                    Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, 0, 1, 0]],
                    "Type": "external",
                    "Annotation": annotation,
                }
                indices = [[i, j + a2, k], [i, j, k + a3]]
                virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                if i == i_val[-1]:
                    c1 = a1 * a2 * a3
                    c2 = a1 * -a2
                    c3 = a1 * a3
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, -c1, 1, 0]],
                        "Type": "external",
                        "Annotation": annotation,
                    }
                    virtual_node_matrix.set_virtual_node_to_index(node, [i, j, k + a3])
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, 1, 0, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, c1, 1, 0]],
                        "Type": "external",
                        "Annotation": annotation,
                    }
                    virtual_node_matrix.set_virtual_node_to_index(node, [i, j + a2, k])
                    node = {
                        Node.VALUE_LABEL_VALUE: [[node_identifier, 1, 0, 0, 0]],
                        Node.VALUE_LABEL_D_DS1: [[node_identifier, 0, c2, a3, 0]],
                        Node.VALUE_LABEL_D_DS2: [[node_identifier, 0, c3, a2, 0]],
                        "Type": "external",
                        "Annotation": annotation,
                    }
                    virtual_node_matrix.set_virtual_node_to_index(node, [i + 1, j, k])
                node_bag.set_node_parameters(
                    [x, d1, d2, d3], node_identifier, internal_node=False
                    )
                node_identifier += 1
                x = add(x, d1)
                ellipse_x[jj] = x
            jj += 1
    # The Thumb-palm connection elements
    # It is against every fiber of my being that I am forced to set these manually
    thumb_angle_degrees = options["Thumb angle"]
    thumb_c = thumb_angle_degrees / 90
    # Back of the thumb
    k_val = [0, 3]
    for k in k_val:
        a1 = -1 if k == 3 else 1
        a3 = -1 if k == 3 else 1
        node_id = virtual_node_matrix.get_virtual_node([0, index_y - 9, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 0.5, -a1 * 0.5, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [0, index_y - 5, k + a3])
        node_id = virtual_node_matrix.get_virtual_node([1, index_y + 1, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, a1 * thumb_c, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [0, index_y - 4, k + a3])

    # Bottom carpal-metacarpal connection
    for k in k_val:
        a1 = -1 if k == 3 else 1
        a2
        node_id = virtual_node_matrix.get_virtual_node([1, index_y + 1, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 0, a1, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a1, -thumb_c, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [1, index_y - 4, k])

        node_id = virtual_node_matrix.get_virtual_node([1, index_y, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 0, a1, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a1 * thumb_c, 0, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [2, index_y - 4, k])

        node_id = virtual_node_matrix.get_virtual_node([0, index_y - 9, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a1 * 0.5, 0.5, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [1, index_y - 5, k])

        node_id = virtual_node_matrix.get_virtual_node([1, index_y - 9, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 0.25, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [2, index_y - 5, k])
    # Intermediate metacarpal connections
    for k in k_val:
        a1 = 2 * thumb_c
        a3 = -1 if k == 3 else 1

        node_id = virtual_node_matrix.get_virtual_node([1, index_y, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a3 * thumb_c, 0, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [4, index_y - 4, k])

        node_id = virtual_node_matrix.get_virtual_node([1, index_y - 9, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 0.25, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [4, index_y - 5, k])

        node_id = virtual_node_matrix.get_virtual_node([2, index_y, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, -a1 * a3, 0.5, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [5, index_y - 4, k])

        node_id = virtual_node_matrix.get_virtual_node([2, index_y - 9, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, 1, 0, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, a1 * a3, 0.5, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [5, index_y - 5, k])

    # Front of the webbing
    for k in k_val:
        a1 = 2 * thumb_c
        # a1 = -1 if k == 3 else 1
        a3 = -1 if k == 3 else 1
        node_id = virtual_node_matrix.get_virtual_node([2, index_y - 9, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, -a1, -a3 * 0.5, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, 1, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [6, index_y - 5, k + a3])

        node_id = virtual_node_matrix.get_virtual_node([2, index_y, k])
        node_id = node_id[Node.VALUE_LABEL_VALUE][0][0]
        node = {
            Node.VALUE_LABEL_VALUE: [[node_id, 1, 0, 0, 0]],
            Node.VALUE_LABEL_D_DS1: [[node_id, 0, a1, -a3 * 0.5, 0]],
            Node.VALUE_LABEL_D_DS2: [[node_id, 0, 0, -1, 0]],
            "Type": "external",
        }
        virtual_node_matrix.set_virtual_node_to_index(node, [6, index_y - 4, k + a3])

    return virtual_node_matrix, node_bag, node_identifier


def create_cube_element(fieldmodule: Fieldmodule, element_identifier: int,
                        node_matrix: list, node_indices: list,) -> list:
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
    A virtual node always has the structure
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
    node_1 = node_matrix[ix][iy][iz]
    node_2 = node_matrix[ix + 1][iy][iz]
    node_3 = node_matrix[ix][iy + 1][iz]
    if node_1 is None or node_2 is None or node_3 is None:
        return -2, None
    n1_ext = True if node_1["Type"] == "external" else False
    n2_ext = True if node_2["Type"] == "external" else False
    n3_ext = True if node_3["Type"] == "external" else False
    # 6 possible cases, one for each face of a cube
    if not n1_ext and not n2_ext and not n3_ext:
        ranges = [[0, 1], [0, 1], [0, 1]] # 1
        order = [2, 1, 0]
    elif not n1_ext and not n2_ext and n3_ext:
        ranges = [[0, 1], [0, 1], [1, 0]] # 2
        order = [1, 2, 0]
    elif n1_ext and n2_ext and n3_ext:
        ranges = [[0, 1], [1, 0], [1, 0]] # 3
        order = [2, 1, 0]
    elif n1_ext and n2_ext and not n3_ext:
        ranges = [[0, 1], [1, 0], [0, 1]] # 4
        order = [1, 2, 0]
    elif n1_ext and not n2_ext and n3_ext:
        ranges = [[1, 0], [0, 1], [1, 0]] # 5
        order = [0, 2, 1]
    else:
        ranges = [[0, 1], [0, 1], [1, 0]] # 6
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
        virtual_node = node_matrix[ix + i][iy + j][iz + k]
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
                    local_scale_factor_id = [
                        global_to_local_scale_factor_ids[global_scale_factor]
                        ]
                    et.append((
                        local_node_ids[global_node_id],
                        value_labels[factor],
                        local_scale_factor_id
                        ))
            value_expression_terms[local_node] = et
    local_to_global_scale_factor_ids = {
        value: key for key, value in global_to_local_scale_factor_ids.items()}
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
                neg_et.append((
                    l_node_id,
                    label,
                    global_to_local_scale_factor_ids[global_scale_factor]
                    ))
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
            virtual_node = node_matrix[ix + i][iy + j][iz + k]
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
                        ret.append((
                            str(global_node_id).zfill(2),
                            value_label_names[factor],
                            str(scale_factor).zfill(4)
                            ))
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
            virtual_node = node_matrix[ix + i][iy + j][iz + k]
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
                        ret.append((
                            str(global_node_id).zfill(2),
                            value_label_names[factor],
                            str(scale_factor).zfill(4)
                        ))
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
        for local_node in value_expression_terms:
            remapEftNodeValueLabelWithNodes(
                eft,
                local_node,
                Node.VALUE_LABEL_VALUE,
                value_expression_terms[local_node]
            )
            remapEftNodeValueLabelWithNodes(
                eft,
                local_node,
                Node.VALUE_LABEL_D_DS1,
                d1_expression_terms[local_node]
            )
            remapEftNodeValueLabelWithNodes(
                eft,
                local_node,
                Node.VALUE_LABEL_D_DS2,
                d2_expression_terms[local_node]
            )
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
                eft,
                local_node,
                Node.VALUE_LABEL_VALUE,
                value_expression_terms[local_node]
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

def setNodeFieldParameters(field, fieldcache, x,
                           d1=None, d2=None, d3=None, d12=None, d13=None):
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
