import math
from itertools import product

from cmlibs.maths.vectorops import (
    add, sub,
    magnitude,
    mult,
    rotate_vector_around_vector,
    set_magnitude,
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
            'Create internal elements': True,
            'Create external elements': False,
            'Internal elements across hand': 1,
            'Elements along carpal': 1,
            'Elements along metacarpal': 2,
            'Elements along proximal phalanx': 2,
            'Elements along middle phalanx': 1,
            'Elements along distal phalanx': 2,
            'finger names': [
                "little finger",
                "ring finger",
                "middle finger",
                "index finger",
                "thumb"],
            'bone names': [
                'carpal',
                'metacarpal',
                'proximal phalanx',
                'middle phalanx',
                'distal phalanx'
                ],
        }
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Thumb angle",
            "Splaying angle",
            'Create internal elements',
            'Create external elements',
            'Internal elements across hand',
            'Elements along carpal',
            'Elements along metacarpal',
            'Elements along proximal phalanx',
            'Elements along middle phalanx',
            'Elements along distal phalanx',
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
            'Elements along carpal',
            'Elements along metacarpal',
            'Elements along proximal phalanx',
            'Elements along middle phalanx',
            'Elements along distal phalanx',
        ]:
            if options[key] < 1:
                options[key] = 1
            if options[key] > 3:
                options[key] = 3
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
        create_internal = options.get('Create internal elements')
        create_external = options.get('Create external elements')
        node_identifier = 1
        hand_elements_along = [
            options['Elements along carpal'],
            options['Elements along metacarpal'],
            options['Elements along proximal phalanx'],
            options['Elements along middle phalanx'],
            options['Elements along distal phalanx']
        ]
        node_network = Node_network(region, fieldcache)
        ix = 2 * sum(hand_elements_along)
        iy = 6 * 5 # 5 columns per finger, + 1 column for palm-thumb connection
        internal_rows = options.get('Internal elements across hand') + 1
        external_rows = 2
        iz = internal_rows + external_rows
        virtual_node_matrix = Virtual_node_matrix([ix, iy, iz])

        node_network, node_identifier = generate_internal_nodes(
            node_network, hand_elements_along, node_identifier, options = options
        )
        if create_internal:
            virtual_node_matrix, node_network, node_identifier = generate_internal_node_matrix(
                hand_elements_along, node_identifier, virtual_node_matrix,
                node_network=node_network, options=options
            )
            if create_external:
                virtual_node_matrix, node_network, node_identifier = \
                generate_external_node_matrix(
                    hand_elements_along, node_identifier, virtual_node_matrix,
                    node_network=node_network, options=options
                )

        finger_names = options['finger names']
        finger_bone_names = options['bone names'][2:]
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
        # elements_along_per_finger = sum(hand_elements_along)
        # for i in range(4):
        #     node_network.remove_node(1 + i * elements_along_per_finger)
        node_network.set_zinc_nodes()
        matrix = virtual_node_matrix.get_matrix()
        for k in range(iz - 1):
            for j in range(iy - 1):
                for i in range(ix - 1):
                    result, element = create_cube_element(
                        fieldmodule, element_identifier, matrix, [i, j, k],
                    )
                    if result == RESULT_OK:
                        # print('element: ' + str(element_identifier))
                        # virtual_node_matrix.print_cube_from_index([i, j, k])
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

        finger_names = options['finger names']
        finger_bone_names = options['bone names'][2:]
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

class Bone():
    def __init__(self):
        pass
    CARPAL = 0
    METACARPAL = 1
    PROX_PHALANX = 2
    MID_PHALANX = 3
    DIST_PHALANX = 4

class Node_network():

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
            # Node.VALUE_LABEL_D_DS1,
            Node.VALUE_LABEL_D_DS2,
            Node.VALUE_LABEL_D_DS3,
        ]
        internal_nodetemplate = nodes.createNodetemplate()
        internal_nodetemplate.defineField(coordinates)
        for value_label in value_labels[1:]:
            internal_nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)
        self.internal_nodetemplate = internal_nodetemplate

        value_labels = [
            Node.VALUE_LABEL_VALUE,
            # Node.VALUE_LABEL_D_DS1,
            # Node.VALUE_LABEL_D_DS2,
            Node.VALUE_LABEL_D_DS3,
        ]
        internal_palm_nodetemplate = nodes.createNodetemplate()
        internal_palm_nodetemplate.defineField(coordinates)
        for value_label in value_labels[1:]:
            internal_palm_nodetemplate.setValueNumberOfVersions(
                coordinates, -1, value_label, 1)
        self.internal_palm_nodetemplate = internal_palm_nodetemplate

    def set_node_parameters(self, node_coordinates: list, node_identifier:int = 0,
                 internal_node = True, palm_node = False) -> None :
        nodes_dict = self.nodes_dict
        assert len(node_coordinates) == 4
        x, d1, d2, d3 = node_coordinates
        nodes_dict[node_identifier] = {
            'parameters': [x, d1, d2, d3],
            'internal': internal_node,
            'palm': palm_node
        }

    def set_zinc_nodes(self):
        node_dict = self.nodes_dict
        coordinates = self.coordinates
        fieldcache = self.fieldcache
        for node_identifier, node_params in node_dict.items():
            is_internal = node_params.get('internal')
            is_palm = node_params.get('palm')
            node_coordinates = node_params.get('parameters')
            if is_internal:
                x, d1, d2, d3 = node_coordinates
                d1 = None
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
            setNodeFieldParameters(coordinates, fieldcache, x, d1 = d1, d2 = d2, d3 = d3)

    def get_node_parameters(self, node_identifier):
        nodes_dict = self.nodes_dict
        node_dict = nodes_dict[node_identifier].get('parameters')
        return node_dict

    def get_node_identifier(self)-> int:
        return self.node_identifier

    def remove_node(self, node_identifier):
        nodes_dict = self.nodes_dict
        nodes_dict.pop(node_identifier)

class Virtual_node_matrix():

    def __init__(self, matrix_dimensions: list):
        ix, iy, iz = matrix_dimensions
        self.node_matrix = [[[None for k in range(iz)] for j in range(iy)] for i in range(ix)]

    def create_internal_vnode (self, node_identifier: int, x: list, annotation: dict):
        virtual_node = {
            Node.VALUE_LABEL_VALUE: [[node_identifier, *x]],
            'node_identifier': node_identifier,
            "Type": 'internal',
            'Annotation': annotation
        }
        return virtual_node

    def create_external_vnode (self, node_identifier: int, x: list, d1: list, d2: list,
                               annotation: dict):
        assert len(x) == len(d1) == len(d2) == 4,  "Scale factor vector must be length 4"
        virtual_node = {
            Node.VALUE_LABEL_VALUE: [[node_identifier, *x]],
            Node.VALUE_LABEL_D_DS1: [[node_identifier, *d1]],
            Node.VALUE_LABEL_D_DS2: [[node_identifier, *d2]],
            'node_identifier': node_identifier,
            "Type": 'external',
            'Annotation': annotation
        }
        return virtual_node

    def set_virtual_node_to_index(self, node: dict, index: list, replace=True):
        i, j, k = index
        node_matrix = self.node_matrix
        if replace:
            node_matrix[i][j][k] = node.copy()
        else:
            if node_matrix[i][j][k] is None:
                node_matrix[i][j][k] = node.copy()
            else:
                if node.get('Type') == 'internal':
                    new_node = node.get(Node.VALUE_LABEL_VALUE)
                    new_node = new_node[0]
                    self.blend_virtual_node(index, new_node)

    def set_virtual_node_to_indices(self, node:dict, indices: list, replace=True):
        for index in indices:
            self.set_virtual_node_to_index(node, index, replace)

    def get_matrix(self) -> list:
        return self.node_matrix

    def blend_virtual_node(self, existing_node_index: list, new_node):
        """
        Only works for blending internal linear nodes for now

        :param self: Description
        :param existing_node_index: Description
        :type existing_node_index: list
        :param new_node: Description
        """
        vertex = self.get_virtual_node(existing_node_index)
        original_vertex = vertex[Node.VALUE_LABEL_VALUE]
        N = len(original_vertex)
        updated_vertex = [*original_vertex]
        updated_vertex.append([*new_node])
        for i in [1, 4]:
            coeff_sum = 0
            for j in range(N+1):
                coeff_sum += updated_vertex[j][i]
            # TODO make a desicion on whether to keep this check or no.
            # assert abs(coeff_sum) == 1, 'Not affine combination'
        vertex[Node.VALUE_LABEL_VALUE] = updated_vertex
        self.set_virtual_node_to_index(vertex, existing_node_index)

    def get_virtual_node(self, index: list) -> dict:
        i, j, k = index
        node_matrix = self.node_matrix
        virtual_node = node_matrix[i][j][k]
        return virtual_node

    def get_node_identifier_at_index(self, index: list) -> int:
        node = self.get_virtual_node(index)
        node_id = node.get('node_identifier')
        return node_id

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
        print ("------------------------------")
        for index in indices:
            i, j, k = index
            print([start_i + i, start_j + j, start_k + k],
                  node_matrix[start_i + i][start_j + j][start_k + k])
        print ("------------------------------")

def generate_internal_nodes(node_network: Node_network, hand_elements_along: list,
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

    hand_dimensions_by_finger = [ # d1, d2, d3
        [ # Finger 1 (little finger)
            [1.0, 0.4, 0.2],  # carpal
            [2.0, 0.4, 0.2],  # metacarpal
            [1.1, 0.4, 0.2],  # proximal phalanx
            [0.7, 0.2, 0.2],  # middle phalanx
            [0.6, 0.2, 0.2],  # distal phalanx
        ],
        [ # Finger 2 (ring finger)
            [1.0, 0.4, 0.2],  # carpal
            [2.0, 0.4, 0.2],  # metacarpal
            [1.5, 0.4, 0.2],  # proximal phalanx
            [1.0, 0.2, 0.2],  # middle phalanx
            [0.7, 0.2, 0.2],  # distal phalanx
        ],
        [ # Finger 3 (middle finger)
            [1.0, 0.4, 0.2],  # carpal
            [2.0, 0.4, 0.2],  # metacarpal
            [1.9, 0.4, 0.2],  # proximal phalanx
            [1.0, 0.2, 0.2],  # middle phalanx
            [0.6, 0.2, 0.2],  # distal phalanx
        ],
        [ # Finger 4 (index finger)
            [1.0, 0.4, 0.2],  # carpal
            [2.0, 0.4, 0.2],  # metacarpal
            [1.9, 0.4, 0.2],  # proximal phalanx
            [0.8, 0.2, 0.2],  # middle phalanx
            [0.6, 0.2, 0.2],  # distal phalanx
        ],
        [ # Finger 5 (thumb finger)
            [0.0, 0.0, 0.0],  # carpal
            [1.0, 0.25, 0.2, 0.2],  # metacarpal
            [1.1, 0.25, 0.2, 0.2],  # proximal phalanx
            [0.0, 0.0, 0.0],  # middle phalanx
            [0.8, 0.25, 0.2, 0.2],  # distal phalanx
        ],
    ]
    carpal_spacing = hand_dimensions_by_finger[0][0][1] * 2.0
    index_carpal_node_id = None
    # create nodes up each finger fastest
    x1 = [1.0, 0.0, 0.0]
    x2 = [0.0, 1.0, 0.0]
    x3 = [0.0, 0.0, 1.0]
    splaying_angle_radians = math.radians(options['Splaying angle'])
    for j in range(4):
        finger_dimensions = hand_dimensions_by_finger[j]
        x = [0.0, j * carpal_spacing, 0.0]
        x1 = [1.0, 0.0, 0.0]
        x2 = [0.0, 1.0, 0.0]
        for bone, bone_dimensions in enumerate(finger_dimensions):
            n_elements_along = hand_elements_along[bone]
            bone_dimensions = finger_dimensions[bone]
            if bone == Bone.METACARPAL:
                x1 = rotate_vector_around_vector(x1, x3, (j - 2) * splaying_angle_radians)
                x2 = rotate_vector_around_vector(x2, x3, (j - 2) * splaying_angle_radians)
            d1 = mult(x1, bone_dimensions[0])
            if bone == Bone.DIST_PHALANX:
                d1 = mult(d1, 1 / (n_elements_along - 1))
            else:
                d1 = mult(d1, 1 / n_elements_along)
            d2 = mult(x2, bone_dimensions[1])
            d3 = mult(x3, bone_dimensions[2])
            if (j == 3) and (bone == Bone.CARPAL):
                index_carpal_node_id = node_identifier
            is_palm = True if bone < 3 else False
            for i_along in range(n_elements_along):
                if bone == Bone.PROX_PHALANX and i_along == 1:
                    is_palm = False
                    d2 = mult(d2, 0.5)
                    # x = add(x, mult(d2, -1.0 + (2.0 / 3.0) * j))
                node_network.set_node_parameters([x, d1, d2, d3], node_identifier,
                                             internal_node = True, palm_node = is_palm)
                x = add(x, d1)
                node_identifier += 1
    thumb_dimensions = hand_dimensions_by_finger[-1]
    thumb_indices = [Bone.METACARPAL, Bone.PROX_PHALANX, Bone.DIST_PHALANX]
    thumb_angle_degrees = options["Thumb angle"]
    # Thumb flexion angle
    angle_radians = math.radians(thumb_angle_degrees)
    x1 = rotate_vector_around_vector(x1, x3, angle_radians)
    x2 = rotate_vector_around_vector(x2, x3, angle_radians)
    # Obtain the starting node position from the metacarpal node
    x, d1, d2, d3 = node_network.get_node_parameters(index_carpal_node_id)
    x = add(x, mult(d2, 1.0))
    x = add(x, mult(x1, 1.5*magnitude(d1)))
    # x = add(x, mult(d1, 0.15))
    for bone in thumb_indices:
        bone_dimensions = thumb_dimensions[bone]
        n_elements_along = hand_elements_along[bone]
        if bone in [Bone.METACARPAL, Bone.PROX_PHALANX]:
            n_elements_along = max(1, n_elements_along - 1)
        d1 = mult(x1, bone_dimensions[0])
        if bone == Bone.DIST_PHALANX:
            d1 = mult(d1, 1 / (n_elements_along - 1))
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

def generate_internal_node_matrix(hand_elements_along, node_identifier,
                                  virtual_node_matrix: Virtual_node_matrix,
                                  node_network: Node_network, options: dict =None):
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
    # TODO calculate these constants as functions of the node parameters
    bone_c_int = 0.5
    bone_c_ext = 0.75
    n_rows_per_node = options.get('Internal elements across hand')
    n_cols_per_node = 1
    def sampling_coefficient(k, n):
        """
         Samples ihe [-1, 1] interval in n segments.

        :param k: Integer from 1 to n + 1
        :param n: Total number of segments
        :return: Real number between [-1, 1]
        """
        return ((2*k - n - 2)/n)
    a0 = 1.0
    parent_node_id = 1
    index_y = 1
    a1 = 0.0
    k_val = [i + 1 for i in range(n_rows_per_node + 1)]
    columns_per_finger = 5
    finger_names = options.get('finger names')
    bone_names = options.get('bone names')
    # Finger
    # Palm and first four fingers
    for finger_index in range(4):
        index_x = 0
        # Carpal, metacarpal and first row of proximal phalanx
        for bone in range(3):
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
                        a2 = 0.0
                        a3 = a0*sampling_coefficient(k, n_rows_per_node)
                        bone_name = bone_names[bone]
                        indices = [[i, j, k]]
                        if bone == Bone.CARPAL:
                            annotation = [bone_name] if k == 1 else ""
                            if (j == index_y + 1 and finger_index == 3) \
                                or (j == index_y + 6 and finger_index == 2):
                                indices.append([i + 1, j + 4, k])
                            node = virtual_node_matrix.create_internal_vnode(
                                parent_node_id, [a0, a1, a2, a3], annotation)
                            virtual_node_matrix.set_virtual_node_to_indices(
                                node, indices, False
                            )
                        elif bone == Bone.METACARPAL:
                            annotation = [bone_name] if k == 1 else ""
                            if (j == index_y + 1 and finger_index == 3) \
                                or (j == index_y + 6 and finger_index == 2): # Webbing
                                if i == index_x:
                                    indices.append([i + 1, j + 4, k])
                                    indices.append([i + 3, j + 4, k])
                                # else:
                                #     indices.append([i + 3, j + 4, k])
                            node = virtual_node_matrix.create_internal_vnode(
                                parent_node_id, [a0, a1, a2, a3], annotation)
                            virtual_node_matrix.set_virtual_node_to_indices(
                                node, indices, False
                            )
                        elif bone == Bone.PROX_PHALANX:
                            n_elements_along = 1
                            finger_name = finger_names[finger_index]
                            if j > index_y + 1 and finger_index < 3:
                                finger_name = finger_names[finger_index + 1]
                            finger_part = bone_name + ' of ' + finger_name
                            annotation = [bone_name, finger_name, finger_part]
                            node = virtual_node_matrix.create_internal_vnode(
                                parent_node_id, [a0, a1, a2, a3], annotation)
                            indices.append([i + 2, j, k])
                            virtual_node_matrix.set_virtual_node_to_indices(
                                node, indices, False
                            )
                            indices
                parent_node_id += 1
            index_x += n_elements_along
        index_x += 2
        a0 = 1
        if hand_elements_along[2] > 1:
            phalanx_bones = [Bone.PROX_PHALANX, Bone.MID_PHALANX, Bone.DIST_PHALANX]
        else:
            phalanx_bones = [Bone.MID_PHALANX, Bone.DIST_PHALANX]
        # Proximal, middle and distal phalanx
        for bone in phalanx_bones:
            n_elements_along = hand_elements_along[bone]
            n_elements_along = n_elements_along - 1 if bone == Bone.PROX_PHALANX \
                  else n_elements_along
            j_val = [index_y, index_y + 1]
            i_val = [index_x + i for i in range(n_elements_along)]
            for i in i_val:
                for j in j_val:
                    for k in k_val:
                        a2 = sampling_coefficient(j - (index_y - 1), n_cols_per_node)
                        a3 = sampling_coefficient(k, n_rows_per_node)
                        finger_name = finger_names[finger_index]
                        bone_name = bone_names[bone]
                        finger_part = bone_name + ' of ' + finger_name
                        annotation = [bone_name, finger_name, finger_part]
                        indices = [[i, j, k]]
                        node = virtual_node_matrix.create_internal_vnode(
                                parent_node_id, [a0, a1, a2, a3], annotation)
                        virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)
                parent_node_id += 1
            index_x += n_elements_along
        index_y += columns_per_finger
    # Thumb
    index_y += columns_per_finger
    index_x = hand_elements_along[0]
    finger_index = 4
    finger_name = finger_names[-1]
    thumb_bones = [Bone.METACARPAL, Bone.PROX_PHALANX, Bone.DIST_PHALANX]
    for bone in thumb_bones:
        n_elements_along = hand_elements_along[bone]
        if bone in [Bone.METACARPAL, Bone.PROX_PHALANX]:
            n_elements_along = max(1, n_elements_along - 1)
        j_val = [index_y, index_y + 1]
        i_val = [index_x + i for i in range(n_elements_along)]
        for i in i_val:
            for j in j_val:
                for k in k_val:
                    a2 = sampling_coefficient(j - (index_y - 1), n_cols_per_node)
                    a3 = sampling_coefficient(k, n_rows_per_node)
                    indices = [[i, j, k]]
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
                    elif bone == Bone.PROX_PHALANX:
                        bone_name = bone_names[bone]
                        finger_part = bone_name + ' of ' + finger_name
                        annotation = [bone_name, finger_name, finger_part]
                    elif bone == Bone.DIST_PHALANX:
                        bone_name = bone_names[bone]
                        finger_part = bone_name + ' of ' + finger_name
                        annotation = [bone_name, finger_name, finger_part]
                    node = virtual_node_matrix.create_internal_vnode(
                                parent_node_id, [a0, a1, a2, a3], annotation)
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices, False)
            parent_node_id += 1
        index_x += n_elements_along

    return virtual_node_matrix, node_network, node_identifier

def generate_external_node_matrix(hand_elements_along,
                                  node_identifier, virtual_node_matrix: Virtual_node_matrix,
                                  node_network: Node_network, options):
    c, mc, pp, mp, dp = hand_elements_along
    palm_elements_along = c + mc + 1
    parent_node_id = sum(hand_elements_along) + 1
    finger_names = options['finger names']
    bone_names = options['bone names']
    # TODO calculate formulas for these numbers, based on the width of the internal boxes.
    a_palm = [4.65 + i * (4.65 - 4.65) / (palm_elements_along - 1) \
                for i in range(palm_elements_along)]
    b_palm = [5.5 + i * (2.5 - 5.5) / (palm_elements_along - 1) \
                for i in range(palm_elements_along)]
    a_finger = 2
    b_finger = 1.75

    ellipses = [None for i in range(sum(hand_elements_along[0:2]) + 1)]
    for n in range(palm_elements_along):
        x, d1, d2, d3 = node_network.get_node_parameters(parent_node_id)
        center = add(x, d2)
        major_axis = mult(d2, -a_palm[n])
        minor_axis = mult(d3, b_palm[n])
        major_ax_mag = magnitude(major_axis)
        # minor_ax_mag = magnitude(minor_axis)
        internal_box_length = 2 * magnitude(d2)
        sampling_along_x = [internal_box_length / major_ax_mag, 0] # between 0 and b
        sampling_along_y = [0.6] # between 0 and a
        angles_along_x = [math.acos(value) for value in sampling_along_x]
        angles_along_y = [math.asin(value) for value in sampling_along_y]
        sampling_angles = sorted(angles_along_x + angles_along_y)
        ellipse_x, ellipse_d2 = sample_ellipse_along_angles(
            center, major_axis, minor_axis, sampling_angles
            )
        ellipse_d1 = [None for i in range(len(ellipse_x)) ]
        ellipses[n] = [ellipse_x, ellipse_d1, ellipse_d2]
        parent_node_id += 1

    for n in range(palm_elements_along - 1):
        ellipse_x, ellipse_d1, _ = ellipses[n]
        next_ellipse_x, next_ellipse_d1, _ = ellipses[n + 1]
        for i in range(len(ellipse_x)):
            d1 = sub(next_ellipse_x[i], ellipse_x[i])
            ellipse_d1[i] = next_ellipse_d1[i] = d1

    parent_node_id = 1
    index_y = 1
    d3 = None
    d1_offset = 0.0
    k_val = [1, 2]
    # Virtual node creation goes up each finger fastest
    for finger_index in range(4):
        index_x = 0
        # Carpal, metacarpal and proximal phalanx
        for bone in range(3):
            n_elements_along = hand_elements_along[bone]
            j_val = [index_y, index_y + 1] if finger_index == 0 else [index_y + 1]
            if bone in [Bone.CARPAL, Bone.METACARPAL]:
                i_val = [index_x + i for i in range(n_elements_along)]
            else:
                i_val = [index_x]
            for i in i_val:
                ellipse = ellipses[i]
                for j in j_val:
                    for k in k_val:
                        index_e = slice(0, 5, 1) if k == 2 else slice(10, 4, -1)
                        ellipse_x = ellipse[0][index_e]
                        ellipse_d1 = ellipse[1][index_e]
                        ellipse_d2 = ellipse[2][index_e]
                        a1 = finger_index if j == index_y else finger_index + 1
                        a2 = -1 if j == index_y else 1
                        a3 = -1 if k == 1 else 1
                        x = ellipse_x[a1]
                        # if bone > 0:
                        # d1 = add(d1, set_magnitude(d1, d1_offset/(c+mc+pp)))
                        d1 = ellipse_d1[a1]
                        d2 = ellipse_d2[a1]
                        indices = [[i, j, k + a3]]
                        if bone == Bone.CARPAL:
                            annotation = ['']
                            if j == 1:
                                indices.append([i, j + a2, k])
                            elif j == 17:
                                indices.append([i, j + a2, k])
                                indices.append([i, j + 4, k + a3])
                            else:
                                indices.append([i, j + 4, k + a3])
                        elif bone == Bone.METACARPAL:
                            annotation = ['']
                            if j == 1:
                                indices.append([i, j + a2, k])
                            elif j == 17:
                                if i > 1:
                                    indices.append([i, j + a2, k])
                                indices.append([i, j, k + a3])
                            else:
                                indices.append([i, j, k + a3])
                                indices.append([i, j + 4, k + a3])
                        elif bone == Bone.PROX_PHALANX:
                            n_elements_along = 1
                            finger_name = finger_names[finger_index] if j == index_y else \
                                finger_names[finger_index + 1]
                            finger_name = finger_names[3] if finger_index == 3 else finger_name
                            bone_name = bone_names[2]
                            finger_part = bone_name + ' of ' + finger_name
                            annotation = [bone_name, finger_name, finger_part]
                            indices.append([i + 2, j, k + a3])
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
                        node = virtual_node_matrix.create_external_vnode(
                            x  = [1, 0, 0, 0],
                            d1 = [0, 1, 0, 0],
                            d2 = [0, 0, 1, 0],
                            node_identifier = node_identifier, annotation = annotation
                        )
                        virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                        node_network.set_node_parameters(
                            [x, d1, d2, d3], node_identifier, internal_node=False
                        )
                        node_identifier += 1
                        x = add(x, d1)
                        ellipse_x[a1] = x
                parent_node_id += 1
            index_x += n_elements_along
        index_x += 2
        if hand_elements_along[2] > 1:
            phalanx_bones = [Bone.PROX_PHALANX, Bone.MID_PHALANX, Bone.DIST_PHALANX]
        else:
            phalanx_bones = [Bone.MID_PHALANX, Bone.DIST_PHALANX]
        # Proximal, middle and distal phalanx
        for bone in phalanx_bones:
            n_elements_along = hand_elements_along[bone]
            n_elements_along = n_elements_along - 1 if bone == Bone.PROX_PHALANX \
                else n_elements_along
            j_val = [index_y, index_y + 1]
            i_val = [index_x + i for i in range(n_elements_along)]
            # Estimate ellipse
            center, d1, d2, d3 = node_network.get_node_parameters(parent_node_id)
            if bone in [Bone.DIST_PHALANX]:
                center = add(center, set_magnitude(d1, d1_offset/(n_elements_along - 1)))
            else:
                center = add(center, set_magnitude(d1, d1_offset/(n_elements_along)))
            major_axis = mult(d2, -a_finger)
            minor_axis = mult(d3, b_finger)
            sampling_angles = [1 * math.pi / 4]
            ellipse_x, ellipse_d2 = sample_ellipse_along_angles(
                center, major_axis, minor_axis, sampling_angles
                )
            ellipse_d1 = [d1 for i in range(len(ellipse_x))]
            ellipse_x = [ellipse_x[i] for i in [3, 0, 2, 1]]
            ellipse_d2 = [ellipse_d2[i] for i in [3, 0, 2, 1]]
            ""
            for i in i_val:
                a1 = 0
                for j in j_val:
                    for k in k_val:
                        a2 = -1 if j == index_y else 1
                        a3 = -1 if k == 1 else 1
                        x = ellipse_x[a1]
                        d1 = ellipse_d1[a1]
                        d2 = ellipse_d2[a1]
                        finger_name = finger_names[finger_index]
                        bone_name = bone_names[bone]
                        finger_part = bone_name + ' of ' + finger_name
                        annotation = [bone_name, finger_name, finger_part]
                        indices = [[i, j + a2, k], [i, j, k + a3]]
                        node = virtual_node_matrix.create_external_vnode(
                            x  = [1, 0, 0, 0],
                            d1 = [0, 1, 0, 0],
                            d2 = [0, 0, 1, 0],
                            node_identifier = node_identifier, annotation = annotation
                        )
                        virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                        node_network.set_node_parameters(
                            [x, d1, d2, d3], node_identifier, internal_node=False
                        )
                        if bone == Bone.DIST_PHALANX and i == i_val[-1]:
                            c0 = 1
                            c1 = c0 * a2 * a3
                            c2 = c0 * a2
                            c3 = c0 * a3
                            annotation = [finger_name]
                            node = virtual_node_matrix.create_external_vnode(
                                x  = [1, 0, 0, 0],
                                d1 = [0, 1, 0, 0],
                                d2 = [0, -c1, 1, 0],
                                node_identifier = node_identifier, annotation = annotation
                            )
                            virtual_node_matrix.set_virtual_node_to_index(node, [i, j, k + a3])
                            node = virtual_node_matrix.create_external_vnode(
                                x  = [1, 0, 0, 0],
                                d1 = [0, 1, 0, 0],
                                d2 = [0, c1, 1, 0],
                                node_identifier = node_identifier, annotation = annotation
                            )
                            virtual_node_matrix.set_virtual_node_to_index(node, [i, j + a2, k])
                            node = virtual_node_matrix.create_external_vnode(
                                x  = [1, 0, 0, 0],
                                d1 = [0, -c2, a3, 0],
                                d2 = [0, c3, a2, 0],
                                node_identifier = node_identifier, annotation = annotation
                            )
                            virtual_node_matrix.set_virtual_node_to_index(node, [i + 1, j, k])
                        node_identifier += 1
                        x = add(x, d1)
                        ellipse_x[a1] = x
                        a1 += 1
                parent_node_id += 1
            index_x += n_elements_along
        index_y += 5
    # Finger-palm connections
    j_val = [3, 8, 13]
    i = c + mc + 2  # First row of PP elements
    bone_name = bone_names[2]
    finger_index = 0
    for j in j_val:
        for k in k_val:
            a3 = -1 if k == 1 else 1
            c1 = 0.25 * a3
            finger_name = finger_names[finger_index]
            finger_part = bone_name + ' of ' + finger_name
            annotation = [bone_name, finger_name, finger_part]
            index = [i, j - 1, k + a3]
            node_id = virtual_node_matrix.get_node_identifier_at_index(index)
            node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, -c1, 0],
                d2 = [0, 0, 1, 0],
                node_identifier = node_id, annotation = annotation
            )
            virtual_node_matrix.set_virtual_node_to_index(node, [i, j - 1, k + a3])
            node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, -c1, 0],
                d2 = [0, a3, 0, 0],
                node_identifier = node_id, annotation = annotation
            )
            virtual_node_matrix.set_virtual_node_to_index(node, [i, j, k])

            finger_name = finger_names[finger_index + 1]
            finger_part = bone_name + ' of ' + finger_name
            annotation = [bone_name, finger_name, finger_part]
            node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, c1, 0],
                d2 = [0, -a3, 0, 0],
                node_identifier = node_id, annotation = annotation
            )
            virtual_node_matrix.set_virtual_node_to_index(node, [i, j + 2, k])

            node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, c1, 0],
                d2 = [0, 0, 1, 0],
                node_identifier = node_id, annotation = annotation
            )
            virtual_node_matrix.set_virtual_node_to_index(node, [i, j + 3, k + a3])
        finger_index += 1
    # Thumb
    index_y += 5
    index_x = c
    finger_index = 4
    thumb_bones = [Bone.METACARPAL, Bone.PROX_PHALANX, Bone.DIST_PHALANX]
    # Metacarpal, proximal and distal phalanx
    for bone in thumb_bones:
        n_elements_along = hand_elements_along[bone]
        if bone in [Bone.METACARPAL, Bone.PROX_PHALANX]:
            n_elements_along = max(1, n_elements_along - 1)
        j_val = [index_y, index_y + 1]
        i_val = [index_x + i for i in range(n_elements_along)]
        # Estimate ellipse
        center, d1, d2, d3 = node_network.get_node_parameters(parent_node_id)
        center = add(center, set_magnitude(d1, d1_offset/n_elements_along))
        major_axis = mult(d2, -a_finger)
        minor_axis = mult(d3, b_finger)
        sampling_angles = [1 * math.pi / 4]
        ellipse_x, ellipse_d2 = sample_ellipse_along_angles(
            center, major_axis, minor_axis, sampling_angles
            )
        ellipse_d1 = [d1 for i in range(len(ellipse_x))]
        ellipse_x = [ellipse_x[i] for i in [3, 0, 2, 1]]
        ellipse_d2 = [ellipse_d2[i] for i in [3, 0, 2, 1]]
        for i in i_val:
            a1 = 0
            for j in j_val:
                for k in k_val:
                    a2 = -1 if j == index_y else 1
                    a3 = -1 if k == 1 else 1
                    x = ellipse_x[a1]
                    d1 = ellipse_d1[a1]
                    d2 = ellipse_d2[a1]
                    finger_name = finger_names[finger_index]
                    bone_name = bone_names[bone]
                    finger_part = bone_name + ' of ' + finger_name
                    annotation = [bone_name, finger_name, finger_part]
                    indices = [[i, j, k + a3]]
                    if bone == Bone.METACARPAL:
                        if j == index_y + 1:
                            indices.append([i, j + a2, k])
                    elif bone == Bone.PROX_PHALANX:
                        indices.append([i, j + a2, k])
                    elif bone == Bone.DIST_PHALANX:
                        indices.append([i, j + a2, k])
                    node = virtual_node_matrix.create_external_vnode(
                        x  = [1, 0, 0, 0],
                        d1 = [0, 1, 0, 0],
                        d2 = [0, 0, 1, 0],
                        node_identifier = node_identifier, annotation = annotation
                    )
                    virtual_node_matrix.set_virtual_node_to_indices(node, indices)
                    node_network.set_node_parameters(
                        [x, d1, d2, d3], node_identifier, internal_node=False
                    )
                    if bone == Bone.DIST_PHALANX and i == i_val[-1]:
                        c0 = 1
                        c1 = c0 * a2 * a3
                        c2 = c0 * a2
                        c3 = c0 * a3
                        annotation = [finger_name]
                        node = virtual_node_matrix.create_external_vnode(
                            x  = [1, 0, 0, 0],
                            d1 = [0, 1, 0, 0],
                            d2 = [0, -c1, 1, 0],
                            node_identifier = node_identifier, annotation = annotation
                        )
                        virtual_node_matrix.set_virtual_node_to_index(node, [i, j, k + a3])
                        node = virtual_node_matrix.create_external_vnode(
                            x  = [1, 0, 0, 0],
                            d1 = [0, 1, 0, 0],
                            d2 = [0, c1, 1, 0],
                            node_identifier = node_identifier, annotation = annotation
                        )
                        virtual_node_matrix.set_virtual_node_to_index(node, [i, j + a2, k])
                        node = virtual_node_matrix.create_external_vnode(
                            x  = [1, 0, 0, 0],
                            d1 = [0, -c2, a3, 0],
                            d2 = [0, c3, a2, 0],
                            node_identifier = node_identifier, annotation = annotation
                        )
                        virtual_node_matrix.set_virtual_node_to_index(node, [i + 1, j, k])
                    node_identifier += 1
                    x = add(x, d1)
                    ellipse_x[a1] = x
                    a1 += 1
            parent_node_id += 1
        index_x += n_elements_along
    # The Thumb-palm connection elements
    # It is against every fiber of my being that I am forced to set these manually
    thumb_angle_degrees = options["Thumb angle"]
    thumb_c = thumb_angle_degrees / 90
    # Back of the thumb
    k_val = [0, 3]
    for k in k_val:
        a1 = -1 if k == 3 else 1
        a3 = -1 if k == 3 else 1
        index = [0, index_y - 9, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 0.5, -a1 * 0.5, 0],
                d2 = [0, 0, 1, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [0, index_y - 5, k + a3])
        index = [1, index_y + 1, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, a1 * thumb_c, 0],
                d2 = [0, 0, 1, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [0, index_y - 4, k + a3])

    # Bottom carpal-metacarpal connection
    for k in k_val:
        a1 = -1 if k == 3 else 1
        index = [1, index_y + 1, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 0, a1, 0],
                d2 = [0, -a1, -thumb_c, 0],
                node_identifier = node_id, annotation = annotation
        )

        virtual_node_matrix.set_virtual_node_to_index(node, [1, index_y - 4, k])

        index = [1, index_y, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 0, a1, 0],
                d2 = [0, -a1 * thumb_c, 0, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [2, index_y - 4, k])

        index = [0, index_y - 9, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, 0, 0],
                d2 = [0, -a1 * 0.5, 0.5, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [1, index_y - 5, k])

        index = [1, index_y - 9, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, 0, 0],
                d2 = [0, 0, 0.25, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [2, index_y - 5, k])
    # Intermediate metacarpal connections
    for k in k_val:
        a1 = 2 * thumb_c
        a3 = -1 if k == 3 else 1
        index = [1, index_y, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, 0, 0],
                d2 = [0, -a3 * thumb_c, 0, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [4, index_y - 4, k])

        index = [1, index_y - 9, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, 0, 0],
                d2 = [0, 0, 0.25, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [4, index_y - 5, k])

        index = [2, index_y, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, 0, 0],
                d2 = [0, -a1 * a3, 0.5, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [5, index_y - 4, k])

        index = [2, index_y - 9, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, 1, 0, 0],
                d2 = [0, a1 * a3, 0.5, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [5, index_y - 5, k])

    # Front of the webbing
    for k in k_val:
        a1 = 2 * thumb_c
        a3 = -1 if k == 3 else 1
        index = [2, index_y - 9, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, -a1, -a3 * 0.5, 0],
                d2 = [0, 0, 1, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [6, index_y - 5, k + a3])

        index = [2, index_y, k]
        node_id = virtual_node_matrix.get_node_identifier_at_index(index)
        node = virtual_node_matrix.create_external_vnode(
                x  = [1, 0, 0, 0],
                d1 = [0, a1, -a3 * 0.5, 0],
                d2 = [0, 0, -1, 0],
                node_identifier = node_id, annotation = annotation
        )
        virtual_node_matrix.set_virtual_node_to_index(node, [6, index_y - 4, k + a3])

    return virtual_node_matrix, node_network, node_identifier

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

def sample_ellipse_along_angles(center: list, major_axis: list, minor_axis: list,
                                 sampling_angles: list):
    """
    From a list of angles on the first quadrant of an ellipse, calculate their reflection
    over the other three quadrants. Then alculate both the position and tangent angle at the
    updated list of angles.

    :param center: Description
    :param major_axis: Description
    :param minor_axis: Description
    :param sampling_angles: Description
    """
    for angle in reversed(sampling_angles):
        new_angle = math.pi - angle
        if new_angle not in sampling_angles:
            sampling_angles.append(new_angle)
    for angle in sampling_angles[:]:
        new_angle = math.pi + angle
        if new_angle not in sampling_angles:
            sampling_angles.append(new_angle)
    sampling_angles.append(2 * math.pi + sampling_angles[0])

    ellipse_x = []
    ellipse_d1 = []
    for i in range(len(sampling_angles) - 1):
        ellipse = sampleEllipsePoints(
            center, major_axis, minor_axis, sampling_angles[i], sampling_angles[i + 1], 1
        )
        ellipse_x.append(ellipse[0][0])
        ellipse_d1.append(ellipse[1][0])
    return ellipse_x, ellipse_d1

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
