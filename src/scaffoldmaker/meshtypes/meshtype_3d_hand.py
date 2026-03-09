import math
import logging

from cmlibs.maths.vectorops import (
    add, cross, distance, dot, magnitude, matrix_mult, matrix_inv, mult, normalize, rejection, set_magnitude, sub)
from cmlibs.utils.zinc.field import find_or_create_field_group, find_or_create_field_coordinates
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.element import Element, Elementbasis, Elementfieldtemplate
from cmlibs.zinc.field import Field, FieldFindMeshLocation, FieldGroup
from cmlibs.zinc.node import Node
from cmlibs.zinc.region import Region
from cmlibs.zinc.fieldmodule import Fieldmodule
from cmlibs.zinc.result import RESULT_OK
from cmlibs.maths.vectorops import rotate_about_z_axis
from scaffoldfitter.fitter import Fitter as GeometryFitter
from scaffoldfitter.fitterstepfit import FitterStepFit
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, findOrCreateAnnotationGroupForTerm, \
    findAnnotationGroupByName
from scaffoldmaker.annotation.vagus_terms import get_vagus_term, get_vagus_marker_term, \
    get_left_vagus_marker_locations_list, get_right_vagus_marker_locations_list
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.utils.constructionobject import ConstructionObject
from scaffoldmaker.utils.eft_utils import remapEftLocalNodes, remapEftNodeValueLabel, remapEftNodeValueLabelWithNodes, \
    setEftScaleFactorIds
from scaffoldmaker.utils.interpolation import (
    evaluateCoordinatesOnCurve, evaluateScalarOnCurve, getCubicHermiteBasis, getCubicHermiteBasisDerivatives,
    getCubicHermiteArcLength, getCubicHermiteCurvature, getCubicHermiteCurvesLength,
    getCubicHermiteTrimmedCurvesLengths, getNearestLocationOnCurve, get_curve_from_points,
    interpolateCubicHermiteDerivative, sampleCubicHermiteCurves, sampleCubicHermiteCurvesSmooth,
    smoothCurveSideCrossDerivatives, track_curve_side_direction)
from scaffoldmaker.utils.read_vagus_data import load_vagus_data
from scaffoldmaker.utils.zinc_utils import (
    define_and_fit_field, find_or_create_field_zero_fibres, fit_hermite_curve, generate_curve_mesh, generate_datapoints,\
    generate_mesh_marker_points)




class MeshType_3d_hand1(Scaffold_base):
    """
    Generates a hermite x bilinear 3-D box network mesh based on data supplied by an input file.
    """

    @classmethod
    def getName(cls):
        return "3D Hand 1"

    @classmethod
    def getParameterSetNames(cls):
        return [
            'Default'
            ]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        baseParameterSetName = 'Human Left Vagus 1' if (parameterSetName == 'Default') else parameterSetName
        options = {
            'Base parameter set': baseParameterSetName,
            'Thumb angle': 90.0,
        }
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            'Thumb angle'
        ]

    @classmethod
    def checkOptions(cls, options):
        dependent_changes = False
        for key, angleRange in {
            'Thumb angle': (0.0, 100.0), 
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
        thumb_angle_degrees = options['Thumb angle']
        #################
        # Setup zinc 
        #################
        fieldmodule = region.getFieldmodule()
        coordinates = find_or_create_field_coordinates(fieldmodule)
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        fieldcache = fieldmodule.createFieldcache()
        mesh3d = fieldmodule.findMeshByDimension(3)
        #################
        # Create bone nodes
        #################
        node_identifier = 1
        carpal_nodes = [[0, 0.2, 0]]

        for i in range(4):
            x = carpal_nodes[-1]
            x = add(x, mult([0, 1, 0], 1))
            carpal_nodes.append(x)
        # Fingers 2 - 4 (finger 1, thumb, is added later)
        finger_dimensions = [
            [1.0, 0.5, 0.3, 0.4],
            [2.0, 0.5, 0.3, 0.2],
            [1.5, 0.2, 0.3, 0.2], 
            [1.0, 0.2, 0.3, 0.2], 
            [1.0, 0.2, 0.3, 0.2]
        ]
        node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[0])
        node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[1])
        node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[2])
        node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[3])
        node_identifier = nodes.getSize() + 1
        # finger_dimensions = [
        #     [0.3, 0.5, 0.2, 0.4],
        #     [2.4, 0.5, 0.2, 0.2],
        #     [1.4, 0.2, 0.2, 0.2], 
        #     [0.8, 0.2, 0.2, 0.2], 
        #     [0.6, 0.2, 0.2, 0.2]
        # ]
        # node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[0])
        # finger_dimensions = [
        #     [0.3, 0.5, 0.2, 0.4],
        #     [2.3, 0.5, 0.2, 0.2],
        #     [1.6, 0.2, 0.2, 0.2], 
        #     [1, 0.2, 0.2, 0.2], 
        #     [0.6, 0.2, 0.2, 0.2]
        # ]
        # node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[1])
        # finger_dimensions = [
        #     [0.3, 0.5, 0.2, 0.4],
        #     [2.0, 0.5, 0.2, 0.2],
        #     [1.5, 0.2, 0.2, 0.2], 
        #     [0.9, 0.2, 0.2, 0.2], 
        #     [0.7, 0.2, 0.2, 0.2]
        # ]
        # node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[2])
        # finger_dimensions = [
        #     [0.3, 0.5, 0.2, 0.4],
        #     [1.9, 0.5, 0.2, 0.2],
        #     [1.2, 0.2, 0.2, 0.2], 
        #     [0.7, 0.2, 0.2, 0.2], 
        #     [0.6, 0.2, 0.2, 0.2]
        # ]
        # node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[3])
        # Finger 1 (thumb)
        # Get the finger 1 metacarpal node
        finger_dimensions = [
            [1.9, 0.4, 0.2, 0.2],
            [1.2, 0.2, 0.2, 0.2], 
            [0.6, 0.2, 0.2, 0.2]
        ]
        # node_identifier = create_thumb_nodes(fieldmodule, 21, finger_dimensions, 5, thumb_angle_degrees)
        #################
        # Create box elements
        #################
        # Get scale factor matrix 
        # There is an extra row of elements on the metacarpal that represent the joint between
        # the palm and the finger, always needs to be there. 
        # Make sure to add 1 manually to the number of metacarpal elements when you add the options
        number_elements = [1, 1+1, 1, 1, 1]
        
        scale_factor_matrix = create_scale_factor_matrix(fieldmodule, number_elements, node_identifier)
        # Let it rip
        element_identifier = 1
        for k in range(3):
            for j in range(18): 
                for i in range(sum(number_elements)):
                    # if i != 2:
                    #     continue 
                    result = create_linear_cube_element(fieldmodule, element_identifier, scale_factor_matrix, i, j, k)
                    if result == RESULT_OK:
                        element_identifier += 1
        return [], None
    


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


def create_scale_factor_matrix(fieldmodule, number_elements, node_identifier):
    c, mc, pp, mp, dp = number_elements
    # mc += 1 #To account for the transitionary element from palm to finger
    scale_factor_matrix = [[[None for k in range(4)] for j in range(19)] for i in range(c+mc+pp+mp+dp+1)]
    # Scale factors for d2 and d3
    bone_w = 1
    skin_w = 1.5
    bone_h = 1
    skin_h = 2
    # hand_node_ids = {
    #     # Carpals
    #     1: {'y': [(0, -skin_w), (1, -bone_w), (2, bone_w), (6, bone_w)]}, 
    #     2: {'y': [(7, bone_w), (11, bone_w)]}, 
    #     3: {'y': [(12, bone_w), (16, bone_w)]}, 
    #     4: {'y': [(17, bone_w), (18, skin_w)]}, 
    #     # Metacarpals 
    #     5: {'y': [(0, -skin_w), (1, -bone_w), (2, bone_w), (6, bone_w)]}, 
    #     6: {'y': [(7, bone_w), (11, bone_w)]}, 
    #     7: {'y': [(12, bone_w), (16, bone_w)]}, 
    #     8: {'y': [(17, bone_w), (18, skin_w)]}, 
    #     # Proximal phalanx
    #     9: {'y': [(0, -skin_w), (1, -bone_w), (2, bone_w), (3, skin_w)]}, 
    #     10: {'y': [(5, -skin_w), (6, -bone_w), (7, bone_w), (8, skin_w)]}, 
    #     11: {'y': [(10, -skin_w), (11, -bone_w), (12, bone_w), (13, skin_w)]}, 
    #     12: {'y': [(15, -skin_w), (16, -bone_w), (17, bone_w), (18, skin_w)]}, 
    #     # Middle phalanx 
    #     13: {'y': [(0, -skin_w), (1, -bone_w), (2, bone_w), (3, skin_w)]}, 
    #     14: {'y': [(5, -skin_w), (6, -bone_w), (7, bone_w), (8, skin_w)]}, 
    #     15: {'y': [(10, -skin_w), (11, -bone_w), (12, bone_w), (13, skin_w)]}, 
    #     16: {'y': [(15, -skin_w), (16, -bone_w), (17, bone_w), (18, skin_w)]}, 
    #     # Distal phalanx
    #     17: {'y': [(0, -skin_w), (1, -bone_w), (2, bone_w), (3, skin_w)]}, 
    #     18: {'y': [(5, -skin_w), (6, -bone_w), (7, bone_w), (8, skin_w)]}, 
    #     19: {'y': [(10, -skin_w), (11, -bone_w), (12, bone_w), (13, skin_w)]}, 
    #     20: {'y': [(15, -skin_w), (16, -bone_w), (17, bone_w), (18, skin_w)]}, 
    # }
    bone_node_ids = {
        # Carpals
        1: {'y': [ (1, -bone_w), (2, bone_w), (6, bone_w)]}, 
        2: {'y': [(7, bone_w), (11, bone_w)]}, 
        3: {'y': [(12, bone_w), (16, bone_w)]}, 
        4: {'y': [(17, bone_w), ]}, 
        # Metacarpals 
        5: {'y': [ (1, -bone_w), (2, bone_w), (6, bone_w)]}, 
        6: {'y': [(7, bone_w), (11, bone_w)]}, 
        7: {'y': [(12, bone_w), (16, bone_w)]}, 
        8: {'y': [(17, bone_w)]}, 
        # Proximal phalanx
        9: {'y': [ (1, -bone_w), (2, bone_w), ]}, 
        10: {'y': [(6, -bone_w), (7, bone_w), ]}, 
        11: {'y': [(11, -bone_w), (12, bone_w), ]}, 
        12: {'y': [(16, -bone_w), (17, bone_w), ]}, 
        # Middle phalanx 
        13: {'y': [(1, -bone_w), (2, bone_w), ]}, 
        14: {'y': [ (6, -bone_w), (7, bone_w), ]}, 
        15: {'y': [ (11, -bone_w), (12, bone_w), ]}, 
        16: {'y': [ (16, -bone_w), (17, bone_w), ]}, 
        # Distal phalanx
        17: {'y': [ (1, -bone_w), (2, bone_w), ]}, 
        18: {'y': [ (6, -bone_w), (7, bone_w), ]}, 
        19: {'y': [ (11, -bone_w), (12, bone_w), ]}, 
        20: {'y': [ (16, -bone_w), (17, bone_w), ]}, 
    }
    # The rows in the x and z direction follow a more basic algorithm, which still depends on the 
    # node_ids, but these can be somewhat automated. 
    # Carpals
    z_vals = [(1, -bone_h), (2, bone_h)]
    for node_id in range(1, 5):
        bone_node_ids[node_id]['x'] = [(i, i/c) for i in range(c)]
        bone_node_ids[node_id]['z'] = z_vals
    # Metacarpals
    for node_id in range(5, 9):
        bone_node_ids[node_id]['x'] = [(c+i, (i/mc)*0.5) for i in range(mc-1)]
        bone_node_ids[node_id]['x'].append((c+mc-1, 0.5))
        bone_node_ids[node_id]['z'] = z_vals
    # Proximal phalanx
    for node_id in range(9, 13):
        bone_node_ids[node_id]['x'] = [(i+c+mc, i/pp) for i in range(pp)]
        bone_node_ids[node_id]['z'] = z_vals
    # Middle phalanx 
    for node_id in range(13, 17):
        bone_node_ids[node_id]['x'] = [(i+c+mc+pp, i/mp) for i in range(mp)]
        bone_node_ids[node_id]['z'] = z_vals
        # Distal phalanx
    for node_id in range(17, 21):
        bone_node_ids[node_id]['x'] = [(i+c+mc+pp+mp, i/mp) for i in range(dp+1)]
        bone_node_ids[node_id]['z'] = z_vals
        
    for node_id, node_factors in bone_node_ids.items():
            for x in node_factors['x']:
                for y in node_factors['y']:
                    for z in node_factors['z']: 
                        i = x[0]
                        j = y[0]
                        k = z[0]
                        a0 = 1
                        a1 = x[1]
                        a2 = y[1]
                        a3 = z[1]
                        if scale_factor_matrix[i][j][k] is None: 
                            scale_factor_matrix[i][j][k] = {
                                Node.VALUE_LABEL_VALUE: [node_id, a0, a1, a2, a3]
                            } 
    ############
    # Skin nodes 
    ############
    skin_node_ids = {
        # Carpals
        1: {'y': [([0], -skin_w), ([2, 6], bone_w)]}, 
        2: {'y': [([7, 11], bone_w)]}, 
        3: {'y': [([12, 16], bone_w)]}, 
        4: {'y': [([17], bone_w), ([18], skin_w)]}, 
        # Metacarpals 
        5: {'y': [([0], -skin_w), ([2, 6], bone_w)]}, 
        6: {'y': [([7, 11], bone_w)]},
        7: {'y': [([12, 16], bone_w)]}, 
        8: {'y': [([17], bone_w), ([18], skin_w)]}, 
        # Proximal phalanx
        9: {'y': [ ([0], -skin_w), ([3], skin_w), ]}, 
        10: {'y': [ ([5], -skin_w), ([8], skin_w), ]}, 
        11: {'y': [ ([10], -skin_w), ([13], skin_w), ]}, 
        12: {'y': [ ([15], -skin_w), ([18], skin_w), ]}, 
        # Middle phalanx 
        13: {'y': [ ([0], -skin_w), ([3], skin_w), ]}, 
        14: {'y': [ ([5], -skin_w), ([8], skin_w), ]}, 
        15: {'y': [ ([10], -skin_w), ([13], skin_w), ]}, 
        16: {'y': [ ([15], -skin_w), ([18], skin_w), ]}, 
        # Distal phalanx
        17: {'y': [ ([0], -skin_w), ([3], skin_w), ]}, 
        18: {'y': [ ([5], -skin_w), ([8], skin_w), ]}, 
        19: {'y': [ ([10], -skin_w), ([13], skin_w), ]}, 
        20: {'y': [ ([15], -skin_w), ([18], skin_w), ]}, 
    }
    z_vals = [(0, -skin_h), (3, skin_h)]
    for node_id in range(1, 5):
        skin_node_ids[node_id]['x'] = [(i, i/c) for i in range(c)]
        skin_node_ids[node_id]['z'] = z_vals
    # Metacarpals
    for node_id in range(5, 9):
        skin_node_ids[node_id]['x'] = [(c+i, (i/mc)*0.5) for i in range(mc-1)]
        skin_node_ids[node_id]['x'].append((c+mc-1, 0.5))
        skin_node_ids[node_id]['z'] = z_vals
    # Proximal phalanx
    for node_id in range(9, 13):
        skin_node_ids[node_id]['x'] = [(i+c+mc, i/pp) for i in range(pp)]
        skin_node_ids[node_id]['z'] = z_vals
    # Middle phalanx 
    for node_id in range(13, 17):
        skin_node_ids[node_id]['x'] = [(i+c+mc+pp, i/mp) for i in range(mp)]
        skin_node_ids[node_id]['z'] = z_vals
    # Distal phalanx
    for node_id in range(17, 21):
        skin_node_ids[node_id]['x'] = [(i+c+mc+pp+mp, i/mp) for i in range(dp+1)]
        skin_node_ids[node_id]['z'] = z_vals
    # Assigning scale factors to the matrix
    for node_id, node_factors in skin_node_ids.items():
        for x in node_factors['x']:
            for z in node_factors['z']: 
                for y in node_factors['y']:
                    a0 = 1
                    a1 = x[1]
                    a2 = y[1]
                    a3 = z[1]
                    node_identifier = add_skin_node(
                            fieldmodule, node_id, node_identifier, [a0, a1, a2, a3])
                    # Assign this node to all the corners, as per the indices described
                    # in the dictionary
                    i = x[0]
                    k = z[0]
                    for y0 in y[0]:
                        j = y0
                        # In these 'corner' cases 
                        # The matrix cell itself is not written on, instead the two 
                        # other nodes at the side are 'pinched together' to make sure the 
                        # skin elements stitch together. 
                        if abs(a2) == skin_w and abs(a3) == skin_h:
                            sign2 = int(math.copysign(1, a2))
                            sign3 = int(math.copysign(1, a3))
                            # node_identifier = add_skin_node(
                                # fieldmodule, node_id, node_identifier, [a0, a1, a2, a3])
                            scale_factor_matrix[i][j-sign2][k] = {
                                Node.VALUE_LABEL_VALUE: [node_identifier-1, 1, 0, 0, 0]
                            }
                            scale_factor_matrix[i][j][k-sign3] = {
                                Node.VALUE_LABEL_VALUE: [node_identifier-1, 1, 0, 0, 0]
                            }
                        else:
                            # To not accidentally overwritte a corner node
                            if scale_factor_matrix[i][j][k] is None: 
                                scale_factor_matrix[i][j][k] = {
                                    Node.VALUE_LABEL_VALUE: [node_identifier-1, 1, 0, 0, 0]
                                } 
    return scale_factor_matrix



def create_linear_cube_element(fieldmodule, element_identifier, scale_factor_matrix, x, y, z):
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    linear_basis = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    # Create adn remap eft
    eft = mesh3d.createElementfieldtemplate(linear_basis)
    scale_factor_ids = {
    }
    local_node = 0
    local_node_ids = {}
    expression_terms = {}
    readable_expression_terms = {}
    n_local_nodes = 0
    n_scale_factors = 0
    value_labels = [
            0, 
            Node.VALUE_LABEL_VALUE,
            Node.VALUE_LABEL_D_DS1, 
            Node.VALUE_LABEL_D_DS2, 
            Node.VALUE_LABEL_D_DS3
        ]
    value_label_names = [0, 'v', 'd1', 'd2', 'd3']
    for k in [0, 1]:
        for j in [0, 1]:
            for i in [0, 1]:
                local_node = 1 + 1*i + 2*j +4*k
                et = []
                ret = []
                corner = scale_factor_matrix[x+i][y+j][z+k]
                if corner is None:
                    return -2 
                corner = corner[Node.VALUE_LABEL_VALUE]
                for factor in range(len(corner)):
                    if factor == 0:
                        # Check for node_ids
                        global_node_id = corner[factor]
                        if global_node_id not in local_node_ids:
                            n_local_nodes += 1
                            local_node_ids[global_node_id] = n_local_nodes
                    else:
                        # Check for scale_factor_ids
                        scale_factor = corner[factor]
                        if scale_factor not in scale_factor_ids:
                            n_scale_factors += 1
                            scale_factor_ids[scale_factor] = n_scale_factors
                        et.append(
                            [local_node_ids[global_node_id], value_labels[factor], scale_factor_ids[scale_factor]]
                        )
                        ret.append(
                            [str(global_node_id).zfill(2), value_label_names[factor], str(scale_factor).zfill(4)]
                        )
                expression_terms[local_node] = et
                readable_expression_terms[local_node] = ret
    setEftScaleFactorIds(eft, [], [], n_scale_factors)
    for local_node in expression_terms:
        remapEftNodeValueLabelWithNodes(
            eft, local_node, Node.VALUE_LABEL_VALUE, expression_terms[local_node]
        )
    remapEftLocalNodes(eft, n_local_nodes, [1, 2, 3, 4, 5, 6, 7, 8])
    # Create element template
    etemplate = mesh3d.createElementtemplate()
    etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
    result = etemplate.defineField(coordinates, -1, eft)
    if result != RESULT_OK:
        return result
    # Create element
    element = mesh3d.createElement(element_identifier, etemplate)
    node_ids = list(local_node_ids.keys())
    element.setNodesByIdentifier(eft, node_ids)
    scale_factors = list(scale_factor_ids.keys())
    element.setScaleFactors(eft, scale_factors)
    # element_identifier += 1
    return RESULT_OK

def add_skin_node(fieldmodule: Fieldmodule, bone_node_id: int, node_identifier: int, scale_factors: list):
    # Zinc setup
    coordinates = find_or_create_field_coordinates(fieldmodule)
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = get_simple_nodetemplate(fieldmodule)
    fieldcache = fieldmodule.createFieldcache()
    # Get position from node
    bone_node = nodes.findNodeByIdentifier(bone_node_id)
    fieldcache.setNode(bone_node)
    a0, a1, a2, a3 = scale_factors
    value_labels = [
            Node.VALUE_LABEL_VALUE,
            Node.VALUE_LABEL_D_DS1, 
            Node.VALUE_LABEL_D_DS2, 
            Node.VALUE_LABEL_D_DS3
        ]
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
    d1 = [1, 0, 0]
    d2 = [0, 0, 0]
    d3 = [0, 0, 0]
    setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
    node_identifier += 1
    return node_identifier

def create_thumb_nodes(fieldmodule, node_identifier, finger_dimensions, metacarpal_node_id, angle_degrees=0):
    """
    Docstring for create_finger_nodes
    
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
    x1 = rotate_about_z_axis(x1, math.radians(angle_degrees))
    x2 = rotate_about_z_axis(x2, math.radians(angle_degrees))
    # Obtaining the starting node position from the metacarpal node
    metacarpal_node = nodes.findNodeByIdentifier(metacarpal_node_id)
    fieldcache.setNode(metacarpal_node)
    node_location = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)[1]
    d2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)[1]
    node_location = add(node_location, d2)
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
        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
        x = add(x, d1)
        node_location = x
        # 
        finger_node_identifier += 1 
    node_identifier += 1
    return node_identifier

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
    """
    """
    coordinates = find_or_create_field_coordinates(fieldmodule)
    value_labels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                        Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
                        Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3]
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

def setNodeFieldVersionDerivatives(field, fieldcache, version, d1, d2, d3, d12=None, d13=None):
    """
    Assign node field parameters d1, d2, d3 of field.
    :param field: Field to assign parameters of.
    :param fieldcache: Fieldcache with node set.
    :param version: Version of d1, d2, d3 >= 1.
    :param d1: Parameters to set for Node.VALUE_LABEL_D_DS1.
    :param d2: Parameters to set for Node.VALUE_LABEL_D_DS2.
    :param d3: Parameters to set for Node.VALUE_LABEL_D_DS3.
    :param d12: Optional parameters to set for Node.VALUE_LABEL_D2_DS1DS2.
    :param d13: Optional parameters to set for Node.VALUE_LABEL_D2_DS1DS3.
    :return:
    """
    field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, version, d1)
    field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, version, d2)
    field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, version, d3)
    if d12:
        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, version, d12)
    if d13:
        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, version, d13)
