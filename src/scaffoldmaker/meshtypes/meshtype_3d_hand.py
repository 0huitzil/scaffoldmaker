import math
import logging


from itertools import product
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
from scaffoldmaker.utils.geometry import sampleEllipsePoints




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
        carpal_nodes = [[0, 0, 0]]

        
        # Fingers 2 - 4 (finger 1, thumb, is added later)
        finger_dimensions = [ #d1, d2, d3, d12
            [1.0, 0.6, 0.3, 0.4], #carpal
            [3.0, 0.6, 0.3, 0.2], #metacarpal
            [1.5, 0.2, 0.3, 0.2], #pp
            [1.0, 0.2, 0.3, 0.2], #m                             p
            [1.0, 0.2, 0.3, 0.2]  #dp
        ]
        d2 = finger_dimensions[0][1]
        for i in range(4):
            x = carpal_nodes[-1]
            x = add(x, mult([0, 1, 0], d2*2))
            carpal_nodes.append(x)
        node_identifier = create_finger_nodes_new(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[0])
        node_identifier = create_finger_nodes_new(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[1])
        node_identifier = create_finger_nodes_new(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[2])
        node_identifier = create_finger_nodes_new(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[3])
        # Thumb
        finger_dimensions = [ #d1, d2, d3, d12
            [1.0, 0.5, 0.3, 0.4], #carpal
            [2.0, 0.6, 0.3, 0.2], #metacarpal
            [1, 0.2, 0.3, 0.2], #pp
            [0.5, 0.2, 0.3, 0.2], #m                             p
            [0.5, 0.2, 0.3, 0.2]  #dp
        ]
        node_identifier = create_thumb_nodes_new(fieldmodule, node_identifier, finger_dimensions[1:4], 17, thumb_angle_degrees)
        # node_identifier = nodes.getSize() + 1
        # node_identifier = create_thumb_nodes(fieldmodule, node_identifier, finger_dimensions[1:], 8)
        # node_identifier = nodes.getSize() + 1
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

        #################
        # Create box elements
        #################
        # Get scale factor matrix 
        # There is an extra row of elements on the metacarpal that represent the joint between
        # the palm and the finger, always needs to be there. 
        # Make sure to add 1 manually to the number of metacarpal elements when you add the options
        number_elements = [1, 2+1, 1, 1, 2]
        c, mc, pp, mp, dp = number_elements
        virtual_node_matrix = [[[None for k in range(4)] for j in range(30)] for i in range(c+mc+pp+mp+dp+1)]
        
        virtual_node_matrix, node_identifier = generate_internal_node_matrix(fieldmodule, number_elements, node_identifier, virtual_node_matrix)

        generate_external_node_matrix(fieldmodule, number_elements, node_identifier, virtual_node_matrix)
        # virtual_node_matrix = create_virtual_node_matrix(fieldmodule, number_elements, node_identifier, skin_elements=False)
        # Let it rip
        # virtual_node_matrix[3][3][1] = virtual_node_matrix[3][2][0] 
        # virtual_node_matrix[3][3][2] = virtual_node_matrix[3][2][3]

        # virtual_node_matrix[3][5][1] = virtual_node_matrix[3][6][0] 
        # virtual_node_matrix[3][5][2] = virtual_node_matrix[3][6][3]
        
        # virtual_node_matrix[3][8][1] = virtual_node_matrix[3][7][0] 
        # virtual_node_matrix[3][8][2] = virtual_node_matrix[3][7][3]

        # virtual_node_matrix[3][10][1] = virtual_node_matrix[3][11][0] 
        # virtual_node_matrix[3][10][2] = virtual_node_matrix[3][11][3]
        
        # virtual_node_matrix[3][13][1] = virtual_node_matrix[3][12][0] 
        # virtual_node_matrix[3][13][2] = virtual_node_matrix[3][12][3]

        # virtual_node_matrix[3][15][1] = virtual_node_matrix[3][16][0] 
        # virtual_node_matrix[3][15][2] = virtual_node_matrix[3][12][3]

        z_len = len(virtual_node_matrix[0][0]) - 1
        y_len = len(virtual_node_matrix[0]) -1
        x_len = len(virtual_node_matrix) - 1
        element_identifier = 1
        for k in range(z_len):
            for j in range(y_len): 
                for i in range(x_len):
        
                    # if i != 3:
                    #     continue 
                    # if j not in [0, 1]:
                    #     continue
                    # if k not in [1]:
                    #     continue
                    result = create_linear_cube_element(fieldmodule, element_identifier, virtual_node_matrix, i, j, k)
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




def generate_internal_node_matrix(fieldmodule, number_elements, node_identifier, node_matrix):
    c, mc, pp, mp, dp = number_elements
    # Finger elements are created in reverse order, from 5 (little) to 1 (thumb)
    # Scale factors for d2 and d3
    bone_w = 1
    bone_h = 1
    a0 = 1
    # Finger 
    node_id = 0
    y = 1
    for f in range(4):
        # Carpal
        node_id += 1
        x = 0
        y_val = [y, y+1] if f == 0 else [y+1]
        x_val = [(x+i, i/c) for i in range(c)]
        for k in [1, 2]:
            for j in y_val:
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a3 = -bone_h if k == 1 else bone_h  
                    if node_matrix[i][j][k] is None: 
                        node_matrix[i][j][k] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            'Type': 'bone'
                        } 
                        node_matrix[i][j+4][k] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            'Type': 'bone'
                        } 
                        node_identifier = add_node(
                                fieldmodule, node_id, node_identifier, [a0, a1, a2, a3])

        x = c
        # Metacarpal
        node_id += 1
        y_val = [y, y+1, y+5]
        x_val = [(x+i, i/mc) for i in range(mc)]
        for k in [1, 2]:
            for j in y_val:
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a3 = -bone_h if k == 1 else bone_h  
                    if node_matrix[i][j][k] is None: 
                                    node_matrix[i][j][k] = {
                                        Node.VALUE_LABEL_VALUE: [node_id, a0, a1, a2, a3], 
                                        'Type': 'bone'
                                    } 
        x = c + mc
        # PP
        node_id += 1
        y_val = [y, y+1]
        x_val = [(x+i, i/pp) for i in range(pp)]
        for k in [1, 2]:
            for j in y_val:
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a3 = -bone_h if k == 1 else bone_h  
                    if node_matrix[i][j][k] is None: 
                                    node_matrix[i][j][k] = {
                                        Node.VALUE_LABEL_VALUE: [node_id, a0, a1, a2, a3], 
                                        'Type': 'bone'
                                    } 
        x = c + mc + pp
        # MP
        node_id += 1
        y_val = [y, y+1]
        x_val = [(x+i, i/mp) for i in range(mp)]
        for k in [1, 2]:
            for j in y_val:
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a3 = -bone_h if k == 1 else bone_h  
                    if node_matrix[i][j][k] is None: 
                                    node_matrix[i][j][k] = {
                                        Node.VALUE_LABEL_VALUE: [node_id, a0, a1, a2, a3], 
                                        'Type': 'bone'
                                    } 
        x = c + mc + pp + mp
        # DP
        node_id += 1
        y_val = [y, y+1]
        x_val = [(x+i, i) for i in range(dp)]
        for k in [1, 2]:
            for j in y_val:
                for i in x_val:
                    a1 = i[1]
                    i = i[0]
                    a2 = -bone_w if j == y else bone_w
                    a3 = -bone_h if k == 1 else bone_h  
                    if node_matrix[i][j][k] is None: 
                                    node_matrix[i][j][k] = {
                                        Node.VALUE_LABEL_VALUE: [node_id, a0, a1, a2, a3], 
                                        'Type': 'bone'
                                    } 
        x = c + mc + pp + mp + dp 
        y += 5
    # Thumb 
    # Metacarpal
    node_id += 1
    y = 22
    for k in [1, 2]:
        for j in [22]:
            for i in [1, 2]:
                a1 = 1/2 
                a2 = bone_w if i == 1 else -bone_w
                a3 = -bone_h if k == 1 else bone_h  
                node_matrix[i][j][k] = {
                                            Node.VALUE_LABEL_VALUE: [node_id, a0, a1, a2, a3], 
                                            'Type': 'bone'
                                        } 
    y += 1
    # PP
    node_id += 1
    for k in [1, 2]:
        for j in [y]:
            for i in [1, 2]:
                a1 = 0 if j == 23 else 1
                a2 = bone_w if i == 1 else -bone_w
                a3 = -bone_h if k == 1 else bone_h  
                node_matrix[i][j][k] = {
                                            Node.VALUE_LABEL_VALUE: [node_id, a0, a1, a2, a3], 
                                            'Type': 'bone'
                                        } 
    y += 1
    # DP
    node_id = 23
    for k in [1, 2]:
        for j in [y, y+1]:
            for i in [1, 2]:
                a1 = 0 if j == y else 1
                a2 = bone_w if i == 1 else -bone_w
                a3 = -bone_h if k == 1 else bone_h  
                node_matrix[i][j][k] = {
                                            Node.VALUE_LABEL_VALUE: [node_id, a0, a1, a2, a3], 
                                            'Type': 'bone'
                                        } 
    return node_matrix, node_identifier


def generate_external_node_matrix(fieldmodule, number_elements, node_identifier, node_matrix):
    c, mc, pp, mp, dp = number_elements
    # Finger elements are created in reverse order, from 5 (little) to 1 (thumb)
    # Palm skin nodes are sampled from an ellipse surrounding the internal node
    x, d1, d2, d3 = getNodeFieldParameters(fieldmodule, 'coordinates', 6)
    center = add(x, d2) # Ellipse center
    a = 5*magnitude(d2)
    b = 2*magnitude(d3)
    major_axis = mult([0, -1, 0], a)
    minor_axis = mult([0, 0, 1], b)
    outer_nodes = sampleEllipsePoints(center, major_axis, minor_axis, math.pi/2, 2*math.pi + math.pi/2, 10)
    upper_ellipse_x = [outer_nodes[0][i] for i in [8, 9, 0, 1, 2]]
    upper_ellipse_d2 = [outer_nodes[1][i] for i in [8, 9, 0, 1, 2]]
    lower_ellipse_x = [outer_nodes[0][i] for i in [7, 6, 5, 4, 3]]
    lower_ellipse_d2 = [outer_nodes[1][i] for i in [7, 6, 5, 4, 3]]
    ellipse_d3 = [sub(lower_ellipse_x[i],upper_ellipse_x[i]) for i in range(5)]
    parent_node = 1
    # Propagate a single node forward 
    a0 = 1
    # Finger 
    node_id = 0
    y = 1
    jj = 0
    # Carpal
    for f in range(4):
        node_id += 1
        x = 0
        y_val = [y, y+1] if f == 0 else [y+1]
        x_val = [x+i for i in range(c)]
        for j in y_val:
            for k in [1, 2]:
                ellipse_x = upper_ellipse_x if k == 2 else lower_ellipse_x
                ellipse_d2 = upper_ellipse_d2 if k == 2 else lower_ellipse_d2
                d1 = getNodeFieldParameters(fieldmodule, 'coordinates', parent_node)[1]
                d1 = mult(d1, 1/c)
                a3 = -1 if k == 1 else 1
                if j == 1 or j == 17:
                    a2 = -1 if j == 1 else 1
                    for i in x_val:
                        node_matrix[i][j+a2][k] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                            Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                                'Type': 'skin'
                        }
                        node_matrix[i][j][k+a3] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                            Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                                'Type': 'skin'
                        }
                        node_loc = ellipse_x[jj]
                        d2 = ellipse_d2[jj]
                        node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                        node_loc = add(node_loc, d1)
                        ellipse_x[jj] = node_loc
                else:
                    for i in x_val:
                        node_matrix[i][j][k+a3] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                            Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                                'Type': 'skin'
                        }
                        node_matrix[i][j+4][k+a3] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                            Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                                'Type': 'skin'
                        }
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
        y_val = [y, y+1] if f == 0 else [y+1]
        x_val = [x+i for i in range(n)]
        z_val = [1, 2]
        for j in y_val:
            for k in z_val:
                ellipse_x = upper_ellipse_x if k == 2 else lower_ellipse_x
                ellipse_d2 = upper_ellipse_d2 if k ==2 else lower_ellipse_d2
                d1 = getNodeFieldParameters(fieldmodule, 'coordinates', parent_node)[1]
                d1 = mult(d1, 1.0/n)
                a3 = -1 if k == 1 else 1
                if j == 1 or j == 17:
                    a2 = -1 if j == 1 else 1
                    for i in x_val:
                        node_matrix[i][j+a2][k] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                            Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                                'Type': 'skin'
                        }
                        node_matrix[i][j][k+a3] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                            Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                                'Type': 'skin'
                        }
                        node_loc = ellipse_x[jj]
                        d2 = ellipse_d2[jj]
                        node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                        node_loc = add(node_loc, d1)
                        ellipse_x[jj] = node_loc
                else:
                    for i in x_val:
                        node_matrix[i][j][k+a3] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                            Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                                'Type': 'skin'
                        }
                        node_matrix[i][j+4][k+a3] = {
                            Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                            Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                            Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                                'Type': 'skin'
                        }
                        node_loc = ellipse_x[jj]
                        d2 = ellipse_d2[jj]
                        d3 = ellipse_d3[jj] if i == x_val[-1] else None
                        node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2, d3)
                        node_loc = add(node_loc, d1)
                        ellipse_x[jj] = node_loc
            jj += 1
        y += 5
        parent_node += 5
    # Finger-palm connections
    y_val = [3, 8, 13]
    z_val = [1, 2]
    i = c+mc-1 # Last row of MC elements
    for j in y_val:
        for k in z_val:
            kk = 0 if k == 1 else 3
            node_id = node_matrix[i][j-1][kk][Node.VALUE_LABEL_VALUE][0]
            # print(i, j, k, node_id)
            # print(i, j+2, k, node_id)
            node_matrix[i][j][k] = {
                    Node.VALUE_LABEL_VALUE: [node_id, 1, 0, 0, 0], 
                    Node.VALUE_LABEL_D_DS1: [node_id, 0, 1, 0.25, 0], 
                    Node.VALUE_LABEL_D_DS2: [node_id, 0, 0, 0, 1], 
                    'Type': 'skin'
            }
            node_matrix[i][j+2][k] = {
                    Node.VALUE_LABEL_VALUE: [node_id, 1, 0, 0, 0], 
                    Node.VALUE_LABEL_D_DS1: [node_id, 0, 1, 0.25, 0], 
                    Node.VALUE_LABEL_D_DS2: [node_id, 0, 0, 0, -1], 
                    'Type': 'skin'
            }
    # Fingers 5 to 2 
    parent_node = 3
    y = 1
    n = pp
    jj = 0
    a = 2
    b = 2
    z_val = [1, 2]
    for f in range(4):
        # PP
        x = c + mc 
        n = pp
        y_val = [y, y+1]
        x_val = [x+i for i in range(n)]
        # Estimate ellipse
        center, d1, d2, d3 = getNodeFieldParameters(fieldmodule, 'coordinates', parent_node)
        major_axis = mult(d2, -a)
        minor_axis = mult(d3, b)
        ellipse = sampleEllipsePoints(center, major_axis, minor_axis, - math.pi/4, 2*math.pi - math.pi/4, 4)
        ellipse_x = [ellipse[0][i] for i in [0, 1, 3, 2]]
        ellipse_d2 = [ellipse[1][i] for i in [0, 1, 3, 2]]
        jj = 0
        for j in y_val:
            for k in z_val:
                d1 = getNodeFieldParameters(fieldmodule, 'coordinates', parent_node)[1]
                d1 = mult(d1, 1/n)
                a2 = -1 if j == y else 1
                a3 = -1 if k == 1 else 1  
                for i in x_val:
                    node_loc = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node_matrix[i][j+a2][k] = {
                        Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                        Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                        Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                            'Type': 'skin'
                    }
                    node_matrix[i][j][k+a3] = {
                        Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                        Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                        Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                            'Type': 'skin'
                    }
                    node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                    node_loc = add(node_loc, d1)
                    ellipse_x[jj] = node_loc
                jj += 1
        
        
        # MP
        parent_node += 1
        x = c + mc + pp
        n = mp
        jj = 0
        y_val = [y, y+1]
        x_val = [x+i for i in range(n)]
        # Estimate ellipse
        center, d1, d2, d3 = getNodeFieldParameters(fieldmodule, 'coordinates', parent_node)
        major_axis = mult(d2, -a)
        minor_axis = mult(d3, b)
        ellipse = sampleEllipsePoints(center, major_axis, minor_axis, -math.pi/4, 2*math.pi - math.pi/4, 4)
        ellipse_x = [ellipse[0][i] for i in [0, 1, 3, 2]]
        ellipse_d2 = [ellipse[1][i] for i in [0, 1, 3, 2]]
        jj = 0
        for j in y_val:
            for k in z_val:
                d1 = getNodeFieldParameters(fieldmodule, 'coordinates', parent_node)[1]
                d1 = mult(d1, 1/n)
                a2 = -1 if j == y else 1
                a3 = -1 if k == 1 else 1  
                for i in x_val:
                    node_loc = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node_matrix[i][j+a2][k] = {
                        Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                        Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                        Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                        'Type': 'skin'
                    }
                    node_matrix[i][j][k+a3] = {
                        Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                        Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                        Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0],  
                        'Type': 'skin'
                    }
                    node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                    node_loc = add(node_loc, d1)
                    ellipse_x[jj] = node_loc
                jj += 1
        # DP
        parent_node += 1
        x = c + mc + pp + mp
        n = dp
        jj = 0
        y_val = [y, y+1]
        x_val = [x+i for i in range(n)]
        # Estimate ellipse
        center, d1, d2, d3 = getNodeFieldParameters(fieldmodule, 'coordinates', parent_node)
        major_axis = mult(d2, -a)
        minor_axis = mult(d3, b)
        ellipse = sampleEllipsePoints(center, major_axis, minor_axis, -math.pi/4, 2*math.pi - math.pi/4, 4)
        ellipse_x = [ellipse[0][i] for i in [0, 1, 3, 2]]
        ellipse_d2 = [ellipse[1][i] for i in [0, 1, 3, 2]]
        jj = 0
        for j in y_val:
            for k in z_val:
                d1 = getNodeFieldParameters(fieldmodule, 'coordinates', parent_node)[1]
                d1 = mult(d1, 1/(n-1))
                a2 = -1 if j == y else 1
                a3 = -1 if k == 1 else 1  
                for i in x_val:
                    node_loc = ellipse_x[jj]
                    d2 = ellipse_d2[jj]
                    node_matrix[i][j+a2][k] = {
                        Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                        Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                        Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                            'Type': 'skin'
                    }
                    node_matrix[i][j][k+a3] = {
                        Node.VALUE_LABEL_VALUE: [node_identifier, 1, 0, 0, 0], 
                        Node.VALUE_LABEL_D_DS1: [node_identifier, 0, 1, 0, 0], 
                        Node.VALUE_LABEL_D_DS2: [node_identifier, 0, 0, 1, 0], 
                            'Type': 'skin'
                    }
                    node_identifier = add_skin_node_new(fieldmodule, node_identifier, node_loc, d1, d2)
                    node_loc = add(node_loc, d1)
                    ellipse_x[jj] = node_loc
                jj += 1
        y += 5
        parent_node += 3

    return node_matrix, node_identifier



def create_linear_cube_element(fieldmodule, element_identifier, scale_factor_matrix, x, y, z):
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    # 
    # Criteria to identify the z axis of the element
    corner_1 = scale_factor_matrix[x][y][z]
    corner_3 = scale_factor_matrix[x+1][y+1][z]
    corner_5 = scale_factor_matrix[x+1][y+1][z+1]
    if corner_1 is None or corner_3 is None or corner_5 is None:
        return -2
    corner_5_skin = True if corner_5['Type'] == 'skin' else False 
    corner_3_skin = True if corner_3['Type'] == 'skin' else False
    corner_1_skin = True if corner_1['Type'] == 'skin' else False 
    if not corner_1_skin and not corner_3_skin:
        ranges = [[0, 1], [0, 1], [0, 1]]
        order = [2, 1, 0]
    elif not corner_1_skin and  corner_3_skin:
        ranges = [[0, 1], [0, 1], [1, 0]]
        order = [1, 2, 0]
    elif  corner_1_skin and not corner_3_skin:
        ranges = [[0, 1], [1, 0], [0, 1]]
        order = [1, 2, 0]
    elif corner_1_skin and corner_3_skin:
        ranges = [[0, 1], [1, 0], [1, 0]]
        order = [2, 1, 0]
    is_bicubic = True if corner_1_skin or corner_3_skin or corner_5_skin else False
    # Obtained the ordered list of indices to parse through
    reordered_ranges = [ranges[idx] for idx in order]
    indices = []
    # Create the cartersian product
    for prod in product(*reordered_ranges):
        # Create a placeholder for [i, j, k]
        row = [0] * len(ranges)
        # Map the generated values back to their correct positions
        for i, val in enumerate(prod):
            original_axis = order[i]
            row[original_axis] = val
        indices.append(row)
    # Extract information from nodes matrix and create expression terms
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
    value_labels = [
            0, 
            Node.VALUE_LABEL_VALUE,
            Node.VALUE_LABEL_D_DS1, 
            Node.VALUE_LABEL_D_DS2, 
            Node.VALUE_LABEL_D_DS3
        ]
    value_label_names = [0, 'v', 'd1', 'd2', 'd3']
    local_node = 0
    for index in indices:
        et = []
        neg_et = []
        ret = []
        i, j, k = index
        local_node += 1
        virtual_node = scale_factor_matrix[x+i][y+j][z+k]
        if virtual_node is None:
            return -2 
        virtual_node = virtual_node[Node.VALUE_LABEL_VALUE]
        # Value expression terms
        for factor in range(len(virtual_node)):
            if factor == 0:
                # Check for node_ids
                global_node_id = virtual_node[factor]
                if global_node_id not in local_node_ids:
                    n_local_nodes += 1
                    local_node_ids[global_node_id] = n_local_nodes
            else:
                # Check for scale_factor_ids
                scale_factor = virtual_node[factor]
                if scale_factor not in global_to_local_scale_factor_ids:
                    n_scale_factors += 1
                    global_to_local_scale_factor_ids[scale_factor] = n_scale_factors
                et.append(
                    [local_node_ids[global_node_id], value_labels[factor], global_to_local_scale_factor_ids[scale_factor]]
                )
                ret.append(
                    [str(global_node_id).zfill(2), value_label_names[factor], str(scale_factor).zfill(4)]
                )
        value_expression_terms[local_node] = et
        readable_expression_terms[local_node] = ret
    local_to_global_node_ids = {value:key for key, value in local_node_ids.items()}
    local_to_global_scale_factor_ids = {value:key for key, value in global_to_local_scale_factor_ids.items()}
    if is_bicubic:
        # Create 'negative' value expression terms 
        for local_node in [1, 2, 3]: 
            neg_et = []
            red_et = []
            et = value_expression_terms[local_node]
            for term in et: 
                l_node_id = term[0]
                label = term[1]
                scale_factor_id = term[2]
                scale_factor = -local_to_global_scale_factor_ids[scale_factor_id]
                if scale_factor not in global_to_local_scale_factor_ids:
                    n_scale_factors += 1
                    global_to_local_scale_factor_ids[scale_factor] = n_scale_factors
                neg_et.append([l_node_id, label, global_to_local_scale_factor_ids[scale_factor]])
                # red_et.append(
                #     [str(local_to_global_node_ids[l_node_id]).zfill(2), value_label_names[label], str(scale_factor).zfill(4)]
                # )
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
            i, j, k = indices[local_node-1]
            virtual_node = scale_factor_matrix[x+i][y+j][z+k]
            virtual_node = virtual_node[label]
            for factor in range(len(virtual_node)):
                if factor == 0:
                    # Check for node_ids
                    global_node_id = virtual_node[factor]
                    if global_node_id not in local_node_ids:
                        n_local_nodes += 1
                        local_node_ids[global_node_id] = n_local_nodes
                else:
                    #Check for scale_factor_ids
                    scale_factor = virtual_node[factor]
                    if scale_factor not in global_to_local_scale_factor_ids:
                        n_scale_factors += 1
                        global_to_local_scale_factor_ids[scale_factor] = n_scale_factors
                    et.append(
                        [local_node_ids[global_node_id], value_labels[factor], global_to_local_scale_factor_ids[scale_factor]]
                    )
                    ret.append(
                        [str(global_node_id).zfill(2), value_label_names[factor], str(scale_factor).zfill(4)]
                    )
            d1_expression_terms[local_node] = et 
            # et = value_expression_terms[local_node][0]
            # l_node_id = et[0]
            # scale_factor_id = global_to_local_scale_factor_ids[1]
            # d1_expression_terms[local_node] = [[l_node_id, label, scale_factor_id]]
            # d1_expression_terms[local_node-4] = [[l_node_id, label, scale_factor_id]]
        # create d2 expression terms 
        d2_expression_terms[1] = value_expression_terms[3] + negative_value_ets[1]
        d2_expression_terms[2] = value_expression_terms[4] + negative_value_ets[2]
        d2_expression_terms[3] = value_expression_terms[3] + negative_value_ets[1]
        d2_expression_terms[4] = value_expression_terms[4] + negative_value_ets[2]
        label = Node.VALUE_LABEL_D_DS2
        for local_node in range(5, 9):
            et = []
            ret = []
            i, j, k = indices[local_node-1]
            virtual_node = scale_factor_matrix[x+i][y+j][z+k]
            virtual_node = virtual_node[label]
            for factor in range(len(virtual_node)):
                if factor == 0:
                    # Check for node_ids
                    global_node_id = virtual_node[factor]
                    if global_node_id not in local_node_ids:
                        n_local_nodes += 1
                        local_node_ids[global_node_id] = n_local_nodes
                else:
                    # Check for scale_factor_ids
                    scale_factor = virtual_node[factor]
                    if scale_factor not in global_to_local_scale_factor_ids:
                        n_scale_factors += 1
                        global_to_local_scale_factor_ids[scale_factor] = n_scale_factors
                    et.append(
                        [local_node_ids[global_node_id], value_labels[factor], global_to_local_scale_factor_ids[scale_factor]]
                    )
                    ret.append(
                        [str(global_node_id).zfill(2), value_label_names[factor], str(scale_factor).zfill(4)]
                    )
            d2_expression_terms[local_node] = et
        # for local_node in range(5, 9):
        #     et = value_expression_terms[local_node][0]
        #     l_node_id = et[0]
        #     label = Node.VALUE_LABEL_D_DS2
        #     scale_factor_id = global_to_local_scale_factor_ids[1]
        #     d2_expression_terms[local_node] = [[l_node_id, label, scale_factor_id]]
            # d2_expression_terms[local_node-4] = [[l_node_id, label, scale_factor_id]]
    # Create and remap eft
    if is_bicubic:
        # Bicubic linear element (skin)
        bicubic_linear_basis = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY)
        bicubic_linear_basis.setFunctionType(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        eft = mesh3d.createElementfieldtemplate(bicubic_linear_basis)
        setEftScaleFactorIds(eft, [], [], n_scale_factors)
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
        remapEftLocalNodes(eft, n_local_nodes, [1, 2, 3, 4, 5, 6, 7, 8])
        # Create element template
        etemplate = mesh3d.createElementtemplate()
        etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        result = etemplate.defineField(coordinates, -1, eft)
        if result != RESULT_OK:
            return result
    else:
        # Trilinear element (bone)
        trilinear_basis = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
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
            return result
    # Create element
    element = mesh3d.createElement(element_identifier, etemplate)
    node_ids = list(local_node_ids.keys())
    element.setNodesByIdentifier(eft, node_ids)
    scale_factors = list(global_to_local_scale_factor_ids.keys())
    element.setScaleFactors(eft, scale_factors)
    # element_identifier += 1
    return RESULT_OK


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
    d1 = node_params[1]
    d2 = [0, 0, 0]
    d3 = [0, 0, 0]
    setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
    node_identifier += 1
    return node_identifier

def add_skin_node(fieldmodule: Fieldmodule, bone_node_id: int, node_identifier: int, scale_factors: list, directions: list):
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
    d1, d2 = directions
    # d2 = set_magnitude(d2, magnitude(mult(node_params[2], scale_factors[2])))
    # d1 = set_magnitude(d1, magnitude(mult(node_params[1], 1-scale_factors[1])))
    d2 = set_magnitude(d2, magnitude(node_params[2])) if magnitude(d2) != 0 else d2
    # d1 = set_magnitude(d1, magnitude(node_params[1])) if magnitude(d1) != 0 else d2
    d3 = [0, 0, 0]
    setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
    node_identifier += 1
    return node_identifier

def create_thumb_nodes_new(fieldmodule, node_identifier, finger_dimensions, metacarpal_node_id, angle_degrees=45):
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
    d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)[1]
    d2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)[1]
    node_location = add(node_location, d2)
    # node_location = add(node_location, d2)
    node_location = add(node_location, mult(d1, 1/6))
    
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
        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
        x = add(x, d1)
        node_location = x
        # 
        finger_node_identifier += 1 
    node_identifier += 1
    return finger_node_identifier

def create_thumb_nodes(fieldmodule, node_identifier, finger_dimensions, metacarpal_node_id, angle_degrees=45):
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
    return finger_node_identifier

def create_finger_nodes_new(fieldmodule, node_identifier, finger_dimensions, starting_location):
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