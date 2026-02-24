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
            'Thumb angle': 30.0,
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
            'Thumb angle': (0.0, 30.0), 
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
        fieldcache = fieldmodule.createFieldcache()
        mesh3d = fieldmodule.findMeshByDimension(3)
        #################
        # Create bone nodes
        #################
        node_identifier = 1
        carpal_nodes = [[0, -0.2, 0]]

        for i in range(4):
            x = carpal_nodes[-1]
            x = add(x, mult([0, 1, 0], -0.8))
            carpal_nodes.append(x)
        # Fingers 2 - 4 (finger 1, thumb, is added later)
        finger_dimensions = [
            [0.3, 0.4, 0.2, 0.4],
            [2.4, 0.4, 0.2, 0.2],
            [1.4, 0.2, 0.2, 0.2], 
            [0.8, 0.2, 0.2, 0.2], 
            [0.6, 0.2, 0.2, 0.2]
        ]
        node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[0])
        finger_dimensions = [
            [0.3, 0.4, 0.2, 0.4],
            [2.3, 0.4, 0.2, 0.2],
            [1.6, 0.2, 0.2, 0.2], 
            [1, 0.2, 0.2, 0.2], 
            [0.6, 0.2, 0.2, 0.2]
        ]
        node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[1])
        finger_dimensions = [
            [0.3, 0.4, 0.2, 0.4],
            [2, 0.4, 0.2, 0.2],
            [1.5, 0.2, 0.2, 0.2], 
            [0.9, 0.2, 0.2, 0.2], 
            [0.7, 0.2, 0.2, 0.2]
        ]
        node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[2])
        finger_dimensions = [
            [0.3, 0.4, 0.2, 0.4],
            [1.9, 0.4, 0.2, 0.2],
            [1.2, 0.2, 0.2, 0.2], 
            [0.7, 0.2, 0.2, 0.2], 
            [0.6, 0.2, 0.2, 0.2]
        ]
        node_identifier = create_finger_nodes(fieldmodule, node_identifier, finger_dimensions, carpal_nodes[3])
        # Finger 1 (thumb)
        # Get the finger 1 metacarpal node
        finger_dimensions = [
            [1.9, 0.4, 0.2, 0.2],
            [1.2, 0.2, 0.2, 0.2], 
            [0.6, 0.2, 0.2, 0.2]
        ]
        node_identifier = create_thumb_nodes(fieldmodule, 21, finger_dimensions, 5, thumb_angle_degrees)

        # #################
        # # Create box elements
        # #################
        element_identifier = 1
        template_1_back, eft_1_back = get_elementtemplate_and_eft_1node_back(fieldmodule)
        template_2_back, eft_2_back = get_elementtemplate_and_eft_2node_back(fieldmodule)
        template_2_front_back, eft_2_front_back = get_elementtemplate_and_eft_2node_front_back(fieldmodule)
        template_3_back, eft_3_back = get_elementtemplate_and_eft_3node_1front_2back(fieldmodule)
        template_2_triangle, eft_2_triangle = get_elementtemplate_and_eft_2node_triangle(fieldmodule)
        template_4, eft_4 = get_elementtemplate_and_eft_4node_2front_2back(fieldmodule)

        element_identifier = create_bone_palm_elements(fieldmodule, element_identifier, 1, [1, 2])
        element_identifier = create_bone_finger_elements(fieldmodule, element_identifier, 9, [1, 1, 1])
        element_identifier = create_bone_palm_elements(fieldmodule, element_identifier, 2, [1, 2])
        element_identifier = create_bone_finger_elements(fieldmodule, element_identifier, 10, [1, 1, 1])
        element_identifier = create_bone_palm_elements(fieldmodule, element_identifier, 3, [1, 2])
        element_identifier = create_bone_finger_elements(fieldmodule, element_identifier, 11, [1, 1, 1])
        element_identifier = create_bone_palm_elements(fieldmodule, element_identifier, 4, [1, 2], last_finger=True)
        element_identifier = create_bone_finger_elements(fieldmodule, element_identifier, 12, [1, 1, 1])
        # Thumb 
        # element_identifier = create_bone_thumb_elements(fieldmodule, element_identifier, 22, 1, [2, 1, 1])
        #
        # Thumb
        
        element = mesh3d.createElement(element_identifier, template_4)
        element.setNodesByIdentifier(eft_4, [1, 1, 21, 5])
        element.setScaleFactors(eft_4, [1, -1, 0, 0])
        element_identifier += 1

        element = mesh3d.createElement(element_identifier, template_3_back)
        element.setNodesByIdentifier(eft_3_back, [21, 5, 21])
        element.setScaleFactors(eft_3_back, [1, -1, 0, 0.5])
        element_identifier += 1

        element = mesh3d.createElement(element_identifier, template_1_back)
        element.setNodesByIdentifier(eft_1_back, [21])
        element.setScaleFactors(eft_1_back, [1, -1, 0.5, 0.9])
        element_identifier += 1

        element = mesh3d.createElement(element_identifier, template_2_front_back)
        element.setNodesByIdentifier(eft_2_front_back, [21, 22])
        element.setScaleFactors(eft_2_front_back, [1, -1, 0.9, 0])
        element_identifier += 1

        element = mesh3d.createElement(element_identifier, template_2_triangle)
        element.setNodesByIdentifier(eft_2_triangle, [21, 5])
        element.setScaleFactors(eft_2_triangle, [1, -1, 0.45, 0.5])
        element_identifier += 1

        element = mesh3d.createElement(element_identifier, template_2_front_back)
        element.setNodesByIdentifier(eft_2_front_back, [22, 23])
        element.setScaleFactors(eft_2_front_back, [1, -1, 0, 0])
        element_identifier += 1

        element = mesh3d.createElement(element_identifier, template_1_back)
        element.setNodesByIdentifier(eft_1_back, [23])
        element.setScaleFactors(eft_1_back, [1, -1, 0, 1])
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


def get_elementtemplate_and_eft_1node_back(fieldmodule):
    """
    Docstring for getElementTempplateTwoNodeBlended
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh3d.createElementfieldtemplate(basis3d)
    # Matrix with corner coordinates as linear combinations of nodes
    # Matrix has 8 rows, with each row containing as many nodes as necessary 
    # With corresponding value labels [value, d1, d2, d3]
    matrix = [
        [[1, 3, 2, 2]],
        [[1, 4, 2, 2]],
        [[1, 3, 1, 2]],
        [[1, 4, 1, 2]],
        [[1, 3, 2, 1]],
        [[1, 4, 2, 1]],
        [[1, 3, 1, 1]],
        [[1, 4, 1, 1]],
    ]
    element_scale_factors = 4
    setEftScaleFactorIds(eft, [], [], element_scale_factors) 
    map_matrix_to_expression_terms_linear(matrix, eft)
    remapEftLocalNodes(eft, 1, [1, 1, 1, 1, 1, 1, 1, 1])
    etemplate = mesh3d.createElementtemplate()
    etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
    result = etemplate.defineField(coordinates, -1, eft)
    eftAndTemplate = []
    if result == RESULT_OK:
        eftAndTemplate = [etemplate, eft]
    return eftAndTemplate

def get_elementtemplate_and_eft_2node_back(fieldmodule):
    """
    Docstring for getElementTempplateTwoNodeBlended
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh3d.createElementfieldtemplate(basis3d)
    # Matrix with corner coordinates as linear combinations of nodes
    # Matrix has 8 rows, with each row containing as many nodes as necessary 
    # With corresponding value labels [value, d1, d2, d3]
    matrix = [
        [[0, 0, 0, 0], [1, 3, 1, 2]],
        [[0, 0, 0, 0], [1, 4, 1, 2]],
        [[1, 3, 1, 2], [0, 0, 0, 0]],
        [[1, 4, 1, 2], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 3, 1, 1]],
        [[0, 0, 0, 0], [1, 4, 1, 1]],
        [[1, 3, 1, 1], [0, 0, 0, 0]],
        [[1, 4, 1, 1], [0, 0, 0, 0]],
    ]
    element_scale_factors = 4
    setEftScaleFactorIds(eft, [], [], element_scale_factors) 
    map_matrix_to_expression_terms_linear(matrix, eft)
    remapEftLocalNodes(eft, 2, [1, 2, 1, 1, 1, 1, 1, 1])
    etemplate = mesh3d.createElementtemplate()
    etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
    result = etemplate.defineField(coordinates, -1, eft)
    eftAndTemplate = []
    if result == RESULT_OK:
        eftAndTemplate = [etemplate, eft]
    return eftAndTemplate

def get_elementtemplate_and_eft_2node_front_back(fieldmodule):
    """
    Docstring for getElementTempplateTwoNodeBlended
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh3d.createElementfieldtemplate(basis3d)
    # Matrix with corner coordinates as linear combinations of nodes
    # Matrix has 8 rows, with each row containing as many nodes as necessary 
    # With corresponding value labels [value, d1, d2, d3]
    matrix = [
        [[1, 3, 2, 2], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 4, 2, 2]],
        [[1, 3, 1, 2], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 4, 1, 2]],
        [[1, 3, 2, 1], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 4, 2, 1]],
        [[1, 3, 1, 1], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 4, 1, 1]],
    ]
    element_scale_factors = 4
    setEftScaleFactorIds(eft, [], [], element_scale_factors) 
    map_matrix_to_expression_terms_linear(matrix, eft)
    remapEftLocalNodes(eft, 2, [1, 2, 1, 1, 1, 1, 1, 1])
    etemplate = mesh3d.createElementtemplate()
    etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
    result = etemplate.defineField(coordinates, -1, eft)
    eftAndTemplate = []
    if result == RESULT_OK:
        eftAndTemplate = [etemplate, eft]
    return eftAndTemplate


def get_elementtemplate_and_eft_3node_triangle(fieldmodule):
    """
    Docstring for getElementTempplateTwoNodeBlended
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh3d.createElementfieldtemplate(basis3d)
    # Matrix with corner coordinates as linear combinations of nodes
    # Matrix has 8 rows, with each row containing as many nodes as necessary 
    # With corresponding value labels [value, d1, d2, d3]
    matrix = [
        [[1, 3, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 2]],
        [[1, 3, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 4, 1, 2], [0, 0, 0, 0]],
        [[1, 3, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 1]],
        [[1, 3, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 4, 1, 1], [0, 0, 0, 0]],
    ]
    element_scale_factors = 4
    setEftScaleFactorIds(eft, [], [], element_scale_factors) 
    map_matrix_to_expression_terms_linear(matrix, eft)
    remapEftLocalNodes(eft, 3, [1, 2, 3, 1, 1, 1, 1, 1])
    etemplate = mesh3d.createElementtemplate()
    etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
    result = etemplate.defineField(coordinates, -1, eft)
    eftAndTemplate = []
    if result == RESULT_OK:
        eftAndTemplate = [etemplate, eft]
    return eftAndTemplate


def get_elementtemplate_and_eft_2node_triangle(fieldmodule):
    """
    Docstring for getElementTempplateTwoNodeBlended
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh3d.createElementfieldtemplate(basis3d)
    # Matrix with corner coordinates as linear combinations of nodes
    # Matrix has 8 rows, with each row containing as many nodes as necessary 
    # With corresponding value labels [value, d1, d2, d3]
    matrix = [
        [[0, 0, 0, 0], [1, 0, 1, 2]],
        [[0, 0, 0, 0], [1, 3, 1, 2]],
        [[0, 0, 0, 0], [1, 0, 1, 2]],
        [[1, 4, 2, 2], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 0, 1, 1]],
        [[0, 0, 0, 0], [1, 3, 1, 1]],
        [[0, 0, 0, 0], [1, 0, 1, 1]],
        [[1, 4, 2, 1], [0, 0, 0, 0]],
    ]
    element_scale_factors = 4
    setEftScaleFactorIds(eft, [], [], element_scale_factors) 
    map_matrix_to_expression_terms_linear(matrix, eft)
    remapEftLocalNodes(eft, 2, [1, 2, 3, 1, 1, 1, 1, 1])
    etemplate = mesh3d.createElementtemplate()
    etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
    result = etemplate.defineField(coordinates, -1, eft)
    eftAndTemplate = []
    if result == RESULT_OK:
        eftAndTemplate = [etemplate, eft]
    return eftAndTemplate

def get_elementtemplate_and_eft_3node_1front_2back(fieldmodule):
    """
    Docstring for getElementTempplateTwoNodeBlended
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh3d.createElementfieldtemplate(basis3d)
    # Matrix with corner coordinates as linear combinations of nodes
    # Matrix has 8 rows, with each row containing as many nodes as necessary 
    # With corresponding value labels [value, d1, d2, d3]
    matrix = [
        [[0, 0, 0, 0], [1, 3, 1, 2], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 2, 2]],
        [[1, 3, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 2]],
        [[0, 0, 0, 0], [1, 3, 1, 1], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 2, 1]],
        [[1, 3, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 1]],
    ]
    element_scale_factors = 4
    setEftScaleFactorIds(eft, [], [], element_scale_factors) 
    map_matrix_to_expression_terms_linear(matrix, eft)
    remapEftLocalNodes(eft, 3, [1, 2, 3, 1, 1, 1, 1, 1])
    etemplate = mesh3d.createElementtemplate()
    etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
    result = etemplate.defineField(coordinates, -1, eft)
    eftAndTemplate = []
    if result == RESULT_OK:
        eftAndTemplate = [etemplate, eft]
    return eftAndTemplate


def get_elementtemplate_and_eft_4node_2front_2back(fieldmodule):
    """
    Docstring for getElementTempplateTwoNodeBlended
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh3d.createElementfieldtemplate(basis3d)
    # Matrix with corner coordinates as linear combinations of nodes
    # Matrix has 8 rows, with each row containing as many nodes as necessary 
    # With corresponding value labels [value, d1, d2, d3]
    matrix = [
        [[0, 0, 0, 0], [1, 3, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 2]],
        [[1, 3, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 2], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 3, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 1]],
        [[1, 3, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 4, 1, 1], [0, 0, 0, 0]],
    ]
    element_scale_factors = 4
    setEftScaleFactorIds(eft, [], [], element_scale_factors) 
    map_matrix_to_expression_terms_linear(matrix, eft)
    remapEftLocalNodes(eft, 4, [1, 2, 3, 4, 1, 1, 1, 1])
    etemplate = mesh3d.createElementtemplate()
    etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
    result = etemplate.defineField(coordinates, -1, eft)
    eftAndTemplate = []
    if result == RESULT_OK:
        eftAndTemplate = [etemplate, eft]
    return eftAndTemplate


def get_elementtemplate_and_eft(fieldmodule):
    """
    Docstring for getElementTempplateTwoNodeBlended
    """
    # Zinc setup
    mesh3d = fieldmodule.findMeshByDimension(3)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh3d.createElementfieldtemplate(basis3d)
    # Matrix with corner coordinates as linear combinations of nodes
    # Matrix has 8 rows, with each row containing as many nodes as necessary 
    # With corresponding value labels [value, d1, d2, d3]
    matrix = [
        [[0, 0, 0, 0]],
        [[0, 0, 0, 0]],
        [[0, 0, 0, 0]],
        [[0, 0, 0, 0]],
        [[0, 0, 0, 0]],
        [[0, 0, 0, 0]],
        [[0, 0, 0, 0]],
        [[0, 0, 0, 0]],
    ]
    element_scale_factors = 2
    setEftScaleFactorIds(eft, [], [], element_scale_factors) 
    map_matrix_to_expression_terms_linear(matrix, eft)
    remapEftLocalNodes(eft, 2, [1, 2, 1, 1, 1, 1, 1, 1])
    etemplate = mesh3d.createElementtemplate()
    etemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
    result = etemplate.defineField(coordinates, -1, eft)
    eftAndTemplate = []
    if result == RESULT_OK:
        eftAndTemplate = [etemplate, eft]
    return eftAndTemplate

def map_matrix_to_expression_terms_linear(matrix, eft):
    ln = 1
    value_labels = [
        Node.VALUE_LABEL_VALUE,
        Node.VALUE_LABEL_D_DS1, 
        Node.VALUE_LABEL_D_DS2, 
        Node.VALUE_LABEL_D_DS3
    ]
    # Set expression terms according to matrix
    for row in matrix:
        expression_terms = []
        for node in range(len(row)):
            for value in range(4):
                scale_factor = row[node][value]
                if scale_factor != 0:
                    # Value is 1, DS1 is 2, DS2 is 3, DS3 is 4
                    expression_terms.append(
                        (node+1, value_labels[value], scale_factor)
                    )
        remapEftNodeValueLabelWithNodes(
                    eft, 
                    ln, 
                    Node.VALUE_LABEL_VALUE,                   
                    expression_terms
                    )
        ln += 1
    if eft.validate():
        return expression_terms
    else:
        return -1

def create_bone_thumb_elements(fieldmodule, element_identifier, metacarpal_node_id, carpal_node_id, element_numbers):
    '''
    Docstring for create_bone_palm_elements

    :param fieldmodule: Description
    :param element_identifier: Description
    :param carpal_node_id: Description
    :param element_numbers: Description
    '''
    #################
    # Zinc setup
    #################
    mesh = fieldmodule.findMeshByDimension(3)
    # Gather efts
    template_4, eft_4 = get_elementtemplate_and_eft_4node_2front_2back(fieldmodule)
    template_3_back, eft_3_back = get_elementtemplate_and_eft_3node_1front_2back(fieldmodule)
    template_2_triangle, eft_2_triangle = get_elementtemplate_and_eft_2node_triangle(fieldmodule)
    template_2_front_back, eft_2_front_back = get_elementtemplate_and_eft_2node_front_back(fieldmodule)
    template_1_back, eft_1_back = get_elementtemplate_and_eft_1node_back(fieldmodule)

    #################
    # Metacarpal elements 
    #################
    # First triangle element
    node_ids = [carpal_node_id, carpal_node_id, metacarpal_node_id, carpal_node_id+4]
    element = mesh.createElement(element_identifier, template_4)
    element.setNodesByIdentifier(eft_4, node_ids)
    element.setScaleFactors(eft_4, [1, -1, 0, 0])
    element_identifier += 1
    # Box element
    node_ids = [metacarpal_node_id, carpal_node_id+4, metacarpal_node_id]
    n_elements = element_numbers[0]
    # Create uniform partition of the unit interval (tip box is always 10% of d1)
    start_val = [(i/n_elements)*0.9 for i in range(n_elements)]
    end_val = [(i/n_elements)*0.9 for i in range(1, n_elements+1)]
    for i in range(n_elements):
        element = mesh.createElement(element_identifier, template_3_back)
        element.setNodesByIdentifier(eft_3_back, node_ids)
        element.setScaleFactors(eft_3_back, [1, -1, start_val[i], end_val[i]])
        element_identifier += 1
    # Tip of box element
    node_ids = [metacarpal_node_id, metacarpal_node_id+1]
    element = mesh.createElement(element_identifier, template_2_front_back)
    element.setNodesByIdentifier(eft_2_front_back, node_ids)
    element.setScaleFactors(eft_2_front_back, [1, -1, end_val[-1], 0])
    element_identifier += 1
    # Second Triangle element
    node_ids = [metacarpal_node_id, carpal_node_id+4]
    element = mesh.createElement(element_identifier, template_2_triangle)
    element.setNodesByIdentifier(eft_2_triangle, node_ids)
    element.setScaleFactors(eft_2_triangle, [1, -1, end_val[0], end_val[0]])
    element_identifier += 1
    #################
    # Proximal phalanx elements
    #################
    # Box element
    # Setup elements
    node_ids = [metacarpal_node_id+1, metacarpal_node_id+2]
    n_elements = element_numbers[1]
    # Create uniform partition
    start_val = [(i/n_elements) for i in range(n_elements)]
    end_val = [(i/n_elements)for i in range(1, n_elements+1)]
    for j in range(n_elements):
        element = mesh.createElement(element_identifier, template_2_front_back)
        element.setNodesByIdentifier(eft_2_front_back, node_ids)
        element.setScaleFactors(eft_2_front_back, [1, -1, 0, 0])
        element_identifier += 1
    
    n_elements = element_numbers[1]
    #################
    # Distal phalanx elements
    #################
    node_ids = [metacarpal_node_id+2]
    n_elements = element_numbers[-1]
    # Create uniform partition
    start_val = [(i/n_elements) for i in range(n_elements)]
    end_val = [(i/n_elements)for i in range(1, n_elements+1)]
    for i in range(n_elements):
        element = mesh.createElement(element_identifier, template_1_back)
        element.setNodesByIdentifier(eft_1_back, node_ids)
        element.setScaleFactors(eft_1_back, [1, -1, start_val[i], end_val[i]])
        element_identifier += 1
    return element_identifier

def create_bone_palm_elements(fieldmodule, element_identifier, carpal_node_id, element_numbers, last_finger = False):
    '''
    Docstring for create_bone_palm_elements

    :param fieldmodule: Description
    :param element_identifier: Description
    :param carpal_node_id: Description
    :param element_numbers: Description
    '''
    # Zinc setup
    mesh = fieldmodule.findMeshByDimension(3)
    # Gather efts (boxes on the last finger only use half the usual nodes)
    if last_finger:
        template_carpal, eft_carpal = get_elementtemplate_and_eft_2node_front_back(fieldmodule)
        template_metacarpal, eft_metacarpal = get_elementtemplate_and_eft_1node_back(fieldmodule)
        template_metacarpal_tip, eft_metacarpal_tip = get_elementtemplate_and_eft_2node_front_back(fieldmodule)
    else:
        template_carpal, eft_carpal = get_elementtemplate_and_eft_4node_2front_2back(fieldmodule)
        template_metacarpal, eft_metacarpal = get_elementtemplate_and_eft_2node_back(fieldmodule)
        template_metacarpal_tip, eft_metacarpal_tip = get_elementtemplate_and_eft_3node_1front_2back(fieldmodule)
    # Setup elements
    node_id = carpal_node_id
    if last_finger: 
        node_ids = [node_id, node_id+4]
    else:
        node_ids = [node_id, node_id+1, node_id+4, node_id+5]
    # Carpal box element (hardcoded to 1 box for now)
    element = mesh.createElement(element_identifier, template_carpal)
    element.setNodesByIdentifier(eft_carpal, node_ids)
    element.setScaleFactors(eft_carpal, [1, -1, 0, 0])
    element_identifier += 1
    # Metacarpal box elements (multiple 2-node boxes + 1 3-node box at the tip)
    node_id = carpal_node_id + 4
    if last_finger: 
        node_ids = [node_id, node_id+4]
    else:
        node_ids = [node_id, node_id+1, node_id+4]
    n_elements = element_numbers[1]
    # Create uniform partition of the unit interval (tip box is always 10% of d1)
    start_val = [(i/n_elements)*0.9 for i in range(n_elements)]
    end_val = [(i/n_elements)*0.9 for i in range(1, n_elements+1)]
    for i in range(n_elements):
        element = mesh.createElement(element_identifier, template_metacarpal)
        element.setNodesByIdentifier(eft_metacarpal, node_ids[0:-1])
        element.setScaleFactors(eft_metacarpal, [1, -1, start_val[i], end_val[i]])
        element_identifier += 1
    element = mesh.createElement(element_identifier, template_metacarpal_tip)
    element.setNodesByIdentifier(eft_metacarpal_tip, node_ids)
    element.setScaleFactors(eft_metacarpal_tip, [1, -1, 0.9, 0])
    element_identifier += 1
    return element_identifier


def create_bone_finger_elements(fieldmodule, element_identifier, prox_phalanx_node_id, element_numbers):
    '''
    Docstring for create_bone_finger_elements

    :param fieldmodule: Description
    :param element_identifier: Description
    :param carpal_node_id: Description
    :param element_numbers: Description
    '''
    # Zinc setup
    mesh = fieldmodule.findMeshByDimension(3)
    # Gather efts
    template_phalanx, eft_phalanx = get_elementtemplate_and_eft_2node_front_back(fieldmodule)
    template_d_phalanx, eft_d_phalanx = get_elementtemplate_and_eft_1node_back(fieldmodule)
    # Proximal (and median) phalanx box element 
    node_id = prox_phalanx_node_id
    # The for loop takes care of the thumb case as well (no median phalanx)
    for i in range(len(element_numbers)-1):
        # Setup elements
        node_ids = [node_id, node_id+4]
        n_elements = element_numbers[i]
        # Create uniform partition
        start_val = [(i/n_elements) for i in range(n_elements)]
        end_val = [(i/n_elements)for i in range(1, n_elements+1)]
        for j in range(n_elements):
            element = mesh.createElement(element_identifier, template_phalanx)
            element.setNodesByIdentifier(eft_phalanx, node_ids)
            element.setScaleFactors(eft_phalanx, [1, -1, 0, 0])
            element_identifier += 1
        node_id = node_id + 4
    # Distal phalanx 
    # Setup elements
    # node_id = node_id + 4
    node_ids = [node_id]
    n_elements = element_numbers[-1]
    # Create uniform partition
    start_val = [(i/n_elements) for i in range(n_elements)]
    end_val = [(i/n_elements)for i in range(1, n_elements+1)]
    for i in range(n_elements):
        element = mesh.createElement(element_identifier, template_d_phalanx)
        element.setNodesByIdentifier(eft_d_phalanx, node_ids)
        element.setScaleFactors(eft_d_phalanx, [1, -1, start_val[i], end_val[i]])
        element_identifier += 1
    return element_identifier

def create_thumb_nodes(fieldmodule, node_identifier, finger_dimensions, metacarpal_node_id, angle_degrees=10):
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
