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
        }
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
        ]

    @classmethod
    def checkOptions(cls, options):
        dependent_changes = False
        return dependent_changes

    @classmethod
    def generateBaseMesh(cls, region: Region, options):
        """

        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        return: list of AnnotationGroup, 
        """

        #################
        # Setup zinc 
        #################

        fieldmodule = region.getFieldmodule()
        coordinates = find_or_create_field_coordinates(fieldmodule)
        fieldcache = fieldmodule.createFieldcache()

        #################
        # Create nodes template
        #################

        value_labels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                        Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
                        Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3]
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        for value_label in value_labels[1:]:
            nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)

        #################
        # Create nodes 
        #################

        nodeIdentifier = 1 
        n = nodes.createNode(nodeIdentifier, nodetemplate)
        fieldcache.setNode(n)
        x0 = [0, 0, 0]
        d1 = [2, 0, 0]
        d2 = [0, 2, 0]
        d3 = [0, 0, 2]
        setNodeFieldParameters(coordinates, fieldcache, x0, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, 2)), mult(d2, 3))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, 2)), mult(d2, 1))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, 2)), mult(d2, -1))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, 2)), mult(d2,  -3))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, 4)), mult(d2, 3))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, 4)), mult(d2, 1))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, 4)), mult(d2, -1))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, 4)), mult(d2,  -3))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # a = 8
        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, a)), mult(d2, 3))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, a)), mult(d2, 1))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, a)), mult(d2, -1))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # nodeIdentifier += 1
        # n = nodes.createNode(nodeIdentifier, nodetemplate)
        # fieldcache.setNode(n)
        # x = add(add(x0, mult(d1, a)), mult(d2,  -3))
        # setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        # Line
        nodeIdentifier += 1
        n = nodes.createNode(nodeIdentifier, nodetemplate)
        fieldcache.setNode(n)
        x = add(add(x0, mult(d1, 1)), mult(d2,  0))
        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        nodeIdentifier += 1
        n = nodes.createNode(nodeIdentifier, nodetemplate)
        fieldcache.setNode(n)
        x = add(add(x0, mult(d1, 2)), mult(d2,  0))
        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        nodeIdentifier += 1
        n = nodes.createNode(nodeIdentifier, nodetemplate)
        fieldcache.setNode(n)
        x = add(add(x0, mult(d1, 4)), mult(d2,  0))
        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)

        #################
        # Create 1D elements 
        #################

        mesh = fieldmodule.findMeshByDimension(1)
        basis = fieldmodule.createElementbasis(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        eft = mesh.createElementfieldtemplate(basis)
        elementtemplate = mesh.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
        result = elementtemplate.defineField(coordinates, -1, eft)
        elementIdentifier = 0


        #################
        # 2D Plate Template (1 node)
        #################
        mesh2d = fieldmodule.findMeshByDimension(2)
        basis2d = fieldmodule.createElementbasis(2, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        eft_plate_1 = mesh2d.createElementfieldtemplate(basis2d)
        # Here is where we start messing with the eft
        setEftScaleFactorIds(eft_plate_1, [], [], 1) 
        ln = 1
        matrix = [
            [[], [], [1]], 
            [[], [], []], 
            [[], [1], [1]],
            [[], [1], []]
        ]
        eft_plate_1.setNumberOfLocalNodes(5)
        for n in range(4):
            remapEftNodeValueLabelWithNodes(
                        eft_plate_1, 
                        ln, 
                        Node.VALUE_LABEL_VALUE,                   
                        [
                            (1, Node.VALUE_LABEL_VALUE, matrix[n][0]),
                            (1, Node.VALUE_LABEL_D_DS2, matrix[n][1]),
                            (1, Node.VALUE_LABEL_D_DS3, matrix[n][2]),
                        ]
                        )
            # remapEftNodeValueLabel(
            #             eft_plate_1, 
            #             [ln], 
            #             Node.VALUE_LABEL_VALUE,                   
            #             [
            #                 (Node.VALUE_LABEL_VALUE, matrix[n][0]),
            #                 (Node.VALUE_LABEL_D_DS2, matrix[n][1]),
            #                 (Node.VALUE_LABEL_D_DS3, matrix[n][2]),
            #             ]
            #             )
            ln += 1
        remapEftLocalNodes(eft_plate_1, 1, [1, 1, 1, 1, 1])
        eft_plate_1.validate()
        etemplate_plate_1 = mesh2d.createElementtemplate()
        etemplate_plate_1.setElementShapeType(Element.SHAPE_TYPE_SQUARE)
        result = etemplate_plate_1.defineField(coordinates, -1, eft_plate_1)


        #################
        # 3D Box Template (1 node)
        #################
        mesh3d = fieldmodule.findMeshByDimension(3)
        basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        eft_box_1 = mesh3d.createElementfieldtemplate(basis3d)
        # Here is where we start messing with the eft
        setEftScaleFactorIds(eft_box_1, [], [], 6) 
        ln = 1
        for n3 in [5, 6]:
            for n2 in [3, 4]:
                for n1 in [1, 2]:
                    remapEftNodeValueLabel(
                        eft_box_1, 
                        [ln], 
                        Node.VALUE_LABEL_VALUE,                   
                        [
                            (Node.VALUE_LABEL_VALUE, []),
                            (Node.VALUE_LABEL_D_DS1, n1),
                            (Node.VALUE_LABEL_D_DS2, n2),
                            (Node.VALUE_LABEL_D_DS3, n3)
                        ]
                        )
                    ln += 1
        remapEftLocalNodes(eft_box_1, 1, [1, 1, 1, 1, 1, 1, 1, 1])
        etemplate_box_1 = mesh3d.createElementtemplate()
        etemplate_box_1.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        result = etemplate_box_1.defineField(coordinates, -1, eft_box_1)

        #################
        # 3D Box Template (2 nodes)
        #################
        mesh3d = fieldmodule.findMeshByDimension(3)
        basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        eft_box_2 = mesh3d.createElementfieldtemplate(basis3d)
        # Here is where we start messing with the eft
        setEftScaleFactorIds(eft_box_2, [], [], 6) 
        ln = 1
        for n3 in [3, 4]:
            for n2 in [1, 2]:
                for n1 in [5, 6]:
                    remapEftNodeValueLabel(
                        eft_box_2, 
                        [ln], 
                        Node.VALUE_LABEL_VALUE,                   
                        [
                            (Node.VALUE_LABEL_VALUE, []),
                            (Node.VALUE_LABEL_D_DS2, [n1, n2]),
                            # (Node.VALUE_LABEL_D_DS2, n2),
                            (Node.VALUE_LABEL_D_DS3, n3)
                        ]
                        )
                    ln += 1
        remapEftLocalNodes(eft_box_2, 2, [1, 2, 1, 2, 1, 2, 1, 2])
        etemplate_box_2 = mesh3d.createElementtemplate()
        etemplate_box_2.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        result = etemplate_box_2.defineField(coordinates, -1, eft_box_2)

        #################
        # 3D Box Template (2 nodes - blended)
        #################

        mesh3d = fieldmodule.findMeshByDimension(3)
        basis3d = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        eft_box_3 = mesh3d.createElementfieldtemplate(basis3d)
        # Here is where we start messing with the eft
        setEftScaleFactorIds(eft_box_3, [], [], 8) 
        ln = 1
        matrix = [
            [1, 2, 3, 4, 3, 4],
            [5, 6, 7, 8, 7, 8],
            [1, 2, 1, 2, 3, 4],
            [5, 6, 5, 6, 7, 8],
            [1, 2, 3, 4, 1, 2],
            [5, 6, 7, 8, 5, 6],
            [1, 2, 1, 2, 1, 2],
            [5, 6, 5, 6, 5, 6],


            # [1, 2, 1, 2, 3, 4],
            # [5, 6, 5, 6, 7, 8],
            # [1, 2, 3, 4, 3, 4],
            # [5, 6, 7, 8, 7, 8],
            # [1, 2, 1, 2, 1, 2],
            # [5, 6, 5, 6, 5, 6],
            # [1, 2, 3, 4, 1, 2],
            # [5, 6, 7, 8, 5, 6],
        ]
        for n in range(8):
            remapEftNodeValueLabelWithNodes(
                        eft_box_3, 
                        ln, 
                        Node.VALUE_LABEL_VALUE,                   
                        [
                            (1, Node.VALUE_LABEL_VALUE, [matrix[n][0]]),
                            (2, Node.VALUE_LABEL_VALUE, [matrix[n][1]]),
                            (1, Node.VALUE_LABEL_D_DS2, [matrix[n][2]]),
                            (2, Node.VALUE_LABEL_D_DS2, [matrix[n][3]]),
                            (1, Node.VALUE_LABEL_D_DS3, [matrix[n][4]]),
                            (2, Node.VALUE_LABEL_D_DS3, [matrix[n][5]])
                        ]
                        )
            ln += 1
        remapEftLocalNodes(eft_box_3, 2, [1, 2, 1, 1, 1, 1, 1, 1])
        etemplate_box_3 = mesh3d.createElementtemplate()
        etemplate_box_3.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        result = etemplate_box_3.defineField(coordinates, -1, eft_box_3)

        #################
        # Box Elements
        #################

        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_1)
        # element.setNodesByIdentifier(eft_box_1, [1])
        # element.setScaleFactors(eft_box_1, [0, 1, -4, -2, -1, 1])
        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_1)
        # element.setNodesByIdentifier(eft_box_1, [1])
        # element.setScaleFactors(eft_box_1, [0, 1, -2, 0, -1, 1])
        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_1)
        # element.setNodesByIdentifier(eft_box_1, [1])
        # element.setScaleFactors(eft_box_1, [0, 1, 0, 2,-1, 1])
        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_1)
        # element.setNodesByIdentifier(eft_box_1, [1])
        # element.setScaleFactors(eft_box_1, [0, 1, 2, 4,-1, 1])
        # elementIdentifier += 1


        # element = mesh3d.createElement(elementIdentifier, etemplate_box_2)
        # element.setNodesByIdentifier(eft_box_2, [1, 2])
        # element.setScaleFactors(eft_box_2, [-1, 1, -1, 1, 3, 0])
        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_2)
        # element.setNodesByIdentifier(eft_box_2, [1, 3])
        # element.setScaleFactors(eft_box_2, [-1, 1, -1, 1, 1, 0])
        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_2)
        # element.setNodesByIdentifier(eft_box_2, [1, 4])
        # element.setScaleFactors(eft_box_2, [-1, 1, -1, 1, -1, 0])
        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_2)
        # element.setNodesByIdentifier(eft_box_2, [1, 5])
        # element.setScaleFactors(eft_box_2, [-1, 1, -1, 1, -3, 0])
        # elementIdentifier += 1
        
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_2)
        # element.setNodesByIdentifier(eft_box_2, [2, 6])
        # element.setScaleFactors(eft_box_2, [-1, 1, -1, 1, 0, 0])
        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_2)
        # element.setNodesByIdentifier(eft_box_2, [3, 7])
        # element.setScaleFactors(eft_box_2, [-1, 1, -1, 1, 0, 0])
        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_2)
        # element.setNodesByIdentifier(eft_box_2, [4, 8])
        # element.setScaleFactors(eft_box_2, [-1, 1, -1, 1, 0, 0])
        # elementIdentifier += 1
        # element = mesh3d.createElement(elementIdentifier, etemplate_box_2)
        # element.setNodesByIdentifier(eft_box_2, [5, 9])
        # element.setScaleFactors(eft_box_2, [-1, 1, -1, 1, 0, 0])
        a = 0.0
        b = 0.25
        scale_factors = [1-a, a, a-1, -a, 1-b, b, b-1, -b]
        element = mesh3d.createElement(elementIdentifier, etemplate_box_3)
        element.setNodesByIdentifier(eft_box_3, [1, 2])
        element.setScaleFactors(eft_box_3, scale_factors)
        elementIdentifier += 1
        a = 0.25
        b = 0.5
        scale_factors = [1-a, a, a-1, -a, 1-b, b, b-1, -b]
        element = mesh3d.createElement(elementIdentifier, etemplate_box_3)
        element.setNodesByIdentifier(eft_box_3, [1, 2])
        element.setScaleFactors(eft_box_3, scale_factors)
        elementIdentifier += 1
        a = 0.5
        b = 0.75
        scale_factors = [1-a, a, a-1, -a, 1-b, b, b-1, -b]
        element = mesh3d.createElement(elementIdentifier, etemplate_box_3)
        element.setNodesByIdentifier(eft_box_3, [1, 2])
        element.setScaleFactors(eft_box_3, scale_factors)
        elementIdentifier += 1
        a = 0.75
        b = 1.0
        scale_factors = [1-a, a, a-1, -a, 1-b, b, b-1, -b]
        element = mesh3d.createElement(elementIdentifier, etemplate_box_3)
        element.setNodesByIdentifier(eft_box_3, [1, 2])
        element.setScaleFactors(eft_box_3, scale_factors)
        elementIdentifier += 1

        # element = mesh3d.createElement(elementIdentifier, etemplate_box_3)
        # element.setNodesByIdentifier(eft_box_3, [2, 6])
        # element.setScaleFactors(eft_box_3, [0.5, -0.5, 1, -1])

        # element = mesh3d.createElement(elementIdentifier, etemplate_plate_1)
        # element.setNodesByIdentifier(eft_plate_1, [1])
        # element.setScaleFactors(eft_plate_1, [-1])
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
