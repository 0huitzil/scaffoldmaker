"""
Generates a 3D body coordinates using tube network mesh.
"""
from cmlibs.maths.vectorops import add, cross, mult, set_magnitude, sub, magnitude, axis_angle_to_rotation_matrix, matrix_vector_mult, matrix_mult, dot, angle 
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.utils.zinc.field import Field, find_or_create_field_coordinates, find_or_create_field_group, find_or_create_field_stored_string, find_or_create_field_finite_element
from cmlibs.utils.zinc.finiteelement import get_maximum_node_identifier
from cmlibs.utils.zinc.region import copy_fitting_data
from cmlibs.zinc.element import Element
from cmlibs.zinc.node import Node
from scaffoldfitter.fitter import Fitter as GeometryFitter
from scaffoldfitter.fitterstepalign import FitterStepAlign
from scaffoldmaker.annotation.annotationgroup import (
    AnnotationGroup, findOrCreateAnnotationGroupForTerm, getAnnotationGroupForTerm, evaluateAnnotationMarkerNearestMeshLocation)
from scaffoldmaker.annotation.body_terms import get_body_term, marker_name_in_terms
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils.interpolation import (
    computeCubicHermiteEndDerivative, getCubicHermiteArcLength, interpolateLagrangeHermiteDerivative,
    sampleCubicHermiteCurvesSmooth, smoothCubicHermiteDerivativesLine)
from scaffoldmaker.utils.networkmesh import NetworkMesh
from scaffoldmaker.utils.tubenetworkmesh import BodyTubeNetworkMeshBuilder, TubeNetworkMeshGenerateData
from scaffoldmaker.utils.human_network_layout import constructNetworkLayoutStructure, humanElementCounts
from scaffoldmaker.utils.zinc_utils import generate_datapoints, find_or_create_field_zero_fibres
# from scaffoldmaker.utils.human_network_layout import 
import math
import logging
logger = logging.getLogger(__name__)

class MeshType_1d_human_body_network_layout1(MeshType_1d_network_layout1):
    """
    Defines body network layout.
    """

    @classmethod
    def getName(cls):
        return "1D Human Body Network Layout 1"

    @classmethod
    def getParameterSetNames(cls):
        return ["Default"]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {}
        options["Base parameter set"] = parameterSetName
        options["Structure"] = constructNetworkLayoutStructure(humanElementCounts)
        options["Define inner coordinates"] = True
        options["Head depth"] = 2.0
        options["Head length"] = 2.2
        options["Head width"] = 2.0
        options["Neck length"] = 1.3
        options["Shoulder drop"] = 1.0
        options["Shoulder width"] = 5.0
        options["Left shoulder flexion degrees"] = 0.0
        options["Right shoulder flexion degrees"] = 0.0
        options["Left shoulder abduction degrees"] = 10.0
        options["Right shoulder abduction degrees"] = 10.0
        options["Left elbow flexion degrees"] = 0.0
        options["Right elbow flexion degrees"] = 0.0
        options["Left arm rotation degrees"] = 0.0
        options["Right arm rotation degrees"] = 0.0
        options["Arm length"] = 7.5
        options["Arm top diameter"] = 1.0
        options["Arm twist angle degrees"] = 0.0
        options["Wrist thickness"] = 0.5
        options["Wrist width"] = 0.7
        options["Left wrist flexion degrees"] = 0.0
        options["Right wrist flexion degrees"] = 0.0
        options["Left wrist deviation degrees"] = 0.0
        options["Right wrist deviation degrees"] = 0.0
        options["Hand length"] = 2.0
        options["Hand thickness"] = 0.2
        options["Hand width"] = 1.0
        options["Thorax length"] = 2.5 
        options["Abdomen length"] = 3.0
        options["Torso depth"] = 2.5
        options["Torso width"] = 3.2
        options["Pelvis drop"] = 1
        options["Pelvis width"] = 2.0
        options["Left leg abduction degrees"] = 10.0
        options["Right leg abduction degrees"] = 10.0
        options["Left hip flexion degrees"] = 0.0
        options["Right hip flexion degrees"] = 0.0
        options["Leg length"] = 11.0
        options["Leg top diameter"] = 2.0
        options["Leg bottom diameter"] = 0.7
        options["Left knee flexion degrees"] = 0.0
        options["Right knee flexion degrees"] = 0.0
        options["Left ankle flexion degrees"] = 90.0
        options["Right ankle flexion degrees"] = 90.0
        options["Foot height"] = 1.25
        options["Foot length"] = 2.0
        options["Foot thickness"] = 0.3
        options["Foot width"] = 1.0
        options["Inner proportion default"] = 0.7
        options["Inner proportion head"] = 0.35
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Head depth",
            "Head length",
            "Head width",
            "Neck length",
            "Shoulder drop",
            "Shoulder width",
            "Left shoulder abduction degrees",
            "Right shoulder abduction degrees",
            "Left shoulder flexion degrees",
            "Right shoulder flexion degrees",
            "Left arm rotation degrees",
            "Right arm rotation degrees",
            "Left elbow flexion degrees",
            "Right elbow flexion degrees",
            "Left wrist flexion degrees",
            "Right wrist flexion degrees",
            "Left wrist deviation degrees",
            "Right wrist deviation degrees",
            "Arm length",
            "Arm top diameter",
            "Arm twist angle degrees",
            "Wrist thickness",
            "Wrist width",
            "Hand length",
            "Hand thickness",
            "Hand width",
            "Thorax length",
            "Abdomen length",
            "Torso depth",
            "Torso width",
            "Pelvis drop",
            "Pelvis width",
            "Left leg abduction degrees",
            "Right leg abduction degrees",
            "Left hip flexion degrees",
            "Right hip flexion degrees",
            "Leg length",
            "Leg top diameter",
            "Leg bottom diameter",
            "Left knee flexion degrees",
            "Right knee flexion degrees",
            "Foot height",
            "Foot length",
            "Foot thickness",
            "Foot width",
            "Left ankle flexion degrees",
            "Right ankle flexion degrees",
            "Inner proportion default",
            "Inner proportion head"
        ]

    @classmethod
    def checkOptions(cls, options):
        dependentChanges = False
        for key in [
            "Head depth",
            "Head length",
            "Head width",
            "Neck length",
            "Shoulder drop",
            "Shoulder width",
            "Arm length",
            "Arm top diameter",
            "Wrist thickness",
            "Wrist width",
            "Hand length",
            "Hand thickness",
            "Hand width",
            "Pelvis drop",
            "Pelvis width",
            "Thorax length",
            "Abdomen length",
            "Torso depth",
            "Torso width",
            "Leg length",
            "Leg top diameter",
            "Leg bottom diameter",
            "Foot height",
            "Foot length",
            "Foot thickness",
            "Foot width"
        ]:
            if options[key] < 0.1:
                options[key] = 0.1
        for key in [
            "Inner proportion default",
            "Inner proportion head"
        ]:
            if options[key] < 0.1:
                options[key] = 0.1
            elif options[key] > 0.9:
                options[key] = 0.9
        for key, angleRange in {
            "Left shoulder abduction degrees": (-100.0, 180.0),
            "Right shoulder abduction degrees": (-100.0, 180.0),
            "Left shoulder flexion degrees": (-60.0, 200.0),
            "Right shoulder flexion degrees": (-60.0, 200.0),
            "Left elbow flexion degrees": (0.0, 120.0),
            "Right elbow flexion degrees": (0.0, 120.0),
            "Left arm rotation degrees": (-90.0, 90.0),
            "Right arm rotation degrees": (-90.0, 90.0),
            "Left wrist flexion degrees": (-30.0, 30.0),
            "Right wrist flexion degrees": (-30.0, 30.0),
            "Left hip flexion degrees": (0.0, 150.0),
            "Right hip flexion degrees": (0.0, 150.0),
            "Left knee flexion degrees": (0.0, 140.0),
            "Right knee flexion degrees": (0.0, 140.0),
            "Left ankle flexion degrees": (60.0, 140.0),
            "Right ankle flexion degrees": (60.0, 140.0),
            "Arm twist angle degrees": (-90.0, 90.0),
            "Left leg abduction degrees": (-20.0, 60.0),
            "Right leg abduction degrees": (-20.0, 60.0)
        }.items():
            if options[key] < angleRange[0]:
                options[key] = angleRange[0]
            elif options[key] > angleRange[1]:
                options[key] = angleRange[1]
        return dependentChanges

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the unrefined mesh.
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: [] empty list of AnnotationGroup, NetworkMesh
        """
        # parameterSetName = options['Base parameter set']
        structure = options["Structure"]
        halfHeadDepth = 0.5 * options["Head depth"]
        headLength = options["Head length"]
        halfHeadWidth = 0.5 * options["Head width"]
        neckLength = options["Neck length"]
        shoulderDrop = options["Shoulder drop"]
        halfShoulderWidth = 0.5 * options["Shoulder width"]
        shoulderLeftFlexionRadians = math.radians(options["Left shoulder flexion degrees"])
        shoulderRightFlexionRadians = math.radians(options["Right shoulder flexion degrees"])
        armLeftAngleRadians = math.radians(options["Left shoulder abduction degrees"])
        armRightAngleRadians = math.radians(options["Right shoulder abduction degrees"])
        elbowLeftFlexionRadians = math.radians(options["Left elbow flexion degrees"])
        elbowRightFlexionRadians = math.radians(options["Right elbow flexion degrees"])
        armLeftRotationRadians = math.radians(options["Left arm rotation degrees"])
        armRightRotationRadians = math.radians(options["Right arm rotation degrees"])
        wristLeftFlexionRadians = math.radians(options["Left wrist flexion degrees"])
        wristRightFlexionRadians = math.radians(options["Right wrist flexion degrees"])
        wristLeftAbductionRadians = math.radians(options["Left wrist deviation degrees"])
        wristRightAbductionRadians = math.radians(options["Right wrist deviation degrees"])
        armLength = options["Arm length"]
        armTopRadius = 0.5 * options["Arm top diameter"]
        armTwistAngleRadians = math.radians(options["Arm twist angle degrees"])
        halfWristThickness = 0.5 * options["Wrist thickness"]
        halfWristWidth = 0.5 * options["Wrist width"]
        handLength = options["Hand length"]
        halfHandThickness = 0.5 * options["Hand thickness"]
        halfHandWidth = 0.5 * options["Hand width"]
        halfTorsoDepth = 0.5 * options["Torso depth"]
        thoraxLength = options["Thorax length"]
        abdomenLength = options["Abdomen length"]
        halfTorsoWidth = 0.5 * options["Torso width"]
        pelvisDrop = options["Pelvis drop"]
        halfPelvisWidth = 0.5 * options["Pelvis width"]
        leftLegAbductionRadians = math.radians(options["Left leg abduction degrees"])
        rightLegAbductionRadians = math.radians(options["Right leg abduction degrees"])
        hipLeftFlexionRadians = math.radians(options["Left hip flexion degrees"])
        hipRightFlexionRadians = math.radians(options["Right hip flexion degrees"])
        legLength = options["Leg length"]
        legTopRadius = 0.5 * options["Leg top diameter"]
        legBottomRadius = 0.5 * options["Leg bottom diameter"]
        kneeLeftFlexionRadians = math.radians(options["Left knee flexion degrees"])
        kneeRightFlexionRadians = math.radians(options["Right knee flexion degrees"])
        ankleLeftFlexionRadians = math.radians( options["Left ankle flexion degrees"])
        ankleRightFlexionRadians = math.radians(options["Right ankle flexion degrees"])
        footHeight = options["Foot height"]
        footLength = options["Foot length"]
        halfFootThickness = 0.5 * options["Foot thickness"]
        halfFootWidth = 0.5 * options["Foot width"]
        innerProportionDefault = options["Inner proportion default"]
        innerProportionHead = options["Inner proportion head"]
        # Store coordinates for kinematic tree markers
        options['Kinematic tree'] = {}
        kinTreeMarkers = {}
        networkMesh = NetworkMesh(structure)
        networkMesh.create1DLayoutMesh(region)
        fieldmodule = region.getFieldmodule()
        mesh = fieldmodule.findMeshByDimension(1)
        # set up element annotations
        bodyGroup = AnnotationGroup(region, get_body_term("body"))
        headGroup = AnnotationGroup(region, get_body_term("head"))
        neckGroup = AnnotationGroup(region, get_body_term("neck"))
        armGroup = AnnotationGroup(region, get_body_term("upper limb"))
        # armToHandGroup = AnnotationGroup(region, ("arm to hand", ""))
        leftArmGroup = AnnotationGroup(region, get_body_term("left upper limb"))
        leftShoulderGroup = AnnotationGroup(region, get_body_term("left shoulder"))
        leftBrachiumGroup = AnnotationGroup(region, get_body_term("left brachium"))
        leftAntebrachiumGroup = AnnotationGroup(region, get_body_term("left antebrachium"))
        # leftElbowGroup = AnnotationGroup(region, get_body_term("left elbow"))
        leftHandGroup = AnnotationGroup(region, get_body_term("left hand"))
        rightArmGroup = AnnotationGroup(region, get_body_term("right upper limb"))
        rightShoulderGroup = AnnotationGroup(region, get_body_term("right shoulder"))
        rightBrachiumGroup = AnnotationGroup(region, get_body_term("right brachium"))
        rightAntebrachiumGroup = AnnotationGroup(region, get_body_term("right antebrachium"))
        # rightElbowGroup = AnnotationGroup(region, get_body_term("right elbow"))
        rightHandGroup = AnnotationGroup(region, get_body_term("right hand"))
        handGroup = AnnotationGroup(region, get_body_term("hand"))
        thoraxGroup = AnnotationGroup(region, get_body_term("thorax"))
        abdomenGroup = AnnotationGroup(region, get_body_term("abdomen"))
        hipGroup = AnnotationGroup(region, get_body_term("hip"))
        legGroup = AnnotationGroup(region, get_body_term("lower limb"))
        legToFootGroup = AnnotationGroup(region, ("leg to foot", ""))
        leftLegGroup = AnnotationGroup(region, get_body_term("left lower limb"))
        leftUpperLegGroup = AnnotationGroup(region, get_body_term("left upper leg"))
        leftLowerLegGroup = AnnotationGroup(region, get_body_term("left lower leg"))
        leftFootGroup = AnnotationGroup(region, get_body_term("left foot"))
        rightLegGroup = AnnotationGroup(region, get_body_term("right lower limb"))
        rightUpperLegGroup = AnnotationGroup(region, get_body_term("right upper leg"))
        rightLowerLegGroup = AnnotationGroup(region, get_body_term("right lower leg"))
        rightFootGroup = AnnotationGroup(region, get_body_term("right foot"))
        footGroup = AnnotationGroup(region, get_body_term("foot"))
        annotationGroups = [bodyGroup, headGroup, neckGroup,
                            thoraxGroup, abdomenGroup, hipGroup,
                            leftShoulderGroup, leftBrachiumGroup, leftAntebrachiumGroup, leftHandGroup, 
                            rightShoulderGroup, rightBrachiumGroup, rightAntebrachiumGroup, rightHandGroup,
                            leftLegGroup, leftUpperLegGroup, leftLowerLegGroup, leftFootGroup,
                            rightLegGroup, rightUpperLegGroup, rightLowerLegGroup, rightFootGroup,
                            armGroup, leftArmGroup, rightArmGroup, handGroup,
                            legGroup, footGroup]
        bodyMeshGroup = bodyGroup.getMeshGroup(mesh)
        elementIdentifier = 1
        headElementsCount = humanElementCounts['headElementsCount']
        meshGroups = [bodyMeshGroup, headGroup.getMeshGroup(mesh)]
        for e in range(headElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1
        neckElementsCount = humanElementCounts['neckElementsCount']
        meshGroups = [bodyMeshGroup, neckGroup.getMeshGroup(mesh)]
        for e in range(neckElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1
        thoraxElementsCount = humanElementCounts['thoraxElementsCount']
        abdomenElementsCount = humanElementCounts['abdomenElementsCount']
        # Setup thorax elements
        meshGroups = [bodyMeshGroup, thoraxGroup.getMeshGroup(mesh)]
        for e in range(thoraxElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1
        # Setup abdomen elements 
        meshGroups = [bodyMeshGroup, abdomenGroup.getMeshGroup(mesh)]
        for e in range(abdomenElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1
        left = 0
        right = 1
        shoulderElementsCount = humanElementCounts['shoulderElementsCount']
        brachiumElementsCount = humanElementCounts['brachiumElementsCount']
        antebrachiumElementsCount = humanElementCounts['antebrachiumElementsCount']
        handElementsCount = humanElementCounts['handElementsCount']
        armToHandElementsCount = shoulderElementsCount + brachiumElementsCount + antebrachiumElementsCount 
        armMeshGroup = armGroup.getMeshGroup(mesh)
        # armToHandMeshGroup = armToHandGroup.getMeshGroup(mesh)
        handMeshGroup = handGroup.getMeshGroup(mesh)
        for side in (left, right):
            sideArmGroup = leftArmGroup if (side == left) else rightArmGroup
            sideShoulderGroup = leftShoulderGroup if (side == left) else rightShoulderGroup
            sideBrachiumGroup = leftBrachiumGroup if (side == left) else rightBrachiumGroup
            sideAntebrachiumGroup = leftAntebrachiumGroup if (side == left) else rightAntebrachiumGroup
            sideHandGroup = leftHandGroup if (side == left) else rightHandGroup
            # Setup shoulder elements
            meshGroups = [bodyMeshGroup, 
                          armMeshGroup, 
                          sideArmGroup.getMeshGroup(mesh), sideShoulderGroup.getMeshGroup(mesh)]
            for e in range(shoulderElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            # Setup brachium elements
            meshGroups = [bodyMeshGroup, 
                          armMeshGroup, 
                          sideArmGroup.getMeshGroup(mesh), sideBrachiumGroup.getMeshGroup(mesh)]
            for e in range(brachiumElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            # Setup antebrachium elements
            meshGroups = [bodyMeshGroup, 
                          armMeshGroup,
                           sideArmGroup.getMeshGroup(mesh), sideAntebrachiumGroup.getMeshGroup(mesh)]
            for e in range(antebrachiumElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            # Setup hand elements
            meshGroups = [bodyMeshGroup, 
                          armMeshGroup, 
                          handMeshGroup, sideHandGroup.getMeshGroup(mesh)]
            for e in range(handElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
        
        hipElementsCount = humanElementCounts['hipElementsCount']
        upperLegElementsCount = humanElementCounts['upperLegElementsCount']
        lowerLegElementsCount = humanElementCounts['lowerLegElementsCount']
        footElementsCount = humanElementCounts['footElementsCount']
        legToFootElementsCount = hipElementsCount + upperLegElementsCount + lowerLegElementsCount
        legMeshGroup = legGroup.getMeshGroup(mesh)
        hipMeshGroup = hipGroup.getMeshGroup(mesh)
        legToFootMeshGroup = legToFootGroup.getMeshGroup(mesh)
        footMeshGroup = footGroup.getMeshGroup(mesh)
        for side in (left, right):
            sideLegGroup = leftLegGroup if (side == left) else rightLegGroup
            sideUpperLegGroup = leftUpperLegGroup if (side == left) else rightUpperLegGroup
            sideLowerLegGroup = leftLowerLegGroup if (side == left) else rightLowerLegGroup
            sideFootGroup = leftFootGroup if (side == left) else rightFootGroup
            # Hip
            meshGroups = [bodyMeshGroup, legMeshGroup, hipMeshGroup,
                          sideLegGroup.getMeshGroup(mesh), sideUpperLegGroup.getMeshGroup(mesh)]
            for e in range(hipElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            # Upper leg
            meshGroups = [bodyMeshGroup, legMeshGroup, 
                          sideLegGroup.getMeshGroup(mesh), sideUpperLegGroup.getMeshGroup(mesh)]
            for e in range(upperLegElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            # Lower leg
            meshGroups = [bodyMeshGroup, legMeshGroup, 
                          sideLegGroup.getMeshGroup(mesh), sideLowerLegGroup.getMeshGroup(mesh)]
            for e in range(lowerLegElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            # Foot
            meshGroups = [bodyMeshGroup, legMeshGroup, footMeshGroup, sideFootGroup.getMeshGroup(mesh)]
            for e in range(footElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1

        # set coordinates (outer)
        fieldcache = fieldmodule.createFieldcache()
        coordinates = find_or_create_field_coordinates(fieldmodule)
        # need to ensure inner coordinates are at least defined:
        cls.defineInnerCoordinates(region, coordinates, options, networkMesh, innerProportion=0.75)
        innerCoordinates = find_or_create_field_coordinates(fieldmodule, "inner coordinates")
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        headScale = headLength / headElementsCount
        nodeIdentifier = 1
        d1 = [headScale, 0.0, 0.0]
        d2 = [0.0, halfHeadWidth, 0.0]
        d3 = [0.0, 0.0, halfHeadDepth]
        id2 = mult(d2, innerProportionHead)
        id3 = mult(d3, innerProportionHead)
        for i in range(headElementsCount):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = [headScale * i, 0.0, 0.0]
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3)
            nodeIdentifier += 1

        neckScale = neckLength / neckElementsCount
        d2 = [0.0, halfHeadWidth, 0.0]
        d3 = [0.0, 0.0, halfHeadWidth]
        id2 = mult(d2, innerProportionHead)
        id3 = mult(d3, innerProportionHead)
        for i in range(neckElementsCount):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = [headLength + neckScale * i, 0.0, 0.0]
            d1 = [0.5 * (headScale + neckScale) if (i == 0) else neckScale, 0.0, 0.0]
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3)
            nodeIdentifier += 1
        armJunctionNodeIdentifier = nodeIdentifier

        thoraxScale = thoraxLength / thoraxElementsCount
        thoraxStartX = headLength + neckLength
        sx = [thoraxStartX, 0.0, 0.0]
        options['Kinematic tree']['thorax_top'] = sx
        kinTreeMarkers['thorax_top'] = nodeIdentifier
        for i in range(thoraxElementsCount):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = [thoraxStartX + thoraxScale * i, 0.0, 0.0]
            if i == 0:
                d1 = [0.5 * (neckScale + thoraxScale), 0.0, 0.0]
                d2 = [0.0, 0.5 * (halfTorsoWidth + halfHeadWidth), 0.0]
                d12 = [0.0, halfTorsoWidth - halfHeadWidth, 0.0]
                d3 = [0.0, 0.0, 0.5 * (halfHeadWidth + halfTorsoDepth)]
                id2 = [0.0, 0.5 * (innerProportionHead * halfHeadWidth + innerProportionDefault * halfTorsoWidth), 0.0]
                id12 = [0.0, innerProportionDefault * halfTorsoWidth - innerProportionHead * halfHeadWidth, 0.0]
                id3 = mult(d3, 0.5 * (innerProportionHead + innerProportionDefault))
            else:
                d1 = [thoraxScale, 0.0, 0.0]
                d2 = [0.0, halfTorsoWidth, 0.0]
                d12 = None
                d3 = [0.0, 0.0, halfTorsoDepth]
                id2 = mult(d2, innerProportionDefault)
                id12 = None
                id3 = mult(d3, innerProportionDefault)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12)
            nodeIdentifier += 1
        
        abdomenScale = abdomenLength / abdomenElementsCount
        d2 = [0.0, halfTorsoWidth, 0.0]
        d3 = [0.0, 0.0, halfTorsoDepth]
        id2 = mult(d2, innerProportionDefault)
        id3 = mult(d3, innerProportionDefault)
        abdomenStartX = thoraxStartX + thoraxLength
        for i in range(abdomenElementsCount + 1):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = [abdomenStartX + abdomenScale * i, 0.0, 0.0]
            d1 = [0.5 * (thoraxScale + abdomenScale) if (i == 0) else abdomenScale, 0.0, 0.0]
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3)
            nodeIdentifier += 1
        legJunctionNodeIdentifier = nodeIdentifier - 1
        px = [abdomenStartX + abdomenLength, 0.0, 0.0]
        # options['Kinematic tree']['lumbar_body'] = px
        # arms
        for side in (left, right):
            # Shoulder rotation
            # rotate shoulder with arm, pivoting about shoulder drop below arm junction on network
            # this has the realistic effect of shoulders becoming narrower with higher angles
            # initial shoulder rotation with arm is negligible, hence:
            armAbductionRadians = armLeftAngleRadians if (side == left) else armRightAngleRadians
            shoulderRotationFactor = 1.0 - math.cos(0.5 * armAbductionRadians)
            # assume shoulder drop is half shrug distance to get limiting shoulder angle for 180 degree arm rotation
            shoulderLimitAngleRadians = math.asin(1.5 * shoulderDrop / halfShoulderWidth)
            shoulderAngleRadians = shoulderRotationFactor * shoulderLimitAngleRadians
            nonHandArmLength = armLength - handLength
            # shoulderElementsCount = 2
            armScale = nonHandArmLength / (armToHandElementsCount - shoulderElementsCount)
            d12_mag = (halfWristThickness - armTopRadius) / (armToHandElementsCount - shoulderElementsCount)
            d13_mag = (halfWristWidth - armTopRadius) / (armToHandElementsCount - shoulderElementsCount)
            armAngle = armAbductionRadians if (side == left) else -armAbductionRadians
            cosArmAngle = math.cos(armAngle)
            sinArmAngle = math.sin(armAngle)
            armStartX = thoraxStartX + shoulderDrop - halfShoulderWidth * math.sin(shoulderAngleRadians)
            armStartY = (halfShoulderWidth if (side == left) else -halfShoulderWidth) * math.cos(shoulderAngleRadians)
            armStart = [armStartX, armStartY, 0.0]
            x = armStart
            armDirn = [cosArmAngle, sinArmAngle, 0.0]
            armSide = [-sinArmAngle, cosArmAngle, 0.0]
            armFront = cross(armDirn, armSide)
            d1 = mult(armDirn, armScale)
            # set arm versions 2 (left) and 3 (right) on arm junction node, and intermediate shoulder node
            sd1 = interpolateLagrangeHermiteDerivative(sx, x, d1, 0.0)
            nx, nd1 = sampleCubicHermiteCurvesSmooth([sx, x], [sd1, d1], 2, derivativeMagnitudeEnd=armScale)[0:2]
            arcLengths = [getCubicHermiteArcLength(nx[i], nd1[i], nx[i + 1], nd1[i + 1]) for i in range(2)]
            sd2_list = []
            sd3_list = []
            sNodeIdentifiers = []
            side_label = 'l' if (side == left) else 'r'
            options['Kinematic tree']['humerus_' + side_label] = nx[1]
            kinTreeMarkers['humerus_' + side_label] = nodeIdentifier
            # Upper shoulder nodes
            for i in range(2):
                sNodeIdentifiers.append(nodeIdentifier if (i > 0) else armJunctionNodeIdentifier)
                node = nodes.findNodeByIdentifier(sNodeIdentifiers[-1])
                fieldcache.setNode(node)
                version = 1 if (i > 0) else 2 if (side == left) else 3
                sd1 = nd1[i]
                sDistance = sum(arcLengths[i:])
                sHalfHeight = armTopRadius + sDistance * -d12_mag
                sHalfDepth = armTopRadius + sDistance * -d13_mag
                sd3 = [0.0, 0.0, sHalfDepth]
                sid3 = mult(sd3, innerProportionDefault)
                sd2 = set_magnitude(cross(sd3, sd1), sHalfHeight)
                sid2 = mult(sd2, innerProportionDefault)
                sd2_list.append(sd2)
                sd3_list.append(sd3)
                if i > 0:
                    for field in (coordinates, innerCoordinates):
                        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, nx[i])
                    nodeIdentifier += 1
                setNodeFieldVersionDerivatives(coordinates, fieldcache, version, sd1, sd2, sd3)
                setNodeFieldVersionDerivatives(innerCoordinates, fieldcache, version, sd1, sid2, sid3)
            # Arm twist
            elementTwistAngle = ((armTwistAngleRadians if (side == left) else -armTwistAngleRadians) /
                                 (armToHandElementsCount - 3))
            # Shoulder rotation
            #Reset flexion angle w.r.t. current node
            shoulderFlexionRadians = shoulderLeftFlexionRadians if (side == left) else shoulderRightFlexionRadians
            ventralFlexion = True
            if shoulderFlexionRadians < 0:
                ventralFlexion = False
                shoulderFlexionRadians = -shoulderFlexionRadians
            #Reset abdution angle w.r.t. current node
            shoulderAbductionRadians =  (1 if (side == left) else -1)*angle([1,0,0], sd1) - armAngle
            upwardAbduction = True
            if shoulderAbductionRadians < 0:
                upwardAbduction = False
                shoulderAbductionRadians = -shoulderAbductionRadians
            # Calculate magntiudes for joint node
            i = 0
            xi = i / (armToHandElementsCount - 2)
            halfWidth = xi * halfWristWidth + (1.0 - xi) * armTopRadius
            halfThickness = xi * halfWristThickness + (1.0 - xi) * armTopRadius
            upperShoulderDir = [nx[1], sd1, sd2, sd3, nd1[1], 0, 0]
            armRotationRadians = armLeftRotationRadians if (side == left) else armRightRotationRadians
            x1 = [0,1,0]
            for i in range(1, 4):
                upperShoulderDir[i] = set_magnitude(upperShoulderDir[i], 1) #normalize frame
            shoulderDir, armDir = getJointRotationFrames( \
                shoulderAbductionRadians, shoulderFlexionRadians, upperShoulderDir, upwardAbduction, ventralFlexion)
            shoulderDir[0] = armStart
            shoulderDir[4:] = [armScale, halfThickness, halfWidth]
            rotationCoeff = 0.2
            x, shoulderd2_mag, shoulderd3_mag, shoulderd12_mag, shoulderd13_mag = getJointNodePosition(
                shoulderAbductionRadians, shoulderFlexionRadians, upperShoulderDir, \
                    shoulderDir, armDir, rotationCoeff, upwardAbduction, ventralFlexion
            )
            armRotationRadians = armLeftRotationRadians if (side == left) else armRightRotationRadians
            jointAbductionMatrix = axis_angle_to_rotation_matrix(mult(shoulderDir[1], 1), armRotationRadians)
            for i in range(2, 4):
                shoulderDir[i] = matrix_vector_mult(jointAbductionMatrix, shoulderDir[i])
                armDir[i] = matrix_vector_mult(jointAbductionMatrix, armDir[i])
            shoulderDirn, shoulderSide, shoulderFront = shoulderDir[1:4]
            d1 = mult(shoulderDirn, armScale)
            d2 = mult(shoulderSide, shoulderd2_mag)
            d3 = mult(shoulderFront, shoulderd3_mag)
            d12 = add(
                mult(shoulderSide, d12_mag), 
                mult(shoulderDirn, shoulderd12_mag)
            )
            d13 = add(
                mult(shoulderFront, d13_mag), 
                mult(shoulderDirn, shoulderd13_mag)
                )
            id2 = mult(d2, innerProportionDefault)
            id3 = mult(d3, innerProportionDefault)
            id12 = mult(d12, innerProportionDefault)
            id13 = mult(d13, innerProportionDefault)
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
            nodeIdentifier += 1
            sd2_list.append(d2)
            sd3_list.append(d3)
            # Adjusting d12 and 13 for the upper Shoulder node
            for i in range(2):
                node = nodes.findNodeByIdentifier(sNodeIdentifiers[i])
                fieldcache.setNode(node)
                version = 1 if (i > 0) else 2 if (side == left) else 3
                sd12 = sub(sd2_list[i + 1], sd2_list[i])
                sd13 = sub(sd3_list[i + 1], sd3_list[i])
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, version, sd12)
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, version, sd13)
                sid12 = mult(sd12, innerProportionDefault)
                sid13 = mult(sd13, innerProportionDefault)
                innerCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, version, sid12)
                innerCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, version, sid13)
            options['Kinematic tree']['ulna_' + side_label] = x
            kinTreeMarkers['ulna_' + side_label] = nodeIdentifier + 1
            # Initial position for arm node
            armDirn, armSide, armFront = armDir[1:4]
            d1 = set_magnitude(armDirn, armScale)
            armStart = add(x, d1)
            armDir[0] = armStart
            armStart = getDistalNodePosition(shoulderAbductionRadians, shoulderFlexionRadians,
                upperShoulderDir, shoulderDir, armDir, 0.2)
            # Setting brachium coordinates
            j = 0
            for i in range(1, brachiumElementsCount):
                xi = i / (armToHandElementsCount - 2)
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                x = add(armStart, mult(d1, j))
                halfThickness = xi * halfWristThickness + (1.0 - xi) * armTopRadius
                halfWidth = xi * halfWristWidth + (1.0 - xi) * armTopRadius
                if i == 0:
                    twistAngle = 0.0
                else:
                    twistAngle = -0.5 * elementTwistAngle + elementTwistAngle * i
                if twistAngle == 0.0:
                    d2 = mult(armSide, halfThickness)
                    d3 = mult(armFront, halfWidth)
                    d12 = mult(armSide, d12_mag)
                    d13 = mult(armFront, d13_mag)
                else:
                    cosTwistAngle = math.cos(twistAngle)
                    sinTwistAngle = math.sin(twistAngle)
                    d2 = sub(mult(armSide, halfThickness * cosTwistAngle),
                             mult(armFront, halfThickness * sinTwistAngle))
                    d3 = add(mult(armFront, halfWidth * cosTwistAngle),
                             mult(armSide, halfWidth * sinTwistAngle))
                    d12 = set_magnitude(d2, d12_mag)
                    d13 = set_magnitude(d3, d13_mag)
                id2 = mult(d2, innerProportionDefault)
                id3 = mult(d3, innerProportionDefault)
                id12 = mult(d12, innerProportionDefault)
                id13 = mult(d13, innerProportionDefault)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
                nodeIdentifier += 1
                j += 1
            # Elbow
            i += 1
            twistAngle = -0.5 * elementTwistAngle + elementTwistAngle * (i)
            if twistAngle == 0.0:
                d2 = armSide 
                d3 = armFront
            else:
                cosTwistAngle = math.cos(twistAngle)
                sinTwistAngle = math.sin(twistAngle)
                d2 = sub(mult(armSide, cosTwistAngle),
                            mult(armFront, sinTwistAngle))
                d3 = add(mult(armFront,  cosTwistAngle),
                            mult(armSide, sinTwistAngle))
            armSide = d2 
            armFront = d3
            # Updating frame of reference wrt flexion angle (using d2 as rotation axis)
            elbowFlexionRadians = elbowLeftFlexionRadians if (side == left) else elbowRightFlexionRadians
            elbowAbductionRadians = 0
            ventralFlexion = True
            if shoulderFlexionRadians < 0:
                ventralFlexion = False
                shoulderFlexionRadians = -shoulderFlexionRadians
            armDir[0] = x
            xi = i / (armToHandElementsCount - 2)
            halfThickness = xi * halfWristThickness + (1.0 - xi) * armTopRadius
            halfWidth = xi * halfWristWidth + (1.0 - xi) * armTopRadius
            rotationCoeff = 0.25
            elbowDir, antebrachiumDir = getJointRotationFrames( \
                0, elbowFlexionRadians, armDir, upwardAbduction, ventralFlexion)
            elbowDir[0] = add(x, d1)
            elbowDir[4:] = [armScale, halfThickness, halfWidth]
            x, elbowd2_mag, elbowd3_mag, elbowd12_mag, elbowd13_mag = getJointNodePosition(
                0, elbowFlexionRadians, armDir, elbowDir, antebrachiumDir, rotationCoeff, upwardAbduction, ventralFlexion)
            elbowDirn, elbowSide, elbowFront = elbowDir[1:4]
            d1 = mult(elbowDirn, armScale)
            d2 = mult(elbowSide, elbowd2_mag)
            d3 = mult(elbowFront, elbowd3_mag)
            d12 = add(
                mult(elbowSide, d12_mag), 
                mult(elbowDirn, elbowd12_mag)
                )
            d13 = add(
                mult(elbowFront, d13_mag), 
                mult(elbowDirn, elbowd13_mag)
                )
            id2 = mult(d2, innerProportionDefault)
            id3 = mult(d3, innerProportionDefault)
            id12 = mult(d12, innerProportionDefault)
            id13 = mult(d13, innerProportionDefault)
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
            options['Kinematic tree']['ulna_' + side_label] = x
            kinTreeMarkers['ulna_' + side_label] - nodeIdentifier
            nodeIdentifier += 1
            # Antebrachium nodes starts after the elbow node
            antebrachiumDirn, antebrachiumSide, antebrachiumFront = antebrachiumDir[1:4]
            d1 = set_magnitude(antebrachiumDirn, armScale)
            antebrachiumStart = add(x, d1)
            antebrachiumDir[0] = antebrachiumStart
            antebrachiumStart = getDistalNodePosition(elbowAbductionRadians, elbowFlexionRadians,
                armDir, elbowDir, antebrachiumDir, rotationCoeff)
            j=0
            for i in range(brachiumElementsCount + 1, armToHandElementsCount - 2):
                xi = (i) / (armToHandElementsCount - 2)
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                x = add(antebrachiumStart, mult(d1, j)) 
                halfThickness = xi * halfWristThickness + (1.0 - xi) * armTopRadius
                halfWidth =  xi * halfWristWidth + (1.0 - xi) * armTopRadius
                if i == 0:
                    twistAngle = armTwistAngleRadians if (side == left) else -armTwistAngleRadians
                else:
                    twistAngle = -0.5 * elementTwistAngle + elementTwistAngle * (i)
                if twistAngle == 0.0:
                    d2 = mult(antebrachiumSide, halfThickness)
                    d3 = mult(antebrachiumFront, halfWidth)
                    d12 = mult(antebrachiumSide, d12_mag)
                    d13 = mult(antebrachiumFront, d13_mag)
                else:
                    cosTwistAngle = math.cos(twistAngle)
                    sinTwistAngle = math.sin(twistAngle)
                    d2 = sub(mult(antebrachiumSide, halfThickness * cosTwistAngle),
                             mult(antebrachiumFront, halfThickness * sinTwistAngle))
                    d3 = add(mult(antebrachiumFront, halfWidth * cosTwistAngle),
                             mult(antebrachiumSide, halfWidth * sinTwistAngle))
                    d12 = set_magnitude(d2, d12_mag)
                    d13 = set_magnitude(d3, d13_mag)
                    if i < (antebrachiumElementsCount - 1):
                        d12 = add(d12, set_magnitude(d3, -halfThickness * elementTwistAngle))
                        d13 = add(d13, set_magnitude(d2, halfWidth * elementTwistAngle))
                id2 = mult(d2, innerProportionDefault)
                id3 = mult(d3, innerProportionDefault)
                id12 = mult(d12, innerProportionDefault)
                id13 = mult(d13, innerProportionDefault)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
                nodeIdentifier += 1
                j += 1
            # Wrist flexion 
            i += 1
            twistAngle = -0.5 * elementTwistAngle + elementTwistAngle * (i)
            if twistAngle == 0.0:
                d2 = antebrachiumSide 
                d3 = antebrachiumFront
            else:
                cosTwistAngle = math.cos(twistAngle)
                sinTwistAngle = math.sin(twistAngle)
                d2 = sub(mult(antebrachiumSide, cosTwistAngle),
                            mult(antebrachiumFront, sinTwistAngle))
                d3 = add(mult(antebrachiumFront,  cosTwistAngle),
                            mult(antebrachiumSide, sinTwistAngle))
            antebrachiumSide = d2 
            antebrachiumFront = d3
            wristFlexionRadians = wristLeftFlexionRadians if (side == left) else wristRightFlexionRadians
            wristAbductionRadians = wristLeftAbductionRadians if (side == left) else -wristRightAbductionRadians
            ventralFlexion = True
            if wristFlexionRadians < 0:
                ventralFlexion = False
                wristFlexionRadians = -wristFlexionRadians
            upwardAbduction = True
            if wristAbductionRadians < 0:
                upwardAbduction = False
                wristAbductionRadians = -wristAbductionRadians
            antebrachiumDir[0] = x
            xi = i / (armToHandElementsCount - 2)
            halfThickness = halfWristThickness
            halfWidth = halfWristWidth
            rotationCoeff = 0.3
            wristDir, handDir = getJointRotationFrames(
                wristAbductionRadians, wristFlexionRadians, antebrachiumDir, upwardAbduction, ventralFlexion
            )
            wristDir[0] = add(x, d1)
            wristDir[4:] = [armScale, halfThickness, halfWidth]
            x, wristd2_mag, wristd3_mag, wristd12_mag, wristd13_mag = getJointNodePosition(
                wristAbductionRadians, wristFlexionRadians, antebrachiumDir, wristDir, handDir, rotationCoeff,  upwardAbduction, ventralFlexion
            )
            wristDirn, wristSide, wristFront = wristDir[1:4]
            handDirn, handSide, handFront = handDir[1:4]
            d1 = mult(handDirn, armScale)
            d2 = mult(wristSide, wristd2_mag)
            d3 = mult(wristFront, wristd3_mag)
            d12 = add(
                mult(wristSide, d12_mag), 
                mult(handDirn, wristd12_mag)
                )
            d13 = add(
                mult(wristFront, d13_mag), 
                mult(handDirn, wristd13_mag)
                )
            id2 = mult(d2, innerProportionDefault)
            id3 = mult(d3, innerProportionDefault)
            id12 = mult(d12, innerProportionDefault)
            id13 = mult(d13, innerProportionDefault)
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
            kinTreeMarkers['hand_' + side_label] = nodeIdentifier
            nodeIdentifier += 1
            options['Kinematic tree']['hand_' + side_label] = x
            
            d1 = set_magnitude(handDirn, armScale)
            handStart = add(x, mult(handDirn, handLength))
            handDir[0] = handStart
            # Hand
            assert handElementsCount == 1
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            hd1 = computeCubicHermiteEndDerivative(x, d1, handStart, d1)
            twistAngle = armTwistAngleRadians if (side == left) else -armTwistAngleRadians
            if twistAngle >= 0.0:
                hd2 = set_magnitude(handSide, halfHandThickness)
                hd3 = set_magnitude(handFront, halfHandWidth)
            else:
                cosTwistAngle = math.cos(twistAngle)
                sinTwistAngle = math.sin(twistAngle)
                hd2 = sub(mult(handSide, halfHandThickness * cosTwistAngle),
                          mult(handFront, halfHandThickness * sinTwistAngle))
                hd3 = add(mult(handFront, halfHandWidth * cosTwistAngle),
                          mult(handSide, halfHandWidth * sinTwistAngle))
            hid2 = mult(hd2, innerProportionDefault)
            hid3 = mult(hd3, innerProportionDefault)
            setNodeFieldParameters(coordinates, fieldcache, handStart, hd1, hd2, hd3)
            setNodeFieldParameters(innerCoordinates, fieldcache, handStart, hd1, hid2, hid3)
            nodeIdentifier += 1
        # legs
        legStartX = abdomenStartX + abdomenLength + pelvisDrop
        nonFootLegLength = legLength - footHeight
        legScale = nonFootLegLength / (legToFootElementsCount - 1) 
        d12_mag = (legBottomRadius - legTopRadius) / (legToFootElementsCount)
        d13_mag = (legBottomRadius - legTopRadius) / (legToFootElementsCount)
        pd3 = [0.0, 0.0, 0.5 * legTopRadius + 0.5 * halfTorsoDepth]
        pid3 = mult(pd3, innerProportionDefault)
        for side in (left, right):
            side_label = 'l' if (side == left) else 'r'
            legAngle = leftLegAbductionRadians if (side == left) else -rightLegAbductionRadians
            cosLegAngle = math.cos(legAngle)
            sinLegAngle = math.sin(legAngle)
            legStartY = halfPelvisWidth if (side == left) else -halfPelvisWidth
            x = legStart = [legStartX, legStartY, 0.0]
            legDirn = [cosLegAngle, sinLegAngle, 0.0]
            legSide = [-sinLegAngle, cosLegAngle, 0.0]
            legFront = cross(legDirn, legSide)
            d1 = mult(legDirn, legScale)
            # set leg versions 2 (left) and 3 (right) on leg junction node
            node = nodes.findNodeByIdentifier(legJunctionNodeIdentifier)
            fieldcache.setNode(node)
            pd1 = interpolateLagrangeHermiteDerivative(px, x, d1, 0.0)
            pd2 = set_magnitude(cross(pd3, pd1), 0.5 * legTopRadius + 0.5 * halfTorsoWidth)
            pid2 = mult(pd2, innerProportionDefault)
            pd12 = sub(mult(legSide, legTopRadius), pd2)
            pd13 = sub([0.0, 0.0, legTopRadius], pd3)
            pid12 = mult(pd12, innerProportionDefault)
            pid13 = mult(pd13, innerProportionDefault)
            version = 2 if (side == left) else 3
            setNodeFieldVersionDerivatives(coordinates, fieldcache, version, pd1, pd2, pd3, pd12, pd13)
            setNodeFieldVersionDerivatives(innerCoordinates, fieldcache, version, pd1, pid2, pid3, pid12, pid13)
            d12 = [-d12_mag * sinLegAngle, d12_mag * cosLegAngle, 0.0]
            id12 = mult(d12, innerProportionDefault)
            d13 = [0.0, 0.0, d13_mag]
            id13 = mult(d13, innerProportionDefault)
            # options['Kinematic tree']['femur_' + side_label] = x
            # Upper leg
            for i in range(hipElementsCount-1):
                xi = i / legToFootElementsCount
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                x = add(legStart, mult(d1, i))
                radius = xi * legBottomRadius + (1.0 - xi) * legTopRadius
                d2 = mult(legSide, radius)
                d3 = mult(legFront, radius)
                d13 = mult(legFront, d13_mag)
                d12 = mult(legSide, d12_mag)
                id2 = mult(d2, innerProportionDefault)
                id3 = mult(d3, innerProportionDefault)
                id13 = mult(d13, innerProportionDefault)
                id12 = mult(d12, innerProportionDefault)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
                nodeIdentifier += 1
            # Frontal hip flexion
            # Updating frame of reference wrt flexion angle (using d2 as rotation axis)
            hipFlexionRadians = hipLeftFlexionRadians if (side == left) else hipRightFlexionRadians
            ventralFlexion = True
            if hipFlexionRadians < 0:
                ventralFlexion = False
                hipFlexionRadians = -hipFlexionRadians
            i += 1
            xi = i / legToFootElementsCount
            radius = xi * legBottomRadius + (1.0 - xi) * legTopRadius
            legDir = [x, legDirn, legSide, legFront, legScale, 0, 0]
            rotationCoeff = 0.18
            hipDir, upperLegDir = getJointRotationFrames( \
                0, hipFlexionRadians, legDir, True, ventralFlexion)
            hipDir[0] = add(x, d1)
            hipDir[4:] = [legScale, radius, radius]
            x, hipd2_mag, hipd3_mag, hipd12_mag, hipd13_mag = getJointNodePosition(
                0, hipFlexionRadians, legDir, hipDir, upperLegDir, rotationCoeff
            )
            hipDirn, hipSide, hipFront = hipDir[1:4]
            d1 = mult(hipDirn, legScale)
            d2 = mult(hipSide, hipd2_mag)
            d3 = mult(hipFront, 0.9*hipd3_mag)
            d12 = mult(hipSide, d12_mag)
            d13 = add(
                mult(hipFront, d13_mag), 
                mult(hipDirn, 0.7*hipd13_mag)
                )
            id2 = mult(d2, innerProportionDefault)
            id3 = mult(d3, innerProportionDefault)
            id12 = mult(d12, innerProportionDefault)
            id13 = mult(d13, innerProportionDefault)
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
            kinTreeMarkers['femur_' + side_label] = nodeIdentifier
            nodeIdentifier += 1
            options['Kinematic tree']['femur_' + side_label] = x
            upperLegDirn, upperLegSide, upperLegFront = upperLegDir[1:4]
            d1 = set_magnitude(upperLegDirn, legScale)
            upperLegStart = add(x, d1)
            upperLegDir[0] = upperLegStart
            upperLegStart = getDistalNodePosition(0, hipFlexionRadians, 
                legDir, hipDir, upperLegDir, rotationCoeff)
            # Rest of upper leg
            j = 0
            for i in range(hipElementsCount, hipElementsCount+upperLegElementsCount-1):
                xi = i / legToFootElementsCount
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                x = add(upperLegStart, mult(d1, j))
                radius = xi * legBottomRadius + (1.0 - xi) * legTopRadius
                d2 = mult(upperLegSide, radius)
                d3 = mult(upperLegFront, radius)
                d12 = mult(upperLegSide, d12_mag)
                d13 = mult(upperLegFront, d13_mag)
                id2 = mult(d2, innerProportionDefault)
                id3 = mult(d3, innerProportionDefault)
                id12 = set_magnitude(d12, d12_mag)
                id13 = set_magnitude(d13, d13_mag)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
                nodeIdentifier += 1
                j += 1
            # knee
            kneeFlexionRadians = kneeLeftFlexionRadians if (side == left) else kneeRightFlexionRadians
            ventralFlexion = False
            if shoulderFlexionRadians < 0:
                ventralFlexion = True
                kneeFlexionRadians = -kneeFlexionRadians
            # # Set coordiantes for joint node
            i += 1
            xi = i / legToFootElementsCount
            radius = xi * legBottomRadius + (1.0 - xi) * legTopRadius
            rotationCoeff = 0.25
            kneeDir, lowerLegDir = getJointRotationFrames(
                0, kneeFlexionRadians, upperLegDir, True, ventralFlexion
            )
            kneeDir[0] = add(x, d1)
            kneeDir[4:] = [legScale, radius, radius]
            x, kneed2_mag, kneed3_mag, kneed12_mag, kneed13_mag = getJointNodePosition(
                0, kneeFlexionRadians, upperLegDir, kneeDir, lowerLegDir, rotationCoeff, True, ventralFlexion)
            kneeDirn, kneeSide, kneeFront = kneeDir[1:4]
            d1 = mult(kneeDirn, legScale)
            d2 = mult(kneeSide, kneed2_mag)
            d3 = mult(kneeFront, kneed3_mag)
            d12 = mult(kneeSide, d12_mag)
            d13 = add(
                set_magnitude(d3, d13_mag),
                set_magnitude(d1, kneed13_mag)
            )
            id2 = mult(d2, innerProportionDefault)
            id3 = mult(d3, innerProportionDefault)
            id12 = mult(d12, innerProportionDefault)
            id13 = mult(d13, innerProportionDefault)
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
            kinTreeMarkers['tibia_' + side_label] = nodeIdentifier
            nodeIdentifier += 1
            options['Kinematic tree']['tibia_' + side_label] = x
            # Lower leg
            lowerLegDirn, lowerLegSide, lowerLegFront = lowerLegDir[1:4]
            d1 = set_magnitude(lowerLegDirn, legScale)
            lowerLegStart = add(x, d1)
            lowerLegDir[0] = lowerLegStart
            lowerLegStart = getDistalNodePosition(0, kneeFlexionRadians,
                upperLegDir, kneeDir, lowerLegDir, rotationCoeff)
            j = 0 
            for i in range(hipElementsCount+upperLegElementsCount, legToFootElementsCount-1):
                xi = i / legToFootElementsCount
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                x = add(lowerLegStart, mult(d1, j))
                radius = xi * legBottomRadius + (1.0 - xi) * legTopRadius
                d2 = set_magnitude(lowerLegSide, radius)
                d3 = set_magnitude(lowerLegFront, radius)
                d12 = set_magnitude(d2, d12_mag)
                d13 = set_magnitude(d3, d13_mag)
                id2 = mult(d2, innerProportionDefault)
                id3 = mult(d3, innerProportionDefault)
                id12 = set_magnitude(d12, d12_mag)
                id13 = set_magnitude(d13, d13_mag)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
                nodeIdentifier += 1
                j+=1
            # foot
            ankleFlexionRadians = ankleLeftFlexionRadians if (side == left) else ankleRightFlexionRadians
            ventralFlexion = True
            if ankleFlexionRadians < 0:
                ventralFlexion = False
                ankleFlexionRadians = -ankleFlexionRadians
            i += 1
            radius = math.sqrt(2)*legBottomRadius 
            rotationCoeff = 0.4
            ankleDir, footDir = getJointRotationFrames(
                0, ankleFlexionRadians, lowerLegDir, True, ventralFlexion
            )
            ankleDir[0] = add(x, d1)
            ankleDir[4:] = [legScale, radius, radius]
            [x, ankled2_mag, ankled3_mag, ankled12_mag, ankled13_mag] = getJointNodePosition(
                0, ankleFlexionRadians, lowerLegDir, ankleDir, footDir, rotationCoeff, True, ventralFlexion
            )
            ankleDirn, ankleSide, ankleFront = ankleDir[1:4]
            footDirn, footSide, footFront = footDir[1:4]
            d1 = mult(footDirn, legScale)
            d2 = set_magnitude(ankleSide, ankled2_mag)
            d3 = set_magnitude(ankleFront, ankled3_mag)
            d12 = set_magnitude(ankleSide, d12_mag)
            d13 = add(
                set_magnitude(ankleFront, d13_mag), 
                set_magnitude(footDirn, ankled13_mag)
            )
            id2 = mult(d2, innerProportionDefault)
            id3 = mult(d3, innerProportionDefault)
            id12 = mult(d12, innerProportionDefault)
            id13 = mult(d13, innerProportionDefault)
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
            nodeIdentifier += 1                 
            # Foot end nodes
            d1 = mult(footDirn, footLength)
            footStart = add(x, d1)
            footDir[0] = footStart
            j = 0 
            for i in range(footElementsCount):
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                x = add(footStart, mult(d1, j))
                d2 = set_magnitude(footSide, halfFootWidth)
                d3 = set_magnitude(footFront, halfFootThickness)
                d12 = set_magnitude(d2, d12_mag)
                d13 = sub(d3, set_magnitude(ankleFront, ankled3_mag))
                id2 = mult(d2, innerProportionDefault)
                id3 = mult(d3, innerProportionDefault)
                id12 = set_magnitude(d12, d12_mag)
                id13 = set_magnitude(d13, d13_mag)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
                nodeIdentifier += 1
                j+=1
            kinTreeMarkers['toes_' + side_label] = nodeIdentifier - 2
            options['Kinematic tree']['toes_' + side_label] = x

        # Kinematic tree markers 
        for marker_name, marker_position in kinTreeMarkers.items():
            marker_group = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, (marker_name, ""), isMarker=True
                )
            marker_group.createMarkerNode(
                nodeIdentifier, element=mesh.findElementByIdentifier(marker_position), xi=[1]
            )
        return annotationGroups, networkMesh

    @classmethod
    def getInteractiveFunctions(cls):
        """
        Edit base class list to include only valid functions.
        """
        interactiveFunctions = super(MeshType_1d_human_body_network_layout1, cls).getInteractiveFunctions()
        for interactiveFunction in interactiveFunctions:
            if interactiveFunction[0] == "Edit structure...":
                interactiveFunctions.remove(interactiveFunction)
                break
        interactiveFunctions = interactiveFunctions + [\
        ("Align to body markers", 
            {"Load transformation parameters into Settings": True},
            lambda region, options, networkMesh, functionOptions, editGroupName:
            cls.alignNetworkLayoutToMarkers(region, options, networkMesh, functionOptions, editGroupName)
            ),
        ]
        return interactiveFunctions

    @classmethod
    def alignNetworkLayoutToMarkers(cls, region, options, networkMesh, functionOptions, editGroupName):

        # Load the data into the current region 
        data_region = region.getParent().findChildByName('data')
        if not data_region.isValid():
            logger.warning('Missing input data')
            return None
        data_fieldmodule = data_region.getFieldmodule()
        with ChangeManager(data_fieldmodule):
            copy_fitting_data(region, data_region)
        del data_region  
        # Setup fitting routine
        fieldmodule = region.getFieldmodule()
        fitter = GeometryFitter(region=region)
        fitter.getInitialFitterStepConfig()
        # Manually run the _loadModel routines
        fitter._discoverModelCoordinatesField()
        fitter._discoverModelFitGroup()
        # 
        zero_fibres = find_or_create_field_zero_fibres(fieldmodule)
        fitter.setFibreField(zero_fibres)
        del zero_fibres
        #
        fitter._discoverFlattenGroup()
        fitter.defineCommonMeshFields()
        # Manually run the _loadData routines
        fitter._discoverDataCoordinatesField()
        fitter._discoverMarkerGroup()
        # Stuff 
        fitter.defineCommonMeshFields()
        fitter.defineDataProjectionFields()
        fitter.initializeFit()
        # Call Align step
        fit1 = FitterStepAlign() 
        fitter.addFitterStep(fit1)
        fit1.setAlignMarkers(True)
        fit1._doAutoAlign()
        # Pass the graphical transformation settings into options
        options['rotation'] = fit1._rotation
        options['scale'] = fit1._scale
        options['translation'] = fit1._translation
        del fit1
        return True, False
    

class MeshType_3d_wholebody2(Scaffold_base):
    """
    Generates a 3-D hermite bifurcating tube network with core representing the human body.
    """

    @classmethod
    def getName(cls):
        return "3D Whole Body 2"

    @classmethod
    def getParameterSetNames(cls):
        return [
            "Default",
            "Human 1 Coarse",
            "Human 1 Medium",
            "Human 1 Fine"
        ]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {}
        useParameterSetName = "Human 1 Coarse" if (parameterSetName == "Default") else parameterSetName
        options["Base parameter set"] = useParameterSetName
        options["Body network layout"] = ScaffoldPackage(MeshType_1d_human_body_network_layout1)
        options["Number of elements along head"] = 2
        options["Number of elements along neck"] = 1
        options["Number of elements along thorax"] = 2
        options["Number of elements along abdomen"] = 2
        options["Number of elements along shoulder"] = 2
        options["Number of elements along brachium"] = 3
        options["Number of elements along antebrachium"] = 2
        options["Number of elements along hand"] = 1
        options["Number of elements along hip"] = 2
        options["Number of elements along upper leg"] = 3
        options["Number of elements along lower leg"] = 3
        options["Number of elements along foot"] = 1
        options["Number of elements around head"] = 12
        options["Number of elements around torso"] = 12
        options["Number of elements around arm"] = 8
        options["Number of elements around leg"] = 8
        options["Number of elements through shell"] = 1
        options["Show trim surfaces"] = False
        options["Use Core"] = True
        options["Number of elements across core box minor"] = 2
        options["Number of elements across core transition"] = 1
        if "Medium" in useParameterSetName:
            options["Number of elements along head"] = 3
            options["Number of elements along neck"] = 2
            options["Number of elements along thorax"] = 3
            options["Number of elements along abdomen"] = 3
            options["Number of elements along shoulder"] = 2
            options["Number of elements along brachium"] = 3
            options["Number of elements along antebrachium"] = 3
            options["Number of elements along hand"] = 1
            options["Number of elements along upper leg"] = 2
            options["Number of elements along lower leg"] = 2
            options["Number of elements along foot"] = 2
            options["Number of elements around head"] = 16
            options["Number of elements around torso"] = 16
            options["Number of elements around leg"] = 12
        elif "Fine" in useParameterSetName:
            options["Number of elements along head"] = 4
            options["Number of elements along neck"] = 2
            options["Number of elements along thorax"] = 4
            options["Number of elements along abdomen"] = 4
            options["Number of elements along shoulder"] = 2
            options["Number of elements along brachium"] = 3
            options["Number of elements along antebrachium"] = 4
            options["Number of elements along hand"] = 2
            options["Number of elements along upper leg"] = 3
            options["Number of elements along lower leg"] = 2
            options["Number of elements along foot"] = 3
            options["Number of elements around head"] = 20
            options["Number of elements around torso"] = 20
            options["Number of elements around arm"] = 12
            options["Number of elements around leg"] = 16
            options["Number of elements through shell"] = 2
            options["Number of elements across core box minor"] = 4

        return options

    @classmethod
    def getOrderedOptionNames(cls):
        optionNames = [
            "Body network layout",
            "Number of elements along head",
            "Number of elements along neck",
            "Number of elements along thorax",
            "Number of elements along abdomen",
            "Number of elements along shoulder",
            "Number of elements along brachium",
            "Number of elements along antebrachium",
            "Number of elements along hand",
            "Number of elements along hip",
            "Number of elements along upper leg",
            "Number of elements along lower leg",
            "Number of elements along foot",
            "Number of elements around head",
            "Number of elements around torso",
            "Number of elements around arm",
            "Number of elements around leg",
            "Number of elements through shell",
            "Show trim surfaces",
            "Use Core",
            "Number of elements across core box minor",
            "Number of elements across core transition"]
        return optionNames

    @classmethod
    def getOptionValidScaffoldTypes(cls, optionName):
        if optionName == "Body network layout":
            return [MeshType_1d_human_body_network_layout1]
        return []

    @classmethod
    def getOptionScaffoldPackage(cls, optionName, scaffoldType, parameterSetName=None):
        """
        :param parameterSetName:  Name of valid parameter set for option Scaffold, or None for default.
        :return: ScaffoldPackage.
        """
        if parameterSetName:
            assert parameterSetName in cls.getOptionScaffoldTypeParameterSetNames(optionName, scaffoldType), \
                "Invalid parameter set " + str(parameterSetName) + " for scaffold " + str(scaffoldType.getName()) + \
                " in option " + str(optionName) + " of scaffold " + cls.getName()
        if optionName == "Body network layout":
            if not parameterSetName:
                parameterSetName = "Default"
            return ScaffoldPackage(MeshType_1d_human_body_network_layout1, defaultParameterSetName=parameterSetName)
        assert False, cls.__name__ + ".getOptionScaffoldPackage:  Option " + optionName + " is not a scaffold"

    @classmethod
    def checkOptions(cls, options):
        dependentChanges = False
        if (options["Body network layout"].getScaffoldType() not in
                cls.getOptionValidScaffoldTypes("Body network layout")):
            options["Body network layout"] = ScaffoldPackage(MeshType_1d_human_body_network_layout1)
        for key in [
            "Number of elements along head",
            "Number of elements along neck",
            "Number of elements along thorax",
            "Number of elements along abdomen",
            "Number of elements along brachium",
            "Number of elements along antebrachium",
            "Number of elements along hand",
            "Number of elements along upper leg",
            "Number of elements along lower leg",
            "Number of elements along foot"
        ]:
            if options[key] < 1:
                options[key] = 1
        minElementsCountAround = None
        for key in [
            "Number of elements around head",
            "Number of elements around torso",
            "Number of elements around arm",
            "Number of elements around leg"
        ]:
            if options[key] < 8:
                options[key] = 8
            elif options[key] % 4:
                options[key] += 4 - (options[key] % 4)
            if (minElementsCountAround is None) or (options[key] < minElementsCountAround):
                minElementsCountAround = options[key]

        if options["Number of elements through shell"] < 0:
            options["Number of elements through shell"] = 1

        if options["Number of elements across core transition"] < 1:
            options["Number of elements across core transition"] = 1

        maxElementsCountCoreBoxMinor = minElementsCountAround // 2 - 2
        for key in [
            "Number of elements across core box minor"
        ]:
            if options[key] < 2:
                options[key] = 2
            elif options[key] > maxElementsCountCoreBoxMinor:
                options[key] = maxElementsCountCoreBoxMinor
                dependentChanges = True
            elif options[key] % 2:
                options[key] += options[key] % 2

        return dependentChanges

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the base hermite-bilinear mesh. See also generateMesh().
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: list of AnnotationGroup, None
        """
        # parameterSetName = options['Base parameter set']
        networkLayout = options["Body network layout"]
        elementsCountAlongHead = options["Number of elements along head"]
        elementsCountAlongNeck = options["Number of elements along neck"]
        elementsCountAlongThorax = options["Number of elements along thorax"]
        elementsCountAlongAbdomen = options["Number of elements along abdomen"]
        elementsCountAlongShoulder = options["Number of elements along shoulder"]
        elementsCountAlongBrachium = options["Number of elements along brachium"]
        elementsCountAlongAntebrachium = options["Number of elements along antebrachium"]
        elementsCountAlongHand = options["Number of elements along hand"]
        elementsCountAlongHip = options["Number of elements along hip"]
        elementsCountAlongUpperLeg = options["Number of elements along upper leg"]
        elementsCountAlongLowerLeg = options["Number of elements along lower leg"]
        elementsCountAlongFoot = options["Number of elements along foot"]
        elementsCountAroundHead = options["Number of elements around head"]
        elementsCountAroundTorso = options["Number of elements around torso"]
        elementsCountAroundArm = options["Number of elements around arm"]
        elementsCountAroundLeg = options["Number of elements around leg"]
        isCore = options["Use Core"]

        layoutRegion = region.createRegion()
        networkLayout.generate(layoutRegion)  # ask scaffold to generate to get user-edited parameters
        layoutAnnotationGroups = networkLayout.getAnnotationGroups()
        networkMesh = networkLayout.getConstructionObject()

        annotationAlongCounts = []
        annotationAroundCounts = []
        defaultCoreBoundaryScalingMode = 1
        annotationCoreBoundaryScalingMode = []
        for layoutAnnotationGroup in layoutAnnotationGroups:
            alongCount = 0
            aroundCount = 0
            coreBoundaryScalingMode = 0
            name = layoutAnnotationGroup.getName()
            if "head" in name:
                alongCount = elementsCountAlongHead
                aroundCount = elementsCountAroundHead
            elif "neck" in name:
                alongCount = elementsCountAlongNeck
                aroundCount = elementsCountAroundHead
            elif "thorax" in name:
                alongCount = elementsCountAlongThorax
                aroundCount = elementsCountAroundTorso
                coreBoundaryScalingMode = 2
            elif "abdomen" in name:
                alongCount = elementsCountAlongAbdomen
                aroundCount = elementsCountAroundTorso
                coreBoundaryScalingMode = 2
            elif "shoulder" in name:
                alongCount = elementsCountAlongShoulder
                aroundCount = elementsCountAroundArm
            elif " brachium" in name:
                alongCount = elementsCountAlongBrachium
                aroundCount = elementsCountAroundArm
            elif " antebrachium" in name:
                alongCount = elementsCountAlongAntebrachium
                aroundCount = elementsCountAroundArm
            elif "hand" in name:
                alongCount = elementsCountAlongHand
                aroundCount = elementsCountAroundArm
            elif "hip" in name:
                alongCount = elementsCountAlongHip
                aroundCount = elementsCountAroundLeg
            elif "upper leg" in name:
                alongCount = elementsCountAlongUpperLeg
                aroundCount = elementsCountAroundLeg
            elif "lower leg" in name:
                alongCount = elementsCountAlongLowerLeg
                aroundCount = elementsCountAroundLeg
            elif "foot" in name:
                alongCount = elementsCountAlongFoot
                aroundCount = elementsCountAroundLeg
            annotationAlongCounts.append(alongCount)
            annotationAroundCounts.append(aroundCount)
            annotationCoreBoundaryScalingMode.append(coreBoundaryScalingMode)

        tubeNetworkMeshBuilder = BodyTubeNetworkMeshBuilder(
            networkMesh,
            targetElementDensityAlongLongestSegment=2.0,  # not used for body
            layoutAnnotationGroups=layoutAnnotationGroups,
            annotationElementsCountsAlong=annotationAlongCounts,
            defaultElementsCountAround=options["Number of elements around head"],
            annotationElementsCountsAround=annotationAroundCounts,
            elementsCountThroughShell=options["Number of elements through shell"],
            isCore=isCore,
            elementsCountTransition=options['Number of elements across core transition'],
            defaultElementsCountCoreBoxMinor=options["Number of elements across core box minor"],
            annotationElementsCountsCoreBoxMinor=[],
            defaultCoreBoundaryScalingMode=defaultCoreBoundaryScalingMode,
            annotationCoreBoundaryScalingMode=annotationCoreBoundaryScalingMode,
            useOuterTrimSurfaces=True)

        meshDimension = 3
        tubeNetworkMeshBuilder.build()
        generateData = TubeNetworkMeshGenerateData(
            region, meshDimension,
            isLinearThroughShell=False,
            isShowTrimSurfaces=options["Show trim surfaces"])
        tubeNetworkMeshBuilder.generateMesh(generateData)
        annotationGroups = generateData.getAnnotationGroups()
        fieldmodule = region.getFieldmodule()
        if isCore:
            mesh = fieldmodule.findMeshByDimension(meshDimension)
            thoraxGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("thorax"))
            abdomenGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("abdomen"))
            coreGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("core"))

            thoracicCavityGroup = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_body_term("thoracic cavity"))
            is_thoracic_cavity = fieldmodule.createFieldAnd(thoraxGroup.getGroup(), coreGroup.getGroup())
            thoracicCavityGroup.getMeshGroup(mesh).addElementsConditional(is_thoracic_cavity)

            abdominalCavityGroup = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_body_term("abdominal cavity"))
            is_abdominal_cavity = fieldmodule.createFieldAnd(abdomenGroup.getGroup(), coreGroup.getGroup())
            abdominalCavityGroup.getMeshGroup(mesh).addElementsConditional(is_abdominal_cavity)

        # Kinematic tree markers 
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        node_identifier = max(1, get_maximum_node_identifier(nodes) + 1)
        coordinates = find_or_create_field_coordinates(fieldmodule)
        stickman_markers = networkLayout._scaffoldSettings['Kinematic tree']
        for marker_name, marker_position in stickman_markers.items():
            marker_group = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, (marker_name, ""), isMarker=True
                )
            marker_group.createMarkerNode(
                node_identifier, coordinates, marker_position
                )
        return annotationGroups, None

    @classmethod
    def defineFaceAnnotations(cls, region, options, annotationGroups):
        """
        Add face annotation groups from the highest dimension mesh.
        Must have defined faces and added subelements for highest dimension groups.

        :param region: Zinc region containing model.
        :param options: Dict containing options. See getDefaultOptions().
        :param annotationGroups: List of annotation groups for top-level elements.
        New face annotation groups are appended to this list.
        """
        isCore = options["Use Core"]

        # create 2-D surface mesh groups, 1-D spinal cord
        fieldmodule = region.getFieldmodule()
        mesh2d = fieldmodule.findMeshByDimension(2)
        mesh1d = fieldmodule.findMeshByDimension(1)

        is_exterior = fieldmodule.createFieldIsExterior()
        is_face_xi3_0 = fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0)

        skinGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, get_body_term("skin epidermis outer surface"))
        is_skin = is_exterior if isCore else fieldmodule.createFieldAnd(
            is_exterior, fieldmodule.createFieldNot(is_face_xi3_0))
        skinGroup.getMeshGroup(mesh2d).addElementsConditional(is_skin)

        leftArmGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("left upper limb"))
        leftArmSkinGroup = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_body_term("left upper limb skin epidermis outer surface"))
        leftArmSkinGroup.getMeshGroup(mesh2d).addElementsConditional(
            fieldmodule.createFieldAnd(leftArmGroup.getGroup(), is_exterior))
        rightArmGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("right upper limb"))
        rightArmSkinGroup = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_body_term("right upper limb skin epidermis outer surface"))
        rightArmSkinGroup.getMeshGroup(mesh2d).addElementsConditional(
            fieldmodule.createFieldAnd(rightArmGroup.getGroup(), is_exterior))
        leftLegGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("left lower limb"))
        leftLegSkinGroup = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_body_term("left lower limb skin epidermis outer surface"))
        leftLegSkinGroup.getMeshGroup(mesh2d).addElementsConditional(
            fieldmodule.createFieldAnd(leftLegGroup.getGroup(), is_exterior))
        rightLegGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("right lower limb"))
        rightLegSkinGroup = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_body_term("right lower limb skin epidermis outer surface"))
        rightLegSkinGroup.getMeshGroup(mesh2d).addElementsConditional(
            fieldmodule.createFieldAnd(rightLegGroup.getGroup(), is_exterior))

        if isCore:
            coreGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("core"))
            shellGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("shell"))
            leftGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("left"))
            rightGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("right"))
            dorsalGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("dorsal"))

            is_core_shell = fieldmodule.createFieldAnd(coreGroup.getGroup(), shellGroup.getGroup())
            is_left_right = fieldmodule.createFieldAnd(leftGroup.getGroup(), rightGroup.getGroup())
            is_left_right_dorsal = fieldmodule.createFieldAnd(is_left_right, dorsalGroup.getGroup())

            neckGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("neck"))
            thoracicCavityGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("thoracic cavity"))
            abdominalCavityGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("abdominal cavity"))
            armGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("upper limb"))
            legGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("lower limb"))

            thoracicCavityBoundaryGroup = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_body_term("thoracic cavity boundary surface"))
            is_thoracic_cavity_boundary = fieldmodule.createFieldAnd(
                thoracicCavityGroup.getGroup(),
                fieldmodule.createFieldOr(
                    fieldmodule.createFieldOr(neckGroup.getGroup(), armGroup.getGroup()),
                    fieldmodule.createFieldOr(shellGroup.getGroup(), abdominalCavityGroup.getGroup())))
            thoracicCavityBoundaryGroup.getMeshGroup(mesh2d).addElementsConditional(is_thoracic_cavity_boundary)

            abdominalCavityBoundaryGroup = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_body_term("abdominal cavity boundary surface"))
            is_abdominal_cavity_boundary = fieldmodule.createFieldAnd(
                abdominalCavityGroup.getGroup(),
                fieldmodule.createFieldOr(
                    thoracicCavityGroup.getGroup(),
                    fieldmodule.createFieldOr(shellGroup.getGroup(), legGroup.getGroup())))
            abdominalCavityBoundaryGroup.getMeshGroup(mesh2d).addElementsConditional(is_abdominal_cavity_boundary)

            diaphragmGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, get_body_term("diaphragm"))
            is_diaphragm = fieldmodule.createFieldAnd(thoracicCavityGroup.getGroup(), abdominalCavityGroup.getGroup())
            diaphragmGroup.getMeshGroup(mesh2d).addElementsConditional(is_diaphragm)

            spinalCordGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, get_body_term("spinal cord"))
            is_spinal_cord = fieldmodule.createFieldAnd(is_core_shell, is_left_right_dorsal)
            spinalCordGroup.getMeshGroup(mesh1d).addElementsConditional(is_spinal_cord)


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


def getJointRotationFrames(jointAbductionRadians, jointFlexionRadians, proximalDir, upwardAbduction=True, ventralFlexion=True):
    """
    Get the directions for the joint and distal nodes after applying a rotation parameter. 
    Joint directions are rotated by half the rotation angle for smoothness
    
    :param jointAbductionRadians: Rotation parameter fixing the d3 (Front) direction
    :param jointFlexionRadians: Rotation parameter fixing the d2 (Side) direction
    :param proximalDir: Parameters for the proximal (before the joint) node
    :param upwardAbduction: If true, abduction is done in the d3 direction. If false, -d3
    :param ventralFlexion: If true, abduction is done in the d2 direction. If false, -d2 
    :return jointDir: Paramters for the joint node
    :return distalDir: Paramters for the distal (after the joint) node
    """
    proximalDirn, proximalSide, proximalFront = proximalDir[1:4]
    ventral = 1 if upwardAbduction else -1 
    jointAbductionMatrix = axis_angle_to_rotation_matrix(mult(proximalFront, -1*ventral), jointAbductionRadians)
    jointHalfAbductionMatrix = axis_angle_to_rotation_matrix(mult(proximalFront, -1*ventral), jointAbductionRadians/2)
    ventral = 1 if ventralFlexion else -1 
    jointFlexionMatrix = axis_angle_to_rotation_matrix(mult(proximalSide, -1*ventral), jointFlexionRadians)
    jointHalfFlexionMatrix = axis_angle_to_rotation_matrix(mult(proximalSide, -1*ventral), jointFlexionRadians/2)
    jointRotationMatrix = matrix_mult(jointFlexionMatrix,jointAbductionMatrix)
    jointHalfRotationMatrix = matrix_mult(jointHalfFlexionMatrix, jointHalfAbductionMatrix)
    # Joint directions (frame is rotated by half the abduction angle)
    jointDirn = matrix_vector_mult(jointHalfRotationMatrix, proximalDirn)
    jointSide = matrix_vector_mult(jointHalfRotationMatrix, proximalSide)
    jointFront = matrix_vector_mult(jointHalfRotationMatrix, proximalFront)
    # Distal directions
    distalDirn = matrix_vector_mult(jointRotationMatrix, proximalDirn)
    distalSide = matrix_vector_mult(jointRotationMatrix, proximalSide)
    distalFront = matrix_vector_mult(jointRotationMatrix, proximalFront)
    jointDir = [0, jointDirn, jointSide, jointFront, 0, 0, 0]
    distalDir = [0, distalDirn, distalSide, distalFront, 0, 0, 0]
    return jointDir, distalDir


def getJointNodePosition(jointAbductionRadians, jointFlexionRadians, \
    proximalDir, jointDir, distalDir, rotationCoeff, upwardAbduction=True, ventralFlexion=True):
    """
    Get the position of the joint node after applying rotations, as well as the magnitude of 
    d2, d3, d12 and d13
    
    :param jointAbductionRadians: Rotation parameter fixing the d3 (Front) direction
    :param jointFlexionRadians: Rotation parameter fixing the d2 (Side) direction
    :param proximalDir: Parameters for the proximal (before the joint) node
    :param jointDir: Paramters for the joint node
    :param distalDir: Paramters for the distal (after the joint) node
    :param rotationCoeff: Higher values move the joint node to give a smoother transition from proximal to distal node
    :param upwardAbduction: If true, abduction is done in the d3 direction. If false, -d3
    :param ventralFlexion: If true, abduction is done in the d2 direction. If false, -d2 
    
    :return jointDir: parameters for the joint node
    """
    ventral = 1 if ventralFlexion else -1 
    upward = 1 if upwardAbduction else -1 
    proximalNodePosition, proximalDirn, proximalSide, proximalFront = proximalDir[0:4]
    jointNodePosition, jointDirn, jointSide, jointFront = jointDir[0:4]
    distalNodePosition, distalDirn, distalSide, distalFront = distalDir[0:4]
    # Flexion node adjustment
    jointAngleRadians = math.pi - jointFlexionRadians
    frontScale = jointDir[6]
    rotationRotFactor = 1*math.sin(jointAngleRadians)     
    jointRotFactor = 1/math.sin(jointAngleRadians/2)
    d13RotFactor = math.sqrt(2)*math.tan(jointFlexionRadians/2)
    rotDisplacementFactor = rotationCoeff*frontScale*rotationRotFactor
    jointNodePosition = adjustJointNodePosition(proximalNodePosition, jointNodePosition, \
        jointFront, distalDirn, rotDisplacementFactor, ventralFlexion)
    d3_mag = frontScale*(jointRotFactor-(rotationCoeff*rotationRotFactor))
    d13_mag = -1*ventral*frontScale*d13RotFactor
    # Abduction node adjustment
    jointAngleRadians = math.pi - jointAbductionRadians
    sideScale = jointDir[5]
    jointSide = mult(jointSide, -1)
    rotationRotFactor = 1*math.sin(jointAngleRadians)     
    jointRotFactor = 1/math.sin(jointAngleRadians/2)
    d12RotFactor = math.sqrt(2)*math.tan(jointAbductionRadians/2)
    rotDisplacementFactor = rotationCoeff*sideScale*rotationRotFactor
    jointNodePosition = adjustJointNodePosition(proximalNodePosition, jointNodePosition, \
        jointSide, distalDirn, rotDisplacementFactor, upwardAbduction)
    d2_mag = sideScale*(jointRotFactor-(rotationCoeff*rotationRotFactor))
    d12_mag = upward*sideScale*d12RotFactor
    return [jointNodePosition, d2_mag, d3_mag, d12_mag, d13_mag]

def getDistalNodePosition(jointAbductionRadians, jointFlexionRadians, proximalDir, jointDir, distalDir, rotationCoeff):
    """
    Get the position of the distal node after applying rotations
    
    :param jointAbductionRadians: Rotation parameter fixing the d3 (Front) direction
    :param jointFlexionRadians: Rotation parameter fixing the d2 (Side) direction
    :param proximalDir: Parameters for the proximal (before the joint) node
    :param jointDir: Paramters for the joint node
    :param distalDir: Paramters for the distal (after the joint) node
    :param rotationCoeff: Higher values move the joint node to give a smoother transition from proximal to distal node
    
    :return distalNodePosition: coordiantes for the distal node
    """
    proximalNodePosition, proximalDirn, proximalSide, proximalFront = proximalDir[0:4]
    jointNodePosition, jointDirn, jointSide, jointFront = jointDir[0:4]
    distalNodePosition, distalDirn, distalSide, distalFront = distalDir[0:4]
    # Flexion
    jointFrontScale = jointDir[6]
    jointAngleRadians = math.pi - jointFlexionRadians
    flexionRotFactor = 1*math.sin(jointAngleRadians)     
    rotDisplacementFactor = rotationCoeff*jointFrontScale*flexionRotFactor
    distalNodePosition = adjustJointNodePosition(jointNodePosition, distalNodePosition, \
        jointDirn, proximalDirn, rotDisplacementFactor)
    # Abduction
    jointSideScale = jointDir[5]
    jointAngleRadians = math.pi - jointAbductionRadians
    abductionRotFactor = 1*math.sin(jointAngleRadians)     
    rotDisplacementFactor = rotationCoeff*jointSideScale*abductionRotFactor
    distalNodePosition = adjustJointNodePosition(jointNodePosition, distalNodePosition, \
        jointDirn, proximalDirn, rotDisplacementFactor)
    return distalNodePosition

def adjustJointNodePosition(proximalNodePosition, nodePosition, d1, d2, dispFactor, ventral = True):
    """
    Displaces a node towards the center of the tube network,
    while preserving the distance between the current node and the previous (proximal) node

    :param proximalNodePosition: Position of the node previous to nodePosition
    :param nodePosition: Original position of the node to be rotated.
    :param lenScale: Distance between the joint node and the proximal/distal node.
    :param d1: Typically jointFront or or jointSide.
    :param d2: Typically distalDirn or jointDirn.
    :param dispFactor: Measure of displacement of the node away from the corner.
    """
    initialDir = sub(nodePosition, proximalNodePosition)
    lenScale = magnitude(initialDir)
    ventral = 1 if (ventral) else -1
    jointAdjustDir = add(
            set_magnitude(d1, ventral*dispFactor), 
            set_magnitude(d2, dispFactor), 
        )
    jointAdjustDir =  add(initialDir, jointAdjustDir)
    jointAdjustDir = set_magnitude(jointAdjustDir, lenScale)
    adjustedNodePosition = add(proximalNodePosition, jointAdjustDir)
    return adjustedNodePosition
