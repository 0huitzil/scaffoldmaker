"""
Generates a 3D body coordinates using tube network mesh.
"""
from cmlibs.maths.vectorops import add, cross, mult, set_magnitude, sub
from cmlibs.utils.zinc.field import Field, find_or_create_field_coordinates
from cmlibs.zinc.element import Element
from cmlibs.zinc.node import Node
from scaffoldmaker.annotation.annotationgroup import (
    AnnotationGroup, findOrCreateAnnotationGroupForTerm, getAnnotationGroupForTerm)
from scaffoldmaker.annotation.body_terms import get_body_term
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils.interpolation import (
    computeCubicHermiteEndDerivative, getCubicHermiteArcLength, interpolateLagrangeHermiteDerivative,
    sampleCubicHermiteCurvesSmooth, smoothCubicHermiteDerivativesLine)
from scaffoldmaker.utils.networkmesh import NetworkMesh
from scaffoldmaker.utils.tubenetworkmesh import BodyTubeNetworkMeshBuilder, TubeNetworkMeshGenerateData
import math
from cmlibs.utils.zinc.finiteelement import get_maximum_node_identifier

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
        options["Structure"] = (
            "(1-2-3-4,"
            "4-5-6.1,"
            "6.2-14-15-16-17-18-19,19-20,"
            "6.3-21-22-23-24-25-26,26-27,"
            "6.1-7-8-9,"
            "9-10-11-12-13.1,"
            "13.2-28-29-30-31-32,32-33-34,"
            "13.3-35-36-37-38-39,39-40-41")
        options["Define inner coordinates"] = True
        options["Head depth"] = 2.0
        options["Head length"] = 2.5
        options["Head width"] = 2.0
        options["Neck length"] = 1.0
        options["Shoulder drop"] = 1.0
        options["Shoulder width"] = 4.5
        options["Arm lateral angle degrees"] = 10.0
        options["Arm length"] = 7.3
        options["Arm top diameter"] = 1.0
        options["Arm twist angle degrees"] = 0.0
        options["Wrist thickness"] = 0.5
        options["Wrist width"] = 0.7
        options["Hand length"] = 2.0
        options["Hand thickness"] = 0.2
        options["Hand width"] = 1.0
        options["Thorax length"] = 2.5
        options["Abdomen length"] = 3.0
        options["Torso depth"] = 2.2
        options["Torso width"] = 3.2
        options["Pelvis drop"] = 1.25
        options["Pelvis width"] = 2.0
        options["Leg lateral angle degrees"] = 10.0
        options["Leg length"] = 9.0
        options["Leg top diameter"] = 2.0
        options["Leg bottom diameter"] = 0.7
        options["Foot height"] = 1.25
        options["Foot length"] = 2.5
        options["Foot thickness"] = 0.3
        options["Foot width"] = 1.0
        options["Inner proportion default"] = 0.7
        options["Inner proportion head"] = 0.7
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
            "Arm lateral angle degrees",
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
            "Leg lateral angle degrees",
            "Leg length",
            "Leg top diameter",
            "Leg bottom diameter",
            "Foot height",
            "Foot length",
            "Foot thickness",
            "Foot width",
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
            "Arm lateral angle degrees": (-60.0, 200.0),
            "Arm twist angle degrees": (-90.0, 90.0),
            "Leg lateral angle degrees": (-20.0, 60.0)
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
        armAngleRadians = math.radians(options["Arm lateral angle degrees"])
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
        legAngleRadians = math.radians(options["Leg lateral angle degrees"])
        legLength = options["Leg length"]
        legTopRadius = 0.5 * options["Leg top diameter"]
        legBottomRadius = 0.5 * options["Leg bottom diameter"]
        footHeight = options["Foot height"]
        footLength = options["Foot length"]
        halfFootThickness = 0.5 * options["Foot thickness"]
        halfFootWidth = 0.5 * options["Foot width"]
        innerProportionDefault = options["Inner proportion default"]
        innerProportionHead = options["Inner proportion head"]

        networkMesh = NetworkMesh(structure)
        networkMesh.create1DLayoutMesh(region)

        fieldmodule = region.getFieldmodule()
        mesh = fieldmodule.findMeshByDimension(1)

        # set up element annotations
        bodyGroup = AnnotationGroup(region, get_body_term("body"))
        headGroup = AnnotationGroup(region, get_body_term("head"))
        neckGroup = AnnotationGroup(region, get_body_term("neck"))
        armGroup = AnnotationGroup(region, get_body_term("upper limb"))
        armToHandGroup = AnnotationGroup(region, ("arm to hand", ""))
        leftArmGroup = AnnotationGroup(region, get_body_term("left upper limb"))
        rightArmGroup = AnnotationGroup(region, get_body_term("right upper limb"))
        handGroup = AnnotationGroup(region, get_body_term("hand"))
        leftHandGroup = AnnotationGroup(region, get_body_term("left hand"))
        rightHandGroup = AnnotationGroup(region, get_body_term("right hand"))
        thoraxGroup = AnnotationGroup(region, get_body_term("thorax"))
        abdomenGroup = AnnotationGroup(region, get_body_term("abdomen"))
        legGroup = AnnotationGroup(region, get_body_term("lower limb"))
        legToFootGroup = AnnotationGroup(region, ("leg to foot", ""))
        leftLegGroup = AnnotationGroup(region, get_body_term("left lower limb"))
        rightLegGroup = AnnotationGroup(region, get_body_term("right lower limb"))
        footGroup = AnnotationGroup(region, get_body_term("foot"))
        leftFootGroup = AnnotationGroup(region, get_body_term("left foot"))
        rightFootGroup = AnnotationGroup(region, get_body_term("right foot"))
        annotationGroups = [bodyGroup, headGroup, neckGroup,
                            armGroup, armToHandGroup, leftArmGroup, rightArmGroup, handGroup,
                            leftHandGroup, rightHandGroup, thoraxGroup, abdomenGroup,
                            legGroup, legToFootGroup, leftLegGroup, rightLegGroup, footGroup,
                            leftFootGroup, rightFootGroup]
        bodyMeshGroup = bodyGroup.getMeshGroup(mesh)
        elementIdentifier = 1
        headElementsCount = 3
        meshGroups = [bodyMeshGroup, headGroup.getMeshGroup(mesh)]
        for e in range(headElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1
        neckElementsCount = 2
        meshGroups = [bodyMeshGroup, neckGroup.getMeshGroup(mesh)]
        for e in range(neckElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1
        left = 0
        right = 1
        armToHandElementsCount = 6
        handElementsCount = 1
        armMeshGroup = armGroup.getMeshGroup(mesh)
        armToHandMeshGroup = armToHandGroup.getMeshGroup(mesh)
        handMeshGroup = handGroup.getMeshGroup(mesh)
        for side in (left, right):
            sideArmGroup = leftArmGroup if (side == left) else rightArmGroup
            sideHandGroup = leftHandGroup if (side == left) else rightHandGroup
            meshGroups = [bodyMeshGroup, armMeshGroup, armToHandMeshGroup, sideArmGroup.getMeshGroup(mesh)]
            for e in range(armToHandElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            meshGroups = [bodyMeshGroup, armMeshGroup, handMeshGroup,
                          sideArmGroup.getMeshGroup(mesh), sideHandGroup.getMeshGroup(mesh)]
            for e in range(handElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
        thoraxElementsCount = 3
        abdomenElementsCount = 4
        meshGroups = [bodyMeshGroup, thoraxGroup.getMeshGroup(mesh)]
        for e in range(thoraxElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1
        meshGroups = [bodyMeshGroup, abdomenGroup.getMeshGroup(mesh)]
        for e in range(abdomenElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1
        legToFootElementsCount = 5
        footElementsCount = 2
        legMeshGroup = legGroup.getMeshGroup(mesh)
        legToFootMeshGroup = legToFootGroup.getMeshGroup(mesh)
        footMeshGroup = footGroup.getMeshGroup(mesh)
        for side in (left, right):
            sideLegGroup = leftLegGroup if (side == left) else rightLegGroup
            sideFootGroup = leftFootGroup if (side == left) else rightFootGroup
            meshGroups = [bodyMeshGroup, legMeshGroup, legToFootMeshGroup, sideLegGroup.getMeshGroup(mesh)]
            for e in range(legToFootElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            meshGroups = [bodyMeshGroup, legMeshGroup, footMeshGroup,
                          sideLegGroup.getMeshGroup(mesh), sideFootGroup.getMeshGroup(mesh)]
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

        headInnerOffset = (1.0 - innerProportionHead) * halfHeadWidth
        headLengthInner = headLength - headInnerOffset
        ix, id1 = sampleCubicHermiteCurvesSmooth(
            [[headInnerOffset, 0.0, 0.0], [headLength, 0.0, 0.0]],
            [[headLengthInner, 0.0, 0.0]] * 2,
            headElementsCount, derivativeMagnitudeEnd=headScale)[:2]
        for i in range(headElementsCount):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = [headScale * i, 0.0, 0.0]
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
            setNodeFieldParameters(innerCoordinates, fieldcache, ix[i], id1[i], id2, id3)
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

        # arms
        # rotate shoulder with arm, pivoting about shoulder drop below arm junction on network
        # this has the realistic effect of shoulders becoming narrower with higher angles
        # initial shoulder rotation with arm is negligible, hence:
        shoulderRotationFactor = 1.0 - math.cos(0.5 * armAngleRadians)
        # assume shoulder drop is half shrug distance to get limiting shoulder angle for 180 degree arm rotation
        shoulderLimitAngleRadians = math.asin(1.5 * shoulderDrop / halfShoulderWidth)
        shoulderAngleRadians = shoulderRotationFactor * shoulderLimitAngleRadians
        armStartX = thoraxStartX + shoulderDrop - halfShoulderWidth * math.sin(shoulderAngleRadians)
        nonHandArmLength = armLength - handLength
        armScale = nonHandArmLength / (armToHandElementsCount - 2)  # 2 == shoulder elements count
        d12_mag = (halfWristThickness - armTopRadius) / (armToHandElementsCount - 2)
        d13_mag = (halfWristWidth - armTopRadius) / (armToHandElementsCount - 2)
        for side in (left, right):
            armAngle = armAngleRadians if (side == left) else -armAngleRadians
            cosArmAngle = math.cos(armAngle)
            sinArmAngle = math.sin(armAngle)
            armStartY = (halfShoulderWidth if (side == left) else -halfShoulderWidth) * math.cos(shoulderAngleRadians)
            x = [armStartX, armStartY, 0.0]
            armDirn = [cosArmAngle, sinArmAngle, 0.0]
            armSide = [-sinArmAngle, cosArmAngle, 0.0]
            armFront = cross(armDirn, armSide)
            d1 = mult(armDirn, armScale)
            # set leg versions 2 (left) and 3 (right) on leg junction node, and intermediate shoulder node
            sd1 = interpolateLagrangeHermiteDerivative(sx, x, d1, 0.0)
            nx, nd1 = sampleCubicHermiteCurvesSmooth([sx, x], [sd1, d1], 2, derivativeMagnitudeEnd=armScale)[0:2]
            arcLengths = [getCubicHermiteArcLength(nx[i], nd1[i], nx[i + 1], nd1[i + 1]) for i in range(2)]
            sd2_list = []
            sd3_list = []
            sNodeIdentifiers = []
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
            sd2_list.append([-armTopRadius * sinArmAngle, armTopRadius * cosArmAngle, 0.0])
            sd3_list.append([0.0, 0.0, armTopRadius])
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
            # main part of arm to wrist
            elementTwistAngle = ((armTwistAngleRadians if (side == left) else -armTwistAngleRadians) /
                                 (armToHandElementsCount - 3))
            for i in range(armToHandElementsCount - 1):
                xi = i / (armToHandElementsCount - 2)
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                x = [armStartX + d1[0] * i, armStartY + d1[1] * i, d1[2] * i]
                halfThickness = xi * halfWristThickness + (1.0 - xi) * armTopRadius
                halfWidth = xi * halfWristWidth + (1.0 - xi) * armTopRadius
                if i == 0:
                    twistAngle = 0.0
                elif i == (armToHandElementsCount - 2):
                    twistAngle = armTwistAngleRadians if (side == left) else -armTwistAngleRadians
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
                    if i < (armToHandElementsCount - 2):
                        d12 = add(d12, set_magnitude(d3, -halfThickness * elementTwistAngle))
                        d13 = add(d13, set_magnitude(d2, halfWidth * elementTwistAngle))
                id2 = mult(d2, innerProportionDefault)
                id3 = mult(d3, innerProportionDefault)
                id12 = mult(d12, innerProportionDefault)
                id13 = mult(d13, innerProportionDefault)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
                nodeIdentifier += 1
            # hand
            assert handElementsCount == 1
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            hx = [armStartX + armLength * cosArmAngle, armStartY + armLength * sinArmAngle, 0.0]
            hd1 = computeCubicHermiteEndDerivative(x, d1, hx, d1)
            twistAngle = armTwistAngleRadians if (side == left) else -armTwistAngleRadians
            if twistAngle == 0.0:
                hd2 = set_magnitude(d2, halfHandThickness)
                hd3 = [0.0, 0.0, halfHandWidth]
            else:
                cosTwistAngle = math.cos(twistAngle)
                sinTwistAngle = math.sin(twistAngle)
                hd2 = sub(mult(armSide, halfHandThickness * cosTwistAngle),
                          mult(armFront, halfHandThickness * sinTwistAngle))
                hd3 = add(mult(armFront, halfHandWidth * cosTwistAngle),
                          mult(armSide, halfHandWidth * sinTwistAngle))
            hid2 = mult(hd2, innerProportionDefault)
            hid3 = mult(hd3, innerProportionDefault)
            setNodeFieldParameters(coordinates, fieldcache, hx, hd1, hd2, hd3)
            setNodeFieldParameters(innerCoordinates, fieldcache, hx, hd1, hid2, hid3)
            nodeIdentifier += 1

        # legs
        legStartX = abdomenStartX + abdomenLength + pelvisDrop
        nonFootLegLength = legLength - footHeight
        legScale = nonFootLegLength / (legToFootElementsCount - 1)
        d12_mag = (legBottomRadius - legTopRadius) / (armToHandElementsCount - 2)
        d13_mag = (legBottomRadius - legTopRadius) / (armToHandElementsCount - 2)

        pd3 = [0.0, 0.0, 0.5 * legTopRadius + 0.5 * halfTorsoDepth]
        pid3 = mult(pd3, innerProportionDefault)
        for side in (left, right):
            legAngle = legAngleRadians if (side == left) else -legAngleRadians
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
            # main part of leg to ankle
            for i in range(legToFootElementsCount):
                xi = i / legToFootElementsCount
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                x = [legStartX + d1[0] * i, legStartY + d1[1] * i, d1[2] * i]
                radius = xi * legBottomRadius + (1.0 - xi) * legTopRadius
                d2 = mult(legSide, radius)
                d3 = [0.0, 0.0, radius]
                id2 = mult(d2, innerProportionDefault)
                id3 = mult(d3, innerProportionDefault)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
                nodeIdentifier += 1
            # foot
            fx = [x,
                  add(add(legStart, mult(legDirn, legLength - 1.5 * halfFootThickness)),
                      [0.0, 0.0, legBottomRadius]),
                  add(add(legStart, mult(legDirn, legLength - halfFootThickness)),
                      [0.0, 0.0, footLength - legBottomRadius])]
            fd1 = smoothCubicHermiteDerivativesLine(
                fx, [d1, [0.0, 0.0, 0.5 * footLength], [0.0, 0.0, 0.5 * footLength]],
                fixAllDirections=True, fixStartDerivative=True)
            fd2 = [d2, mult(legSide, halfFootWidth), mult(legSide, halfFootWidth)]
            fd3 = [d3,
                   set_magnitude(sub(legFront, legDirn),
                                 math.sqrt(2.0 * halfFootThickness * halfFootThickness) + legBottomRadius),
                   set_magnitude(cross(fd1[2], fd2[2]), halfFootThickness)]
            fd12 = sub(fd2[2], fd2[1])
            fd13 = sub(fd3[2], fd3[1])
            fid12 = mult(fd12, innerProportionDefault)
            fid13 = mult(fd13, innerProportionDefault)
            for i in range(1, 3):
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                setNodeFieldParameters(coordinates, fieldcache, fx[i], fd1[i], fd2[i], fd3[i], fd12, fd13)
                fid2 = mult(fd2[i], innerProportionDefault)
                fid3 = mult(fd3[i], innerProportionDefault)
                setNodeFieldParameters(innerCoordinates, fieldcache, fx[i], fd1[i], fid2, fid3, fid12, fid13)
                nodeIdentifier += 1

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
        return interactiveFunctions


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
        options["Number of elements along head"] = 4
        options["Number of elements along neck"] = 1
        options["Number of elements along thorax"] = 2
        options["Number of elements along abdomen"] = 2
        options["Number of elements along arm to hand"] = 5
        options["Number of elements along hand"] = 1
        options["Number of elements along leg to foot"] = 4
        options["Number of elements along foot"] = 2
        options["Number of elements around head"] = 12
        options["Number of elements around torso"] = 12
        options["Number of elements around arm"] = 8
        options["Number of elements around leg"] = 8
        options["Number of elements through shell"] = 1
        options["Show trim surfaces"] = False
        options["Use Core"] = True
        options["Define body coordinates"] = True
        options["Number of elements across core box minor"] = 2
        options["Number of elements across core transition"] = 1
        if "Medium" in useParameterSetName:
            options["Number of elements along head"] = 6
            options["Number of elements along neck"] = 2
            options["Number of elements along thorax"] = 3
            options["Number of elements along abdomen"] = 3
            options["Number of elements along arm to hand"] = 6
            options["Number of elements along hand"] = 1
            options["Number of elements along leg to foot"] = 6
            options["Number of elements along foot"] = 2
            options["Number of elements around head"] = 16
            options["Number of elements around torso"] = 16
            options["Number of elements around leg"] = 12
        elif "Fine" in useParameterSetName:
            options["Number of elements along head"] = 8
            options["Number of elements along neck"] = 2
            options["Number of elements along thorax"] = 4
            options["Number of elements along abdomen"] = 4
            options["Number of elements along arm to hand"] = 8
            options["Number of elements along hand"] = 2
            options["Number of elements along leg to foot"] = 8
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
            "Number of elements along arm to hand",
            "Number of elements along hand",
            "Number of elements along leg to foot",
            "Number of elements along foot",
            "Number of elements around head",
            "Number of elements around torso",
            "Number of elements around arm",
            "Number of elements around leg",
            "Number of elements through shell",
            "Show trim surfaces",
            "Use Core",
            "Define body coordinates",
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
            "Number of elements along arm to hand",
            "Number of elements along hand",
            "Number of elements along leg to foot",
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
            options["Number of elements through shell"] = 0

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
        elementsCountAlongArmToHand = options["Number of elements along arm to hand"]
        elementsCountAlongHand = options["Number of elements along hand"]
        elementsCountAlongLegToFoot = options["Number of elements along leg to foot"]
        elementsCountAlongFoot = options["Number of elements along foot"]
        elementsCountAroundHead = options["Number of elements around head"]
        elementsCountAroundTorso = options["Number of elements around torso"]
        elementsCountAroundArm = options["Number of elements around arm"]
        elementsCountAroundLeg = options["Number of elements around leg"]
        shell_count = options["Number of elements through shell"]
        core = options["Use Core"]
        define_body_coordinates = options["Define body coordinates"]

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
            elif "arm to hand" in name:
                alongCount = elementsCountAlongArmToHand
                aroundCount = elementsCountAroundArm
            elif "hand" in name:
                alongCount = elementsCountAlongHand
                aroundCount = elementsCountAroundArm
            elif "leg to foot" in name:
                alongCount = elementsCountAlongLegToFoot
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
            shell_count=shell_count,
            core=core,
            transition_count=options['Number of elements across core transition'],
            defaultElementsCountCoreBoxMinor=options["Number of elements across core box minor"],
            annotationElementsCountsCoreBoxMinor=[],
            defaultCoreBoundaryScalingMode=defaultCoreBoundaryScalingMode,
            annotationCoreBoundaryScalingMode=annotationCoreBoundaryScalingMode,
            useOuterTrimSurfaces=True)

        meshDimension = 3 if (shell_count or core) else 2
        tubeNetworkMeshBuilder.build()

        generateData = TubeNetworkMeshGenerateData(
            region, meshDimension,
            isLinearThroughShell=False,
            isShowTrimSurfaces=options["Show trim surfaces"])
        tubeNetworkMeshBuilder.generateMesh(generateData)
        annotationGroups = generateData.getAnnotationGroups()

        if define_body_coordinates:
            # Generate network layout with default parameters
            materialNetworkLayout = ScaffoldPackage(MeshType_1d_human_body_network_layout1)
            tmp_region = region.createRegion()
            tmp_layoutRegion = region.createRegion()

            materialNetworkLayout.generate(tmp_layoutRegion)
            materialAnnotationGroups = materialNetworkLayout.getAnnotationGroups()
            materialNetworkMesh = materialNetworkLayout.getConstructionObject()

            materialTubeNetworkMeshBuilder = BodyTubeNetworkMeshBuilder(
                materialNetworkMesh,
                targetElementDensityAlongLongestSegment=2.0,  # not used for body
                layoutAnnotationGroups=materialAnnotationGroups,
                annotationElementsCountsAlong=annotationAlongCounts,
                defaultElementsCountAround=options["Number of elements around head"],
                annotationElementsCountsAround=annotationAroundCounts,
                shell_count=shell_count,
                core=core,
                transition_count=options['Number of elements across core transition'],
                defaultElementsCountCoreBoxMinor=options["Number of elements across core box minor"],
                annotationElementsCountsCoreBoxMinor=[],
                defaultCoreBoundaryScalingMode=defaultCoreBoundaryScalingMode,
                annotationCoreBoundaryScalingMode=annotationCoreBoundaryScalingMode,
                useOuterTrimSurfaces=True
                )
            materialTubeNetworkMeshBuilder.build()
            generateData = TubeNetworkMeshGenerateData(
                tmp_region, meshDimension,
                coordinateFieldName= 'body coordinates',
                isLinearThroughShell=False,
                isShowTrimSurfaces=options["Show trim surfaces"]
            )
            materialTubeNetworkMeshBuilder.generateMesh(generateData)
            sir = tmp_region.createStreaminformationRegion()
            srm = sir.createStreamresourceMemory()
            tmp_region.write(sir)
            result, buffer = srm.getBuffer()
            sir = region.createStreaminformationRegion()
            srm = sir.createStreamresourceMemoryBuffer(buffer)
            region.read(sir)

            del tmp_region
            del tmp_layoutRegion

        if core and shell_count:
            fieldmodule = region.getFieldmodule()
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

        # Acupoint Markers
        if define_body_coordinates:
            markers = {
                'acupoints MRI marked': {
                    'CV 3': [9.44381737, 0.09147829, 0.8876839],
                    'CV 8': [7.83127382, 0.12361765, 1.08804391],
                    'CV 6': [8.30446066, 0.06688622, 1.05713872],
                    'CV 4': [8.67412665, 0.01013162, 1.01804633],
                    'CV 12': [6.87585944, 0.06476163, 1.09911749],
                    'CV 16': [5.93671651, 0.06011876, 1.09924028],
                    'CV 22': [3.74413887, 0.0, 1.05942668],
                    'CV 23': [3.13355261, 0.07057897, 1.02214324],
                    'ST 2': [1.59104285, 0.47543993, 0.87969434],
                    'ST 4': [2.17922261, 0.50514232, 0.86297596],
                    'ST 8': [0.49166133, 0.85273753, 0.30654797],
                    'ST 7': [1.72602954, 0.91588599, 0.40117577],
                    'ST 9': [3.17136731, 0.55866116, 0.87627511],
                    'ST 12': [3.54533943, 1.23147096, 0.65080642],
                    'ST 42': [18.47753193, 2.49083596, 0.68054094],
                    'ST 41': [18.15566906, 2.46590857, 0.46543254],
                    'ST 40': [16.11555381, 2.38334378, 0.49072118],
                    'ST 39': [16.32973223, 2.23284498, 0.5652165],
                    'ST 38': [16.14582098, 2.20811977, 0.5755302],
                    'ST 37': [15.81003563, 2.21186028, 0.57727574],
                    'ST 36': [15.33081341, 2.14370939, 0.60667124],
                    'ST 35': [14.4266394, 2.00537959, 0.66570164],
                    'ST 34': [13.38871861, 1.89779027, 0.70671515],
                    'ST 31': [10.11177136, 0.86862968, 0.99168125],
                    'ST 30': [9.82071265, 0.52791772, 0.94011348],
                    'ST 29': [9.41495744, 0.49709795, 0.98090112],
                    'ST 25': [7.87059459, 0.40734316, 1.06180279],
                    'ST 19': [6.10700368, 0.34233873, 1.07474167],
                    'LU 11': [10.80726383, 3.26130975, 0.32508615],
                    'LU 10': [10.04888395, 3.05846559, 0.23709169],
                    'LU 9': [9.77129635, 3.03949844, 0.2888774],
                    'LU 8': [9.57888526, 3.03613395, 0.31991817],
                    'LU 7': [9.38342091, 3.11321592, 0.35946485],
                    'LU 5': [6.90981289, 2.66539703, 0.43044795],
                    'LU 4': [5.4049995, 2.6315806, 0.417228],
                    'PC 4': [8.60209793, 2.68525948, 0.13916884],
                    'PC 5': [8.92150523, 2.74710497, 0.10206435],
                    'PC 6': [9.23857263, 2.82005477, 0.1090493],
                    'PC 8': [10.50087315, 3.12224257, 0.0],
                    'PC 7': [9.76098301, 2.92587756, 0.02735956],
                    'PC 3': [7.13278782, 2.40578404, 0.24929415],
                    'PC 2': [5.15067052, 2.19737716, 0.44433819],
                    'HT 8': [10.39336487, 3.12685156, -0.20675109],
                    'HT 7': [9.7195212, 2.92515129, -0.09452412],
                    'HT 5': [9.5098563, 2.87500292, -0.07905245],
                    'HT 3': [7.10940996, 2.33042702, -0.05197881],
                    'HT 2': [6.48344499, 2.2000118, -0.11153598],
                    'TE 4': [9.79901486, 3.42743883, -0.06251571],
                    'TE 5': [9.28943078, 3.36245825, 0.07238324],
                    'TE 20': [1.29689626, 0.98936663, -0.14457594],
                    'TE 21': [1.539824, 0.98722299, 0.15846088],
                    'LR 3': [18.61745036, 2.3603018, 1.11204306],
                    'LR 4': [18.2504691, 2.16693344, 0.40722805],
                    'LR 5': [17.62047153, 1.90515477, 0.3243277],
                    'LR 8': [14.40229275, 1.05199822, -0.28797623],
                    'LR 9': [13.09703873, 0.68161402, -0.11645029],
                    'LR 11': [11.35440432, 0.75838307, 0.82427243],
                    'LR 12': [9.49762403, 0.73049734, 1.01321165],
                    'LR 13': [7.48908528, 1.43041834, 0.49142466],
                    'LR 14': [6.02920488, 1.47870657, 0.41751418],
                    'SI 8': [6.91712908, 2.78106033, -0.41366881],
                    'SI 19': [1.78201945, 0.98379608, 0.17839895],
                    'SP 3': [18.94567004, 2.03816213, 1.08807395],
                    'SP 4': [18.79396939, 2.0800367, 0.86101794],
                    'KI 3': [18.26986016, 1.99101135, -0.19256278],
                    'SP 6': [17.57071019, 1.77276004, 0.03399734],
                    'SP 9': [15.08357203, 1.26217402, 0.35592386],
                    'SP 10': [13.42864816, 1.12577442, 0.66393366],
                    'SP 12': [9.24618859, 0.3801029, 0.9732377],
                    'SP 14': [8.14440056, 0.72971298, 0.98234917],
                    'KI 2': [18.70332291, 2.0657965, 0.48758987],
                    'KI 4': [18.41756518, 2.04687653, -0.22241579],
                    'KI 5': [18.55356161, 2.00670279, -0.06172022],
                    'KI 6': [18.51049149, 2.01513086, 0.23430724],
                    'KI 7': [17.81454311, 1.87368192, -0.19043686],
                    'KI 10': [14.18011519, 1.21194075, -0.57166767],
                    'KI 11': [10.25831032, 0.4220775, 0.81846982],
                    'KI 21': [6.30401959, 0.18992875, 1.09233802],
                    'KI 9': [16.89005058, 1.62206688, -0.13988937],
                    'SP 5': [16.8198876, 1.66503017, 0.28279719],
                    'KI 22': [5.56034162, 0.32243287, 1.0776843],
                    'ST 10': [3.49925581, 0.26650539, 0.99841812],
                    'KI 27': [4.04284927, 0.42359786, 1.01190095],
                    'ST 11': [3.58526366, 0.45755977, 0.94637163]
                    },
                'acupoints estimated': {
                    'CV 5': [8.48836072, 0.03854181, 1.03935278],
                    'CV 7': [8.06684749, 0.09570813, 1.07536863],
                    'CV 9': [7.59295184, 0.10690546, 1.09664793],
                    'CV 10': [7.35263978, 0.09136233, 1.09824045],
                    'CV 11': [7.11306787, 0.07765819, 1.09872991],
                    'CV 13': [6.64373237, 0.0637709, 1.09914434],
                    'CV 14': [6.40972872, 0.06301292, 1.0991646],
                    'CV 15': [6.17387995, 0.06191902, 1.09919341],
                    'CV 17': [5.5922152, 0.05614352, 1.09934041],
                    'CV 18': [5.23794686, 0.04953861, 1.09948914],
                    'CV 19': [4.86558591, 0.04406606, 1.09959803],
                    'CV 20': [4.47755871, 0.03356456, 1.09651467],
                    'CV 21': [4.11216999, 0.01714463, 1.08055393],
                    'ST 3': [1.89747854, 0.50336044, 0.86401766],
                    'ST 5': [2.04130226, 0.6861498, 0.72706153],
                    'ST 6': [1.8888332, 0.82212685, 0.56907672],
                    'ST 13': [3.89454034, 1.03890404, 0.77457742],
                    'ST 14': [4.22383135, 0.84612088, 0.90491033],
                    'ST 15': [4.60104737, 0.68324975, 0.99185615],
                    'ST 16': [5.02552282, 0.56881885, 1.02829181],
                    'ST 17': [5.40671048, 0.49684494, 1.04584686],
                    'ST 18': [5.75835379, 0.42477188, 1.06076854],
                    'ST 20': [6.40303514, 0.35297321, 1.07311711],
                    'ST 21': [6.6998415, 0.36402238, 1.07137298],
                    'ST 22': [6.99732858, 0.37559108, 1.0694851],
                    'ST 23': [7.2936106, 0.38776684, 1.06742956],
                    'ST 24': [7.5830016, 0.39966603, 1.06593455],
                    'ST 26': [8.23640588, 0.4225156, 1.04917211],
                    'ST 27': [8.61306003, 0.4422566, 1.03136744],
                    'ST 28': [9.00399107, 0.47256131, 1.0094366],
                    'ST 32': [11.26877428, 1.14489624, 0.9307543],
                    'ST 33': [12.34625262, 1.53788107, 0.8400051],
                    'LU 6': [8.14748782, 2.86984755, 0.39393856],
                    'HT 4': [8.31279497, 2.59900911, -0.01662202],
                    'HT 6': [9.61524225, 2.90025298, -0.08746153],
                    'TE 6': [8.78099279, 3.25624019, 0.20232643],
                    'TE 7': [8.2887269, 3.04955545, 0.35589014],
                    'TE 8': [7.7883216, 2.78830937, 0.40223281],
                    'TE 9': [7.29426279, 2.64134913, 0.40313431],
                    'TE 10': [6.79174031, 2.51895376, 0.40748],
                    'TE 11': [6.30284018, 2.35566551, 0.38660194],
                    'TE 12': [5.79997656, 1.59976179, 0.01838571],
                    'TE 13': [5.30426421, 1.94528256, 0.1515157],
                    'TE 14': [4.86649264, 1.8307346, 0.0924826],
                    'TE 15': [4.471294, 1.6517736, 0.07603018],
                    'TE 16': [3.32839544, 1.56313246, 0.21535987],
                    'TE 17': [3.19923319, 1.42432832, 0.04042955],
                    'TE 18': [2.42446499, 0.99362348, -0.11196153],
                    'TE 19': [1.86441236, 0.9854186, -0.16925695],
                    'LR 6': [16.56827943, 1.60115454, 0.28279811],
                    'LR 7': [15.50178116, 1.27064771, 0.10039782],
                    'LR 10': [12.23220271, 0.59465431, 0.46258774],
                    'SI 9': [6.48458421, 2.63422307, -0.44107974],
                    'SI 10': [6.06487929, 2.36922733, -0.42347517],
                    'SI 11': [5.64874161, 1.5231722, -0.33326436],
                    'SI 12': [5.25839876, 1.55576126, -0.25320347],
                    'SI 13': [4.86945216, 1.83139823, -0.09244522],
                    'SI 14': [4.48411625, 1.63934855, -0.06756318],
                    'SI 15': [3.38087664, 1.7104708, -0.12433524],
                    'SI 16': [3.25676645, 1.48717149, 0.1443772],
                    'SI 17': [3.16167953, 1.33476803, 0.11300109],
                    'SI 18': [2.22677053, 0.99538971, 0.09518997],
                    'SP 7': [16.75137667, 1.61699226, 0.22295071],
                    'SP 8': [15.9096577, 1.4715752, 0.34552016],
                    'SP 11': [11.54773933, 0.6797523, 0.74320783],
                    'SP 13': [8.69004765, 0.56977527, 1.02250827],
                    'KI 8': [17.35010278, 1.74381684, -0.16389554],
                    'KI 12': [9.77913707, 0.37406309, 0.88874852],
                    'KI 13': [9.31957803, 0.34729652, 0.95728876],
                    'KI 14': [8.91276321, 0.32169396, 1.00505933],
                    'KI 15': [8.52797445, 0.30237087, 1.03858099],
                    'KI 16': [8.15309728, 0.2843141, 1.06360273],
                    'KI 17': [7.78383634, 0.26398854, 1.08102012],
                    'KI 18': [7.41277037, 0.24105792, 1.08760187],
                    'KI 19': [7.03682766, 0.22181881, 1.08952015],
                    'KI 20': [6.66717905, 0.20556111, 1.09101287],
                    'KI 23': [5.27108759, 0.32522664, 1.07731891],
                    'KI 24': [4.96938751, 0.3299846, 1.07666903],
                    'KI 25': [4.65606349, 0.34639873, 1.07359997],
                    'KI 26': [4.34605377, 0.38052982, 1.05178341]
                    }
            }

            # Create annotation groups
            fieldmodule = region.getFieldmodule()
            nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            node_identifier = get_maximum_node_identifier(nodes) + 1
            body_coordinate_field = fieldmodule.findFieldByName('body coordinates')
            for group_name, markers_dict in markers.items():
                marker_group = findOrCreateAnnotationGroupForTerm(
                    annotationGroups, region, (group_name, "")
                    )
                nodeset_group = marker_group.getNodesetGroup(nodes)
                for marker_name, marker_position in markers_dict.items():
                    marker_group = findOrCreateAnnotationGroupForTerm(
                        annotationGroups, region, (marker_name, ""), isMarker=True
                        )
                    marker_node = marker_group.createMarkerNode(
                        node_identifier, body_coordinate_field, marker_position
                        )
                    nodeset_group.addNode(marker_node)
                    meridian_name = "".join(filter(str.isalpha, marker_name))
                    meridian_group = findOrCreateAnnotationGroupForTerm(
                    annotationGroups, region, (meridian_name, "")
                        )
                    meridian_nodeset_group = meridian_group.getNodesetGroup(nodes)
                    meridian_nodeset_group.addNode(marker_node)

                    node_identifier += 1
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
        is_core = options["Use Core"]
        shell_count = options["Number of elements through shell"]
        BodyTubeNetworkMeshBuilder.defineFaceAnnotations(region, annotationGroups, is_core, shell_count)

        # create 2-D surface mesh groups, 1-D spinal cord
        fieldmodule = region.getFieldmodule()
        mesh2d = fieldmodule.findMeshByDimension(2)
        mesh1d = fieldmodule.findMeshByDimension(1)

        is_exterior = fieldmodule.createFieldIsExterior()
        is_face_xi3_0 = fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0)

        skinGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, get_body_term("skin epidermis outer surface"))
        one = fieldmodule.createFieldConstant(1.0)
        is_skin = (is_exterior if is_core else fieldmodule.createFieldAnd(
            is_exterior, fieldmodule.createFieldNot(is_face_xi3_0))) if (is_core or shell_count) else one
        skinGroup.getMeshGroup(mesh2d).addElementsConditional(is_skin)

        skin_groups = [
            'right lower limb',
            'right upper limb',
            'right hand',
            'right foot',
            'left lower limb',
            'left upper limb',
            'left hand',
            'left foot',
        ]

        for annotation_group_name in skin_groups:
            annotation_group =  getAnnotationGroupForTerm(
            annotationGroups, get_body_term(annotation_group_name))
            skin_annotation_group = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region,
            get_body_term(annotation_group_name +  " skin epidermis outer surface"))
            skin_annotation_group.getMeshGroup(mesh2d).addElementsConditional(
            fieldmodule.createFieldAnd(annotation_group.getGroup(), is_skin))

        if is_core and shell_count:
            # define cavity surfaces, diaphragm and spinal cord
            headGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("head"))
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

            # now excluding head from spinal cord
            spinalCordGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, get_body_term("spinal cord"))
            is_spinal_cord = fieldmodule.createFieldAnd(
                fieldmodule.createFieldAnd(is_core_shell, is_left_right_dorsal),
                fieldmodule.createFieldNot(headGroup.getGroup()))
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
