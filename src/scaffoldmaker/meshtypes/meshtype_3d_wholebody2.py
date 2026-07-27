"""
Generates a 3D body coordinates using tube network mesh.
"""
import math

from cmlibs.maths.vectorops import add, cross, mult, set_magnitude, sub
from cmlibs.utils.zinc.field import Field, find_or_create_field_coordinates
from cmlibs.zinc.element import Element
from cmlibs.zinc.node import Node

from scaffoldmaker.annotation.annotationgroup import (
    AnnotationGroup,
    findOrCreateAnnotationGroupForTerm,
    getAnnotationGroupForTerm,
)
from scaffoldmaker.annotation.body_terms import get_body_term
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils.interpolation import (
    computeCubicHermiteEndDerivative,
    getCubicHermiteArcLength,
    interpolateLagrangeHermiteDerivative,
    sampleCubicHermiteCurvesSmooth,
    smoothCubicHermiteDerivativesLine,
)
from scaffoldmaker.utils.networkmesh import NetworkMesh
from scaffoldmaker.utils.tubenetworkmesh import (
    BodyTubeNetworkMeshBuilder,
    TubeNetworkMeshGenerateData,
)


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
            "1-2-3-4,"
            "4-5-6.1," 
            "6.2-14-15-16-17-18-19,19-20,"
            "6.3-21-22-23-24-25-26,26-27,"
            "6.1-7-8-9,"
            "9-10-11-12-13.1,"
            "13.2-28-29-30-31-32,32-33-34,"
            "13.3-35-36-37-38-39,39-40-41")
        options["Define inner coordinates"] = True
        options["Head depth"] = 2.0
        options["Head length"] = 2.2
        options["Head width"] = 2.0
        options["Neck length"] = 1.3
        options["Shoulder drop"] = 1.0
        options["Shoulder width"] = 4.5
        options["Arm lateral angle degrees"] = 10.0
        options["Arm length"] = 7.5
        options["Arm top diameter"] = 1.0
        options["Arm twist angle degrees"] = 0.0
        options["Wrist thickness"] = 0.5
        options["Wrist width"] = 0.7
        options["Hand length"] = 2.0
        options["Hand thickness"] = 0.2
        options["Hand width"] = 1.0
        options["Thorax length"] = 2.5
        options["Abdomen length"] = 3.0
        options["Torso depth"] = 2.5
        options["Torso width"] = 3.2
        options["Pelvis drop"] = 1.5
        options["Pelvis width"] = 2.0
        options["Leg lateral angle degrees"] = 10.0
        options["Leg length"] = 10.0
        options["Leg top diameter"] = 2.0
        options["Leg bottom diameter"] = 0.7
        options["Foot height"] = 1.25
        options["Foot length"] = 2.5
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
        thoraxGroup = AnnotationGroup(region, get_body_term("thorax"))
        abdomenGroup = AnnotationGroup(region, get_body_term("abdomen"))
        legGroup = AnnotationGroup(region, get_body_term("lower limb"))
        legToFootGroup = AnnotationGroup(region, ("leg to foot", ""))
        leftLegGroup = AnnotationGroup(region, get_body_term("left lower limb"))
        rightLegGroup = AnnotationGroup(region, get_body_term("right lower limb "))
        footGroup = AnnotationGroup(region, get_body_term("foot"))
        annotationGroups = [bodyGroup, headGroup, neckGroup,
                            armGroup, armToHandGroup, leftArmGroup, rightArmGroup, handGroup,
                            thoraxGroup, abdomenGroup,
                            legGroup, legToFootGroup, leftLegGroup, rightLegGroup, footGroup]
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
            meshGroups = [bodyMeshGroup, armMeshGroup, armToHandMeshGroup, sideArmGroup.getMeshGroup(mesh)]
            for e in range(armToHandElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            meshGroups = [bodyMeshGroup, armMeshGroup, handMeshGroup, sideArmGroup.getMeshGroup(mesh)]
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
            meshGroups = [bodyMeshGroup, legMeshGroup, legToFootMeshGroup, sideLegGroup.getMeshGroup(mesh)]
            for e in range(legToFootElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1
            meshGroups = [bodyMeshGroup, legMeshGroup, footMeshGroup, sideLegGroup.getMeshGroup(mesh)]
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
        options["Number of elements along head"] = 2
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
        options["Number of elements across core box minor"] = 2
        options["Number of elements across core transition"] = 1
        if "Medium" in useParameterSetName:
            options["Number of elements along head"] = 3
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
            options["Number of elements along head"] = 4
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
        elementsCountAlongArmToHand = options["Number of elements along arm to hand"]
        elementsCountAlongHand = options["Number of elements along hand"]
        elementsCountAlongLegToFoot = options["Number of elements along leg to foot"]
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

        if isCore:
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

        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        node_identifier = max(1, nodes.getSize() + 1)
        coordinates = find_or_create_field_coordinates(fieldmodule)
        stickman_markers = {
            'Level with the inferior border of the third left costal cartilage': 
            [3.335508199875883,
            0.1419426774928394,
            1.0058769780409123],
            'Internal surface of right half of manubrium sternum at level of superior border of right second sternocostal joint': 
            [2.8940751018966235,
            -0.18712796617174582,
            0.9835753858523458],
            'Level of lower anterior left sided border of T4 vertebra, level with sternal end of left second costal cartilage':
            [2.969765706123864,
            0.0652676018770293,
            -0.168790054929805],
            'Anterior to lower border of T12 vertebra': [5.174004005660772,
            -0.10239119476082352,
            -0.04710701062573053],
            'Left anterior inferior margin of C3 vertebra': [1.3959716638454522,
            0.21332281654425764,
            0.23737024216855437],
            'Right anterior inferior margin of C3 vertebra': [1.3811879287545805,
            -0.05694756224152475,
            0.2052309660320172],
            'At level of lower aspect of medial lip of intertubercular/bicipital groove of left humerus where teres major muscle inserts': [2.789194792265203,
            2.266229775333413,
            0.14732045085104334],
            'At level of lower aspect of medial lip of intertubercular/bicipital groove of right humerus where teres major muscle inserts': [2.7480963245891887,
            -2.228763440128004,
            0.16433015101655338],
            '1cm distal to right elbow joint': [2.9157106115170324,
            -4.709059051776541,
            -0.13051241811798053],
            '1cm below left elbow joint at level of neck of radius': [2.8571677194756817,
            4.833965779583744,
            -0.0330166239493983],
            '1cm below right elbow joint at level of neck of radius': [2.8676431276484013,
            -4.831826373195813,
            -0.07401390799843152],
            'Level of 4th lumbar vertebra': [6.591200615895968,
            0.019566106089660255,
            0.17973744795705707],
            'Anterior to left sacroiliac joint': [7.2086913943739965,
            0.6421065650069326,
            -0.07400065850933421],
            'Anterior to right sacroiliac joint': [7.211406782389645,
            -0.4934018811132983,
            -0.020785158202611825],
            'Adductor tubercle on left femur': [11.56211458217606,
            0.8021641995060096,
            0.5362054659586485],
            'Adductor tubercle on right femur': [11.541922030388077,
            -0.9023746287136986,
            0.5362181766616896],
            'External opening of left carotid canal': [0.6498051852164451,
            0.575691120353834,
            0.10657661599247906],
            'External opening of right carotid canal': [0.6645325664947056,
            -0.3601880703751057,
            0.1695025247374499],
            'Posterior to left neck of mandible': [0.6824610606516195,
            0.7922385840500666,
            0.21448471041672057],
            'Posterior to right neck of mandible': [0.6886348593614651,
            -0.6436143121253458,
            0.2526448439916521],
            'Left wrist joint radial aspect': [2.878442174487298,
            6.826011961729395,
            -0.03283714319600871],
            'Pisiform bone of left wrist': [2.923256062734168,
            6.997125164948838,
            -0.2589401158843421],
            'Pisiform bone of right wrist': [2.913810080890774,
            -7.021011800072831,
            -0.23349857798551685],
            'Upper margin of left greater sciatic foramen': [7.59985718259373,
            0.6021018788316292,
            -0.07057219901411621],
            'Upper margin of right greater sciatic foramen': [7.53959174624729,
            -0.4335320248673514,
            -0.03905019290668944],
            'Diaphragm aortic hiatus (at the level of the lower border of the twelfth thoracic vertebra and the adjacent disc, slightly to the left of the midline)': [5.192734171108048,
            0.03592503462421992,
            -0.024623434792406897],
            'Just below diaphragm aortic hiatus': [5.22607476781567,
            0.03415991794404058,
            -0.020745848295911862],
            'Posterior to left inguinal ligament': [7.902213450747673,
            0.7953633684978062,
            0.6803835610690554],
            'Posterior to right inguinal ligament': [7.823187869270758,
            -0.6428349010815655,
            0.7820474729137357],
            'Left adductor canal': [10.91387499109519,
            0.9972616351177913,
            0.4724471584476272],
            'Right adductor canal': [10.837033354522978,
            -1.0733110342732344,
            0.48444748861323617],
            'Soleal line on proximal posterior right tibia': [12.219826253434894,
            -1.2658715415249273,
            0.1978051156155505],
            'Soleal line on proximal posterior left tibia': [12.237939993609814,
            1.208460057086663,
            0.26319597721397325],
            'Posterior to right sternoclavicular joint, level of superior border of T3 vertebra': [2.7674017279723695,
            -0.31432665363909235,
            0.7061420876644424],
            '1cm distal to left elbow joint': [3.0340932935962215,
            4.351433710862189,
            -0.12402269500325752],
            'Level of junction of T3 and T4 vertebrae': [2.9912888881422557,
            -0.06482935570680645,
            -0.07658792118820353],
            'Lower border of tendon of left teres major muscle': [3.055021617084833,
            1.2049667053157183,
            -0.1593928401563386],
            'Lower border of tendon of right teres major muscle': [2.9660560889919254,
            -1.35601521129089,
            0.04908356321018263],
            'Midway between left anterior superior iliac spine and the pubic symphysis': [7.421473814900645,
            0.8323030571731636,
            0.6314622640464007],
            'Midway between right anterior superior iliac spine and the pubic symphysis': [7.372773037624518,
            -0.815951151538602,
            0.6253752171090131],
            'Near junction of middle and distal thirds of the left femur': [10.414856588536098,
            1.3359670965315038,
            0.40062058275282836],
            'Near junction of middle and distal thirds of the right femur': [10.44658161534392,
            -1.3248493629720506,
            0.46475454826850443],
            'Outer border of left first rib': [2.648280330685333,
            0.6530321318772863,
            0.28949543951993584],
            'Outer border of right first rib': [2.629185037684495,
            -0.7872324578177926,
            0.31934123895055],
            'Posterior to the centrepoint of the manubrium sternum, level of inferior border of T3 vertebra': [2.900265854263726,
            -0.06469009045656827,
            0.6885585697024087],
            'Posterior to upper border of right sternoclavicular joint': [2.765953673771387,
            -0.3436428937178644,
            0.6904196165678982],
            'Right wrist joint radial aspect': [2.9720832351918314,
            -6.381849411552032,
            0.07969069354757924]}
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
        rightLegGroup = getAnnotationGroupForTerm(annotationGroups, get_body_term("right lower limb "))
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
