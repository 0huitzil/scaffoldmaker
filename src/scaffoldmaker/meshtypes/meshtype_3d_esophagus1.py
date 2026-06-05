"""
Generates a 3-D esophagus mesh along the central line,
with variable numbers of elements around, along and through
wall, with variable radius and thickness along.
"""

import copy
import math

from cmlibs.maths.vectorops import magnitude, normalize
from cmlibs.utils.zinc.field import findOrCreateFieldGroup, findOrCreateFieldStoredString, \
    findOrCreateFieldStoredMeshLocation
from cmlibs.zinc.element import Element
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, getAnnotationGroupForTerm, \
    findOrCreateAnnotationGroupForTerm
from scaffoldmaker.annotation.esophagus_terms import get_esophagus_term
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils import geometry
from scaffoldmaker.utils import interpolation as interp
from scaffoldmaker.utils import tubemesh
from scaffoldmaker.utils.zinc_utils import exnode_string_from_nodeset_field_parameters,\
    get_nodeset_path_field_parameters

def getDefaultNetworkLayoutScaffoldPackage(cls, parameterSetName):
    assert parameterSetName in cls.getParameterSetNames()  # make sure parameter set is in list of parameters of parent scaffold
    if parameterSetName in ("Default", "Human 1"):
        return ScaffoldPackage(MeshType_1d_network_layout1, {
            'scaffoldSettings': {
                "Structure": "1-2-3-4-5"
            },
            'meshEdits': exnode_string_from_nodeset_field_parameters(
                ['coordinates'],
                [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
                 Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3], [[
                    (1, [[0.394,-100.872,1402.818], [-0.035,12.367,-48.020], [8.730,-0.526,-0.142], [0.645,-0.149,-0.084], [-0.272,-4.224,-1.088], [-0.172,-1.936,0.474]]),
                    (2, [[0.520,-86.043,1340.066], [0.501,16.682,-77.602], [9.142,-0.799,-0.113], [0.176,-0.353,-0.027], [-0.465,-5.159,-1.112], [-0.217,0.101,-0.821]]),
                    (3, [[1.368,-67.733,1247.932], [0.235,-3.685,-89.672], [9.061,-1.366,0.080], [-0.816,-0.266,0.913], [-0.714,-4.722,0.192], [-0.191,0.388,4.146]]),
                    (4, [[0.361,-91.057,1165.531], [-2.499,-24.560,-49.102], [7.540,-1.290,0.261], [-1.321,2.063,3.911], [-0.806,-4.269,2.176], [0.019,0.649,0.615]]),
                    (5, [[8.225,-110.042,1130.494], [17.674,-15.065,-22.956], [6.092,0.940,4.074], [1.229,3.363,5.229], [-0.669,-3.559,1.821], [0.256,-3.730,2.087]])
                ]]),

            'userAnnotationGroups': [
                {
                    '_AnnotationGroup': True,
                    'dimension': 1,
                    'identifierRanges': '1-4',
                    'name': get_esophagus_term('esophagus')[0],
                    'ontId': get_esophagus_term('esophagus')[1]
                },
                {
                    '_AnnotationGroup': True,
                    'dimension': 1,
                    'identifierRanges': '1',
                    'name': get_esophagus_term('cervical part of esophagus')[0],
                    'ontId': get_esophagus_term('cervical part of esophagus')[1]
                },
                {
                    '_AnnotationGroup': True,
                    'dimension': 1,
                    'identifierRanges': '2-3',
                    'name': get_esophagus_term('thoracic part of esophagus')[0],
                    'ontId': get_esophagus_term('thoracic part of esophagus')[1]
                },
                {
                    '_AnnotationGroup': True,
                    'dimension': 1,
                    'identifierRanges': '4',
                    'name': get_esophagus_term('abdominal part of esophagus')[0],
                    'ontId': get_esophagus_term('abdominal part of esophagus')[1]
                }]
        })


class MeshType_3d_esophagus1(Scaffold_base):
    """
    Generates a 3-D esophagus mesh with variable numbers of elements around, along the central line, and through wall.
    The esophagus is created by a function that generates an elliptical tube segment and uses tubemesh to map the
    segment along a network layout profile.
    """

    @staticmethod
    def getName():
        return '3D Esophagus 1'

    @staticmethod
    def getParameterSetNames():
        return [
            'Default',
            'Human 1']

    @classmethod
    def getDefaultOptions(cls, parameterSetName='Default'):
        options = {
            'Network layout': getDefaultNetworkLayoutScaffoldPackage(cls, parameterSetName),
            'Number of elements around': 8,
            'Number of elements along': 20,
            'Number of elements through wall': 1,
            'Wall thickness': 1.2,
            'Mucosa relative thickness': 0.35,
            'Submucosa relative thickness': 0.15,
            'Circular muscle layer relative thickness': 0.25,
            'Longitudinal muscle layer relative thickness': 0.25,
            'Use cross derivatives': False,
            'Use linear through wall': True,
            'Refine': False,
            'Refine number of elements around': 1,
            'Refine number of elements along': 1,
            'Refine number of elements through wall': 1
            }
        return options

    @staticmethod
    def getOrderedOptionNames():
        return [
            'Network layout',
            'Number of elements around',
            'Number of elements along',
            'Number of elements through wall',
            'Wall thickness',
            'Mucosa relative thickness',
            'Submucosa relative thickness',
            'Circular muscle layer relative thickness',
            'Longitudinal muscle layer relative thickness',
            'Use cross derivatives',
            'Use linear through wall',
            'Refine',
            'Refine number of elements around',
            'Refine number of elements along',
            'Refine number of elements through wall']

    @classmethod
    def getOptionValidScaffoldTypes(cls, optionName):
        if optionName == 'Network layout':
            return [MeshType_1d_network_layout1]
        return []

    @classmethod
    def getOptionScaffoldTypeParameterSetNames(cls, optionName, scaffoldType):
        if optionName == 'Network layout':
            return cls.getParameterSetNames()
        assert scaffoldType in cls.getOptionValidScaffoldTypes(optionName), \
            cls.__name__ + '.getOptionScaffoldTypeParameterSetNames.  ' + \
            'Invalid option \'' + optionName + '\' scaffold type ' + scaffoldType.getName()
        return scaffoldType.getParameterSetNames()

    @classmethod
    def getOptionScaffoldPackage(cls, optionName, scaffoldType, parameterSetName=None):
        """
        :param parameterSetName:  Name of valid parameter set for option Scaffold, or None for default.
        :return: ScaffoldPackage.
        """
        if parameterSetName:
            assert parameterSetName in cls.getOptionScaffoldTypeParameterSetNames(optionName, scaffoldType), \
                'Invalid parameter set ' + str(parameterSetName) + ' for scaffold ' + str(scaffoldType.getName()) + \
                ' in option ' + str(optionName) + ' of scaffold ' + cls.getName()
        if optionName == 'Network layout':
            if not parameterSetName:
                parameterSetName = "Default"
            return getDefaultNetworkLayoutScaffoldPackage(cls, parameterSetName)
        assert False, cls.__name__ + '.getOptionScaffoldPackage:  Option ' + optionName + ' is not a scaffold'

    @classmethod
    def checkOptions(cls, options):
        if not options['Network layout'].getScaffoldType() in cls.getOptionValidScaffoldTypes('Network layout'):
            options['Network layout'] = cls.getOptionScaffoldPackage('Network layout', MeshType_1d_network_layout1)
        if options['Number of elements through wall'] != (1 or 4):
            options['Number of elements through wall'] = 4
        for key in [
            'Number of elements around',
            'Number of elements along',
            'Number of elements through wall',
            'Refine number of elements around',
            'Refine number of elements along',
            'Refine number of elements through wall']:
            if options[key] < 1:
                options[key] = 1
        if options['Wall thickness'] < 0.0:
            options['Wall thickness'] = 0.0

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the base tricubic Hermite mesh. See also generateMesh().
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: list of AnnotationGroup, None
        """
        nextNodeIdentifier = 1
        nextElementIdentifier = 1
        esophagusTermsAlong = ['esophagus', 'cervical part of esophagus', 'thoracic part of esophagus',
                               'abdominal part of esophagus']
        geometricNetworkLayout = options['Network layout']
        geometricNetworkLayout = EsophagusNetworkLayout(region, geometricNetworkLayout, esophagusTermsAlong)

        annotationGroups, nextNodeIdentifier, nextElementIdentifier = \
            createEsophagusMesh3d(region, options, geometricNetworkLayout, nextNodeIdentifier, nextElementIdentifier,
                                  flatCoordinates=True, materialCoordinates=True)[:3]

        return annotationGroups, None

    @classmethod
    def refineMesh(cls, meshrefinement, options):
        """
        Refine source mesh into separate region, with change of basis.
        :param meshrefinement: MeshRefinement, which knows source and target region.
        :param options: Dict containing options. See getDefaultOptions().
        """
        refineElementsCountAround = options['Refine number of elements around']
        refineElementsCountAlong = options['Refine number of elements along']
        refineElementsCountThroughWall = options['Refine number of elements through wall']

        meshrefinement.refineAllElementsCubeStandard3d(refineElementsCountAround, refineElementsCountAlong,
                                                       refineElementsCountThroughWall)
        return

    @classmethod
    def defineFaceAnnotations(cls, region, options, annotationGroups):
        '''
        Add face annotation groups from the highest dimension mesh.
        Must have defined faces and added subelements for highest dimension groups.
        :param region: Zinc region containing model.
        :param options: Dict containing options. See getDefaultOptions().
        :param annotationGroups: List of annotation groups for top-level elements.
        New face annotation groups are appended to this list.
        '''
        # Create 2d surface mesh groups
        fm = region.getFieldmodule()
        mesh2d = fm.findMeshByDimension(2)

        esophagusGroup = getAnnotationGroupForTerm(annotationGroups, get_esophagus_term("esophagus"))
        is_exterior = fm.createFieldIsExterior()
        is_exterior_face_xi3_1 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
        is_exterior_face_xi3_0 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI3_0))
        is_esophagus = esophagusGroup.getGroup()
        is_serosa = fm.createFieldAnd(is_esophagus, is_exterior_face_xi3_1)
        serosa = findOrCreateAnnotationGroupForTerm(annotationGroups, region, get_esophagus_term("serosa of esophagus"))
        serosa.getMeshGroup(mesh2d).addElementsConditional(is_serosa)
        is_mucosaInnerSurface = fm.createFieldAnd(is_esophagus, is_exterior_face_xi3_0)
        mucosaInnerSurface = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                get_esophagus_term("luminal surface of esophagus"))
        mucosaInnerSurface.getMeshGroup(mesh2d).addElementsConditional(is_mucosaInnerSurface)

def createEsophagusMesh3d(region, options, networkLayout, nextNodeIdentifier, nextElementIdentifier,
                          flatCoordinates=False, materialCoordinates=False):
    """
    Generates an esophagus scaffold in the region using a network layout and parameter options.
    :param region: Region to create elements in.
    :param options: Parameter options for esophagus scaffold.
    :param networkLayout: Network layout describing path of the esophagus.
    :param nextNodeIdentifier: Next node identifier to use.
    :param nextElementIdentifier: Next element identifier to use.
    :param flatCoordinates: Create flat coordinates if True.
    :param materialCoordinates: Create material coordinates if True.
    :return annotationGroups, nodeIdentifier, elementIdentifier, nodeIdDistal, xDistal, d1Distal, d2Distal, d3Distal
    """
    elementsCountAround = options['Number of elements around']
    elementsCountAlong = options['Number of elements along']
    elementsCountThroughWall = options['Number of elements through wall']
    wallThickness = options['Wall thickness']
    mucosaRelThickness = options['Mucosa relative thickness']
    submucosaRelThickness = options['Submucosa relative thickness']
    circularRelThickness = options['Circular muscle layer relative thickness']
    longitudinalRelThickness = options['Longitudinal muscle layer relative thickness']
    useCrossDerivatives = options['Use cross derivatives']
    useCubicHermiteThroughWall = not(options['Use linear through wall'])

    # Esophagus coordinates
    lengthToDiameterRatio = 15
    wallThicknessToDiameterRatio = 0.15
    relativeThicknessListEsoCoordinates = [1.0 / elementsCountThroughWall for n3 in range(elementsCountThroughWall)]

    esophagusTermsAlong =\
        ['esophagus', 'cervical part of esophagus', 'thoracic part of esophagus', 'abdominal part of esophagus']

    networkLayoutLength = networkLayout.arcLengthOfGroupsAlong[0]
    cx = networkLayout.cxGroups[0]
    cd1 = networkLayout.cd1Groups[0]
    cd2 = networkLayout.cd2Groups[0]
    cd12 = networkLayout.cd12Groups[0]
    cd3 = networkLayout.cd3Groups[0]
    cd13 = networkLayout.cd13Groups[0]
    arcLengthOfGroupsAlong = networkLayout.arcLengthOfGroupsAlong

    # Sample network layout
    sx, sd1, se, sxi, ssf = interp.sampleCubicHermiteCurves(cx, cd1, elementsCountAlong)
    sd2, sd12 = interp.interpolateSampleCubicHermite(cd2, cd12, se, sxi, ssf)
    sd3, sd13 = interp.interpolateSampleCubicHermite(cd3, cd13, se, sxi, ssf)

    elementAlongLength = networkLayoutLength / elementsCountAlong

    elementsCountAlongGroups = []
    groupLength = 0.0
    e = 0
    elementsCount = 1
    length = elementAlongLength
    for i in range(1, len(esophagusTermsAlong)):
        groupLength += arcLengthOfGroupsAlong[i]
        if e == elementsCountAlong - 2:
            elementsCount += 1
            elementsCountAlongGroups.append(elementsCount)
        else:
            while length < groupLength:
                elementsCount += 1
                e += 1
                length += elementAlongLength

            # check which end is grouplength closer to
            distToUpperEnd = abs(length - groupLength)
            distToLowerEnd = abs(groupLength - (length - elementsCountAlong))
            if distToLowerEnd < distToUpperEnd:
                elementsCount -= 1
                elementsCountAlongGroups.append(elementsCount)
                e -= 1
                length -= elementAlongLength
            else:
                elementsCountAlongGroups.append(elementsCount)
        elementsCount = 0

    majorRadiusElementList = sd2
    minorRadiusElementList = sd3

    # Create annotation groups along esophagus
    esophagusGroup = AnnotationGroup(region, get_esophagus_term("esophagus"))
    cervicalGroup = AnnotationGroup(region, get_esophagus_term("cervical part of esophagus"))
    thoracicGroup = AnnotationGroup(region, get_esophagus_term("thoracic part of esophagus"))
    abdominalGroup = AnnotationGroup(region, get_esophagus_term("abdominal part of esophagus"))

    annotationGroupAlong = [[esophagusGroup, cervicalGroup],
                            [esophagusGroup, thoracicGroup],
                            [esophagusGroup, abdominalGroup]]

    annotationGroupsAlong = []
    for i in range(len(elementsCountAlongGroups)):
        elementsCount = elementsCountAlongGroups[i]
        for n in range(elementsCount):
            annotationGroupsAlong.append(annotationGroupAlong[i])

    annotationGroupsAround = []
    for i in range(elementsCountAround):
        annotationGroupsAround.append([])

    # Groups through wall
    longitudinalMuscleGroup = AnnotationGroup(region,
                                              get_esophagus_term("esophagus smooth muscle longitudinal layer"))
    circularMuscleGroup = AnnotationGroup(region, get_esophagus_term("esophagus smooth muscle circular layer"))
    submucosaGroup = AnnotationGroup(region, get_esophagus_term("submucosa of esophagus"))
    mucosaGroup = AnnotationGroup(region, get_esophagus_term("esophagus mucosa"))

    if elementsCountThroughWall == 1:
        relativeThicknessList = [1.0]
        annotationGroupsThroughWall = [[]]
    else:
        relativeThicknessList = [mucosaRelThickness, submucosaRelThickness,
                                 circularRelThickness, longitudinalRelThickness]
        annotationGroupsThroughWall = [[mucosaGroup],
                                       [submucosaGroup],
                                       [circularMuscleGroup],
                                       [longitudinalMuscleGroup]]

    xToSample = []
    d1ToSample = []
    for n2 in range(elementsCountAlong + 1):
        # Create inner points
        cx = [0.0, 0.0, elementAlongLength * n2]
        axis1 = [magnitude(majorRadiusElementList[n2]), 0.0, 0.0]
        axis2 = [0.0, magnitude(minorRadiusElementList[n2]), 0.0]
        xInner, d1Inner = geometry.createEllipsePoints(cx, 2 * math.pi, axis1, axis2,
                                                       elementsCountAround, startRadians=0.0)
        xToSample += xInner
        d1ToSample += d1Inner

    d2ToSample = [[0.0, 0.0, elementAlongLength]] * (elementsCountAround * (elementsCountAlong+1))

    # Sample along length
    xInnerRaw = []
    d2InnerRaw = []
    xToWarp = []
    d1ToWarp = []
    d2ToWarp = []
    flatWidthList = []
    xiList = []

    for n1 in range(elementsCountAround):
        xForSamplingAlong = []
        d2ForSamplingAlong = []
        for n2 in range(elementsCountAlong + 1):
            idx = n2 * elementsCountAround + n1
            xForSamplingAlong.append(xToSample[idx])
            d2ForSamplingAlong.append(d2ToSample[idx])
        xSampled, d2Sampled = interp.sampleCubicHermiteCurves(xForSamplingAlong, d2ForSamplingAlong,
                                                              elementsCountAlong, arcLengthDerivatives=True)[0:2]
        xInnerRaw.append(xSampled)
        d2InnerRaw.append(d2Sampled)

    # Re-arrange sample order & calculate dx_ds1 and dx_ds3 from dx_ds2
    for n2 in range(elementsCountAlong + 1):
        xAround = []
        d2Around = []

        for n1 in range(elementsCountAround):
            x = xInnerRaw[n1][n2]
            d2 = d2InnerRaw[n1][n2]
            xAround.append(x)
            d2Around.append(d2)

        d1Around = []
        for n1 in range(elementsCountAround):
            v1 = xAround[n1]
            v2 = xAround[(n1 + 1) % elementsCountAround]
            d1 = d2 = [v2[c] - v1[c] for c in range(3)]
            arcLengthAround = interp.computeCubicHermiteArcLength(v1, d1, v2, d2, True)
            dx_ds1 = [c * arcLengthAround for c in normalize(d1)]
            d1Around.append(dx_ds1)
        d1Smoothed = interp.smoothCubicHermiteDerivativesLoop(xAround, d1Around)

        xToWarp += xAround
        d1ToWarp += d1Smoothed
        d2ToWarp += d2Around

        # Flat width and xi
        flatWidth = 0.0
        xiFace = []
        for n1 in range(elementsCountAround):
            v1 = xAround[n1]
            d1 = d1Smoothed[n1]
            v2 = xAround[(n1 + 1) % elementsCountAround]
            d2 = d1Smoothed[(n1 + 1) % elementsCountAround]
            flatWidth += interp.getCubicHermiteArcLength(v1, d1, v2, d2)
        flatWidthList.append(flatWidth)

        for n1 in range(elementsCountAround + 1):
            xi = 1.0 / elementsCountAround * n1
            xiFace.append(xi)
        xiList.append(xiFace)

    # Project reference point for warping onto network layout
    sxRefList, sd1RefList, sd2ProjectedListRef, zRefList = \
        tubemesh.getPlaneProjectionOnCentralPath(xToWarp, elementsCountAround, elementsCountAlong,
                                                 networkLayoutLength, sx, sd1, sd2, sd12)

    # Warp points
    segmentAxis = [0.0, 0.0, 1.0]
    closedProximalEnd = False

    innerRadiusAlong = []
    for n2 in range(elementsCountAlong + 1):
        firstNodeAlong = xToWarp[n2 * elementsCountAround]
        midptSegmentAxis = [0.0, 0.0, elementAlongLength * n2]
        radius = magnitude(firstNodeAlong[c] - midptSegmentAxis[c] for c in range(3))
        innerRadiusAlong.append(radius)

    xWarpedList, d1WarpedList, d2WarpedList, d3WarpedUnitList = \
        tubemesh.warpSegmentPoints(xToWarp, d1ToWarp, d2ToWarp, segmentAxis, sxRefList, sd1RefList,
                                   sd2ProjectedListRef, elementsCountAround, elementsCountAlong,
                                   zRefList)

    # Create coordinates and derivatives
    transitElementList = [0]*elementsCountAround
    xList, d1List, d2List, d3List, curvatureList, localIdxDistal, xDistal, d1Distal, d2Distal, d3Distal = \
        tubemesh.extrudeSurfaceCoordinates(xWarpedList, d1WarpedList, d2WarpedList, d3WarpedUnitList,
                                         [wallThickness]*(elementsCountAlong+1), relativeThicknessList,
                                         elementsCountAround, elementsCountAlong, elementsCountThroughWall,
                                         transitElementList, outward=False)

    # Create flat coordinates
    if flatCoordinates:
        xFlat, d1Flat, d2Flat = tubemesh.createFlatCoordinates(xiList, flatWidthList, length, wallThickness,
                                                               relativeThicknessList, elementsCountAround,
                                                               elementsCountAlong, elementsCountThroughWall,
                                                               transitElementList)
    else:
        xFlat = d1Flat = d2Flat = []

    # Create colon coordinates
    if materialCoordinates:
        xEso, d1Eso, d2Eso = \
            tubemesh.createOrganCoordinates(xiList, relativeThicknessListEsoCoordinates, lengthToDiameterRatio,
                                            wallThicknessToDiameterRatio, elementsCountAround, elementsCountAlong,
                                            elementsCountThroughWall, transitElementList)
    else:
        xEso = d1Eso = d2Eso = []

    # Create nodes and elements
    nodeIdentifier, elementIdentifier, annotationGroups, nodeIdDistal = \
        tubemesh.createNodesAndElements(region, xList, d1List, d2List, d3List, xFlat, d1Flat, d2Flat,
                                        xEso, d1Eso, d2Eso, "esophagus coordinates", elementsCountAround,
                                        elementsCountAlong, elementsCountThroughWall, annotationGroupsAround,
                                        annotationGroupsAlong, annotationGroupsThroughWall,
                                        nextNodeIdentifier, nextElementIdentifier,
                                        useCubicHermiteThroughWall, useCrossDerivatives, closedProximalEnd,
                                        localIdxDistal)

    # annotation fiducial points
    fm = region.getFieldmodule()
    fm.beginChange()
    mesh = fm.findMeshByDimension(3)
    cache = fm.createFieldcache()

    markerGroup = findOrCreateFieldGroup(fm, "marker")
    markerName = findOrCreateFieldStoredString(fm, name="marker_name")
    markerLocation = findOrCreateFieldStoredMeshLocation(fm, mesh, name="marker_location")

    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

    markerNames = ["proximodorsal midpoint on serosa of upper esophageal sphincter",
                   "proximoventral midpoint on serosa of upper esophageal sphincter",
                   "distal point of lower esophageal sphincter serosa on the greater curvature of stomach",
                   "distal point of lower esophageal sphincter serosa on the lesser curvature of stomach"]

    totalElements = elementIdentifier
    radPerElementAround = math.pi * 2.0 / elementsCountAround
    elementAroundHalfPi = int(0.25 * elementsCountAround)
    xi1HalfPi = (math.pi * 0.5 - radPerElementAround * elementAroundHalfPi)/radPerElementAround
    elementAroundPi = int(0.5 * elementsCountAround)
    xi1Pi = (math.pi - radPerElementAround * elementAroundPi)/radPerElementAround

    markerElementIdentifiers = [elementsCountAround * elementsCountThroughWall - elementAroundHalfPi,
                                elementAroundHalfPi + 1 + elementsCountAround * (elementsCountThroughWall - 1),
                                totalElements - elementsCountAround,
                                totalElements - elementsCountAround + elementAroundPi]

    markerXis = [[1.0 - xi1HalfPi, 0.0, 1.0],
                 [xi1HalfPi, 0.0, 1.0],
                 [0.0, 1.0, 1.0],
                 [xi1Pi, 1.0, 1.0]]

    esophagusNodesetGroup = esophagusGroup.getNodesetGroup(nodes)
    for n in range(len(markerNames)):
        markerGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                         get_esophagus_term(markerNames[n]), isMarker=True)
        markerElement = mesh.findElementByIdentifier(markerElementIdentifiers[n])
        markerXi = markerXis[n]
        markerNode = markerGroup.createMarkerNode(nodeIdentifier, element=markerElement, xi=markerXi)
        esophagusNodesetGroup.addNode(markerNode)
        nodeIdentifier += 1

    fm.endChange()

    return annotationGroups, nodeIdentifier, elementIdentifier, nodeIdDistal, xDistal, d1Distal, d2Distal, d3Distal

class EsophagusNetworkLayout:
    """
    Generates sampled network layout for esophagus scaffold.
    """
    def __init__(self, region, networkLayout, termsAlong=[None]):
        """
        :param region: Zinc region to define model in.
        :param networkLayout: Network layout subscaffold from meshtype_1d_network_layout1
        :param termsAlong: Annotation terms along length of network layout
        """
        # Extract length of each group along esophagus from network layout
        cxGroups = []
        cd1Groups = []
        cd2Groups = []
        cd3Groups = []
        cd12Groups = []
        cd13Groups = []

        tmpRegion = region.createRegion()
        networkLayout.generate(tmpRegion)
        tmpFieldmodule = tmpRegion.getFieldmodule()
        tmpNodes = tmpFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        tmpCoordinates = tmpFieldmodule.findFieldByName('coordinates')
        arcLengthOfGroupsAlong = []

        for termName in termsAlong:
            tmpGroup = tmpFieldmodule.findFieldByName(termName).castGroup() if termName else None
            tmpNodeset = tmpGroup.getNodesetGroup(tmpNodes) if tmpGroup else tmpNodes

            cxGroup, cd1Group, cd2Group, cd3Group, cd12Group, cd13Group = get_nodeset_path_field_parameters(
                tmpNodeset, tmpCoordinates,
                [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                 Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3,
                 Node.VALUE_LABEL_D2_DS1DS2, Node.VALUE_LABEL_D2_DS1DS3])

            arcLength = 0.0
            for e in range(len(cxGroup) - 1):
                arcLength += interp.getCubicHermiteArcLength(cxGroup[e], cd1Group[e],
                                                             cxGroup[e + 1], cd1Group[e + 1])
            arcLengthOfGroupsAlong.append(arcLength)

            if termName == "esophagus":
                cxGroups.append(cxGroup)
                cd1Groups.append(cd1Group)
                cd2Groups.append(cd2Group)
                cd3Groups.append(cd3Group)
                cd12Groups.append(cd12Group)
                cd13Groups.append(cd13Group)

            del tmpNodeset
            del tmpGroup

        del tmpCoordinates
        del tmpNodes
        del tmpFieldmodule
        del tmpRegion

        self.arcLengthOfGroupsAlong = arcLengthOfGroupsAlong
        self.cxGroups = cxGroups
        self.cd1Groups = cd1Groups
        self.cd2Groups = cd2Groups
        self.cd3Groups = cd3Groups
        self.cd12Groups = cd12Groups
        self.cd13Groups = cd13Groups
