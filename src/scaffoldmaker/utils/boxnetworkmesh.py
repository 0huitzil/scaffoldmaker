"""
Specialisation of Network Mesh for building 3-D box meshes.
"""
from cmlibs.maths.vectorops import magnitude, mult
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.node import Node
from scaffoldmaker.utils.eft_utils import remapEftLocalNodes, remapEftNodeValueLabelVersion, setEftScaleFactorIds
from scaffoldmaker.utils.interpolation import interpolateSampleCubicHermite, sampleCubicHermiteCurvesSmooth
from scaffoldmaker.utils.meshgeneratedata import MeshGenerateData
from scaffoldmaker.utils.networkmesh import (
    NetworkMesh, NetworkMeshBuilder, NetworkMeshJunction, NetworkMeshSegment, pathValueLabels)
from scaffoldmaker.utils.zinc_utils import get_nodeset_path_ordered_field_parameters
import math


class BoxNetworkMeshGenerateData(MeshGenerateData):
    """
    Data for passing to BoxNetworkMesh generateMesh functions.
    """

    def __init__(self, region, coordinateFieldName="coordinates", startNodeIdentifier=1, startElementIdentifier=1):
        super(BoxNetworkMeshGenerateData, self).__init__(
            region, 3, coordinateFieldName, startNodeIdentifier, startElementIdentifier)
        self._nodetemplates = {}  # map from #versions to nodetemplate
        self._hermiteBilinearBasis = self._fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        self._hermiteBilinearBasis.setFunctionType(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        self._elementtemplatesEfts = {}  # map from (startVersion, endVersion) to (elementtemplate, eft)

    def getNodetemplate(self, versionsCount):
        """
        :param versionsCount: Number of derivative versions for node template.
        :return: Zinc Nodetemplate
        """
        nodetemplate = self._nodetemplates.get(versionsCount)
        if not nodetemplate:
            nodetemplate = self._nodes.createNodetemplate()
            nodetemplate.defineField(self._coordinates)
            for valueLabel in pathValueLabels[1:]:
                nodetemplate.setValueNumberOfVersions(self._coordinates, -1, valueLabel, versionsCount)
            self._nodetemplates[versionsCount] = nodetemplate
        return nodetemplate

    def getElementtemplateAndEft(self, startEndVersions):
        """
        Get element template and element field template for start/end versions.
        :param startEndVersions: (startVersion, endVersion)
        :return: (Zinc Elementtemplate, Zinc Elementfieldtemplate)
        """
        elementtemplateEft = self._elementtemplatesEfts.get(startEndVersions)
        if elementtemplateEft:
            return elementtemplateEft
        eft = self._mesh.createElementfieldtemplate(self._hermiteBilinearBasis)
        setEftScaleFactorIds(eft, [1], [])
        ln = 1
        for n3 in range(2):
            s3 = [1] if (n3 == 0) else []
            for n2 in range(2):
                s2 = [1] if (n2 == 0) else []
                for n1 in range(2):
                    v = startEndVersions[n1]
                    remapEftNodeValueLabelVersion(
                        eft, [ln], Node.VALUE_LABEL_VALUE,
                        [(Node.VALUE_LABEL_VALUE, 1, []),
                         (Node.VALUE_LABEL_D_DS2, v, s2),
                         (Node.VALUE_LABEL_D_DS3, v, s3)])
                    remapEftNodeValueLabelVersion(
                        eft, [ln], Node.VALUE_LABEL_D_DS1,
                        [(Node.VALUE_LABEL_D_DS1, v, []),
                         (Node.VALUE_LABEL_D2_DS1DS2, v, s2),
                         (Node.VALUE_LABEL_D2_DS1DS3, v, s3)])
                    ln += 1
        ln_map = [1, 2, 1, 2, 1, 2, 1, 2]
        remapEftLocalNodes(eft, 2, ln_map)
        elementtemplate = self._mesh.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        elementtemplate.defineField(self._coordinates, -1, eft)
        elementtemplateEft = (elementtemplate, eft)
        self._elementtemplatesEfts[startEndVersions] = elementtemplateEft
        return elementtemplateEft


class BoxNetworkMeshSegment(NetworkMeshSegment):
    """
    Builds a box mesh for network segment.
    """

    def __init__(self, networkSegment, pathParametersList):
        """
        :param networkSegment: NetworkSegment this is built from.
        :param pathParametersList: [pathParameters] or [outerPathParameters, innerPathParameters]
        """
        super(BoxNetworkMeshSegment, self).__init__(networkSegment, pathParametersList)
        self._sampledBoxCoordinates = None

    def sample(self, fixedElementsCountAlong, targetElementLength):
        elementsCountAlong = (fixedElementsCountAlong if fixedElementsCountAlong else
                              max(1, math.ceil(self.getSampleLength() / targetElementLength)))
        if self._isLoop and (elementsCountAlong < 2):
            elementsCountAlong = 2
        pathParameters = self._pathParametersList[0]
        sx, sd1, pe, pxi, psf = sampleCubicHermiteCurvesSmooth(pathParameters[0], pathParameters[1], elementsCountAlong)
        sd2, sd12 = interpolateSampleCubicHermite(pathParameters[2], pathParameters[3], pe, pxi, psf)
        sd3, sd13 = interpolateSampleCubicHermite(pathParameters[4], pathParameters[5], pe, pxi, psf)
        self._sampledBoxCoordinates = (sx, sd1, sd2, sd12, sd3, sd13)

    def getSampledD1(self, nodeIndex):
        return self._sampledBoxCoordinates[1][nodeIndex]

    def setSampledD1(self, nodeIndex, d1):
        self._sampledBoxCoordinates[1][nodeIndex] = d1

    def generateMesh(self, generateData: BoxNetworkMeshGenerateData):
        segmentNodes = self._networkSegment.getNetworkNodes()
        layoutNodeVersions = self._networkSegment.getNodeVersions()
        sx, sd1, sd2, sd12, sd3, sd13 = self._sampledBoxCoordinates
        elementsCountAlong = len(self._sampledBoxCoordinates[0]) - 1
        annotationMeshGroups = generateData.getAnnotationMeshGroups(self._annotationTerms)
        coordinates = generateData.getCoordinates()
        fieldcache = generateData.getFieldcache()
        mesh = generateData.getMesh()
        nodes = generateData.getNodes()
        lastVersion = None
        lastNodeIdentifier = None
        for n in range(elementsCountAlong + 1):
            nodeIdentifier = None
            versionsCount = 1
            version = 1
            junction = None
            if n in [0, elementsCountAlong]:
                junction = self._junctions[0 if (n == 0) else 1]
                nLayout = 0 if (n == 0) else -1
                segmentNode = segmentNodes[nLayout]
                versionsCount = segmentNode.getVersionsCount()
                version = layoutNodeVersions[nLayout]
                nodeIdentifier = junction.getNodeIdentifier()
            if nodeIdentifier is None:
                nodeIdentifier = generateData.nextNodeIdentifier()
                nodetemplate = generateData.getNodetemplate(versionsCount)
                node = nodes.createNode(nodeIdentifier, nodetemplate)
                if junction:
                    junction.setNodeIdentifier(nodeIdentifier)
            else:
                node = nodes.findNodeByIdentifier(nodeIdentifier)
            # note following will set shared versions of coordinates or derivatives multiple times
            # junction.sample should have averaged derivatives from all adjoining segments so this is harmless
            fieldcache.setNode(node)
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, sx[n])  # only one value version
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, version, sd1[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, version, sd2[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, version, sd12[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, version, sd3[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, version, sd13[n])

            if n > 0:
                startEndVersions = (lastVersion, version)
                elementtemplate, eft = generateData.getElementtemplateAndEft(startEndVersions)
                elementIdentifier = generateData.nextElementIdentifier()
                element = mesh.createElement(elementIdentifier, elementtemplate)
                nids = [lastNodeIdentifier, nodeIdentifier]
                element.setNodesByIdentifier(eft, nids)
                element.setScaleFactors(eft, [-1.0])
                for annotationMeshGroup in annotationMeshGroups:
                    annotationMeshGroup.addElement(element)

            lastVersion = version
            lastNodeIdentifier = nodeIdentifier


class BoxNetworkMeshJunction(NetworkMeshJunction):
    """
    Describes junction between multiple segments, some in, some out.
    """

    def __init__(self, inSegments: list, outSegments: list):
        """
        :param inSegments: List of inward BoxNetworkMeshSegment.
        :param outSegments: List of outward BoxNetworkMeshSegment.
        """
        super(BoxNetworkMeshJunction, self).__init__(inSegments, outSegments)
        self._nodeIdentifier = None  # set by adjacent segment

    def getNodeIdentifier(self):
        """
        :return: Identifier of generated node at junction, or None if not set.
        """
        return self._nodeIdentifier

    def setNodeIdentifier(self, nodeIdentifier):
        """
        Store the node identifier to reference from adjacent segments.
        :param nodeIdentifier: Identifier of generated node at junction.
        """
        self._nodeIdentifier = nodeIdentifier

    def sample(self, targetElementLength):
        """
        Blend common derivatives across junctions prior to generateMesh.
        :param targetElementLength: Ignored here.
        """
        s = 0
        nodeVersionSegmentIndexes = {}  # map from version number to list of (segment, nodeIndex) using it
        for segment in self._segments:
            inward = self._segmentsIn[s]
            nodeIndex = -1 if inward else 0
            nodeVersion = segment.getNetworkSegment().getNodeVersions()[nodeIndex]
            nodeVersionSegmentIndex = nodeVersionSegmentIndexes.get(nodeVersion)
            if not nodeVersionSegmentIndex:
                nodeVersionSegmentIndexes[nodeVersion] = nodeVersionSegmentIndex = []
            nodeVersionSegmentIndex.append((segment, nodeIndex))
            s += 1
        for nodeVersion, segmentIndexes in nodeVersionSegmentIndexes.items():
            count = len(segmentIndexes)
            if count < 2:
                continue  # no blending required
            # harmonic mean magnitude
            d1MagRecSum = 0.0
            for segment, nodeIndex in segmentIndexes:
                d1 = segment.getSampledD1(nodeIndex)
                d1Mag = magnitude(d1)
                d1MagRecSum += 1.0 / d1Mag
            d1MagMean = count / d1MagRecSum
            d1Mean = mult(d1, d1MagMean / d1Mag)
            for segment, nodeIndex in segmentIndexes:
                segment.setSampledD1(nodeIndex, d1Mean)

    def generateMesh(self, generateData: BoxNetworkMeshGenerateData):
        pass  # nothing to do for box network


class BoxNetworkMeshBuilder(NetworkMeshBuilder):

    def __init__(self, networkMesh: NetworkMesh, targetElementDensityAlongLongestSegment: float,
                 layoutAnnotationGroups: list = [], annotationElementsCountsAlong: list = []):
        super(BoxNetworkMeshBuilder, self).__init__(
            networkMesh, targetElementDensityAlongLongestSegment, layoutAnnotationGroups, annotationElementsCountsAlong)

    def createSegment(self, networkSegment):
        """
        Create box segment object for building box mesh from networkSegment.
        :param networkSegment: A network segment from the underlying NetworkMesh
        :return: BoxNetworkMeshSegment.
        """
        pathParametersList = [get_nodeset_path_ordered_field_parameters(
            self._layoutNodes, self._layoutCoordinates, pathValueLabels,
            networkSegment.getNodeIdentifiers(), networkSegment.getNodeVersions())]
        return BoxNetworkMeshSegment(networkSegment, pathParametersList)

    def createJunction(self, inSegments, outSegments):
        """
        :param inSegments: List of inward BoxNetworkMeshSegment.
        :param outSegments: List of outward BoxNetworkMeshSegment.
        :return: A BoxNetworkMeshJunction.
        """
        return BoxNetworkMeshJunction(inSegments, outSegments)
