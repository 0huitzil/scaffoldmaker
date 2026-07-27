"""
Utility class for defining network meshes from 1-D connectivity and lateral axes, with continuity control.
"""

from cmlibs.utils.zinc.field import find_or_create_field_coordinates
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from cmlibs.maths.vectorops import cross, magnitude, mult, normalize, rejection, sub
from scaffoldmaker.utils.constructionobject import ConstructionObject
from scaffoldmaker.utils.interpolation import (
    gaussWt4, gaussXi4, getCubicHermiteCurvesLength, interpolateCubicHermiteDerivative)
from scaffoldmaker.utils.meshgeneratedata import MeshGenerateData
from scaffoldmaker.utils.tracksurface import TrackSurface
from abc import ABC, abstractmethod
from enum import Enum
import logging
import math
import sys


logger = logging.getLogger(__name__)


pathValueLabels = [
    Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
    Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
    Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3]


class NetworkNode:
    """
    Describes a single node in a network, storing number of versions etc.
    """

    def __init__(self, nodeIdentifier):
        self._nodeIdentifier = nodeIdentifier
        self._inSegments = []  # segments leading in i.e. ending on this node
        self._outSegments = []  # segments heading out i.e. starting on this node
        self._interiorSegment = None
        self._versionsCount = 0
        self._versionsUsed = set()  # set of version numbers actually used by elements
        self._posX = None  # integer x-coordinate in default layout
        self._x = [None] * 3  # coordinates in default layout

    def getNodeIdentifier(self):
        return self._nodeIdentifier

    def getInteriorSegment(self):
        """
        :return: NetworkSegment this node is interior to i.e. not start nor end, or None if at start or end.
        """
        return self._interiorSegment

    def setInteriorSegment(self, interiorSegment):
        """
        :param interiorSegment: NetworkSegment to set node as interior in, or None to clear.
        """
        self._interiorSegment = interiorSegment

    def addInSegment(self, segment):
        self._inSegments.append(segment)

    def getInSegments(self):
        return self._inSegments

    def addOutSegment(self, segment):
        self._outSegments.append(segment)

    def getOutSegments(self):
        return self._outSegments

    def getVersionsCount(self):
        return self._versionsCount

    def defineVersion(self, nodeVersion):
        self._versionsCount = max(self._versionsCount, nodeVersion)
        self._versionsUsed.add(nodeVersion)

    def checkVersions(self) -> bool:
        """
        :return: True if all node version numbers are in use.
        """
        versionsOK = True
        for nodeVersion in range(1, self._versionsCount + 1):
            if nodeVersion not in self._versionsUsed:
                print("Warning: Node", self._nodeIdentifier, "does not use version", nodeVersion, "of", self._versionsCount,
                      file=sys.stderr)
                versionsOK = False
        return versionsOK

    def getPosX(self):
        return self._posX

    def setPosX(self, posX):
        self._posX = posX

    def getX(self):
        return self._x

    def setX(self, x: list):
        self._x = x


class NetworkSegmentEndStyle(Enum):
    PLAIN = 0
    PATCH = 1
    DOME = 2

    @classmethod
    def fromStartChar(cls, startChar):
        if startChar == '#':
            return cls.PATCH
        if startChar == '(':
            return cls.DOME
        return cls.PLAIN

    @classmethod
    def fromEndChar(cls, endChar):
        if endChar == '#':
            return cls.PATCH
        if endChar == ')':
            return cls.DOME
        return cls.PLAIN

    def get_lower_name(self):
        """
        :return: Lower case style name.
        """
        return self.name.lower()


class NetworkSegment:
    """
    Describes a segment of a network between junctions as a sequence of nodes with node derivative versions.
    """

    def __init__(self, networkNodes: list, nodeVersions: list, endStyles=None):
        """
        :param networkNodes: List of NetworkNodes from start to end. Must be at least 2.
        :param nodeVersions: List of node versions to use for derivatives at network nodes.
        :param endStyles: Optional list of two NetworkSegmentEndStyle values.
        """
        assert isinstance(networkNodes, list) and (len(networkNodes) > 1) and (len(nodeVersions) == len(networkNodes))
        self._networkNodes = networkNodes
        self._nodeVersions = nodeVersions
        if endStyles:
            assert len(endStyles) == 2
            assert all(isinstance(endStyle, NetworkSegmentEndStyle) for endStyle in endStyles)
            self._endStyles = endStyles
        else:
            self._endStyles = [NetworkSegmentEndStyle.PLAIN] * 2
        self._elementIdentifiers = [None] * (len(networkNodes) - 1)
        for networkNode in networkNodes[1:-1]:
            networkNode.setInteriorSegment(self)

    def getNetworkNodes(self):
        """
        :return: List of NetworkNode in order along segment.
        """
        return self._networkNodes

    def getNodeIdentifiers(self):
        """
        :return: List of node identifiers in order along segment.
        """
        return [networkNode.getNodeIdentifier() for networkNode in self._networkNodes]

    def getNodeVersions(self):
        """
        :return: List of node version number in node order along segment.
        """
        return self._nodeVersions

    def getElementIdentifiers(self):
        """
        :return: List of element identifiers along segment. Note: identifiers are all None until mesh is generated
        """
        return self._elementIdentifiers

    def setElementIdentifier(self, index, elementIdentifier):
        self._elementIdentifiers[index] = elementIdentifier

    def hasLayoutElementsInMeshGroup(self, meshGroup):
        """
        NetworkSegment must have elementIdentifiers first!
        :param meshGroup: 1D Mesh group to check membership of.
        :return: True if any element in segment is in 1D mesh group.
        """
        for elementIdentifier in self._elementIdentifiers:
            element = meshGroup.findElementByIdentifier(elementIdentifier)
            if element.isValid():
                return True
        return False

    def isCyclic(self):
        """
        Determine by advancing from end node through each out segment recursively whether network cycles back to
        this segment.
        """
        return False  # not implemented, assume not cyclic

    def getEndStyle(self, index):
        """
        :param index: 0 for start, 1 or -1 for end.
        :return: NetworkSegmentEndStyle (PLAIN, PATCH, DOME etc.) at the requested end index.
        """
        assert index in (0, 1, -1)
        return self._endStyles[index]

    def setEndStyle(self, index, endStyle):
        """
        :param index: 0 for start, 1 or -1 for end.
        :param endStyle: NetworkSegmentEndStyle (PLAIN, PATCH, DOME etc.
        """
        assert index in (0, 1, -1)
        assert isinstance(endStyle, NetworkSegmentEndStyle)
        self._endStyles[index] = endStyle

    def getEndStyles(self):
        """
        :return: list of start, end NetworkSegmentEndStyle types (PLAIN, PATCH, DOME etc.)
        """
        return self._endStyles

    def split(self, splitNetworkNode):
        """
        Split segment to finish at splitNetworkNode, returning remainder as a new NetworkSegment.
        :param splitNetworkNode: NetworkNode to split segment at.
        :return:  New segment after split.
        """
        index = self._networkNodes.index(splitNetworkNode, 1, -1)  # throws exception if not an interior node
        splitNetworkNode.setInteriorSegment(None)
        nextSegment = NetworkSegment(self._networkNodes[index:], self._nodeVersions[index:])
        self._networkNodes = self._networkNodes[:index + 1]
        self._nodeVersions = self._nodeVersions[:index + 1]
        return nextSegment


class NetworkMesh(ConstructionObject):
    """
    Defines a 1-D network with lateral axes, and utility functions for fleshing into higher dimensional meshes.
    """

    def __init__(self, structureString):
        self._networkNodes = {}
        self._networkSegments = []
        self.build(structureString)
        self._region = None  # set when generated

    def build(self, structureString):
        """
        Set up network structure from structure string description.
        :param structureString: Comma-separated sequences of dash-separated node identifiers defining connectivity and
        continuity throughout network.
        Each listing of a given node identifier is either followed by a dot . and the version number, or version 1 is
        assumed. Each version gives a full set of all derivatives.
        Each sequence is a single contiguous segment until it is split by another sequence referencing an interior
        node.
        Example: "1-2-4-5-6,3-4.2-7,5-8" makes the following structure:
                  7
                 /
            1-2-4-5-6
               /   \
              3     8
        where C1 continuity is maintained along each sequence of connected nodes, e.g. 1-2-4-5-6 all use version 1 of
        d1 and side derivatives; 3-4.2-7 uses version 2 at node 4, and 5-8 uses version 1 at node 5 so both output
        segments at node 5 leave with the same longitudinal derivative and use the same side derivatives.
        """
        self._networkNodes = {}
        self._networkSegments = []
        sequenceStrings = structureString.split(",")
        for rawSequenceString in sequenceStrings:
            sequenceString = rawSequenceString.strip()
            # check for end style strings
            endStyles = [NetworkSegmentEndStyle.PLAIN] * 2
            if sequenceString:
                startChar = sequenceString[0]
                if not startChar.isnumeric():
                    endStyles[0] = NetworkSegmentEndStyle.fromStartChar(startChar)
                    if endStyles[0] == NetworkSegmentEndStyle.PLAIN:
                        logger.warning('NetworkMesh: invalid start char ' + startChar + ' in sequence string. Skipping.')
                    sequenceString = sequenceString[1:]
                    if (startChar == '#') and sequenceString and (sequenceString[0] == '-'):
                        # allow old patch structure string with #- previously used by uterus scaffold
                        sequenceString = sequenceString[1:]
            if sequenceString:
                endChar = sequenceString[-1]
                if not endChar.isnumeric():
                    endStyles[1] = NetworkSegmentEndStyle.fromEndChar(endChar)
                    if endStyles[1] == NetworkSegmentEndStyle.PLAIN:
                        logger.warning('NetworkMesh: invalid end char ' + endChar + ' in sequence string. Skipping')
                    sequenceString = sequenceString[:-1]
            nodeIdentifiers = []
            nodeVersions = []
            nodeVersionStrings = sequenceString.split("-")
            if len(nodeVersionStrings) < 2:
                print("Network mesh: Ignoring empty or single node sequence", sequenceString, file=sys.stderr)
                continue
            try:
                for nodeVersionString in nodeVersionStrings:
                    values = [int(s) for s in nodeVersionString.split(".", maxsplit=1)]
                    nodeIdentifiers.append(values[0])
                    nodeVersions.append(values[1] if (len(values) == 2) else 1)
            except ValueError:
                print("Network mesh: Skipping invalid sequence", sequenceString, file=sys.stderr)
                continue
            sequenceNodes = []
            sequenceVersions = []
            for n in range(len(nodeIdentifiers)):
                nodeIdentifier = nodeIdentifiers[n]
                nodeVersion = nodeVersions[n]
                networkNode = existingNetworkNode = self._networkNodes.get(nodeIdentifier)
                if networkNode:
                    interiorSegment = networkNode.getInteriorSegment()
                    if interiorSegment:
                        nextSegment = interiorSegment.split(networkNode)
                        index = self._networkSegments.index(interiorSegment) + 1
                        self._networkSegments.insert(index, nextSegment)
                else:
                    networkNode = NetworkNode(nodeIdentifier)
                    self._networkNodes[nodeIdentifier] = networkNode
                networkNode.defineVersion(nodeVersion)
                sequenceNodes.append(networkNode)
                sequenceVersions.append(nodeVersion)
                if (len(sequenceNodes) > 1) and (existingNetworkNode or (nodeIdentifier == nodeIdentifiers[-1])):
                    networkSegment = NetworkSegment(sequenceNodes, sequenceVersions, endStyles)
                    self._networkSegments.append(networkSegment)
                    sequenceNodes = sequenceNodes[-1:]
                    sequenceVersions = sequenceVersions[-1:]

        # warn about nodes without all versions in use
        for networkNode in self._networkNodes.values():
            networkNode.checkVersions()

        # set up in/out connectivity
        for networkSegment in self._networkSegments:
            segmentNodes = networkSegment.getNetworkNodes()
            segmentNodes[0].addOutSegment(networkSegment)
            segmentNodes[-1].addInSegment(networkSegment)

        # validate end styles
        for networkSegment in self._networkSegments:
            # check only plain end style on junctions
            networkNodes = networkSegment.getNetworkNodes()
            for index in [0, -1]:
                endStyle = networkSegment.getEndStyle(index)
                if endStyle != NetworkSegmentEndStyle.PLAIN:
                    networkNode = networkNodes[index]
                    segmentsCount = len(networkNode.getInSegments()) + len(networkNode.getOutSegments())
                    if segmentsCount > 1:
                        logger.warning('NetworkMesh: End style ' + endStyle.name +
                                       ' at node ' + str(networkNode.getNodeIdentifier()) +
                                       ' not permitted on a junction. Using PLAIN end style instead.')
                        networkSegment.setEndStyle(index, NetworkSegmentEndStyle.PLAIN)
                    elif (index == -1) and (endStyle == NetworkSegmentEndStyle.PATCH) and (segmentsCount == 1):
                        logger.warning('NetworkMesh: End style ' + endStyle.name +
                                       ' at node ' + str(networkNode.getNodeIdentifier()) +
                                       ' only implemented on inlet to junction. Using PLAIN end style instead.')
                        networkSegment.setEndStyle(index, NetworkSegmentEndStyle.PLAIN)

        # assign integer posX coordinates
        for _ in range(len(self._networkSegments)):  # limit total iterations so no endless loop
            changeCount = 0
            for networkSegment in self._networkSegments:
                segmentNodes = networkSegment.getNetworkNodes()
                posX = segmentNodes[0].getPosX()
                if posX is None:
                    posX = 0
                for node in segmentNodes:
                    existingPosX = node.getPosX()
                    if (existingPosX is None) or ((existingPosX < posX) and (
                            (node != segmentNodes[-1]) or not networkSegment.isCyclic())):
                        node.setPosX(posX)
                        changeCount += 1
                    posX += 1
            if changeCount == 0:
                break

        maxPosX = -1
        for node in self._networkNodes.values():
            maxPosX = max(maxPosX, node.getPosX())

        posNodes = []
        for posX in range(0, maxPosX + 1):
            posNodes.append([])
        for node in self._networkNodes.values():
            posNodes[node.getPosX()].append(node)
        maxCountY = 0
        for posX in range(0, maxPosX + 1):
            maxCountY = max(maxCountY, len(posNodes[posX]))
        for posX in range(0, maxPosX + 1):
            xx = float(posX)
            nodes = posNodes[posX]
            countY = len(nodes)
            rangeY = float(countY - 1)
            for iy in range(countY):
                x = [xx, rangeY * (-0.5 + iy / rangeY) if (countY > 1) else 0.0, 0.0]
                nodes[iy].setX(x)

    def getNetworkNodes(self):
        """
        :return: dict mapping node identifier to NetworkNode
        """
        return self._networkSegments

    def getNetworkSegments(self):
        """
        :return: List of segments.
        """
        return self._networkSegments

    def getRegion(self):
        """
        :return: Zinc Region containing generated network layout, or None if not generated
        """
        return self._region

    def create1DLayoutMesh(self, region):
        """
        Expects region Fieldmodule ChangeManager to be in effect.
        :param region: Zinc region to create network layout in.
        """
        self._region = region
        fieldmodule = region.getFieldmodule()
        coordinates = find_or_create_field_coordinates(fieldmodule)

        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplates = [None]
        fieldcache = fieldmodule.createFieldcache()
        derivativeValueLabels = [
            Node.VALUE_LABEL_D_DS1,
            Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
            Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3]
        nodetemplate = None
        for nodeIdentifier, networkNode in self._networkNodes.items():
            versionsCount = networkNode.getVersionsCount()
            # print("Node", nodeIdentifier, "versions", versionsCount)
            if versionsCount < len(nodetemplates):
                nodetemplate = nodetemplates[versionsCount]
            else:
                for v in range(len(nodetemplates), versionsCount + 1):
                    nodetemplate = nodes.createNodetemplate()
                    nodetemplate.defineField(coordinates)
                    for valueLabel in derivativeValueLabels:
                        nodetemplate.setValueNumberOfVersions(coordinates, -1, valueLabel, v)
                    nodetemplates.append(nodetemplate)
            node = nodes.createNode(networkNode.getNodeIdentifier(), nodetemplate)
            fieldcache.setNode(node)
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, networkNode.getX())
            d1 = []
            for nodeVersion in range(1, versionsCount + 1):
                interiorSegment = networkNode.getInteriorSegment()
                if interiorSegment:
                    segmentNodes = interiorSegment.getNetworkNodes()
                    index = segmentNodes.index(networkNode)
                    prevNetworkNode = segmentNodes[index - 1]
                    nextNetworkNode = segmentNodes[index + 1]
                else:
                    prevNetworkNode = None
                    for inNetworkSegment in networkNode.getInSegments():
                        if inNetworkSegment.getNodeVersions()[-1] == nodeVersion:
                            prevNetworkNode = inNetworkSegment.getNetworkNodes()[-2]
                            break
                    nextNetworkNode = None
                    for outNetworkSegment in networkNode.getOutSegments():
                        if outNetworkSegment.getNodeVersions()[0] == nodeVersion:
                            nextNetworkNode = outNetworkSegment.getNetworkNodes()[1]
                            break
                if prevNetworkNode or nextNetworkNode:
                    if prevNetworkNode and nextNetworkNode:
                        d1 = mult(sub(nextNetworkNode.getX(), prevNetworkNode.getX()), 0.5)
                    elif prevNetworkNode:
                        d1 = sub(networkNode.getX(), prevNetworkNode.getX())
                    else:
                        d1 = sub(nextNetworkNode.getX(), networkNode.getX())
                    d3 = [0.0, 0.0, 0.1]
                    d2 = cross(d3, normalize(d1))
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, nodeVersion, d1)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, nodeVersion, d2)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, nodeVersion, d3)
                else:
                    print("Warning: No data to define derivative version", nodeVersion, "at node", nodeIdentifier, ".",
                          file=sys.stderr)

        mesh = fieldmodule.findMeshByDimension(1)
        elementIdentifier = 1
        elementtemplates = {}  # dict (startVersion, endVersion) -> Elementtemplate
        for networkSegment in self._networkSegments:
            segmentNodes = networkSegment.getNetworkNodes()
            segmentNodeVersions = networkSegment.getNodeVersions()
            segmentElementCount = len(segmentNodes) - 1
            for e in range(segmentElementCount):
                startVersion = segmentNodeVersions[e]
                endVersion = segmentNodeVersions[e + 1]
                # print("Element", nextElementIdentifier, "versions", (startVersion, endVersion))
                elementtemplate_eft = elementtemplates.get((startVersion, endVersion))
                if elementtemplate_eft:
                    elementtemplate, eft = elementtemplate_eft
                else:
                    elementtemplate = mesh.createElementtemplate()
                    elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
                    elementbasis = fieldmodule.createElementbasis(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
                    eft = mesh.createElementfieldtemplate(elementbasis)
                    if startVersion != 1:
                        eft.setTermNodeParameter(2, 1, 1, Node.VALUE_LABEL_D_DS1, startVersion)
                    if endVersion != 1:
                        eft.setTermNodeParameter(4, 1, 2, Node.VALUE_LABEL_D_DS1, endVersion)
                    elementtemplate.defineField(coordinates, -1, eft)
                    elementtemplates[(startVersion, endVersion)] = elementtemplate, eft
                element = mesh.createElement(elementIdentifier, elementtemplate)
                element.setNodesByIdentifier(
                    eft, [segmentNodes[e].getNodeIdentifier(), segmentNodes[e + 1].getNodeIdentifier()])
                networkSegment.setElementIdentifier(e, elementIdentifier)
                elementIdentifier += 1


class NetworkMeshSegment(ABC):
    """
    Base class for building a mesh from a NetworkSegment.
    """

    def __init__(self, networkSegment, pathParametersList):
        """
        :param networkSegment: NetworkSegment this is built from.
        :param pathParametersList: [pathParameters] or [outerPathParameters, innerPathParameters]
        Segment length is determined from the first/primary path only.
        """
        self._networkSegment = networkSegment
        self._pathParametersList = pathParametersList
        self._pathsCount = len(pathParametersList)
        self._annotationTerms = []
        self._junctions = []  # start, end junctions. Set when junctions are created.
        self._isLoop = False
        self._lengthParameters = self._calculateLengthParameters()
        self._leftRightSwap = False

    def addAnnotationTerm(self, annotationTerm):
        """
        Add annotation term to this segment. Should not already be present; this is not checked.
        :param annotationTerm: Annotation term (name: str, ontId: str)
        """
        self._annotationTerms.append(annotationTerm)

    def getAnnotationTerms(self):
        return self._annotationTerms

    def getSampleLength(self):
        """
        Get sample length of the segment's primary path.
        :return: Length.
        """
        return self._lengthParameters[0][-1][0]

    def getNetworkSegment(self):
        """
        :return: Original NetworkSegment this is building from.
        """
        return self._networkSegment

    def getPathParameters(self, pathIndex=0):
        """
        :return: Path parameters (x, d1, d2, d12, d3, d13) for path index.
        """
        if pathIndex > len(self._pathParametersList):
            return None
        return self._pathParametersList[pathIndex]

    def getPathsCount(self):
        return self._pathsCount

    def getJunctions(self):
        """
        Get the junctions at start and end of segment.
        :return: List of NetworkMeshJunction-derived objects.
        """
        return self._junctions

    def setJunctions(self, junctions):
        """
        Called by NetworkMeshBuilder so segment knows how it starts and ends.
        :param junctions: list of 2 NetworkMeshJunctions-derived objects at start and end of segment.
        """
        self._junctions = junctions
        self._isLoop = junctions[0] == junctions[1]

    def isLoop(self):
        """
        Only valid after call to self.setJunctions()
        :return: True if segment is a loop, otherwise False.
        """
        return self._isLoop

    def _calculateLengthParameters(self):
        """
        Calculate scalar field values and derivatives giving effective length along segment central path
        with allowance for mean change in lateral axes, used for sampling elements along segment.
        Uses first (outer) path parameters only.
        Calculated in constructor and stored.
        :return: Length parameters (lx[], ld1[])
        """
        px, pd1, pd2, pd12, pd3, pd13 = self._pathParametersList[0]
        lx = [[0.0]]
        totalLength = 0.0
        for e in range(len(px) - 1):
            ax, ad1, ad2, ad12, ad3, ad13 = px[e], pd1[e], pd2[e], pd12[e], pd3[e], pd13[e]
            bx, bd1, bd2, bd12, bd3, bd13 = px[e + 1], pd1[e + 1], pd2[e + 1], pd12[e + 1], pd3[e + 1], pd13[e + 1]
            length = 0.0
            for i in range(4):
                d1 = interpolateCubicHermiteDerivative(ax, ad1, bx, bd1, gaussXi4[i])
                gd1 = magnitude(d1)
                gd2 = magnitude(rejection(interpolateCubicHermiteDerivative(ad2, ad12, bd2, bd12, gaussXi4[i]), d1))
                gd3 = magnitude(rejection(interpolateCubicHermiteDerivative(ad3, ad13, bd3, bd13, gaussXi4[i]), d1))
                gds = 0.5 * (gd2 + gd3)
                length += gaussWt4[i] * math.sqrt(gd1 * gd1 + gds * gds)
            totalLength += length
            lx.append([totalLength])
        ld = []
        for n in range(len(px)):
            d1 = magnitude(pd1[n])
            d2 = 0.5 * (magnitude(pd12[n]) + magnitude(pd13[n]))
            ld.append([math.sqrt(d1 * d1 + d2 * d2)])
        return (lx, ld)

    def getLengthParameters(self):
        """
        Get scalar field parameter (value, derivative) giving effective length along segment.
        :return: Length parameters (lx[], ld[])
        """
        return self._lengthParameters

    @abstractmethod
    def sample(self, fixedElementsCountAlong, targetElementLength):
        """
        Override to resample curve/raw data to final coordinates.
        :param fixedElementsCountAlong: Fixed number of elements along > 0 or None to use targetElementLength.
        Implementations may enforce a higher minimum number.
        :param targetElementLength: Target element size along length of segment/junction.
        """
        pass

    def setLeftRightSwap(self, swap: bool):
        """
        :param swap: True to swap sense of left and right for this segment.
        """
        self._leftRightSwap = swap

    def isLeftRightSwap(self):
        """
        Query whether left-right sense of segment is swapped.
        :return: True if swapped, False if not.
        """
        return self._leftRightSwap

    @abstractmethod
    def generateMesh(self, generateData: MeshGenerateData):
        """
        Override to generate mesh for segment.
        :param generateData: MeshGenerateData or derived object.
        :return:
        """
        pass


class NetworkMeshJunction(ABC):
    """
    Base class for building a mesh at a junction between segments, some in, some out.
    """

    def __init__(self, inSegments: list, outSegments: list):
        """
        :param inSegments: List of inward NetworkMeshSegment-derived objects.
        :param outSegments: List of outward NetworkMeshSegment-derived objects.
        """
        self._segments = inSegments + outSegments
        self._segmentsCount = len(self._segments)
        self._segmentsIn = [segment in inSegments for segment in self._segments]

    def getNetworkSegments(self):
        """
        :return: List of network segments at junction in order with "in" segments first.
        """
        return self._networkSegments

    def getSegmentsCount(self):
        return self._segmentsCount

    def getSegmentsIn(self):
        """
        :return: List of boolean true if segment is inward, false if not, in supplied order.
        """
        return self._segmentsIn

    def getSegments(self):
        """
        :return: List of NetworkMeshSegment-derived objects joined at junction, in supplied order.
        """
        return self._segments

    @abstractmethod
    def sample(self, targetElementLength):
        """
        Override to resample curve/raw data to final coordinates.
        :param targetElementLength: Target element size along length of segment/junction.
        """
        pass

    @abstractmethod
    def generateMesh(self, generateData: MeshGenerateData):
        """
        Override to generate mesh for junction.
        :param generateData: MeshGenerateData-derived object.
        :return:
        """
        pass


class NetworkMeshBuilder(ABC):
    """
    Abstract base class for building meshes from a NetworkMesh network layout.
    """

    def __init__(self, networkMesh: NetworkMesh, targetElementDensityAlongLongestSegment: float,
                 layoutAnnotationGroups, annotationElementsCountsAlong=[]):
        """
        Abstract base class for building meshes from a NetworkMesh network layout.
        :param networkMesh: Description of the topology of the network layout.
        :param targetElementDensityAlongLongestSegment: Real value which longest segment path in network is divided by
        to get target element length, which is used to determine numbers of elements along except when set for a segment
        through annotationElementsCountsAlong.
        :param layoutAnnotationGroups: Annotation groups defined on the layout to mirror on the final mesh.
        :param annotationElementsCountsAlong: List in same order as layoutAnnotationGroups, specifying fixed number of
        elements along segment with any elements in the annotation group. Client must ensure exclusive map from
        segments. Groups with zero value or past end of this list use the targetElementDensityAlongLongestSegment.
        """
        self._networkMesh = networkMesh
        self._targetElementDensityAlongLongestSegment = targetElementDensityAlongLongestSegment
        self._layoutAnnotationGroups = layoutAnnotationGroups
        self._annotationElementsCountsAlong = annotationElementsCountsAlong
        self._layoutRegion = networkMesh.getRegion()
        layoutFieldmodule = self._layoutRegion.getFieldmodule()
        self._layoutNodes = layoutFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self._layoutCoordinates = layoutFieldmodule.findFieldByName("coordinates").castFiniteElement()
        self._layoutMesh = layoutFieldmodule.findMeshByDimension(1)
        self._segments = {}  # map from NetworkSegment to NetworkMeshSegment-derived object
        self._longestSegmentLength = 0.0
        self._targetElementLength = 1.0
        self._junctions = {}  # map from NetworkNode to NetworkMeshJunction-derived object

    def getMetadata(self):
        """
        Override to add any metadata when used as a scaffold construction object.
        :return: Dictionary of metadata.
        """
        return {}

    @abstractmethod
    def createSegment(self, networkSegment):
        """
        Factory method for creating a NetworkMeshSegment.
        Override in concrete class to create a derived Segment for building the mesh of required type.
        :param networkSegment: A network segment from the underlying NetworkMesh
        :return: An object derived from NetworkMeshSegment.
        """
        return None

    def _createSegments(self):
        """
        Create objects for building segment meshes.
        """
        self._segments = {}
        self._longestSegmentLength = 0.0
        for networkSegment in self._networkMesh.getNetworkSegments():
            # derived class makes the segment of its required type
            self._segments[networkSegment] = segment = self.createSegment(networkSegment)
            segmentLength = segment.getSampleLength()
            if segmentLength > self._longestSegmentLength:
                self._longestSegmentLength = segmentLength
            for layoutAnnotationGroup in self._layoutAnnotationGroups:
                if networkSegment.hasLayoutElementsInMeshGroup(layoutAnnotationGroup.getMeshGroup(self._layoutMesh)):
                    segment.addAnnotationTerm(layoutAnnotationGroup.getTerm())
        if self._longestSegmentLength > 0.0:
            self._targetElementLength = self._longestSegmentLength / self._targetElementDensityAlongLongestSegment

    @abstractmethod
    def createJunction(self, inSegments, outSegments):
        """
        Factory method for creating a NetworkMeshJunction.
        Override in concrete class to create a derived Junction for building the mesh of required type.
        :param inSegments: List of inward NetworkMeshSegment-derived objects.
        :param outSegments: List of outward NetworkMeshSegment-derived objects.
        :return: An object derived from NetworkMeshJunction.
        """
        return None

    def _createJunctions(self):
        """
        Create objects for building junction meshes between segments.
        Must have called self.createSegments() first.
        """
        self._junctions = {}
        for networkSegment in self._networkMesh.getNetworkSegments():
            segment = self._segments[networkSegment]
            segmentNodes = networkSegment.getNetworkNodes()
            segmentJunctions = []
            for nodeIndex in (0, -1):
                segmentNode = segmentNodes[nodeIndex]
                junction = self._junctions.get(segmentNode)
                if not junction:
                    inSegments = [self._segments[networkSegment] for networkSegment in segmentNode.getInSegments()]
                    outSegments = [self._segments[networkSegment] for networkSegment in segmentNode.getOutSegments()]
                    junction = self.createJunction(inSegments, outSegments)
                    self._junctions[segmentNode] = junction
                segmentJunctions.append(junction)
            segment.setJunctions(segmentJunctions)

    def _sampleSegments(self):
        """
        Sample coordinates in segments to fit surrounding junctions.
        Must have called self.createJunctions() first.
        """
        for networkSegment in self._networkMesh.getNetworkSegments():
            fixedElementsCountAlong = None
            i = 0
            for layoutAnnotationGroup in self._layoutAnnotationGroups:
                if i >= len(self._annotationElementsCountsAlong):
                    break
                if self._annotationElementsCountsAlong[i] > 0:
                    if networkSegment.hasLayoutElementsInMeshGroup(
                            layoutAnnotationGroup.getMeshGroup(self._layoutMesh)):
                        fixedElementsCountAlong = self._annotationElementsCountsAlong[i]
                        break
                i += 1
            segment = self._segments[networkSegment]
            segment.sample(fixedElementsCountAlong, self._targetElementLength)

    def _sampleJunctions(self):
        """
        Sample coordinates in junctions to fit surrounding junctions.
        Optionally blend common derivatives across simple junctions.
        Must have called self.sampleSegments() first.
        """
        sampledJunctions = set()
        for networkSegment in self._networkMesh.getNetworkSegments():
            segment = self._segments[networkSegment]
            for junction in segment.getJunctions():
                if junction not in sampledJunctions:
                    junction.sample(self._targetElementLength)
                    sampledJunctions.add(junction)

    def build(self):
        """
        Build coordinates for network mesh.
        """
        self._createSegments()
        self._createJunctions()
        self._sampleSegments()
        self._sampleJunctions()

    def setAnnotationLeftRightSwap(self, name, swap):
        """
        Call after build but before generate mesh to swap left-right sense of segments with annotation name.
        :param group_name: Name of annotation group.
        :param swap: True to swap sense of left and right.
        """
        for segment in self._segments.values():
            for annotationTerm in segment.getAnnotationTerms():
                if annotationTerm[0] == name:
                    segment.setLeftRightSwap(swap)
                    break

    def setLeftRightSwap(self, swap):
        """
        Call after build but before generate mesh to swap left-right sense of segments with annotation names
        not containing 'left' or 'right'.
        :param swap: True to swap sense of left and right.
        """
        for segment in self._segments.values():
            for annotationTerm in segment.getAnnotationTerms():
                if ('left' in annotationTerm[0]) or ('right' in annotationTerm[0]):
                    break
            else:
                 segment.setLeftRightSwap(swap)

    def generateMesh(self, generateData: MeshGenerateData):
        """
        Generate mesh from segments and junctions, in order of segments.
        Must have called self.build() first.
        Assumes ChangeManager active for region/fieldmodule.
        :param generateData: MeshGenerateData-derived object.
        """
        generatedJunctions = set()
        for networkSegment in self._networkMesh.getNetworkSegments():
            segment = self._segments[networkSegment]
            junctions = segment.getJunctions()
            if junctions[0] not in generatedJunctions:
                junctions[0].generateMesh(generateData)
                generatedJunctions.add(junctions[0])
            if NetworkSegmentEndStyle.PATCH in networkSegment.getEndStyles():
                continue  # so as not to make patch mesh twice
            segment.generateMesh(generateData)
            if junctions[1] not in generatedJunctions:
                junctions[1].generateMesh(generateData)
                generatedJunctions.add(junctions[1])
