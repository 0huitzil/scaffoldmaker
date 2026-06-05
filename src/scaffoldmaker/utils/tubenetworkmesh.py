"""
Specialisation of Network Mesh for building 2-D and 3-D tube mesh networks.
"""
from cmlibs.maths.vectorops import add, cross, dot, magnitude, mult, normalize, set_magnitude, sub, rejection
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.node import Node
from scaffoldmaker.utils.eft_utils import (
    addTricubicHermiteSerendipityEftParameterScaling, determineCubicHermiteSerendipityEft, HermiteNodeLayoutManager)
from scaffoldmaker.utils.interpolation import (
    computeCubicHermiteDerivativeScaling, computeCubicHermiteEndDerivative, computeCubicHermiteStartDerivative,
    DerivativeScalingMode, evaluateCoordinatesOnCurve, getCubicHermiteTrimmedCurvesLengths, getNearestLocationOnCurve,
    interpolateCubicHermite, interpolateCubicHermiteDerivative,
    interpolateHermiteLagrangeDerivative, interpolateLagrangeHermiteDerivative,
    interpolateSampleCubicHermite, sampleCubicHermiteCurves,
    sampleCubicHermiteCurvesSmooth, smoothCubicHermiteDerivativesLine, smoothCubicHermiteDerivativesLoop,
    smoothCurveSideCrossDerivatives, getNearestLocationBetweenCurves)
from scaffoldmaker.utils.networkmesh import NetworkMesh, NetworkMeshBuilder, NetworkMeshGenerateData, \
    NetworkMeshJunction, NetworkMeshSegment, pathValueLabels
from scaffoldmaker.utils.tracksurface import TrackSurface
from scaffoldmaker.utils.zinc_utils import get_nodeset_path_ordered_field_parameters
import copy
import math


class TubeNetworkMeshGenerateData(NetworkMeshGenerateData):
    """
    Data for passing to TubeNetworkMesh generateMesh functions.
    """

    def __init__(self, region, meshDimension, coordinateFieldName="coordinates",
                 startNodeIdentifier=1, startElementIdentifier=1, isLinearThroughShell=False, isShowTrimSurfaces=False):
        """
        :param isLinearThroughShell: Callers should only set if 3-D with no core.
        :param isShowTrimSurfaces: Tells junction generateMesh to make 2-D trim surfaces.
        """
        super(TubeNetworkMeshGenerateData, self).__init__(
            region, meshDimension, coordinateFieldName, startNodeIdentifier, startElementIdentifier)
        self._isLinearThroughShell = isLinearThroughShell
        self._isShowTrimSurfaces = isShowTrimSurfaces
        self._trimAnnotationGroupCount = 0  # incremented to make unique annotation group names for trim surfaces

        # get node template for standard and cross nodes
        self._nodetemplate = self._nodes.createNodetemplate()
        self._nodetemplate.defineField(self._coordinates)
        self._nodetemplate.setValueNumberOfVersions(self._coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        self._nodetemplate.setValueNumberOfVersions(self._coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        if (meshDimension == 3) and not isLinearThroughShell:
            self._nodetemplate.setValueNumberOfVersions(self._coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)

        # get element template and eft for standard case
        self._standardElementtemplate = self._mesh.createElementtemplate()
        self._standardElementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE if (meshDimension == 3)
                                                     else Element.SHAPE_TYPE_SQUARE)
        self._elementbasis = self._fieldmodule.createElementbasis(
            meshDimension, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY)
        if (meshDimension == 3) and isLinearThroughShell:
            self._elementbasis.setFunctionType(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        self._standardEft = self._mesh.createElementfieldtemplate(self._elementbasis)
        self._standardElementtemplate.defineField(self._coordinates, -1, self._standardEft)

        d3Defined = (meshDimension == 3) and not isLinearThroughShell
        self._nodeLayoutManager = HermiteNodeLayoutManager()
        self._nodeLayout5Way = self._nodeLayoutManager.getNodeLayout5Way12(d3Defined)
        self._nodeLayout6Way = self._nodeLayoutManager.getNodeLayout6Way12(d3Defined)
        self._nodeLayout8Way = self._nodeLayoutManager.getNodeLayout8Way12(d3Defined)
        self._nodeLayoutFlipD2 = self._nodeLayoutManager.getNodeLayoutRegularPermuted(
            d3Defined, limitDirections=[None, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], [[0.0, 0.0, 1.0]]] if d3Defined
            else [None, [[0.0, 1.0], [0.0, -1.0]]])
        self._nodeLayoutFlipD1D2 = self._nodeLayoutManager.getNodeLayoutRegularPermuted(
            d3Defined, limitDirections=[[[-1.0, 0.0, 0.0]], [[0.0, -1.0, 0.0]], [[0.0, 0.0, 1.0]]] if d3Defined
            else [[[-1.0, 0.0]], [[0.0, -1.0]]])
        self._nodeLayoutSwapMinusD1D2 = self._nodeLayoutManager.getNodeLayoutRegularPermuted(
            d3Defined, limitDirections=[[[0.0, 1.0, 0.0]], [[-1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]] if d3Defined
            else [[[0.0, 1.0]], [[-1.0, 0.0]]])
        self._nodeLayoutSwapD1MinusD2 = self._nodeLayoutManager.getNodeLayoutRegularPermuted(
            d3Defined, limitDirections=[[[0.0, -1.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]] if d3Defined
            else [[[0.0, -1.0]], [[1.0, 0.0]]])
        self._nodeLayoutTrifurcation = None
        self._nodeLayoutTransition = self._nodeLayoutManager.getNodeLayoutRegularPermuted(
            d3Defined, limitDirections=[None, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], None])
        self._nodeLayoutTransitionTriplePoint = None

        # annotation groups created if core:
        self._coreGroup = None
        self._shellGroup = None
        # annotation groups created on demand:
        self._leftGroup = None
        self._rightGroup = None
        self._dorsalGroup = None
        self._ventralGroup = None

    def getStandardElementtemplate(self):
        return self._standardElementtemplate, self._standardEft

    def createElementfieldtemplate(self):
        """
        Create a new standard element field template for modifying.
        """
        return self._mesh.createElementfieldtemplate(self._elementbasis)

    def getNodeLayout5Way(self):
        return self._nodeLayout5Way

    def getNodeLayout6Way(self):
        return self._nodeLayout6Way

    def getNodeLayout8Way(self):
        return self._nodeLayout8Way

    def getNodeLayoutFlipD2(self):
        return self._nodeLayoutFlipD2

    def getNodeLayoutFlipD1D2(self):
        return self._nodeLayoutFlipD1D2

    def getNodeLayoutSwapMinusD1D2(self):
        return self._nodeLayoutSwapMinusD1D2

    def getNodeLayoutSwapD1MinusD2(self):
        return self._nodeLayoutSwapD1MinusD2

    def getNodeLayoutTrifurcation(self, location):
        """
        Special node layout for generating core elements for trifurcation. There are two layouts specific to
        left-hand side and right-hand side of the solid core cross-section: LHS (location = 1); and RHS (location = 2).
        :param location: Location identifier.
        :return: Node layout.
        """
        if location == 1:  # Left-hand side
            limitDirections = [None, [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], None]
        elif location == 2:  # Right-hand side
            limitDirections = [None, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], None]

        self._nodeLayoutTrifurcation = self._nodeLayoutManager.getNodeLayout6Way12(True, limitDirections)
        return self._nodeLayoutTrifurcation

    def getNodeLayoutTransition(self):
        """
        Node layout for generating core transition elements, excluding at triple points.
        """
        return self._nodeLayoutTransition

    def getNodeLayoutTransitionTriplePoint(self, location):
        """
        Special node layout for generating core transition elements at triple points.
        There are four layouts specific to each corner of the core box: Top left (location = 1);
        top right (location = -1); bottom left (location = 2); and bottom right (location = -2).
        :param location: Location identifier identifying four corners of solid core box.
        :return: Node layout.
        """
        nodeLayouts = self._nodeLayoutManager.getNodeLayoutTriplePoint()
        assert location in [1, -1, 2, -2, 0]
        if location == 1:  # "Top Left"
            nodeLayout = nodeLayouts[0]
        elif location == -1:  # "Top Right"
            nodeLayout = nodeLayouts[1]
        elif location == 2:  # "Bottom Left"
            nodeLayout = nodeLayouts[2]
        elif location == -2:  # "Bottom Right"
            nodeLayout = nodeLayouts[3]
        else:
            nodeLayout = self._nodeLayoutTransition

        self._nodeLayoutTransitionTriplePoint = nodeLayout
        return self._nodeLayoutTransitionTriplePoint

    def getNodeLayoutBifurcation6WayTriplePoint(self, segmentsIn, sequence, maxMajorSegment, top):
        return self._nodeLayoutManager.getNodeLayoutBifurcation6WayTriplePoint(
            segmentsIn, sequence, maxMajorSegment, top)

    def getNodetemplate(self):
        return self._nodetemplate

    def isLinearThroughShell(self):
        return self._isLinearThroughShell

    def isShowTrimSurfaces(self):
        return self._isShowTrimSurfaces

    def getCoreMeshGroup(self):
        if not self._coreGroup:
            self._coreGroup = self.getOrCreateAnnotationGroup(("core", ""))
        return self._coreGroup.getMeshGroup(self._mesh)

    def getShellMeshGroup(self):
        if not self._shellGroup:
            self._shellGroup = self.getOrCreateAnnotationGroup(("shell", ""))
        return self._shellGroup.getMeshGroup(self._mesh)

    def getLeftMeshGroup(self):
        if not self._leftGroup:
            self._leftGroup = self.getOrCreateAnnotationGroup(("left", ""))
        return self._leftGroup.getMeshGroup(self._mesh)

    def getRightMeshGroup(self):
        if not self._rightGroup:
            self._rightGroup = self.getOrCreateAnnotationGroup(("right", ""))
        return self._rightGroup.getMeshGroup(self._mesh)

    def getDorsalMeshGroup(self):
        if not self._dorsalGroup:
            self._dorsalGroup = self.getOrCreateAnnotationGroup(("dorsal", ""))
        return self._dorsalGroup.getMeshGroup(self._mesh)

    def getVentralMeshGroup(self):
        if not self._ventralGroup:
            self._ventralGroup = self.getOrCreateAnnotationGroup(("ventral", ""))
        return self._ventralGroup.getMeshGroup(self._mesh)

    def getNewTrimAnnotationGroup(self):
        self._trimAnnotationGroupCount += 1
        return self.getOrCreateAnnotationGroup(("trim surface " + "{:03d}".format(self._trimAnnotationGroupCount), ""))

    def resolveEftCoreBoundaryScaling(self, eft, scalefactors, nodeParameters, nodeIdentifiers, mode):
        """
        Resolve differences in d3 scaling across core-shell boundary by one of 2 modes.
        Works for 8-noded tricubic Hermite serendipity basis only.
        :param eft: Element field template to modify.
        :param scalefactors: Existing scalefactors for use with eft.
        :param nodeParameters: List over 8 (3-D) local nodes in Zinc ordering of 4 parameter vectors
        x, d1, d2, d3 each with 3 components.
        :param nodeIdentifiers: List over 8 3-D local nodes giving global node identifiers.
        :param mode: 1 to set scale factors, 2 to add version 2 to d3 for the boundary nodes and assigning
        values to that version equal to the scale factors x version 1.
        :return: New eft, new scalefactors.
        """
        assert mode in (1, 2)
        eft, scalefactors, addScalefactors = addTricubicHermiteSerendipityEftParameterScaling(
            eft, scalefactors, nodeParameters, [5, 6, 7, 8], Node.VALUE_LABEL_D_DS3, version=mode)
        if mode == 2:
            nodetemplate = self._nodes.createNodetemplate()
            n = 4
            for nodeIdentifier, scalefactor in zip(nodeIdentifiers[4:], addScalefactors):
                node = self._nodes.findNodeByIdentifier(nodeIdentifier)
                nodetemplate.defineFieldFromNode(self._coordinates, node)
                versionsCount = nodetemplate.getValueNumberOfVersions(self._coordinates, -1, Node.VALUE_LABEL_D_DS3)
                if versionsCount == 1:
                    # make version 2 of d3 at the node and assign to it
                    nodetemplate.setValueNumberOfVersions(self._coordinates, -1, Node.VALUE_LABEL_D_DS3, 2)
                    # merge clears the current parameters so need to re-set
                    node.merge(nodetemplate)
                    self._fieldcache.setNode(node)
                    for valueLabel, value in zip(
                            (Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                             Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3), nodeParameters[n]):
                        self._coordinates.setNodeParameters(self._fieldcache, -1, valueLabel, 1, value)
                    d3_v2 = mult(nodeParameters[n][3], scalefactor)
                    self._coordinates.setNodeParameters(self._fieldcache, -1, Node.VALUE_LABEL_D_DS3, 2, d3_v2)
                n += 1
        return eft, scalefactors


class TubeNetworkMeshSegment(NetworkMeshSegment):

    def __init__(self, networkSegment, pathParametersList, elementsCountAround, elementsCountThroughShell,
                 isCore=False, elementsCountCoreBoxMinor: int=2, elementsCountTransition: int=1,
                 coreBoundaryScalingMode: int=1):
        """
        :param networkSegment: NetworkSegment this is built from.
        :param pathParametersList: [pathParameters] if 2-D or [outerPathParameters, innerPathParameters] if 3-D
        :param elementsCountAround: Number of elements around this segment.
        :param elementsCountThroughShell: Number of elements between inner and outer tube if 3-D, 1 if 2-D.
        :param isCore: True for generating a solid core inside the tube, False for regular tube network.
        :param elementsCountCoreBoxMinor: Number of elements across core box minor axis.
        :param elementsCountTransition: Number of elements across transition zone between core box elements and
        shell elements.
        """
        super(TubeNetworkMeshSegment, self).__init__(networkSegment, pathParametersList)
        self._isCore = isCore
        self._elementsCountAround = elementsCountAround
        self._elementsCountCoreBoxMajor = (elementsCountAround // 2) - elementsCountCoreBoxMinor
        self._elementsCountCoreBoxMinor = elementsCountCoreBoxMinor
        self._elementsCountTransition = elementsCountTransition
        self._coreBoundaryScalingMode = coreBoundaryScalingMode
        self._networkSegment = networkSegment

        assert elementsCountThroughShell > 0
        self._elementsCountThroughShell = elementsCountThroughShell
        self._rawTubeCoordinatesList = []
        self._rawTrackSurfaceList = []
        for pathParameters in pathParametersList:
            px, pd1, pd2, pd12 = getPathRawTubeCoordinates(pathParameters, self._elementsCountAround)
            self._rawTubeCoordinatesList.append((px, pd1, pd2, pd12))
            nx, nd1, nd2, nd12 = [], [], [], []
            for i in range(len(px)):
                nx += px[i]
                nd1 += pd1[i]
                nd2 += pd2[i]
                nd12 += pd12[i]
            self._rawTrackSurfaceList.append(TrackSurface(len(px[0]), len(px) - 1, nx, nd1, nd2, nd12, loop1=True))
        # list[pathsCount][4] of sx, sd1, sd2, sd12; all [nAlong][nAround]:
        self._sampledTubeCoordinates = [[[], [], [], []] for p in range(self._pathsCount)]
        self._rimCoordinates = None  # these are just shell coordinates; with core there may also be transition coords
        self._rimNodeIds = None
        self._rimElementIds = None  # [e2][e3][e1]

        self._boxCoordinates = None
        self._transitionCoordinates = None
        self._boxNodeIds = None  # [along][major][minor]
        self._boxBoundaryNodeIds = None
        # boxNodeIds that form the boundary of the solid core, rearranged in circular format
        self._boxBoundaryNodeToBoxId = None
        # lookup table that translates box boundary node ids in a circular format to box node ids in
        # [nAlong][nAcrossMajor][nAcrossMinor] format.
        self._boxElementIds = None  # [along][major][minor]

    def getNetworkSegment(self):
        return self._networkSegment

    def getCoreBoundaryScalingMode(self):
        return self._coreBoundaryScalingMode

    def getElementsCountAround(self):
        return self._elementsCountAround

    def getRawTubeCoordinates(self, pathIndex=0):
        return self._rawTubeCoordinatesList[pathIndex]

    def getIsCore(self):
        return self._isCore

    def getElementsCountCoreBoxMajor(self):
        return self._elementsCountCoreBoxMajor

    def getElementsCountCoreBoxMinor(self):
        return self._elementsCountCoreBoxMinor

    def getElementsCountAcrossTransition(self):
        return self._elementsCountTransition

    def getRawTrackSurface(self, pathIndex=0):
        return self._rawTrackSurfaceList[pathIndex]

    def _sampleTubeCoordinates(self, fixedElementsCountAlong, targetElementLength, transitionFactor=3.0):
        """
        Generate sampled outer, inner tube coordinates optionally trimmed to start/end surfaces.
        Element sizes are constant size at the max length, but compressed if trimmed.
        Lateral cross sections in untrimmed areas are in d2-d3 plane of the network layout,
        hence elements are generally bigger on the outside and smaller on the inside of curves.
        Algorithm uses a finite transition region at trimmed ends based on the trim range, so trimming is local.
        :param fixedElementsCountAlong: Number of elements in resampled coordinates, or None to use targetElementLength.
        :param targetElementLength: Target element length or None to use fixedElementsCountAlong.
        Length is determined from mean trimmed length, subject to a minimum for the configuration.
        :param transitionFactor: Factor > 1.0 multiplying range of trimmed lengths at each end to complete
        local element size transition over.
        """
        lx, ld = self._lengthParameters
        # print("lx", lx, "ld", ld)
        rawNodesCountAlong = len(self._rawTubeCoordinatesList[0][0])
        rawElementsCountAlong = rawNodesCountAlong - 1

        # work out trim lengths of outer, inner tubes
        startCurveLocations = []
        startLengths = []
        sumStartLengths = 0.0
        endCurveLocations = []
        endLengths = []
        sumEndLengths = 0.0
        for p in range(self._pathsCount):
            px, pd1, pd2, pd12 = self._rawTubeCoordinatesList[p]
            startTrimSurface = self._junctions[0].getTrimSurfaces(self)[p]
            endTrimSurface = self._junctions[1].getTrimSurfaces(self)[p]

            for q in range(self._elementsCountAround):
                cx = [px[p][q] for p in range(rawNodesCountAlong)]
                cd2 = [pd2[p][q] for p in range(rawNodesCountAlong)]
                startCurveLocation = (0, 0.0)
                startLength = 0.0
                if startTrimSurface:
                    surfacePosition, curveLocation, intersects = startTrimSurface.findNearestPositionOnCurve(cx, cd2)
                    if intersects:
                        startCurveLocation = curveLocation
                        startLength = evaluateCoordinatesOnCurve(lx, ld, startCurveLocation)[0]
                startCurveLocations.append(startCurveLocation)
                startLengths.append(startLength)
                sumStartLengths += startLength
                endCurveLocation = (rawElementsCountAlong, 1.0)
                endLength = lx[-1][0]
                if endTrimSurface:
                    surfacePosition, curveLocation, intersects = endTrimSurface.findNearestPositionOnCurve(cx, cd2)
                    if intersects:
                        endCurveLocation = curveLocation
                        endLength = evaluateCoordinatesOnCurve(lx, ld, endCurveLocation)[0]
                endCurveLocations.append(endCurveLocation)
                endLengths.append(endLength)
                sumEndLengths += endLength

        minStartLength = min(startLengths)
        maxStartLength = max(startLengths)
        minEndLength = min(endLengths)
        maxEndLength = max(endLengths)
        maxLength = maxEndLength - minStartLength
        # minimum number applies to fixedElementsCountAlong and targetElementLength
        minimumElementsCountAlong = 2 if (self._isLoop or ((self._junctions[0].getSegmentsCount() > 2) and
                                                           (self._junctions[1].getSegmentsCount() > 2))) else 1
        # small fudge factor on targetElementLength so whole numbers chosen on centroid don't go one higher:
        elementsCountAlong = max(minimumElementsCountAlong, fixedElementsCountAlong if fixedElementsCountAlong else
            math.ceil(maxLength * 0.9999 / targetElementLength))
        startTransitionSize = transitionFactor * (maxStartLength - minStartLength)
        endTransitionSize = transitionFactor * (maxEndLength - minEndLength)
        if (startTransitionSize + endTransitionSize) > maxLength:
            # reduce transitions in proportion to fit mean length
            startTransitionSize *= maxLength / (startTransitionSize + endTransitionSize)
            endTransitionSize = maxLength - startTransitionSize
        maxElementLength = maxLength / elementsCountAlong
        # print("maxLength", maxLength, "maxElementLength", maxElementLength)
        # for parametric coordinate [0,1] over startTransitionSize, these are the start derivatives
        # where mean would have value startTransitionSize at both ends
        startTransitionXiSpacing = (maxElementLength / startTransitionSize) if (startTransitionSize > 0.0) else 0.0
        endTransitionXiSpacing = (maxElementLength / endTransitionSize) if (endTransitionSize > 0.0) else 0.0
        dStart = [[None] * self._elementsCountAround for _ in range(self._pathsCount)]
        dEnd = [[None] * self._elementsCountAround for _ in range(self._pathsCount)]
        startTransitionEndLength = minStartLength + startTransitionSize
        endTransitionStartLength = maxEndLength - endTransitionSize
        # print("Transition start min", minStartLength, "max", maxStartLength, "size", startTransitionSize, "end", startTransitionEndLength)
        # print("Transition end min", minEndLength, "max", maxEndLength, "size", endTransitionSize, "end", endTransitionStartLength)
        for p in range(self._pathsCount):
            startTrimSurface = self._junctions[0].getTrimSurfaces(self)[p]
            if startTrimSurface:
                for q in range(self._elementsCountAround):
                    ls = startLengths[p * self._elementsCountAround + q]
                    dStart[p][q] = interpolateLagrangeHermiteDerivative(
                        [ls], [startTransitionEndLength], [startTransitionSize], 0.0)[0]
                    # print("    p", p, "q", q, "dStart", dStart[p][q])
            endTrimSurface = self._junctions[1].getTrimSurfaces(self)[p]
            if endTrimSurface:
                for q in range(self._elementsCountAround):
                    le = endLengths[p * self._elementsCountAround + q]
                    dEnd[p][q] = interpolateHermiteLagrangeDerivative(
                        [endTransitionStartLength], [endTransitionSize], [le], 1.0)[0]
                    # print("    p", p, "q", q, "dEnd", dEnd[p][q])

        tubeGenerator = TubeEllipseGenerator()

        for n in range(elementsCountAlong + 1):

            # get point parameters at mean sampling points
            lm = minStartLength + n * maxElementLength
            curveLocation = getNearestLocationOnCurve(lx, ld, [lm])[0]

            for p in range(self._pathsCount):
                cx, cd1, cd2, cd12, cd3, cd13 = self._pathParametersList[p]
                px, pd1 = evaluateCoordinatesOnCurve(cx, cd1, curveLocation, derivative=True)
                pd2, pd12 = evaluateCoordinatesOnCurve(cd2, cd12, curveLocation, derivative=True)
                pd3, pd13 = evaluateCoordinatesOnCurve(cd3, cd13, curveLocation, derivative=True)
                ex, ed1, ed2, ed12 = tubeGenerator.generate(px, pd1, pd2, pd12, pd3, pd13, self._elementsCountAround,
                                                            maxElementLength)
                startTrimSurface = self._junctions[0].getTrimSurfaces(self)[p]
                endTrimSurface = self._junctions[1].getTrimSurfaces(self)[p]
                startTransition = (startTransitionSize > 0.0) and (lm < startTransitionEndLength)
                endTransition = (endTransitionSize > 0.0) and (lm > endTransitionStartLength)
                if startTransition or endTransition:
                    for q in range(self._elementsCountAround):
                        if startTransition:
                            ls = startLengths[p * self._elementsCountAround + q]
                            xi = n * startTransitionXiSpacing
                            v1, d1, v2, d2 = [ls], [dStart[p][q]], [startTransitionEndLength], [startTransitionSize]
                        else:  # if endTransition:
                            le = endLengths[p * self._elementsCountAround + q]
                            xi = 1.0 - (elementsCountAlong - n) * endTransitionXiSpacing
                            v1, d1, v2, d2 = [endTransitionStartLength], [endTransitionSize], [le], [dEnd[p][q]]
                        lt = interpolateCubicHermite(v1, d1, v2, d2, xi)[0]
                        ltd = interpolateCubicHermiteDerivative(v1, d1, v2, d2, xi)[0]
                        qCurveLocation = getNearestLocationOnCurve(lx, ld, [lt])[0]
                        px, pd1 = evaluateCoordinatesOnCurve(cx, cd1, qCurveLocation, derivative=True)
                        pd2, pd12 = evaluateCoordinatesOnCurve(cd2, cd12, qCurveLocation, derivative=True)
                        pd3, pd13 = evaluateCoordinatesOnCurve(cd3, cd13, qCurveLocation, derivative=True)
                        qex, qed1, qed2, qed12 = tubeGenerator.generate(
                            px, pd1, pd2, pd12, pd3, pd13, self._elementsCountAround,
                            ltd * maxElementLength / (startTransitionSize if startTransition else endTransitionSize))
                        for ev, qv in zip((ex, ed1, ed2, ed12), (qex, qed1, qed2, qed12)):
                            ev[q] = qv[q]
                    # recalculate d1 around rings
                    # first smooth to get d1 with new directions not tangential to surface
                    ted1 = smoothCubicHermiteDerivativesLoop(ex, ed1)
                    # constraint to be tangential to tube surface
                    ted1 = [rejection(ted1[q], normalize(cross(ed1[q], ed2[q])))
                            for q in range(self._elementsCountAround)]
                    # smooth magnitudes only
                    ed1 = smoothCubicHermiteDerivativesLoop(ex, ted1, fixAllDirections=True)

                for lst, ev in zip(self._sampledTubeCoordinates[p], (ex, ed1, ed2, ed12)):
                    lst.append(ev)

        # smooth d2, d12

    def sample(self, fixedElementsCountAlong, targetElementLength):
        self._sampleTubeCoordinates(fixedElementsCountAlong, targetElementLength)

        elementsCountAlong = len(self._sampledTubeCoordinates[0][0]) - 1

        if self._dimension == 2:
            # copy first sampled tube coordinates, but insert single-entry 'n3' index after n2
            self._rimCoordinates = (
                [[ring] for ring in self._sampledTubeCoordinates[0][0]],
                [[ring] for ring in self._sampledTubeCoordinates[0][1]],
                [[ring] for ring in self._sampledTubeCoordinates[0][2]],
                None)
        else:
            shellFactor = 1.0 / self._elementsCountThroughShell
            ox, od1, od2 = self._sampledTubeCoordinates[0][0:3]
            ix, id1, id2 = self._sampledTubeCoordinates[1][0:3]
            rx, rd1, rd2, rd3 = [], [], [], []
            for n2 in range(elementsCountAlong + 1):
                coreCentre, arcCentre = self._determineCentrePoints(n2)
                for r in (rx, rd1, rd2, rd3):
                    r.append([])
                otx, otd1, otd2 = ox[n2], od1[n2], od2[n2]
                itx, itd1, itd2 = ix[n2], id1[n2], id2[n2]
                # wx, wd3 = self._determineShellCoordinates(otx, otd1, otd2, itx, itd1, itd2, coreCentre, arcCentre)
                wd3 = [mult(sub(otx[n1], itx[n1]), shellFactor) for n1 in range(self._elementsCountAround)]
                for n3 in range(self._elementsCountThroughShell + 1):
                    oFactor = n3 / self._elementsCountThroughShell
                    iFactor = 1.0 - oFactor
                    for r in (rx, rd1, rd2, rd3):
                        r[n2].append([])
                    for n1 in range(self._elementsCountAround):
                        if n3 == 0:
                            x, d1, d2 = itx[n1], itd1[n1], itd2[n1]
                        elif n3 == self._elementsCountThroughShell:
                            x, d1, d2 = otx[n1], otd1[n1], otd2[n1]
                        else:
                            x = add(mult(itx[n1], iFactor), mult(otx[n1], oFactor))
                            # x = wx[n3][n1]
                            d1 = add(mult(itd1[n1], iFactor), mult(otd1[n1], oFactor))
                            d2 = add(mult(itd2[n1], iFactor), mult(otd2[n1], oFactor))
                        d3 = wd3[n1]
                        # d3 = wd3[n3][n1]
                        for r, value in zip((rx, rd1, rd2, rd3), (x, d1, d2, d3)):
                            r[n2][n3].append(value)
            self._rimCoordinates = rx, rd1, rd2, rd3

        self._rimNodeIds = [None] * (elementsCountAlong + 1)
        self._rimElementIds = [None] * elementsCountAlong
        self._boxElementIds = [None] * elementsCountAlong

        if self._isCore:
            # sample coordinates for the solid core
            self._sampleCoreCoordinates(elementsCountAlong)

    def _sampleCoreCoordinates(self, elementsCountAlong):
        """
        Black box function for sampling coordinates for the solid core.
        :param elementsCountAlong: A number of elements along a segment.
        """
        boxx, boxd1, boxd3 = [], [], []
        transx, transd1, transd3 = [], [], []
        for n2 in range(elementsCountAlong + 1):
            coreCentre, arcCentre = self._determineCentrePoints(n2)
            cbx, cbd1, cbd3, ctx, ctd1, ctd3 = self._generateCoreCoordinates(n2, coreCentre)
            for lst, value in zip((boxx, boxd1, boxd3, transx, transd1, transd3),
                                  (cbx, cbd1, cbd3, ctx, ctd1, ctd3)):
                lst.append(value)
        boxd2, transd2 = self._determineCoreD2Derivatives(boxx, boxd1, boxd3, transx, transd1, transd3)
        self._boxCoordinates = boxx, boxd1, boxd2, boxd3
        self._transitionCoordinates = transx, transd1, transd2, transd3
        self._boxNodeIds = [None] * (elementsCountAlong + 1)

    def _determineCentrePoints(self, n2):
        """
        Calculates coordinates for the centre of the solid core based on outer and inner tube coordinates.
        :param n2: Index for elements along the tube.
        :return: Coordinates of the solid core.
        """
        ox = self._sampledTubeCoordinates[0][0][n2]
        ix = self._sampledTubeCoordinates[1][0][n2]
        cp = []

        for x in [ox, ix]:
            P0 = x[self._elementsCountAround // 4]
            P1 = x[self._elementsCountAround // 4 * 3]
            midpoint = mult(add(P0, P1), 0.5)
            cp.append(midpoint)

        coreCentres, arcCentres = [], []
        for i in range(self._elementsCountAround):
            tol = 1e-10  # tolerance to avoid float division zero
            OP = magnitude(sub(ox[i], cp[0]))
            IP = magnitude(sub(ix[i], cp[1]))
            distBetweenOuterAndInner = magnitude(sub(cp[1], cp[0]))
            distBetweenOuterAndInner = tol if distBetweenOuterAndInner == 0 else distBetweenOuterAndInner
            outerBase = (OP ** 2 - IP ** 2 - distBetweenOuterAndInner ** 2) / (2 * distBetweenOuterAndInner)
            circularArcRadius = math.sqrt(outerBase ** 2 + OP ** 2)
            distBetweenCoreAndInner = circularArcRadius - outerBase - distBetweenOuterAndInner

            directionVector = sub(cp[1], cp[0])
            if directionVector[0] == 0 and directionVector[1] == 0 and directionVector[2] == 0:
                directionVector = [tol, tol, tol]
            scaledDV = mult(normalize(directionVector), (distBetweenOuterAndInner + distBetweenCoreAndInner))
            c = add(scaledDV, cp[0])
            coreCentres.append(c)

            dvi = [-d for d in directionVector]
            mag = magnitude(dvi)
            tol = 1e-10
            scaleDVI = mult(normalize(dvi), outerBase) if mag > tol else [0, 0, 0]
            ac = add(scaleDVI, cp[0])
            arcCentres.append(ac)

        coreCentre = [sum(e) / len(e) for e in zip(*coreCentres)]
        arcCentre = [sum(e) / len(e) for e in zip(*arcCentres)]

        return coreCentre, arcCentre

    def _getRadialCoreCrossing(self, n1Start, n1Half, n2, centre, centreNormal=None):
        """
        Get start, middle and end coordinates across solid core.
        :param n1Start: Start index around rim coordinates.
        :param n1Half: If True, initial n1 is halfway between n1Start and n1Start + 1
        :param n2: Index along rim coordinates.
        :param centre: Centre coordinates.
        :param centreNormal: Optional normal to remove component in direction from centre.
        :return: Radial rx, rd1 (across), rd2(up) for points at n1, centre and opposite n1.
        """
        ix = self._rimCoordinates[0][n2][0]
        id1 = self._rimCoordinates[1][n2][0]
        id3 = self._rimCoordinates[3][n2][0]
        n1End = n1Start + self._elementsCountAround // 2
        if n1Half:
            nx = self._rimCoordinates[0][n2][1]
            nd1 = self._rimCoordinates[1][n2][1]
            ax, ad2 = evaluateCoordinatesOnCurve(ix, id1, (n1Start, 0.5), loop=True, derivative=True)
            tx = evaluateCoordinatesOnCurve(nx, nd1, (n1Start, 0.5), loop=True)
            ad1 = sub(ix, tx)
            cx, cd2 = evaluateCoordinatesOnCurve(ix, id1, (n1End, 0.5), loop=True)
            tx = evaluateCoordinatesOnCurve(nx, nd1, (n1End, 0.5), loop=True)
            cd1 = sub(tx, ix)
            cd2 = [-d for d in cd2]  # since these go around in RH sense
        else:
            ax = copy.copy(ix[n1Start])
            ad1 = [-d for d in id3[n1Start]]
            ad2 = id1[n1Start]
            cx = copy.copy(ix[n1End])
            cd1 = copy.copy(id3[n1End])
            cd2 = [-d for d in id1[n1End]]  # since these go around in RH sense
        bx = centre
        # following gives the core the expected S-shape distortion when twisted
        bd1a = interpolateHermiteLagrangeDerivative(ax, set_magnitude(ad1, magnitude(sub(bx, ax))), bx, 1.0)
        bd1c = interpolateLagrangeHermiteDerivative(bx, cx, set_magnitude(cd1, magnitude(sub(cx, bx))), 0.0)
        bd1 = mult(add(bd1a, bd1c), 0.5)
        bd2 = mult(add(ad2, cd2), 0.5)
        rx = [ax, bx, cx]
        rd1_us = [ad1, bd1, cd1]
        rd2 = [ad2, bd2, cd2]
        rd1 = smoothCubicHermiteDerivativesLine(rx, rd1_us, fixAllDirections=True)
        if centreNormal:
            # constrain centre derivatives to be orthogonal to centreNormal
            rd1[1] = rejection(rd1[1], centreNormal)
            rd2[1] = rejection(rd2[1], centreNormal)
            rd1 = smoothCubicHermiteDerivativesLine(rx, rd1, fixAllDirections=True)
        return rx, rd1, rd2

    def _generateCoreCoordinates(self, n2, centre):
        r"""
        Sample core box and transition elements by sampling radial crossings along
        major -, minor |, diag1 /, diag2 \ directions.
        From these 3x3 array of points at the corners, centres and mid-sides of the box are determined,
        then the actual box elements are resampled from these.
        Transition elements are determined by blending from the outside of the box to the shell.
        :param n2: Index along segment.
        :param centre: Centre coordinates of core.
        :return: box coordinates cbx, cbd1, cbd3, transition coordinates ctx, ctd1, ctd3.
        Box coordinates are over [minorBoxNodeCount][majorBoxNodeCount]. Transition coordinates are over [n3][n1] and
        are only non-empty for at least 2 transition elements as they are the layers between the box and the shell.
        """
        # sample radially across major, minor and both diagonals, like a Union Jack
        major_n1 = 0
        major_x, major_d1, major_d3 = self._getRadialCoreCrossing(
            major_n1, (self._elementsCountCoreBoxMinor % 2) == 1, n2, centre)
        minor_n1 = -((self._elementsCountAround + 3) // 4)
        minor_x, minor_d3, minor_d1 = self._getRadialCoreCrossing(
            minor_n1, (self._elementsCountCoreBoxMajor % 2) == 1, n2, centre)
        minor_d1 = [[-d for d in v] for v in minor_d1]
        major_d3[1] = minor_d3[1]
        minor_d1[1] = major_d1[1]
        centreNormal = normalize(cross(major_d1[1], major_d3[1]))
        majorBoxSize = self._elementsCountCoreBoxMajor
        minorBoxSize = self._elementsCountCoreBoxMinor
        diag1_n1 = minorBoxSize // -2
        diag1_x, diag1_d1, diag1_d3 = self._getRadialCoreCrossing(diag1_n1, False, n2, centre, centreNormal)
        diag2_n1 = diag1_n1 + minorBoxSize
        diag2_x, diag2_d1, diag2_d3 = self._getRadialCoreCrossing(diag2_n1, False, n2, centre, centreNormal)

        # sample to sides and corners of core box
        majorXi = (2.0 * self._elementsCountTransition /
                   (self._elementsCountCoreBoxMajor + 2 * self._elementsCountTransition))
        majorXiR = 1.0 - majorXi
        minorXi = (2.0 * self._elementsCountTransition /
                   (self._elementsCountCoreBoxMinor + 2 * self._elementsCountTransition))
        minorXiR = 1.0 - minorXi
        # following expression adjusted to look best across all cases
        boxDiagonalSize = math.sqrt(majorBoxSize * majorBoxSize + minorBoxSize * minorBoxSize)
        diagXi = self._elementsCountTransition / (0.5 * boxDiagonalSize + self._elementsCountTransition + 0.5)
        diagXiR = 1.0 - diagXi

        # 3x3 nodes (2x2 elements) giving extents of core box
        tripleAngle = math.pi / 3.0
        cosTripleAngle = math.cos(tripleAngle)
        sinTripleAngle = math.sin(tripleAngle)

        e00x, e00dr = evaluateCoordinatesOnCurve(diag1_x, diag1_d1, (0, diagXi), derivative=True)
        e00dc = set_magnitude(add(mult(normalize(diag1_d3[0]), diagXiR), mult(normalize(diag1_d3[1]), diagXi)), magnitude(e00dr))
        e00d1 = add(mult(e00dr, cosTripleAngle), mult(e00dc, -sinTripleAngle))
        e00d3 = add(mult(e00dr, cosTripleAngle), mult(e00dc, sinTripleAngle))

        e02x, e02dr = evaluateCoordinatesOnCurve(diag2_x, diag2_d1, (1, diagXiR), derivative=True)
        e02dc = set_magnitude(add(mult(normalize(diag2_d3[1]), diagXi), mult(normalize(diag2_d3[2]), diagXiR)), magnitude(e02dr))
        e02d1 = add(mult(e02dr, cosTripleAngle), mult(e02dc, sinTripleAngle))
        e02d3 = add(mult(e02dr, -cosTripleAngle), mult(e02dc, sinTripleAngle))

        e20x, e20dr = evaluateCoordinatesOnCurve(diag2_x, diag2_d1, (0, diagXi), derivative=True)
        e20dc = set_magnitude(add(mult(normalize(diag2_d3[0]), diagXiR), mult(normalize(diag2_d3[1]), diagXi)), magnitude(e20dr))
        e20d1 = add(mult(e20dr, cosTripleAngle), mult(e20dc, sinTripleAngle))
        e20d3 = add(mult(e20dr, -cosTripleAngle), mult(e20dc, sinTripleAngle))

        e22x, e22dr = evaluateCoordinatesOnCurve(diag1_x, diag1_d1, (1, diagXiR), derivative=True)
        e22dc = set_magnitude(add(mult(normalize(diag1_d3[1]), diagXi), mult(normalize(diag1_d3[2]), diagXiR)), magnitude(e22dr))
        e22d1 = add(mult(e22dr, cosTripleAngle), mult(e22dc, -sinTripleAngle))
        e22d3 = add(mult(e22dr, cosTripleAngle), mult(e22dc, sinTripleAngle))

        ex = [
            e00x,
            evaluateCoordinatesOnCurve(minor_x, minor_d3, (0, minorXi)),
            e02x,
            evaluateCoordinatesOnCurve(major_x, major_d1, (0, majorXi)),
            centre,
            evaluateCoordinatesOnCurve(major_x, major_d1, (1, majorXiR)),
            e20x,
            evaluateCoordinatesOnCurve(minor_x, minor_d3, (1, minorXiR)),
            e22x
        ]
        majorScale = (2.0 / majorBoxSize) if (majorBoxSize > 0.0) else 1.0
        minorScale = (2.0 / minorBoxSize) if (minorBoxSize > 0.0) else 1.0
        major_ed1 = [
            mult(interpolateCubicHermiteDerivative(major_x[0], major_d1[0], major_x[1], major_d1[1], majorXi), majorXiR),
            mult(major_d1[1], majorXiR),
            mult(interpolateCubicHermiteDerivative(major_x[1], major_d1[1], major_x[2], major_d1[2], majorXiR), majorXiR)]
        minor_ed3 = [
            mult(interpolateCubicHermiteDerivative(minor_x[0], minor_d3[0], minor_x[1], minor_d3[1], minorXi), minorXiR),
            mult(minor_d3[1], minorXiR),
            mult(interpolateCubicHermiteDerivative(minor_x[1], minor_d3[1], minor_x[2], minor_d3[2], minorXiR), minorXiR)]
        mag_ed3 = (magnitude(minor_ed3[1]) * minorScale * majorXi + magnitude(major_d3[0]) * majorXiR) / minorScale
        major_ed3 = [
            set_magnitude(add(mult(normalize(major_d3[0]), majorXiR), mult(normalize(major_d3[1]), majorXi)), mag_ed3),
            minor_ed3[1],
            set_magnitude(add(mult(normalize(major_d3[1]), majorXi), mult(normalize(major_d3[2]), majorXiR)), mag_ed3)]
        mag_ed1 = (magnitude(major_ed1[1]) * majorScale * minorXi + magnitude(minor_d1[0]) * minorXiR) / majorScale
        minor_ed1 = [
            set_magnitude(add(mult(normalize(minor_d1[0]), minorXiR), mult(normalize(minor_d1[1]), minorXi)), mag_ed1),
            major_ed1[1],
            set_magnitude(add(mult(normalize(minor_d1[1]), minorXi), mult(normalize(minor_d1[2]), minorXiR)), mag_ed1)]
        ed1 = [
            smoothCubicHermiteDerivativesLine(
                [ex[0], ex[1]], [e00d1, minor_ed1[0]], fixStartDirection=True, fixEndDerivative=True)[0],
            minor_ed1[0],
            smoothCubicHermiteDerivativesLine(
                [ex[1], ex[2]], [minor_ed1[0], e02d1], fixStartDerivative=True, fixEndDirection=True)[1],
            major_ed1[0],
            major_ed1[1],
            major_ed1[2],
            smoothCubicHermiteDerivativesLine(
                [ex[6], ex[7]], [e20d1, minor_ed1[2]], fixStartDirection=True, fixEndDerivative=True)[0],
            minor_ed1[2],
            smoothCubicHermiteDerivativesLine(
                [ex[7], ex[8]], [minor_ed1[2], e22d1], fixStartDerivative=True, fixEndDirection=True)[1],
        ]
        ed3 = [
            smoothCubicHermiteDerivativesLine(
                [ex[0], ex[3]], [e00d3, major_ed3[0]], fixStartDirection=True, fixEndDerivative=True)[0],
            minor_ed3[0],
            smoothCubicHermiteDerivativesLine(
                [ex[2], ex[5]], [e02d3, major_ed3[2]], fixStartDirection=True, fixEndDerivative=True)[0],
            major_ed3[0],
            major_ed3[1],
            major_ed3[2],
            smoothCubicHermiteDerivativesLine(
                [ex[3], ex[6]], [major_ed3[0], e20d3], fixStartDerivative=True, fixEndDirection=True)[1],
            minor_ed3[2],
            smoothCubicHermiteDerivativesLine(
                [ex[5], ex[8]], [major_ed3[2], e22d3], fixStartDerivative=True, fixEndDirection=True)[1],
        ]

        # Create an empty list for the core with dimensions of (major - 1) x (minor - 1)
        cbx, cbd1, cbd3, ctx, ctd1, ctd3 = [], [], [], [], [], []

        trackSurface = TrackSurface(2, 2, ex, ed1, ed3)
        for m in range(majorBoxSize + 1):
            majorProportion = m / majorBoxSize
            row_x = []
            row_d1 = []
            row_d3 = []
            for n in range(minorBoxSize + 1):
                minorProportion = (n / minorBoxSize) if (minorBoxSize > 0) else 0.0
                position = trackSurface.createPositionProportion(majorProportion, minorProportion)
                x, d1, d3 = trackSurface.evaluateCoordinates(position, derivatives=True)
                row_x.append(x)
                row_d1.append(mult(d1, majorScale))
                row_d3.append(mult(d3, minorScale))
            cbx.append(row_x)
            cbd1.append(row_d1)
            cbd3.append(row_d3)

        if self._elementsCountTransition > 1:
            for i in range(self._elementsCountTransition - 1):
                for lst in (ctx, ctd1, ctd3):
                    lst.append([None] * self._elementsCountAround)
            ix = self._rimCoordinates[0][n2][0]
            id1 = self._rimCoordinates[1][n2][0]
            id3 = self._rimCoordinates[3][n2][0]
            start_bn3 = minorBoxSize // 2
            topLeft_n1 = minorBoxSize - start_bn3
            topRight_n1 = topLeft_n1 + majorBoxSize
            bottomRight_n1 = topRight_n1 + minorBoxSize
            bottomLeft_n1 = bottomRight_n1 + majorBoxSize
            for n1 in range(self._elementsCountAround):
                if n1 <= topLeft_n1:
                    bn1 = 0
                    bn3 = start_bn3 + n1
                    if n1 < topLeft_n1:
                        start_d1 = cbd3[bn1][bn3]
                        start_d3 = [-d for d in cbd1[bn1][bn3]]
                    else:
                        start_d1 = add(cbd3[bn1][bn3], cbd1[bn1][bn3])
                        start_d3 = sub(cbd3[bn1][bn3], cbd1[bn1][bn3])
                elif n1 <= topRight_n1:
                    bn1 = n1 - topLeft_n1
                    bn3 = minorBoxSize
                    if n1 < topRight_n1:
                        start_d1 = cbd1[bn1][bn3]
                        start_d3 = cbd3[bn1][bn3]
                    else:
                        start_d1 = sub(cbd1[bn1][bn3], cbd3[bn1][bn3])
                        start_d3 = add(cbd1[bn1][bn3], cbd3[bn1][bn3])
                elif n1 <= bottomRight_n1:
                    bn1 = majorBoxSize
                    bn3 = minorBoxSize - (n1 - topRight_n1)
                    if n1 < bottomRight_n1:
                        start_d1 = [-d for d in cbd3[bn1][bn3]]
                        start_d3 = cbd1[bn1][bn3]
                    else:
                        start_d1 = [-d for d in add(cbd1[bn1][bn3], cbd3[bn1][bn3])]
                        start_d3 = sub(cbd1[bn1][bn3], cbd3[bn1][bn3])
                elif n1 <= bottomLeft_n1:
                    bn1 = majorBoxSize - (n1 - bottomRight_n1)
                    bn3 = 0
                    if n1 < bottomLeft_n1:
                        start_d1 = [-d for d in cbd1[bn1][bn3]]
                        start_d3 = [-d for d in cbd3[bn1][bn3]]
                    else:
                        start_d1 = sub(cbd3[bn1][bn3], cbd1[bn1][bn3])
                        start_d3 = [-d for d in add(cbd1[bn1][bn3], cbd3[bn1][bn3])]
                else:
                    bn1 = 0
                    bn3 = n1 - bottomLeft_n1
                    start_d1 = cbd3[bn1][bn3]
                    start_d3 = [-d for d in cbd1[bn1][bn3]]
                start_x = cbx[bn1][bn3]

                nx = [start_x, ix[n1]]
                nd3before = [[self._elementsCountTransition * d for d in start_d3], id3[n1]]
                nd3 = [nd3before[0], computeCubicHermiteEndDerivative(nx[0], nd3before[0], nx[1], nd3before[1])]
                tx, td3, pe, pxi, psf = sampleCubicHermiteCurvesSmooth(
                    nx, nd3, self._elementsCountTransition,
                    derivativeMagnitudeStart=magnitude(nd3[0]) / self._elementsCountTransition,
                    derivativeMagnitudeEnd=magnitude(nd3[1]) / self._elementsCountTransition)
                delta_id1 = sub(id1[n1], start_d1)
                td1 = interpolateSampleCubicHermite([start_d1, id1[n1]], [delta_id1, delta_id1], pe, pxi, psf)[0]

                for n3 in range(1, self._elementsCountTransition):
                    ctx[n3 - 1][n1] = tx[n3]
                    ctd1[n3 - 1][n1] = td1[n3]
                    ctd3[n3 - 1][n1] = td3[n3]

            # smooth td1 around:
            for n3 in range(1, self._elementsCountTransition):
                ctd1[n3 - 1] = smoothCubicHermiteDerivativesLoop(ctx[n3 - 1], ctd1[n3 - 1], fixAllDirections=False)

        return cbx, cbd1, cbd3, ctx, ctd1, ctd3

    def _determineCoreD2Derivatives(self, boxx, boxd1, boxd3, transx, transd1, transd3):
        """
        Compute d2 derivatives for the solid core.
        :param boxx, boxd1, boxd3: Coordinates and derivatives (d1 & d3) of the core box nodes.
        :param transx, transd1, transd3: Coordinates and derivatives (d1 & d3) of the core transition nodes.
        :return: D2 derivatives of box and rim components of the core.
        """
        elementsCountAlong = len(boxx)
        coreBoxMajorNodesCount = len(boxx[0])
        coreBoxMinorNodesCount = len(boxx[0][0])

        boxd2 = [[[None for _ in range(coreBoxMinorNodesCount)] for _ in range(coreBoxMajorNodesCount)]
                 for _ in range(elementsCountAlong)]
        transd2 = [[[None for _ in range(self._elementsCountAround)] for _ in range(self._elementsCountTransition - 1)]
                   for _ in range(elementsCountAlong)]

        # compute core d2 directions by weighting with 1/distance from inner coordinates

        def get_d2(n2, x):
            sum_weight = 0.0
            sum_d2 = [0.0, 0.0, 0.0]
            ix = self._rimCoordinates[0][n2][0]
            id2 = self._rimCoordinates[2][n2][0]
            for i in range(len(ix)):
                distance_sq = 0.0
                for c in range(3):
                    delta = x[c] - ix[i][c]
                    distance_sq += delta * delta
                if distance_sq == 0.0:
                    return id2[i]
                weight = 1.0 / math.sqrt(distance_sq)
                sum_weight += weight
                for c in range(3):
                    sum_d2[c] += weight * id2[i][c]
            return [sum_d2[c] / sum_weight for c in range(3)]

        for m in range(coreBoxMajorNodesCount):
            for n in range(coreBoxMinorNodesCount):
                tx, td2 = [], []
                for n2 in range(elementsCountAlong):
                    x = boxx[n2][m][n]
                    tx.append(x)
                    td2.append(get_d2(n2, x))
                # td2 = smoothCubicHermiteDerivativesLine(tx, td2, fixAllDirections=True)
                for n2 in range(elementsCountAlong):
                    boxd2[n2][m][n] = td2[n2]

        if self._elementsCountTransition > 1:
            for n3 in range(self._elementsCountTransition - 1):
                for n1 in range(self._elementsCountAround):
                    tx, td2 = [], []
                    for n2 in range(elementsCountAlong):
                        x = transx[n2][n3][n1]
                        tx.append(x)
                        td2.append(get_d2(n2, x))
                    # td2 = smoothCubicHermiteDerivativesLine(tx, td2, fixAllDirections=True)
                    for n2 in range(elementsCountAlong):
                        transd2[n2][n3][n1] = td2[n2]

        return boxd2, transd2

    def _determineShellCoordinates(self, ox, od1, od2, ix, id1, id2, coreCentre, arcCentre):
        """
        Calculates rim coordinates and d3 derivatives based on the centre point of the solid core.
        :param ox, od1, od2: Coordinates and (d1 and d2) derivatives for outermost rim.
        :param ix, id1, id2: Coordinates and (d1 and d2) derivatives for innermost rim.
        :param coreCentre: Centre point of the solid core.
        :param arcCetnre: Centre point of the arc that passes through the core centre, inner rim and outer rim.
        :return: Coordinates and d3 derivatives for rim nodes.
        """
        wx, wd3 = [], []

        # check if the cross-section of cylinder is regular shaped or irregular.
        dist1a = sub(ix[0], coreCentre)
        dist1b = sub(ix[self._elementsCountAround // 2], coreCentre)
        dist2a = sub(ix[self._elementsCountAround // 4], coreCentre)
        dist2b = sub(ix[self._elementsCountAround // 4 * 3], coreCentre)
        tol = 1e-3
        if abs(magnitude(dist1a) - magnitude(dist1b)) > tol or \
                abs(magnitude(dist2a) - magnitude(dist2b)) > tol:
            isRegular = False
        else:
            isRegular = True

        # Calculate d3 derivatives
        tx, td3 = [], []
        for n1 in range(self._elementsCountAround):
            if isRegular:
                tol = 1e-10
                dist = sub(arcCentre, coreCentre)
                if magnitude(dist) > tol:
                    if dist > [tol, tol, tol]:
                        oc = sub(ox[n1], arcCentre)
                        ic = sub(ix[n1], arcCentre)
                    else:
                        oc = add(mult(oc[n1], -1), arcCentre)
                        ic = add(mult(ic[n1], -1), arcCentre)
                    ot = cross(oc, od1[n1])
                    it = cross(ic, id1[n1])
                else:
                    ot, it = cross(od1[n1], od2[n1]), cross(id1[n1], id2[n1])
                scalefactor = magnitude(sub(ox[n1], ix[n1])) / self._elementsCountThroughShell
                od3 = mult(normalize(ot), scalefactor)
                id3 = mult(normalize(it), scalefactor)
            else:
                shellFactor = 1.0 / self._elementsCountThroughShell
                od3 = id3 = mult(sub(ox[n1], ix[n1]), shellFactor)

            txm, td3m, pe, pxi, psf = sampleCubicHermiteCurves(
                [ix[n1], ox[n1]], [id3, od3], self._elementsCountThroughShell, arcLengthDerivatives=True)

            td3m = smoothCubicHermiteDerivativesLine(txm, td3m, fixStartDirection=True, fixEndDirection=True)

            tx.append(txm)
            td3.append(td3m)

        for n3 in range(self._elementsCountThroughShell + 1):
            wx.append([])
            wd3.append([])
            for n1 in range(self._elementsCountAround):
                wx[n3].append(tx[n1][n3])
                wd3[n3].append(td3[n1][n3])

        return wx, wd3

    def _createBoxBoundaryNodeIdsList(self, startSkipCount=None, endSkipCount=None):
        """
        Creates a list (in a circular format similar to other rim node id lists) of core box node ids that are
        located at the boundary of the core.
        This list is used to easily stitch inner rim nodes with box nodes.
        :param startSkipCount: Row in from start that node ids are for.
        :param endSkipCount: Row in from end that node ids are for.
        :return: A list of box node ids stored in a circular format, and a lookup list that translates indexes used in
        boxBoundaryNodeIds list to indexes that can be used in boxCoordinates list.
        """
        boxBoundaryNodeIds = []
        boxBoundaryNodeToBoxId = []
        elementsCountAlong = len(self._rimCoordinates[0]) - 1

        coreBoxMajorNodesCount = self._elementsCountCoreBoxMajor + 1
        coreBoxMinorNodesCount = self._elementsCountCoreBoxMinor + 1
        for n2 in range(elementsCountAlong + 1):
            if (n2 < startSkipCount) or (n2 > elementsCountAlong - endSkipCount) or self._boxNodeIds[n2] is None:
                boxBoundaryNodeIds.append(None)
                boxBoundaryNodeToBoxId.append(None)
                continue
            else:
                boxBoundaryNodeIds.append([])
                boxBoundaryNodeToBoxId.append([])
            for n3 in range(coreBoxMajorNodesCount):
                if n3 == 0 or n3 == coreBoxMajorNodesCount - 1:
                    ids = self._boxNodeIds[n2][n3] if n3 == 0 else self._boxNodeIds[n2][n3][::-1]
                    n1List = list(range(coreBoxMinorNodesCount)) if n3 == 0 else (
                        list(range(coreBoxMinorNodesCount - 1, -1, -1)))
                    boxBoundaryNodeIds[n2] += [ids[c] for c in range(coreBoxMinorNodesCount)]
                    for n1 in n1List:
                        boxBoundaryNodeToBoxId[n2].append([n3, n1])
                else:
                    for n1 in [-1, 0]:
                        boxBoundaryNodeIds[n2].append(self._boxNodeIds[n2][n3][n1])
                        boxBoundaryNodeToBoxId[n2].append([n3, n1])

            start = self._elementsCountCoreBoxMajor - 2
            idx = self._elementsCountCoreBoxMinor + 2
            for n in range(int(start), -1, -1):
                boxBoundaryNodeIds[n2].append(boxBoundaryNodeIds[n2].pop(idx + 2 * n))
                boxBoundaryNodeToBoxId[n2].append(boxBoundaryNodeToBoxId[n2].pop(idx + 2 * n))

            nloop = self._elementsCountCoreBoxMinor // 2
            for _ in range(nloop):
                boxBoundaryNodeIds[n2].insert(len(boxBoundaryNodeIds[n2]), boxBoundaryNodeIds[n2].pop(0))
                boxBoundaryNodeToBoxId[n2].insert(len(boxBoundaryNodeToBoxId[n2]),
                                                      boxBoundaryNodeToBoxId[n2].pop(0))

        return boxBoundaryNodeIds, boxBoundaryNodeToBoxId

    @classmethod
    def blendSampledCoordinates(cls, segment1, nodeIndexAlong1, segment2, nodeIndexAlong2):
        nodesCountAround = segment1._elementsCountAround
        nodesCountRim = len(segment1._rimCoordinates[0][nodeIndexAlong1])
        if ((nodesCountAround != segment2._elementsCountAround) or
                (nodesCountRim != len(segment2._rimCoordinates[0][nodeIndexAlong2]))):
            return  # can't blend unless these match

        if segment1._isCore and segment2._isCore:
            coreBoxMajorNodesCount = len(segment1._boxCoordinates[0][nodeIndexAlong1])
            coreBoxMinorNodesCount = len(segment1._boxCoordinates[0][nodeIndexAlong1][0])
            nodesCountTransition = len(segment1._transitionCoordinates[0][nodeIndexAlong1]) if segment1._transitionCoordinates else 0
            if ((coreBoxMajorNodesCount != len(segment2._boxCoordinates[0][nodeIndexAlong2])) or
                    (coreBoxMinorNodesCount != len(segment2._boxCoordinates[0][nodeIndexAlong2][0])) or
                    (nodesCountTransition != (len(segment2._transitionCoordinates[0][nodeIndexAlong2])
                    if segment1._transitionCoordinates else 0))):
                return  # can't blend unless these match
            # blend core coordinates
            s1d2 = segment1._boxCoordinates[2][nodeIndexAlong1]
            s2d2 = segment2._boxCoordinates[2][nodeIndexAlong2]
            for m in range(coreBoxMajorNodesCount):
                for n in range(coreBoxMinorNodesCount):
                    # harmonic mean magnitude
                    s1d2Mag = magnitude(s1d2[m][n])
                    s2d2Mag = magnitude(s2d2[m][n])
                    d2Mag = 2.0 / ((1.0 / s1d2Mag) + (1.0 / s2d2Mag))
                    s2d2[m][n] = s1d2[m][n] = mult(s1d2[m][n], d2Mag / s1d2Mag)
            for n3 in range(nodesCountTransition):
                s1d2 = segment1._transitionCoordinates[2][nodeIndexAlong1][n3]
                s2d2 = segment2._transitionCoordinates[2][nodeIndexAlong2][n3]
                for n1 in range(nodesCountAround):
                    # harmonic mean magnitude
                    s1d2Mag = magnitude(s1d2[n1])
                    s2d2Mag = magnitude(s2d2[n1])
                    d2Mag = 2.0 / ((1.0 / s1d2Mag) + (1.0 / s2d2Mag))
                    s2d2[n1] = s1d2[n1] = mult(s1d2[n1], d2Mag / s1d2Mag)
        elif segment1._isCore or segment2._isCore:
            return  # can't blend if both don't have core

        # blend rim coordinates
        for n3 in range(nodesCountRim):
            s1d2 = segment1._rimCoordinates[2][nodeIndexAlong1][n3]
            s2d2 = segment2._rimCoordinates[2][nodeIndexAlong2][n3]
            for n1 in range(nodesCountAround):
                # harmonic mean magnitude
                s1d2Mag = magnitude(s1d2[n1])
                s2d2Mag = magnitude(s2d2[n1])
                d2Mag = 2.0 / ((1.0 / s1d2Mag) + (1.0 / s2d2Mag))
                s2d2[n1] = s1d2[n1] = mult(s1d2[n1], d2Mag / s1d2Mag)

    def getSampledElementsCountAlong(self):
        return len(self._sampledTubeCoordinates[0][0]) - 1

    def getSampledTubeCoordinatesRing(self, pathIndex, nodeIndexAlong):
        """
        Get a ring of sampled coordinates at the supplied node index.
        :param pathIndex: 0 for outer/primary, 1 or -1 for inner/secondary.
        :param nodeIndexAlong: Node index from 0 to self._elementsCountAlong, or negative to count from end.
        :return: sx[nAround]
        """
        return self._sampledTubeCoordinates[pathIndex][0][nodeIndexAlong]

    def getElementsCountShell(self):
        """
        :return: Number of elements through the non-core shell.
        """
        return max(1, len(self._rimCoordinates[0][0]) - 1)

    def getElementsCountRim(self):
        """
        :return: Number of elements radially outside core box if core is on,
        otherwise same as number through shell.
        """
        elementsCountRim = self.getElementsCountShell()
        if self._isCore:
            elementsCountRim += self._elementsCountTransition
        return elementsCountRim

    def getNodesCountRim(self):
        """
        :return: Number of transition + shell nodes
        """
        nodesCountRim = len(self._rimCoordinates[0][0])
        if self._isCore:
            nodesCountRim += (self._elementsCountTransition - 1)
        return nodesCountRim

    def getRimCoordinatesListAlong(self, n1, n2List, n3):
        """
        Get list of parameters for n2 indexes along segment at given n1, n3.
        :param n1: Node index around segment.
        :param n2List: List of node indexes along segment.
        :param n3: Node index from inner to outer rim.
        :return: [x[], d1[], d2[], d3[]]. d3[] may be None
        """
        paramsList = []
        for i in range(4):
            params = []
            for n2 in n2List:
                params.append(self._rimCoordinates[i][n2][n3][n1] if self._rimCoordinates[i] else None)
            paramsList.append(params)
        return paramsList

    def getBoxCoordinatesListAlong(self, n1, n2List, n3):
        """
        Get a list of parameters for solid core box for n2 indexes along segment at given n1, n3.
        :param n1: Node index around segment.
        :param n2List: List of node indexes along segment.
        :param n3: Node index from inner to outer rim.
        :return: [x[], d1[], d2[], d3[]].
        """
        paramsList = []
        for i in range(4):
            params = []
            for n2 in n2List:
                params.append(self._boxCoordinates[i][n2][n3][n1] if self._boxCoordinates[i] else None)
            paramsList.append(params)

        return paramsList

    def getTransitionCoordinatesListAlong(self, n1, n2List, n3):
        """
        Get a list of parameters for core transition nodes for n2 indexes along segment at given n1, n3.
        :param n1: Node index around segment.
        :param n2List: List of node indexes along segment.
        :param n3: Node index from inner to outer rim.
        :return: [x[], d1[], d2[], d3[]].
        """
        paramsList = []
        for i in range(4):
            params = []
            for n2 in n2List:
                params.append(self._transitionCoordinates[i][n2][n3][n1] if self._transitionCoordinates[i]
                              else None)
            paramsList.append(params)

        return paramsList

    def getCoreBoxMajorNodesCount(self):
        return len(self._boxCoordinates[0][0])

    def getCoreBoxMinorNodesCount(self):
        return len(self._boxCoordinates[0][0][0])

    def getElementsCountTransition(self):
        return self._elementsCountTransition

    def getBoxCoordinates(self, n1, n2, n3):
        return (self._boxCoordinates[0][n2][n3][n1], self._boxCoordinates[1][n2][n3][n1],
                self._boxCoordinates[2][n2][n3][n1], self._boxCoordinates[3][n2][n3][n1])

    def getBoxNodeIds(self, n1, n2, n3):
        """
        Get a box node ID for n2 index along segment at given n1, n3.
        :param n1: Node index across major axis (y-axis).
        :param n2: Node index along segment.
        :param n3: Node index across minor axis (z-axis).
        :return: Node identifier.
        """
        return self._boxNodeIds[n2][n3][n1]

    def getBoxNodeIdsSlice(self, n2):
        """
        Get slice of box node IDs at n2 index along segment.
        :param n2: Node index along segment, including negative indexes from end.
        :return: Node IDs arrays, or None if not set.
        """
        return self._boxNodeIds[n2]

    def getBoxBoundaryNodeIds(self, n1, n2):
        """
        Get a node ID around the core box for n2 index along segment at a given n1.
        :param n1: Node index around the core box.
        :param n2: Node index along segment.
        :return: Node identifier.
        """
        return self._boxBoundaryNodeIds[n2][n1]

    def getBoxBoundaryNodeToBoxId(self, n1, n2):
        """
        Translate box boundary node indexes to core box node indexes in the form of major- and minor axes.
        :param n1: Node index around the core box.
        :param n2: Node index along segment.
        :return: n3 (major axis) and n1 (minor axis) indexes used in boxCoordinates.
        """
        return self._boxBoundaryNodeToBoxId[n2][n1]

    def getTriplePointIndexes(self):
        """
        Get a node ID at triple points (special four corners) of the solid core.
        :return: A list of circular (n1) indexes used to identify triple points.
        """
        elementsCountAround = self._elementsCountAround
        nodesCountAcrossMinorHalf = self.getCoreBoxMinorNodesCount() // 2
        triplePointIndexesList = []

        for n in range(0, elementsCountAround, elementsCountAround // 2):
            triplePointIndexesList.append(n + nodesCountAcrossMinorHalf)
            triplePointIndexesList.append((n - nodesCountAcrossMinorHalf) % elementsCountAround)

        return triplePointIndexesList

    def getTriplePointLocation(self, e1):
        """
        Determines the location of a specific triple point relative to the solid core box.
        There are four locations: Top left (location = 1); top right (location = -1); bottom left (location = 2);
        and bottom right (location = -2). Location is None if not located at any of the four specified locations.
        :return: Location identifier.
        """
        en = self._elementsCountCoreBoxMinor // 2
        em = self._elementsCountCoreBoxMajor // 2
        ec = self._elementsCountAround // 4

        lftColumnElements = list(range(0, ec - em)) + list(range(3 * ec + em, self._elementsCountAround))
        topRowElements = list(range(ec - em, ec + em))
        rhtColumnElements = list((range(2 * ec - en, 2 * ec + en)))
        btmRowElements = list(range(3 * ec - em, 3 * ec + em))

        idx = len(lftColumnElements) // 2
        if e1 == topRowElements[0] or e1 == lftColumnElements[idx - 1]:
            location = 1  # "TopLeft"
        elif e1 == topRowElements[-1] or e1 == rhtColumnElements[0]:
            location = -1  # "TopRight"
        elif e1 == btmRowElements[-1] or e1 == lftColumnElements[idx]:
            location = 2  # "BottomLeft"
        elif e1 == btmRowElements[0] or e1 == rhtColumnElements[-1]:
            location = -2  # "BottomRight"
        else:
            location = 0

        return location

    def getRimCoordinates(self, n1, n2, n3):
        """
        Get rim parameters (transition through shell) parameters at a point.
        This was what rim coordinates should have been.
        :param n1: Node index around.
        :param n2: Node index along segment.
        :param n3: Node index from first core transition row or inner to outer shell.
        :return: x, d1, d2, d3
        """
        transitionNodeCount = (len(self._transitionCoordinates[0][0])
                               if (self._transitionCoordinates and self._transitionCoordinates[0]) else 0)
        if n3 < transitionNodeCount:
            return (self._transitionCoordinates[0][n2][n3][n1],
                    self._transitionCoordinates[1][n2][n3][n1],
                    self._transitionCoordinates[2][n2][n3][n1],
                    self._transitionCoordinates[3][n2][n3][n1] if self._transitionCoordinates[3] else None)
        sn3 = n3 - transitionNodeCount
        return (self._rimCoordinates[0][n2][sn3][n1],
                self._rimCoordinates[1][n2][sn3][n1],
                self._rimCoordinates[2][n2][sn3][n1],
                self._rimCoordinates[3][n2][sn3][n1] if self._rimCoordinates[3] else None)

    def getRimNodeId(self, n1, n2, n3):
        """
        Get a rim node ID for a point.
        :param n1: Node index around.
        :param n2: Node index along segment.
        :param n3: Node index from inner to outer rim.
        :return: Node identifier.
        """
        return self._rimNodeIds[n2][n3][n1]

    def getRimElementId(self, e1, e2, e3):
        """
        Get a rim element ID.
        :param e1: Element index around.
        :param e2: Element index along segment.
        :param e3: Element index from inner to outer rim.
        :return: Element identifier.
        """
        return self._rimElementIds[e2][e3][e1]

    def setRimElementId(self, e1, e2, e3, elementIdentifier):
        """
        Set a rim element ID. Only called by adjacent junctions.
        :param e1: Element index around.
        :param e2: Element index along segment.
        :param e3: Element index from inner to outer rim.
        :param elementIdentifier: Element identifier.
        """
        if not self._rimElementIds[e2]:
            elementsCountRim = self.getElementsCountRim()
            self._rimElementIds[e2] = [[None] * self._elementsCountAround for _ in range(elementsCountRim)]
        self._rimElementIds[e2][e3][e1] = elementIdentifier

    def getBoxElementId(self, e1, e2, e3):
        """
        Get a box element ID.
        :param e1: Element index across core box major / d2 direction.
        :param e2: Element index along segment.
        :param e3: Element index across core box minor / d3 direction.
        :return: Element identifier.
        """
        return self._boxElementIds[e2][e3][e1]

    def setBoxElementId(self, e1, e2, e3, elementIdentifier):
        """
        Set a box element ID. Only called by adjacent junctions.
        :param e1: Element index across core box major / d2 direction.
        :param e2: Element index along segment.
        :param e3: Element index across core box minor / d3 direction.
        :param elementIdentifier: Element identifier.
        """
        if not self._boxElementIds[e2]:
            self._boxElementIds[e2] = [
                [None] * self._elementsCountCoreBoxMinor for _ in range(self._elementsCountCoreBoxMajor)]
        self._boxElementIds[e2][e3][e1] = elementIdentifier

    def _addBoxElementsToMeshGroup(self, e1Start, e1Limit, e3Start, e3Limit, meshGroup):
        """
        Add ranges of box elements to mesh group.
        :param e1Start: Start element index in major / d2 direction.
        :param e1Limit: Limit element index in major / d2 direction.
        :param e3Start: Start element index in minor / d3 direction.
        :param e3Limit: Limit element index in minor / d3 direction.
        :param meshGroup: Zinc MeshGroup to add elements to.
        """
        # print("Add box elements", e1Start, e1Limit, e3Start, e3Limit, meshGroup.getName())
        elementsCountAlong = self.getSampledElementsCountAlong()
        mesh = meshGroup.getMasterMesh()
        for e2 in range(elementsCountAlong):
            boxSlice = self._boxElementIds[e2]
            if boxSlice:
                # print(boxSlice[e1Start:e1Limit])
                for elementIdentifiersList in boxSlice[e1Start:e1Limit]:
                    for elementIdentifier in elementIdentifiersList[e3Start:e3Limit]:
                        element = mesh.findElementByIdentifier(elementIdentifier)
                        meshGroup.addElement(element)

    def _addRimElementsToMeshGroup(self, e1Start, e1Limit, e3Start, e3Limit, meshGroup):
        """
        Add ranges of rim elements to mesh group.
        :param e1Start: Start element index around. Can be negative which supports wrapping.
        :param e1Limit: Limit element index around.
        :param e3Start: Start element index rim.
        :param e3Limit: Limit element index rim.
        :param meshGroup: Zinc MeshGroup to add elements to.
        """
        # print("Add rim elements", e1Start, e1Limit, e3Start, e3Limit, meshGroup.getName())
        elementsCountAlong = self.getSampledElementsCountAlong()
        mesh = meshGroup.getMasterMesh()
        for e2 in range(elementsCountAlong):
            rimSlice = self._rimElementIds[e2]
            if rimSlice:
                for elementIdentifiersList in rimSlice[e3Start:e3Limit]:
                    partElementIdentifiersList = elementIdentifiersList[e1Start:e1Limit] if (e1Start >= 0) else (
                            elementIdentifiersList[e1Start:] + elementIdentifiersList[:e1Limit])
                    if None in elementIdentifiersList:
                        break
                    for elementIdentifier in partElementIdentifiersList:
                        element = mesh.findElementByIdentifier(elementIdentifier)
                        meshGroup.addElement(element)

    def addCoreElementsToMeshGroup(self, meshGroup):
        """
        Ensure all core elements in core box or rim arrays are in mesh group.
        :param meshGroup: Zinc MeshGroup to add elements to.
        """
        if not self._isCore:
            return
        self._addBoxElementsToMeshGroup(0, self._elementsCountCoreBoxMajor,
                                        0, self._elementsCountCoreBoxMinor, meshGroup)
        self._addRimElementsToMeshGroup(0, self._elementsCountAround,
                                        0, self._elementsCountTransition, meshGroup)

    def addShellElementsToMeshGroup(self, meshGroup):
        """
        Add elements in the shell to mesh group.
        :param meshGroup: Zinc MeshGroup to add elements to.
        """
        elementsCountRim = self.getElementsCountRim()
        elementsCountShell = self._elementsCountThroughShell
        e3ShellStart = elementsCountRim - elementsCountShell
        self._addRimElementsToMeshGroup(0, self._elementsCountAround, e3ShellStart, elementsCountRim, meshGroup)

    def addAllElementsToMeshGroup(self, meshGroup):
        """
        Add all elements in the segment to mesh group.
        :param meshGroup: Zinc MeshGroup to add elements to.
        """
        self.addCoreElementsToMeshGroup(meshGroup)
        self.addShellElementsToMeshGroup(meshGroup)

    def addSideD2ElementsToMeshGroup(self, side: bool, meshGroup):
        """
        Add elements to the mesh group on side of +d2 or -d2, often matching left and right.
        Only works with even numbers around and phase starting at +d2.
        :param side: False for +d2 direction, True for -d2 direction.
        :param meshGroup: Zinc MeshGroup to add elements to.
        """
        if self._isCore:
            e1Start = (self._elementsCountCoreBoxMajor // 2) if side else 0
            e1Limit = self._elementsCountCoreBoxMajor if side else ((self._elementsCountCoreBoxMajor + 1) // 2)
            self._addBoxElementsToMeshGroup(e1Start, e1Limit, 0, self._elementsCountCoreBoxMinor, meshGroup)
        e1Start = (self._elementsCountAround // 4) if side else -((self._elementsCountAround + 2) // 4)
        e1Limit = e1Start + (self._elementsCountAround // 2)
        if (self._elementsCountAround % 4) == 2:
            e1Limit += 1
        self._addRimElementsToMeshGroup(e1Start, e1Limit, 0, self.getElementsCountRim(), meshGroup)

    def addSideD3ElementsToMeshGroup(self, side: bool, meshGroup):
        """
        Add elements to the mesh group on side of +d3 or -d3, often matching anterior/ventral and posterior/dorsal.
        Only works with even numbers around and phase starting at +d2.
        :param side: False for +d3 direction, True for -d3 direction.
        :param meshGroup: Zinc MeshGroup to add elements to.
        """
        if self._isCore:
            e3Start = 0 if side else (self._elementsCountCoreBoxMinor // 2)
            e3Limit = ((self._elementsCountCoreBoxMinor + 1) // 2) if side else self._elementsCountCoreBoxMinor
            self._addBoxElementsToMeshGroup(0, self._elementsCountCoreBoxMajor, e3Start, e3Limit, meshGroup)
        e1Start = (self._elementsCountAround // 2) if side else 0
        e1Limit = e1Start + (self._elementsCountAround // 2)
        self._addRimElementsToMeshGroup(e1Start, e1Limit, 0, self.getElementsCountRim(), meshGroup)

    def getRimNodeIdsSlice(self, n2):
        """
        Get slice of rim node IDs.
        :param n2: Node index along segment, including negative indexes from end.
        :return: Node IDs arrays through rim and around, or None if not set.
        """
        return self._rimNodeIds[n2]

    def getRimIndexNodeLayoutSpecial(self, generateData, rimIndex):
        return None

    def generateMesh(self, generateData: TubeNetworkMeshGenerateData, n2Only=None):
        """
        :param n2Only: If set, create nodes only for that single n2 index along. Must be >= 0!
        """
        # keeping this code to enable display of raw segment trim surfaces for future diagnostics
        # if (not n2Only) and generateData.isShowTrimSurfaces():
        #     dimension = generateData.getMeshDimension()
        #     nodeIdentifier, elementIdentifier = generateData.getNodeElementIdentifiers()
        #     faceIdentifier = elementIdentifier if (dimension == 2) else None
        #     annotationGroup = generateData.getNewTrimAnnotationGroup()
        #     nodeIdentifier, faceIdentifier = \
        #         self._rawTrackSurfaceList[0].generateMesh(generateData.getRegion(), nodeIdentifier, faceIdentifier,
        #                                               group_name=annotationGroup.getName())
        #     if dimension == 2:
        #         elementIdentifier = faceIdentifier
        #     generateData.setNodeElementIdentifiers(nodeIdentifier, elementIdentifier)

        elementsCountAlong = len(self._rimCoordinates[0]) - 1
        elementsCountRim = self.getElementsCountRim()
        elementsCountTransition = self.getElementsCountTransition()
        coordinates = generateData.getCoordinates()
        fieldcache = generateData.getFieldcache()
        startSkipCount = 1 if (self._junctions[0].getSegmentsCount() > 2) else 0
        endSkipCount = 1 if (self._junctions[1].getSegmentsCount() > 2) else 0

        # create nodes
        nodes = generateData.getNodes()
        isLinearThroughShell = generateData.isLinearThroughShell()
        nodetemplate = generateData.getNodetemplate()
        for n2 in range(elementsCountAlong + 1) if (n2Only is None) else [n2Only]:
            if (n2 < startSkipCount) or (n2 > elementsCountAlong - endSkipCount):
                if self._isCore:
                    self._boxNodeIds[n2] = None
                self._rimNodeIds[n2] = None
                continue
            if self._isCore:
                if self._rimNodeIds[n2] and self._boxNodeIds[n2]:
                    continue
            else:
                if self._rimNodeIds[n2]:
                    continue
            # get shared nodes from single adjacent segment, including loop on itself
            # only handles one in, one out
            if n2 == 0:
                if self._junctions[0].getSegmentsCount() == 2:
                    segments = self._junctions[0].getSegments()
                    if self._isCore:
                        boxNodeIds = segments[0].getBoxNodeIdsSlice(-1)
                        if boxNodeIds:
                            self._boxNodeIds[n2] = boxNodeIds
                    rimNodeIds = segments[0].getRimNodeIdsSlice(-1)
                    if rimNodeIds:
                        self._rimNodeIds[n2] = rimNodeIds
                        continue
            if n2 == elementsCountAlong:
                if self._junctions[1].getSegmentsCount() == 2:
                    segments = self._junctions[1].getSegments()
                    if self._isCore:
                        boxNodeIds = segments[1].getBoxNodeIdsSlice(0)
                        if boxNodeIds:
                            self._boxNodeIds[n2] = boxNodeIds
                    rimNodeIds = segments[1].getRimNodeIdsSlice(0)
                    if rimNodeIds:
                        self._rimNodeIds[n2] = rimNodeIds
                        continue

            # create core box nodes
            if self._boxCoordinates:
                self._boxNodeIds[n2] = [] if self._boxNodeIds[n2] is None else self._boxNodeIds[n2]
                coreBoxMajorNodesCount = self.getCoreBoxMajorNodesCount()
                coreBoxMinorNodesCount = self.getCoreBoxMinorNodesCount()
                for n3 in range(coreBoxMajorNodesCount):
                    self._boxNodeIds[n2].append([])
                    rx = self._boxCoordinates[0][n2][n3]
                    rd1 = self._boxCoordinates[1][n2][n3]
                    rd2 = self._boxCoordinates[2][n2][n3]
                    rd3 = self._boxCoordinates[3][n2][n3]
                    for n1 in range(coreBoxMinorNodesCount):
                        nodeIdentifier = generateData.nextNodeIdentifier()
                        node = nodes.createNode(nodeIdentifier, nodetemplate)
                        fieldcache.setNode(node)
                        for nodeValue, rValue in zip([Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                                                      Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3],
                                                     [rx[n1], rd1[n1], rd2[n1], rd3[n1]]):
                            coordinates.setNodeParameters(fieldcache, -1, nodeValue, 1, rValue)
                        self._boxNodeIds[n2][n3].append(nodeIdentifier)

            # create rim nodes and transition nodes (if there are more than 1 layer of transition)
            self._rimNodeIds[n2] = [] if self._rimNodeIds[n2] is None else self._rimNodeIds[n2]
            nodesCountRim = self.getNodesCountRim()
            for n3 in range(nodesCountRim):
                n3p = n3 - (elementsCountTransition - 1) if self._isCore else n3
                if self._isCore and elementsCountTransition > 1 and n3 < (elementsCountTransition - 1):
                    # transition coordinates
                    rx = self._transitionCoordinates[0][n2][n3]
                    rd1 = self._transitionCoordinates[1][n2][n3]
                    rd2 = self._transitionCoordinates[2][n2][n3]
                    rd3 = self._transitionCoordinates[3][n2][n3]
                else:
                    # rim coordinates
                    rx = self._rimCoordinates[0][n2][n3p]
                    rd1 = self._rimCoordinates[1][n2][n3p]
                    rd2 = self._rimCoordinates[2][n2][n3p]
                    rd3 = None if isLinearThroughShell else self._rimCoordinates[3][n2][n3p]
                ringNodeIds = []
                for n1 in range(self._elementsCountAround):
                    nodeIdentifier = generateData.nextNodeIdentifier()
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                    fieldcache.setNode(node)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, rx[n1])
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, rd1[n1])
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, rd2[n1])
                    if rd3:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, rd3[n1])
                    ringNodeIds.append(nodeIdentifier)
                self._rimNodeIds[n2].append(ringNodeIds)

        # create a new list containing box node ids are located at the boundary
        if self._isCore:
            self._boxBoundaryNodeIds, self._boxBoundaryNodeToBoxId = (
                self._createBoxBoundaryNodeIdsList(startSkipCount, endSkipCount))

        if n2Only is not None:
            return

        # create elements
        annotationMeshGroups = generateData.getAnnotationMeshGroups(self._annotationTerms)
        mesh = generateData.getMesh()
        elementtemplateStd, eftStd = generateData.getStandardElementtemplate()
        for e2 in range(startSkipCount, elementsCountAlong - endSkipCount):
            self._boxElementIds[e2] = []
            self._rimElementIds[e2] = []
            e2p = e2 + 1
            if self._isCore:
                # create box elements
                elementsCountAcrossMinor = self.getCoreBoxMinorNodesCount() - 1
                elementsCountAcrossMajor = self.getCoreBoxMajorNodesCount() - 1
                for e3 in range(elementsCountAcrossMajor):
                    e3p = e3 + 1
                    elementIds = []
                    for e1 in range(elementsCountAcrossMinor):
                        nids = []
                        for n1 in [e1, e1 + 1]:
                            nids += [self._boxNodeIds[e2][e3][n1], self._boxNodeIds[e2][e3p][n1],
                                     self._boxNodeIds[e2p][e3][n1], self._boxNodeIds[e2p][e3p][n1]]
                        elementIdentifier = generateData.nextElementIdentifier()
                        element = mesh.createElement(elementIdentifier, elementtemplateStd)
                        element.setNodesByIdentifier(eftStd, nids)
                        for annotationMeshGroup in annotationMeshGroups:
                            annotationMeshGroup.addElement(element)
                        elementIds.append(elementIdentifier)
                    self._boxElementIds[e2].append(elementIds)

                # create core transition elements first layer after box
                triplePointIndexesList = self.getTriplePointIndexes()
                ringElementIds = []
                for e1 in range(self._elementsCountAround):
                    nids, nodeParameters, nodeLayouts = [], [], []
                    n1p = (e1 + 1) % self._elementsCountAround
                    location = self.getTriplePointLocation(e1)
                    nodeLayoutTransition = generateData.getNodeLayoutTransition()
                    nodeLayoutTransitionTriplePoint = generateData.getNodeLayoutTransitionTriplePoint(location)
                    for n2 in [e2, e2 + 1]:
                        for n1 in [e1, n1p]:
                            nids += [self._boxBoundaryNodeIds[n2][n1]]
                            n3c, n1c = self._boxBoundaryNodeToBoxId[n2][n1]
                            nodeParameters.append(self.getBoxCoordinates(n1c, n2, n3c))
                            nodeLayouts.append(nodeLayoutTransitionTriplePoint if n1 in triplePointIndexesList else
                                               nodeLayoutTransition)
                    for n2 in [e2, e2 + 1]:
                        for n1 in [e1, n1p]:
                            nids += [self._rimNodeIds[n2][0][n1]]
                            nodeParameters.append(self.getRimCoordinates(n1, n2, 0))
                            nodeLayouts.append(None)
                    eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                    if self._elementsCountTransition == 1:
                        eft, scalefactors = generateData.resolveEftCoreBoundaryScaling(
                            eft, scalefactors, nodeParameters, nids, self._coreBoundaryScalingMode)
                    elementtemplate = mesh.createElementtemplate()
                    elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
                    elementtemplate.defineField(coordinates, -1, eft)
                    elementIdentifier = generateData.nextElementIdentifier()
                    element = mesh.createElement(elementIdentifier, elementtemplate)
                    element.setNodesByIdentifier(eft, nids)
                    if scalefactors:
                        element.setScaleFactors(eft, scalefactors)
                    for annotationMeshGroup in annotationMeshGroups:
                        annotationMeshGroup.addElement(element)
                    ringElementIds.append(elementIdentifier)
                self._rimElementIds[e2].append(ringElementIds)

            # create regular rim elements - all elements outside first transition layer
            elementsCountRimRegular = elementsCountRim - 1 if self._isCore else elementsCountRim
            for e3 in range(elementsCountRimRegular):
                ringElementIds = []
                lastTransition = self._isCore and (e3 == (self._elementsCountTransition - 2))
                for e1 in range(self._elementsCountAround):
                    elementtemplate = elementtemplateStd
                    eft = eftStd
                    n1p = (e1 + 1) % self._elementsCountAround
                    nids = []
                    for n3 in [e3, e3 + 1] if (self._dimension == 3) else [0]:
                        nids += [self._rimNodeIds[e2][n3][e1], self._rimNodeIds[e2][n3][n1p],
                                 self._rimNodeIds[e2p][n3][e1], self._rimNodeIds[e2p][n3][n1p]]
                    elementIdentifier = generateData.nextElementIdentifier()
                    scalefactors = []
                    if lastTransition:
                        # get node parameters for computing scale factors
                        nodeParameters = []
                        for n3 in (e3, e3 + 1):
                            for n2 in (e2, e2 + 1):
                                for n1 in (e1, n1p):
                                    nodeParameters.append(self.getRimCoordinates(n1, n2, n3))
                        eft = generateData.createElementfieldtemplate()
                        eft, scalefactors = generateData.resolveEftCoreBoundaryScaling(
                            eft, scalefactors, nodeParameters, nids, self._coreBoundaryScalingMode)
                        elementtemplateTransition = mesh.createElementtemplate()
                        elementtemplateTransition.setElementShapeType(Element.SHAPE_TYPE_CUBE)
                        elementtemplateTransition.defineField(coordinates, -1, eft)
                        elementtemplate = elementtemplateTransition
                    element = mesh.createElement(elementIdentifier, elementtemplate)
                    element.setNodesByIdentifier(eft, nids)
                    if scalefactors:
                        element.setScaleFactors(eft, scalefactors)
                    for annotationMeshGroup in annotationMeshGroups:
                        annotationMeshGroup.addElement(element)
                    ringElementIds.append(elementIdentifier)
                self._rimElementIds[e2].append(ringElementIds)

    def generateJunctionRimElements(self, junction, generateData):
        """
        Generates rim elements for junction part of the segment after segment nodes and elements have been made.
        """
        mesh = generateData.getMesh()
        meshDimension = generateData.getMeshDimension()
        elementtemplate = mesh.createElementtemplate()
        elementtemplate.setElementShapeType(
            Element.SHAPE_TYPE_CUBE if (meshDimension == 3) else Element.SHAPE_TYPE_SQUARE)
        d3Defined = (meshDimension == 3) and not generateData.isLinearThroughShell()
        coordinates = generateData.getCoordinates()

        elementsCountRim = self.getElementsCountRim()
        elementsCountAround = self.getElementsCountAround()
        elementsCountTransition = self.getElementsCountTransition()

        elementsCountAlong = self.getSampledElementsCountAlong()
        s = junction._segments.index(self)
        e2 = (elementsCountAlong - 1) if junction._segmentsIn[s] else 0
        n2 = (elementsCountAlong - 1) if junction._segmentsIn[s] else 1

        # Create elements
        if junction._isCore:
            boxBoundaryNodeIds, boxBoundaryNodeToBoxId = junction._createBoxBoundaryNodeIdsList(s)
            # create box elements
            junction._generateBoxElements(s, n2, mesh, elementtemplate, coordinates, self, generateData)
            # create core transition elements
            junction._generateTransitionElements(s, n2, mesh, elementtemplate, coordinates, self, generateData,
                elementsCountAround, boxBoundaryNodeIds, boxBoundaryNodeToBoxId)

        # create regular rim elements
        elementsCountRimRegular = elementsCountRim - 1 if self._isCore else elementsCountRim
        annotationMeshGroups = generateData.getAnnotationMeshGroups(self.getAnnotationTerms())
        eftList = [None] * elementsCountAround
        scalefactorsList = [None] * elementsCountAround

        for e3 in range(elementsCountRimRegular):
            rim_e3 = e3 + 1 if junction._isCore else e3
            for e1 in range(elementsCountAround):
                n1p = (e1 + 1) % elementsCountAround
                nids = []
                nodeParameters = []
                nodeLayouts = []
                elementIdentifier = generateData.nextElementIdentifier()
                lastTransition = junction._isCore and (elementsCountTransition == (rim_e3 + 1))
                needParameters = (e3 == 0) or lastTransition
                for n3 in [e3, e3 + 1] if (meshDimension == 3) else [e3]:
                    for n1 in [e1, n1p]:
                        nids.append(self.getRimNodeId(n1, n2, n3))
                        if needParameters:
                            rimCoordinates = self.getRimCoordinates(n1, n2, n3)
                            nodeParameters.append(rimCoordinates if d3Defined else (
                                rimCoordinates[0], rimCoordinates[1], rimCoordinates[2], None))
                            nodeLayouts.append(None)
                    for n1 in [e1, n1p]:
                        rimIndex = junction._segmentNodeToRimIndex[s][n1]
                        nids.append(junction._rimNodeIds[n3][rimIndex])
                        if needParameters:
                            nodeParameters.append(
                                (junction._rimCoordinates[0][n3][rimIndex],
                                 junction._rimCoordinates[1][n3][rimIndex],
                                 junction._rimCoordinates[2][n3][rimIndex],
                                 junction._rimCoordinates[3][n3][rimIndex] if d3Defined else None))
                            nodeLayouts.append(junction.getRimIndexNodeLayout(generateData, rimIndex))
                    if not junction._segmentsIn[s]:
                        for a in [nids, nodeParameters, nodeLayouts] if (needParameters) else [nids]:
                            a[-4], a[-2] = a[-2], a[-4]
                            a[-3], a[-1] = a[-1], a[-3]

                # exploit efts being same through the rim
                eft = eftList[e1]
                scalefactors = scalefactorsList[e1]
                if not eft:
                    eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                    eftList[e1] = eft
                    scalefactorsList[e1] = scalefactors
                if lastTransition:
                    # need to generate eft again otherwise modifying object in eftList, mucking up outer layers
                    eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                    eft, scalefactors = generateData.resolveEftCoreBoundaryScaling(
                        eft, scalefactors, nodeParameters, nids, self.getCoreBoundaryScalingMode())

                elementtemplate.defineField(coordinates, -1, eft)
                element = mesh.createElement(elementIdentifier, elementtemplate)
                element.setNodesByIdentifier(eft, nids)
                if scalefactors:
                    element.setScaleFactors(eft, scalefactors)
                self.setRimElementId(e1, e2, rim_e3, elementIdentifier)
                for annotationMeshGroup in annotationMeshGroups:
                    annotationMeshGroup.addElement(element)


class PatchTubeNetworkMeshSegment(TubeNetworkMeshSegment):
    """
    New class derived from TubeNetworkMeshSegment, used for making a patch to cover a segment opening. Only work for
    inlet segment with a junction on the other end. Note that outlet segment and core cases are not implemented yet.
    """

    def __init__(self, networkSegment, pathParametersList, elementsCountAround, elementsCountThroughShell,
                 isCore=False, elementsCountCoreBoxMinor: int=2, elementsCountTransition: int=1,
                 coreBoundaryScalingMode: int=1):
        super(PatchTubeNetworkMeshSegment, self).__init__(
            networkSegment, pathParametersList, elementsCountAround, elementsCountThroughShell,
            isCore, elementsCountCoreBoxMinor, elementsCountTransition, coreBoundaryScalingMode)

        self._patchCoordinates = None
        self._patchRimNodeIds = None
        self._patchElementIds = None

    def sample(self, fixedElementsCountAlong, targetElementLength):
        """
        Samples coordinates along (dorsal/ventral) and around (left/right) patch. Geometry of the patch is derived from
        a combined track surface of the tube segments that are coming in tangentially. The trim surface from the inlet
        segment that is entering normally to the first two segments is used to define the boundary of the patch.
        """
        outer = 0
        inner = 1

        junction = self._junctions[1]
        connectedSegments = junction.getSegments()

        # ML: hard code for uterus layout
        segment1 = connectedSegments[0]  # left tube
        segment2 = connectedSegments[1]  # right tube
        segment3 = connectedSegments[2]  # patch tube
        elementsCountAroundSegmentOut = segment3.getElementsCountAround()

        sampleElementCount = 20

        # make a combined track surface from the surfaces of segments 1 and 2
        sxAlongPatchAllLayers = []
        sd1AlongPatchAllLayers = []
        sd2AlongPatchAllLayers = []

        for layer in (outer, inner):
            xCombinedTrackSurface = []
            d1CombinedTrackSurface = []
            d2CombinedTrackSurface = []
            trimSurfaces1 = junction.getTrimSurfaces(segment1)[layer]
            trimSurfaces2 = junction.getTrimSurfaces(segment2)[layer]
            trimSurfaces3 = junction.getTrimSurfaces(segment3)[layer]

            xAlongCombinedTrackSurfaceMid = []
            d1AlongCombinedTrackSurfaceMid = []

            for s in (1, 0):
                segment = connectedSegments[s]
                elementsCountAroundSegmentIn = segment.getElementsCountAround()
                halfElementsCountAroundSegmentIn = elementsCountAroundSegmentIn // 2
                rawTrackSurface = segment.getRawTrackSurface(layer)
                xRawTrackSurface = rawTrackSurface._nx
                d1RawTrackSurface = rawTrackSurface._nd1
                d2RawTrackSurface = rawTrackSurface._nd2

                elementsCountAlongSegmentIn = int(len(xRawTrackSurface) / elementsCountAroundSegmentIn - 1)
                # Extract coordinates from top half of the inlet tubes
                if s:  # ML: we know that segment 2 is in desired direction
                    for i in range(elementsCountAlongSegmentIn + 1):
                        baseCount = i * elementsCountAroundSegmentIn
                        xCombinedTrackSurface += \
                            xRawTrackSurface[baseCount + elementsCountAroundSegmentIn // 4 - 1:
                                             baseCount + 3 * elementsCountAroundSegmentIn // 4 + 2]
                        d1CombinedTrackSurface += \
                            d1RawTrackSurface[baseCount + elementsCountAroundSegmentIn // 4 - 1:
                                              baseCount + 3 * elementsCountAroundSegmentIn // 4 + 2]
                        d2CombinedTrackSurface += \
                            d2RawTrackSurface[baseCount + elementsCountAroundSegmentIn // 4 - 1:
                                              baseCount + 3 * elementsCountAroundSegmentIn // 4 + 2]
                        if i == elementsCountAlongSegmentIn:
                            xAlongCombinedTrackSurfaceMid += \
                                xRawTrackSurface[baseCount + elementsCountAroundSegmentIn // 4 - 1:
                                                 baseCount + 3 * elementsCountAroundSegmentIn // 4 + 2]
                            d1AlongCombinedTrackSurfaceMid += \
                                d1RawTrackSurface[baseCount + elementsCountAroundSegmentIn // 4 - 1:
                                                 baseCount + 3 * elementsCountAroundSegmentIn // 4 + 2]

                else:  # ML: we know that segment 1 is in opposite direction
                    xAlongAround = []
                    d1AlongAround = []
                    d2AlongAround = []
                    for i in range(elementsCountAlongSegmentIn + 1):
                        baseCount = i * elementsCountAroundSegmentIn
                        xAround = []
                        d1Around = []
                        d2Around = []

                        if i == 0:
                            xAround += xRawTrackSurface[elementsCountAroundSegmentIn // 4 + 1::-1]
                            d1Around += d1RawTrackSurface[elementsCountAroundSegmentIn // 4 + 1::-1]
                            d2Around += d2RawTrackSurface[elementsCountAroundSegmentIn // 4 + 1::-1]
                        else:
                            xAround += \
                                xRawTrackSurface[baseCount + elementsCountAroundSegmentIn // 4 + 1: baseCount - 1: -1]
                            d1Around += \
                                d1RawTrackSurface[baseCount + elementsCountAroundSegmentIn // 4 + 1: baseCount - 1: -1]
                            d2Around += \
                                d2RawTrackSurface[baseCount + elementsCountAroundSegmentIn // 4 + 1: baseCount - 1: -1]

                        xAround += \
                            xRawTrackSurface[baseCount + elementsCountAroundSegmentIn - 1:
                                             baseCount + 3 * elementsCountAroundSegmentIn // 4 - 2: -1]
                        d1Around += \
                            d1RawTrackSurface[baseCount + elementsCountAroundSegmentIn - 1:
                                              baseCount + 3 * elementsCountAroundSegmentIn // 4 - 2: -1]
                        d2Around += \
                            d2RawTrackSurface[baseCount + elementsCountAroundSegmentIn - 1:
                                              baseCount + 3 * elementsCountAroundSegmentIn // 4 - 2: -1]

                        xAlongAround.append(xAround)
                        d1Around = [mult(c, -1) for c in d1Around]
                        d2Around = [mult(c, -1) for c in d2Around]
                        d1AlongAround.append(d1Around)
                        d2AlongAround.append(d2Around)

                    for i in range(len(xAlongAround) - 2, -1, -1):
                        xCombinedTrackSurface += xAlongAround[i]
                        d1CombinedTrackSurface += d1AlongAround[i]
                        d2CombinedTrackSurface += d2AlongAround[i]

            combinedTrackSurface = TrackSurface(halfElementsCountAroundSegmentIn + 2,
                                                elementsCountAlongSegmentIn * 2,
                                                xCombinedTrackSurface, d1CombinedTrackSurface, d2CombinedTrackSurface,
                                                loop1=False)

            # find intersection between trim surfaces and combined track surface
            # Trim surface 3
            xCurveBetweenTrackAndTrim3, d1CurveBetweenTrackAndTrim3, cProportions, loop = \
                trimSurfaces3.findIntersectionCurve(combinedTrackSurface, curveElementsCount=sampleElementCount)

            # Use curve between track+trim 3 and middle of combined track to find starting and ending pt of mid-plane of
            # combined track surface
            startLocationFirstHalf = (0, 0.0)
            nLocation, oLocation = getNearestLocationBetweenCurves(
                xAlongCombinedTrackSurfaceMid, d1AlongCombinedTrackSurfaceMid,
                xCurveBetweenTrackAndTrim3, d1CurveBetweenTrackAndTrim3, nLoop=False,
                oLoop=True, startLocation=startLocationFirstHalf)[0:2]

            xA = evaluateCoordinatesOnCurve(xAlongCombinedTrackSurfaceMid, d1AlongCombinedTrackSurfaceMid,
                                            nLocation, False, derivative=False)

            startLocationSecondHalf = (len(xAlongCombinedTrackSurfaceMid) // 2 + 1, 0.0)
            nLocation, oLocation = getNearestLocationBetweenCurves(
                xAlongCombinedTrackSurfaceMid, d1AlongCombinedTrackSurfaceMid,
                xCurveBetweenTrackAndTrim3, d1CurveBetweenTrackAndTrim3, nLoop=False,
                oLoop=True, startLocation=startLocationSecondHalf)[0:2]

            xB = evaluateCoordinatesOnCurve(xAlongCombinedTrackSurfaceMid, d1AlongCombinedTrackSurfaceMid,
                                            nLocation, False, derivative=False)

            # Sample points along mid-plane of combined track surface
            positionA = combinedTrackSurface.findNearestPosition(xA)
            proportionA = combinedTrackSurface.getProportion(positionA)

            positionB = combinedTrackSurface.findNearestPosition(xB)
            proportionB = combinedTrackSurface.getProportion(positionB)

            startProportion = proportionA if proportionA[0] < proportionB[0] else proportionB
            endProportion = proportionB if proportionA[0] < proportionB[0] else proportionA

            sxMidPlane, sd1MidPlane, sd2MidPlane, sd3MidPlane, sProportionsMidPlane = \
                combinedTrackSurface.createHermiteCurvePoints(startProportion[0], startProportion[1],
                                                              endProportion[0], endProportion[1],
                                                              halfElementsCountAroundSegmentIn)

            sxAlongTubeBothSides = []
            sxAlongPatchInLayer = []
            sd1AlongPatchInLayer = []
            sd2AlongPatchInLayer = []

            for s in (1, 0):
                # Inlet trim surface
                inletTrimSurfaces = trimSurfaces2 if s else trimSurfaces1
                startPosition = inletTrimSurfaces.createPositionProportion(0.0, 0.0)
                xCurveBetweenTrackAndTrim1, d1CurveBetweenTrackAndTrim1, cProportions, loop = \
                    inletTrimSurfaces.findIntersectionCurve(combinedTrackSurface, startPosition,
                                                            curveElementsCount=sampleElementCount)

                # Search for intersection pt on first half of curve1
                startLocationFirstHalf = (0, 0.0)
                location1, otherLocation = \
                    getNearestLocationBetweenCurves(xCurveBetweenTrackAndTrim1, d1CurveBetweenTrackAndTrim1,
                                                    xCurveBetweenTrackAndTrim3, d1CurveBetweenTrackAndTrim3,
                                                    nLoop=False, oLoop=True, startLocation=startLocationFirstHalf)[0:2]

                xCurve1, dCurve1 = evaluateCoordinatesOnCurve(xCurveBetweenTrackAndTrim1, d1CurveBetweenTrackAndTrim1,
                                                              location1, False, derivative=True)

                # Search for intersection pt on second half of curve1
                startLocationSecondHalf = (len(xCurveBetweenTrackAndTrim1)-2, 0.0)
                location2, otherLocation = \
                    getNearestLocationBetweenCurves(xCurveBetweenTrackAndTrim1, d1CurveBetweenTrackAndTrim1,
                                                    xCurveBetweenTrackAndTrim3, d1CurveBetweenTrackAndTrim3,
                                                    nLoop=False, oLoop=True,
                                                    startLocation=startLocationSecondHalf)[0:2]
                xCurve2, dCurve2 = evaluateCoordinatesOnCurve(xCurveBetweenTrackAndTrim1, d1CurveBetweenTrackAndTrim1,
                                                              location2, False, derivative=True)

                nx = []
                nd1 = []
                nx.append(xCurve1)
                nd1.append(dCurve1)
                for i in range(location1[0] + 1, location2[0] + 1):
                    nx.append(xCurveBetweenTrackAndTrim1[i])
                    nd1.append(d1CurveBetweenTrackAndTrim1[i])
                nx.append(xCurve2)
                nd1.append(dCurve2)

                sxAlongTubeSide = sampleCubicHermiteCurves(nx, nd1, halfElementsCountAroundSegmentIn)[0]

                # check order
                positionA = combinedTrackSurface.findNearestPosition(sxAlongTubeSide[0])
                proportionA = combinedTrackSurface.getProportion(positionA)
                positionB = combinedTrackSurface.findNearestPosition(sxAlongTubeSide[-1])
                proportionB = combinedTrackSurface.getProportion(positionB)
                if proportionA[0] > proportionB[0]:
                    sxAlongTubeSide.reverse()
                sxAlongTubeBothSides.append(sxAlongTubeSide)

            # Sample across the patch from segment 1 to segment 2 but forcing paths to go through sampled points
            # around mid-plane
            nodesAround = (elementsCountAroundSegmentOut - 2 * halfElementsCountAroundSegmentIn) // 2 + 1
            xEnd = []
            dEnd = []
            for i in range(len(sxAlongTubeSide)):
                sxAlongPatch = []
                sd1AlongPatch = []
                sd2AlongPatch = []
                sd3AlongPatch = []
                for j in range(2):
                    if j == 0:
                        startPosition = combinedTrackSurface.findNearestPosition(sxAlongTubeBothSides[0][i])
                        startProportion = combinedTrackSurface.getProportion(startPosition)
                        startDerivative = None
                        endProportion = sProportionsMidPlane[i]
                        endDerivativeMag = \
                            magnitude(sub(sxMidPlane[i], sxAlongTubeBothSides[0][i]))/ \
                            ((elementsCountAroundSegmentOut - 2 * halfElementsCountAroundSegmentIn) // 4)
                        endDerivative = set_magnitude(sd2MidPlane[i], endDerivativeMag)

                        xEnd.append(sxMidPlane[i])
                        dEnd.append(endDerivative)

                    else:
                        startProportion = sProportionsMidPlane[i]
                        startDerivativeMag = \
                            magnitude(sub(sxAlongTubeBothSides[1][i], sxMidPlane[i])) / \
                            ((elementsCountAroundSegmentOut - 2 * halfElementsCountAroundSegmentIn) // 4)
                        startDerivative = set_magnitude(sd2MidPlane[i], startDerivativeMag)
                        endPosition = combinedTrackSurface.findNearestPosition(sxAlongTubeBothSides[1][i])
                        endProportion = combinedTrackSurface.getProportion(endPosition)
                        endDerivative = None

                    sxAlongPatchSide, sd1AlongPatchSide, sd2AlongPatchSide, sd3AlongPatchSide = \
                        combinedTrackSurface.createHermiteCurvePoints(
                            startProportion[0], startProportion[1], endProportion[0], endProportion[1],
                            ((elementsCountAroundSegmentOut - 2 * halfElementsCountAroundSegmentIn) // 4),
                            derivativeStart=startDerivative, derivativeEnd=endDerivative)[0:4]

                    sxAlongPatch += sxAlongPatchSide if j else sxAlongPatchSide[0:-1]
                    sd1AlongPatch += sd1AlongPatchSide if j else sd1AlongPatchSide[0:-1]
                    sd2AlongPatch += sd2AlongPatchSide if j else sd2AlongPatchSide[0:-1]
                    sd3AlongPatch += sd3AlongPatchSide if j else sd3AlongPatchSide[0:-1]

                sxAlongPatchInLayer.append(sxAlongPatch)
                sd1AlongPatchInLayer.append(sd1AlongPatch)
                sd2AlongPatchInLayer.append(sd2AlongPatch)

            # Smooth d2 and arrange directions
            for n1 in range(nodesAround):
                xAround = []
                d2Around = []
                for n2 in range(halfElementsCountAroundSegmentIn + 1):
                    xAround.append(sxAlongPatchInLayer[n2][n1])
                    d2Around.append(sd2AlongPatchInLayer[n2][n1])
                d2Around = smoothCubicHermiteDerivativesLine(xAround, d2Around)
                for n2 in range(halfElementsCountAroundSegmentIn + 1):
                    if n2 < halfElementsCountAroundSegmentIn // 2 + 1:
                        sd2AlongPatchInLayer[n2][n1] = [-c for c in d2Around[n2]]
                    else:
                        sd2AlongPatchInLayer[n2][n1] = d2Around[n2]
                        sd1AlongPatchInLayer[n2][n1] = [-c for c in sd1AlongPatchInLayer[n2][n1]]

            sxAlongOrdered = []
            sd1AlongOrdered = []
            sd2AlongOrdered = []
            for n2 in range(halfElementsCountAroundSegmentIn + 1):
                sxAlong = sxAlongPatchInLayer[n2]
                sd1Along = sd1AlongPatchInLayer[n2]
                sd2Along = sd2AlongPatchInLayer[n2]
                if n2 > halfElementsCountAroundSegmentIn // 2:
                    sxAlong.reverse()
                    sd1Along.reverse()
                    sd2Along.reverse()
                sxAlongOrdered.append(sxAlong)
                sd1AlongOrdered.append(sd1Along)
                sd2AlongOrdered.append(sd2Along)

            sxAlongPatchAllLayers.append(sxAlongOrdered)
            sd1AlongPatchAllLayers.append(sd1AlongOrdered)
            sd2AlongPatchAllLayers.append(sd2AlongOrdered)

        rx, rd1, rd2, rd3 = [], [], [], []
        shellFactor = 1.0 / self._elementsCountThroughShell
        for n2 in range(len(sxAlongPatchAllLayers[0])):
            for r in (rx, rd1, rd2, rd3):
                r.append([])
            otx, otd1, otd2 = sxAlongPatchAllLayers[0][n2], sd1AlongPatchAllLayers[0][n2], sd2AlongPatchAllLayers[0][n2]
            itx, itd1, itd2 = sxAlongPatchAllLayers[1][n2], sd1AlongPatchAllLayers[1][n2], sd2AlongPatchAllLayers[1][n2]
            wd3 = [mult(sub(otx[n1], itx[n1]), shellFactor) for n1 in range(len(otx))]
            for n3 in range(self._elementsCountThroughShell + 1):
                oFactor = n3 / self._elementsCountThroughShell
                iFactor = 1.0 - oFactor
                for r in (rx, rd1, rd2, rd3):
                    r[n2].append([])
                for n1 in range(len(otx)):
                    if n3 == 0:
                        x, d1, d2 = itx[n1], itd1[n1], itd2[n1]
                    elif n3 == self._elementsCountThroughShell:
                        x, d1, d2 = otx[n1], otd1[n1], otd2[n1]
                    else:
                        x = add(mult(itx[n1], iFactor), mult(otx[n1], oFactor))
                        d1 = add(mult(itd1[n1], iFactor), mult(otd1[n1], oFactor))
                        d2 = add(mult(itd2[n1], iFactor), mult(otd2[n1], oFactor))
                    d3 = wd3[n1]
                    for r, value in zip((rx, rd1, rd2, rd3), (x, d1, d2, d3)):
                        r[n2][n3].append(value)
            allCoordinates = rx, rd1, rd2, rd3

        # Extract patch coordinates
        r = copy.deepcopy(allCoordinates)
        for i in range(4):
            r[i].pop(0)
            r[i].pop()
            for n2 in range(len(r[i])):
                for n3 in range(self._elementsCountThroughShell + 1):
                    r[i][n2][n3].pop(0)
                    r[i][n2][n3].pop()
        self._patchCoordinates = r[0], r[1], r[2], r[3]

        self._patchNodeIds = [None] * (halfElementsCountAroundSegmentIn - 1)
        self._patchElementIds = [None] * (halfElementsCountAroundSegmentIn - 2)

        # Create rim coordinates that are structured same way as the segment tube indices [n2][n3][n1], calculating
        # annular indexing of two layers (n2), from inlet to outlet direction with the inner layer corner points being
        # represented 3 times. n3 goes from inner to outer wall. n1 starts from the d2 direction defined by the network
        # layout
        patch = 0
        rim = 1
        self._rimCoordinates = []
        for i in range(4):
            countN2 = len(allCoordinates[i])
            halfCountN2 = len(allCoordinates[i]) // 2
            sParamRing = []
            # "rings" from outermost boundary of patch and rim
            for ring in (patch, rim):
                startIdx = 1 if ring == patch else 0
                endIdx = -2 if ring == patch else -1
                sParamLayer = []
                for n3 in range(self._elementsCountThroughShell + 1):
                    sParamRingAround = []
                    sParam = allCoordinates[i]
                    for n2 in range(halfCountN2, 0, -1):
                        if i == 0 or i == 3:
                            sParamRingAround.append(sParam[n2][n3][startIdx])
                        elif i == 1:  # d1
                            sParamRingAround.append(allCoordinates[2][n2][n3][startIdx])  # becomes d2
                        elif i == 2:  # d2
                            sParamRingAround.append([-c for c in allCoordinates[1][n2][n3][startIdx]])  # becomes -d1
                    # triple points at bottom right
                    if i == 0 or i == 3:
                        sParamRingAround.append(sParam[startIdx][n3][startIdx])
                    elif i == 1:
                        sParamRingAround.append(add(allCoordinates[1][startIdx][n3][startIdx],
                                                    allCoordinates[2][startIdx][n3][startIdx]))  # d1 + d2
                    elif i == 2:
                        sParamRingAround.append(add([-c for c in allCoordinates[1][startIdx][n3][startIdx]],
                                                    allCoordinates[2][startIdx][n3][startIdx]))  # -d1 + d2
                    # straight bottom
                    sParamRingAround += sParam[startIdx][n3][1: -1]
                    # triple pts on bottom left
                    if i == 0 or i == 3:
                        sParamRingAround.append(sParam[startIdx][n3][endIdx])
                    elif i == 1:  # d1
                        sParamRingAround.append(add(allCoordinates[1][startIdx][n3][endIdx],
                                                    [-c for c in allCoordinates[2][startIdx][n3][endIdx]]))  # becomes d1 - d2
                    elif i == 2:  # d2
                        sParamRingAround.append(add(allCoordinates[1][startIdx][n3][endIdx],
                                                    allCoordinates[2][startIdx][n3][endIdx]))  # becomes d1 + d2

                    # Up on left side
                    for n2 in range(1, halfCountN2 + 1):
                        if i == 0 or i == 3:
                            sParamRingAround.append(sParam[n2][n3][endIdx])
                        elif i == 1:  # d1
                            sParamRingAround.append([-c for c in allCoordinates[2][n2][n3][endIdx]])  # becomes -d2
                        elif i == 2:  # d2
                            sParamRingAround.append(allCoordinates[1][n2][n3][endIdx])  # becomes d1
                    for n2 in range(halfCountN2 + 1, countN2 - 1):
                        if i == 0 or i == 3:
                            sParamRingAround.append(sParam[n2][n3][startIdx])
                        elif i == 1:  # d1
                            sParamRingAround.append(allCoordinates[2][n2][n3][startIdx])  # becomes d2
                        elif i == 2:  # d2
                            sParamRingAround.append([-c for c in allCoordinates[1][n2][n3][startIdx]])  # becomes -d1
                    # triple pts on top left
                    if i == 0 or i == 3:
                        sParamRingAround.append(sParam[endIdx][n3][startIdx])
                    elif i == 1:
                        sParamRingAround.append(add(allCoordinates[1][endIdx][n3][startIdx],
                                                    allCoordinates[2][endIdx][n3][startIdx]))  # d1 + d2
                    elif i == 2:
                        sParamRingAround.append(add([-c for c in allCoordinates[1][endIdx][n3][startIdx]],
                                                    allCoordinates[2][endIdx][n3][startIdx]))  # -d1 + d2

                    # straight across top
                    sParamRingAround += sParam[endIdx][n3][1: -1]

                    # Triple points top right
                    if i == 0 or i == 3:
                        sParamRingAround.append(sParam[endIdx][n3][endIdx])
                    elif i == 1:
                        sParamRingAround.append(add(allCoordinates[1][endIdx][n3][endIdx],
                                                    [-c for c in allCoordinates[2][endIdx][n3][endIdx]]))  # d1 - d2
                    elif i == 2:
                        sParamRingAround.append(add(allCoordinates[1][endIdx][n3][endIdx],
                                                    allCoordinates[2][endIdx][n3][endIdx]))  # d1 + d2
                    # down right top half
                    for n2 in range(countN2 - 2, halfCountN2, -1):
                        if i == 0 or i == 3:
                            sParamRingAround.append(sParam[n2][n3][endIdx])
                        elif i == 1:  # d1
                            sParamRingAround.append([-c for c in allCoordinates[2][n2][n3][endIdx]])  # becomes -d2
                        elif i == 2:  # d2
                            sParamRingAround.append(allCoordinates[1][n2][n3][endIdx])  # becomes d1
                    sParamLayer.append(sParamRingAround)
                sParamRing.append(sParamLayer)
            self._rimCoordinates.append(sParamRing)

    def getSampledTubeCoordinatesRing(self, pathIndex, nodeIndexAlong):
        """
        Get a ring of rim coordinates at the supplied node index.
        :param pathIndex: 0 for outer/primary, 1 or -1 for inner/secondary.
        :param nodeIndexAlong: Node index from 0 to self._elementsCountAlong, or negative to count from end.
        :return: sx[nAround]
        """
        pathIndexPatch = 0 if pathIndex else 1
        return self._rimCoordinates[0][nodeIndexAlong][pathIndexPatch]

    def generateMesh(self, generateData: TubeNetworkMeshGenerateData, n2Only=None):
        """
        :param n2Only: Ignored. Always makes whole patch.
        """
        # create nodes
        coordinates = generateData.getCoordinates()
        fieldcache = generateData.getFieldcache()
        nodes = generateData.getNodes()
        isLinearThroughShell = generateData.isLinearThroughShell()
        nodetemplate = generateData.getNodetemplate()

        elementsCountThroughWall = len(self._patchCoordinates[0][0]) - 1
        elementsCountAlong = len(self._patchCoordinates[0]) - 1
        elementsCountAround = len(self._patchCoordinates[0][0][0]) - 1
        for n2 in range(elementsCountAlong + 1):
            self._patchNodeIds[n2] = [] if self._patchNodeIds[n2] is None else self._patchNodeIds[n2]
            for n3 in range(elementsCountThroughWall + 1):
                # patch coordinates
                rx = self._patchCoordinates[0][n2][n3]
                rd1 = self._patchCoordinates[1][n2][n3]
                rd2 = self._patchCoordinates[2][n2][n3]
                rd3 = None if isLinearThroughShell else self._patchCoordinates[3][n2][n3]

                nodeIds = []
                for n1 in range(elementsCountAround + 1):
                    nodeIdentifier = generateData.nextNodeIdentifier()
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                    fieldcache.setNode(node)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, rx[n1])
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, rd1[n1])
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, rd2[n1])
                    if rd3:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, rd3[n1])
                    nodeIds.append(nodeIdentifier)
                self._patchNodeIds[n2].append(nodeIds)

        self._rimNodeIds = []
        nodesAlongPatch = elementsCountAlong + 1
        sNodeId = self._patchNodeIds
        nodesAround = []
        for n3 in range(self._elementsCountThroughShell + 1):
            nodeIdPatch = []
            for n2 in range(nodesAlongPatch // 2, 0, -1):
                nodeIdPatch.append(sNodeId[n2][n3][0])
            nodeIdPatch += [sNodeId[0][n3][0], sNodeId[0][n3][0]]
            nodeIdPatch += sNodeId[0][n3]
            nodeIdPatch += [sNodeId[0][n3][-1], sNodeId[0][n3][-1]]
            for n2 in range(nodesAlongPatch // 2):
                nodeIdPatch.append(sNodeId[n2 + 1][n3][-1])
            for n2 in range(nodesAlongPatch // 2, nodesAlongPatch - 2):
                nodeIdPatch.append(sNodeId[n2 + 1][n3][0])

            nodeIdPatch += [sNodeId[-1][n3][0], sNodeId[-1][n3][0]]
            nodeIdPatch += sNodeId[-1][n3]
            nodeIdPatch += [sNodeId[-1][n3][-1], sNodeId[-1][n3][-1]]
            for n2 in range(nodesAlongPatch - 2, nodesAlongPatch // 2, -1):
                nodeIdPatch.append(sNodeId[n2][n3][-1])
            nodesAround.append(nodeIdPatch)
        self._rimNodeIds.append(nodesAround)
        self._rimNodeIds.append([])

        # create elements
        annotationMeshGroups = generateData.getAnnotationMeshGroups(self._annotationTerms)
        mesh = generateData.getMesh()
        elementtemplateStd, eftStd = generateData.getStandardElementtemplate()
        nodeLayoutFlipD1D2 = generateData.getNodeLayoutFlipD1D2()

        for e2 in range(elementsCountAlong):
            self._patchElementIds[e2] = [] if self._patchElementIds[e2] is None else self._patchElementIds[e2]
            for e3 in range(elementsCountThroughWall):
                patchElementIds = []
                for e1 in range(elementsCountAround):
                    elementtemplate = elementtemplateStd
                    eft = eftStd
                    nids = []
                    if e2 < elementsCountAlong // 2:
                        for n3 in [e3, e3 + 1] if (self._dimension == 3) else [0]:
                            nids += [self._patchNodeIds[e2 + 1][n3][e1],
                                     self._patchNodeIds[e2 + 1][n3][e1 + 1],
                                     self._patchNodeIds[e2][n3][-(e1 + 1) if e2 == elementsCountAlong // 2 else e1],
                                     self._patchNodeIds[e2][n3][-(e1 + 2) if e2 == elementsCountAlong // 2 else e1 + 1]]

                        elementIdentifier = generateData.nextElementIdentifier()
                        element = mesh.createElement(elementIdentifier, elementtemplate)
                        element.setNodesByIdentifier(eft, nids)
                    else:
                        for n3 in [e3, e3 + 1]:
                            nids += [self._patchNodeIds[e2][n3][-(e1 + 1) if e2 == elementsCountAlong // 2 else e1],
                                     self._patchNodeIds[e2][n3][-(e1 + 2) if e2 == elementsCountAlong // 2 else e1 + 1],
                                     self._patchNodeIds[e2 + 1][n3][e1],
                                     self._patchNodeIds[e2 + 1][n3][e1 + 1]]
                        nodeParameters = []
                        nodeLayouts = []
                        for n3 in (e3, e3 + 1):
                            for n2 in (e2, e2 + 1):
                                for n1 in (e1, e1 + 1):
                                    nodeParameters.append(self.getPatchCoordinates(n1, n2, n3, isLinearThroughShell))
                                    nodeLayouts.append(nodeLayoutFlipD1D2 if n2 == elementsCountAlong // 2 else
                                                       None)
                        elementIdentifier = generateData.nextElementIdentifier()
                        eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                        elementtemplate = mesh.createElementtemplate()
                        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
                        elementtemplate.defineField(coordinates, -1, eft)
                        element = mesh.createElement(elementIdentifier, elementtemplate)
                        element.setNodesByIdentifier(eft, nids)
                        if scalefactors:
                            element.setScaleFactors(eft, scalefactors)
                    for annotationMeshGroup in annotationMeshGroups:
                        annotationMeshGroup.addElement(element)
                    patchElementIds.append(elementIdentifier)
                self._patchElementIds[e2].append(patchElementIds)

        self._rimElementIds = [None] * (len(self._patchElementIds) // 2 + 2)
        elementsAlongPatch = elementsCountAlong
        sElementId = self._patchElementIds

        elementsAlongPatchRim = len(self._patchElementIds) // 2  # Number of rings in patch
        elementsAroundPatch = len(self._patchElementIds[0][0])

        if elementsAroundPatch:
            for i in range(elementsAlongPatchRim):
                for e3 in range(self._elementsCountThroughShell):
                    if sElementId:
                        elementIdPatch = []
                        e2StartIdx = elementsAlongPatch // 2 - i - 1
                        e2EndIdx = elementsAlongPatch // 2 + i
                        e1StartIdx = elementsAlongPatchRim - (i + 1)
                        e1EndIdx = elementsAroundPatch - e1StartIdx - 1

                        if e1EndIdx > e1StartIdx:
                            for e2 in range(elementsAlongPatch // 2 - 1, e2StartIdx, -1):
                                elementIdPatch.append(sElementId[e2][e3][e1StartIdx])
                            elementIdPatch += [sElementId[e2StartIdx][e3][e1StartIdx]] * (4 + (elementsAlongPatchRim - 1 - i) * 2)

                            elementIdPatch += sElementId[e2StartIdx][e3][e1StartIdx + 1:e1EndIdx]
                            elementIdPatch += [sElementId[e2StartIdx][e3][e1EndIdx]] * (4 + (elementsAlongPatchRim - 1 - i) * 2)

                            for e2 in range(e2StartIdx + 1, e2StartIdx + (e2EndIdx - e2StartIdx) // 2 + 1):
                                elementIdPatch.append(sElementId[e2][e3][e1EndIdx])
                            for e2 in range(e2StartIdx + (e2EndIdx - e2StartIdx) // 2 + 1, e2EndIdx):
                                elementIdPatch.append(sElementId[e2][e3][e1StartIdx])

                            elementIdPatch += [sElementId[e2EndIdx][e3][e1StartIdx]] * (4 + (elementsAlongPatchRim - 1 - i) * 2)
                            elementIdPatch += sElementId[e2EndIdx][e3][e1StartIdx + 1:e1EndIdx]
                            elementIdPatch += [sElementId[e2EndIdx][e3][e1EndIdx]] * (4 + (elementsAlongPatchRim - 1 - i) * 2)
                            for e2 in range(e2EndIdx - 1, e2StartIdx + (e2EndIdx - e2StartIdx) // 2, -1):
                                elementIdPatch.append(sElementId[e2][e3][e1EndIdx])
                            self._rimElementIds[i] = [elementIdPatch]

    def getPatchCoordinates(self, n1, n2, n3, isLinearThroughShell):
        """
        Get patch parameters at a point.
        :param n1: Node index around.
        :param n2: Node index along patch.
        :param n3: Node index from inner to outer shell.
        :param isLinearThroughShell: True for linear through shell so no d3 is returned.
        :return: x, d1, d2, d3
        """
        return (self._patchCoordinates[0][n2][n3][n1],
                self._patchCoordinates[1][n2][n3][n1],
                self._patchCoordinates[2][n2][n3][n1],
                None if isLinearThroughShell else self._patchCoordinates[3][n2][n3][n1])

    def getRimIndexNodeLayoutSpecial(self, generateData, rimIndex):
        """
        Return special 5 way node layout at 3-segment rim nodes to keep roundedness around the rim.
        """
        return generateData.getNodeLayout5Way()

    def generateJunctionRimElements(self, junction, generateData):
        """
        Generates rim elements for junction part of the segment after segment nodes and elements have been made.
        Overwrites base class due to differences in derivative directions for nodes on patch and the need to handle
        two elements at each corner with collapsed nodes. The two elements at each corner is replaced by a regular
        element using all 8 corners, resulting in four less elements compared to tube segment case.
        """
        elementsAlongPatchSegment = len(self._patchCoordinates[0]) - 1
        elementsAroundPatchSegment = len(self._patchCoordinates[0][0][0]) - 1

        mesh = generateData.getMesh()
        meshDimension = generateData.getMeshDimension()
        elementtemplate = mesh.createElementtemplate()
        elementtemplate.setElementShapeType(
            Element.SHAPE_TYPE_CUBE if (meshDimension == 3) else Element.SHAPE_TYPE_SQUARE)
        isLinearThroughShell = generateData.isLinearThroughShell()
        d3Defined = (meshDimension == 3) and not isLinearThroughShell
        coordinates = generateData.getCoordinates()
        nodes = generateData.getNodes()
        fieldcache = generateData.getFieldcache()

        nodeLayoutFlipD1D2 = generateData.getNodeLayoutFlipD1D2()
        nodeLayoutSwapMinusD1D2 = generateData.getNodeLayoutSwapMinusD1D2()
        nodeLayoutSwapD1MinusD2 = generateData.getNodeLayoutSwapD1MinusD2()

        elementsCountRim = self.getElementsCountRim()
        elementsCountAround = self.getElementsCountAround()
        elementsCountRimRegular = elementsCountRim - 1 if junction._isCore else elementsCountRim
        elementsCountTransition = self.getElementsCountTransition()

        annotationMeshGroups = generateData.getAnnotationMeshGroups(self.getAnnotationTerms())
        eftList = [None] * elementsCountAround
        scalefactorsList = [None] * elementsCountAround

        elementsCountAlong = self.getSampledElementsCountAlong()
        s = junction._segments.index(self)
        e2 = (elementsCountAlong - 1) if junction._segmentsIn[s] else 0
        n2 = (elementsCountAlong - 1) if junction._segmentsIn[s] else 1

        # smooth directions of midpoints
        for n3 in range(elementsCountRimRegular + 1):
            for i in range(2):
                nids = []
                sx = []
                sd1 = []
                for n1 in range(elementsAlongPatchSegment // 2 + 1 if i == 0 else
                                elementsCountAround // 2 + elementsAlongPatchSegment // 2 + 1,
                                elementsAlongPatchSegment // 2 + elementsAroundPatchSegment + 4 if i == 0 else
                                elementsCountAround - elementsAlongPatchSegment // 2):
                    rimIndex = junction._segmentNodeToRimIndex[s][n1]
                    nids.append(junction._rimNodeIds[n3][rimIndex])
                    sx.append(junction._rimCoordinates[0][n3][rimIndex])
                    if (i == 0 and n1 < elementsAlongPatchSegment // 2 + elementsAroundPatchSegment + 2) or \
                            (i and n1 > elementsCountAround - elementsAlongPatchSegment // 2 - 1):
                        sd1.append(junction._rimCoordinates[1][n3][rimIndex])
                    else:
                         sd1.append([-1.0 * c for c in junction._rimCoordinates[2][n3][rimIndex]])
                sd1Smoothed = smoothCubicHermiteDerivativesLine(sx, sd1, fixStartDerivative=True, fixEndDerivative=True)
                for n1 in range(1, len(nids) - 1):
                    node = nodes.findNodeByIdentifier(nids[n1])
                    fieldcache.setNode(node)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, sd1Smoothed[n1])

        # Recalculate d3 if not linear through shell
        if not isLinearThroughShell:
            shellFactor = 1.0 / self._elementsCountThroughShell
            for n1 in range(elementsCountAround):
                rimIndex = junction._segmentNodeToRimIndex[s][n1]
                ox = junction._rimCoordinates[0][-1][rimIndex]
                ix = junction._rimCoordinates[0][0][rimIndex]
                d3 = mult(sub(ox, ix), shellFactor)
                for n3 in range(elementsCountRimRegular + 1):
                    nid = junction._rimNodeIds[n3][rimIndex]
                    node = nodes.findNodeByIdentifier(nid)
                    fieldcache.setNode(node)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)

        for e3 in range(elementsCountRimRegular):
            rim_e3 = e3 + 1 if junction._isCore else e3
            for e1 in range(elementsCountAround):
                n1p = (e1 + 1) % elementsCountAround
                isFirstCorner = (e1 == elementsAlongPatchSegment // 2 or \
                                 e1 == elementsAlongPatchSegment // 2 + 2 + elementsAroundPatchSegment or \
                                 e1 == elementsCountAround // 2 + elementsAlongPatchSegment // 2 or \
                                 e1 == elementsCountAround - elementsAlongPatchSegment // 2 - 2)
                isSecondCorner = (e1 == elementsAlongPatchSegment // 2 + 1 or \
                                  e1 == elementsAlongPatchSegment // 2 + elementsAroundPatchSegment + 3 or \
                                  e1 == elementsCountAround // 2 + elementsAlongPatchSegment // 2 + 1 or \
                                  e1 == elementsCountAround - elementsAlongPatchSegment // 2 - 1)
                # Skip the second corner element
                if isSecondCorner:
                    continue

                nids = []
                nodeParameters = []
                nodeLayouts = []
                elementIdentifier = generateData.nextElementIdentifier()
                lastTransition = junction._isCore and (elementsCountTransition == (rim_e3 + 1))
                needParameters = (e3 == 0) or lastTransition
                for n3 in [e3, e3 + 1] if (meshDimension == 3) else [e3]:
                    for n1 in [e1, n1p]:  # patch side
                        rimIndex = junction._segmentNodeToRimIndex[s][n1]
                        if isFirstCorner and n1 == n1p:
                            rimIndex = junction._segmentNodeToRimIndex[s][n1 + 1]
                            nids.append(junction._rimNodeIds[n3][rimIndex])
                        else:
                            nids.append(self.getRimNodeId(n1, n2, n3))
                        if needParameters:
                            if isFirstCorner and n1 == n1p:
                                nodeParameters.append(
                                    (junction._rimCoordinates[0][n3][rimIndex],
                                     junction._rimCoordinates[1][n3][rimIndex],
                                     junction._rimCoordinates[2][n3][rimIndex],
                                     junction._rimCoordinates[3][n3][rimIndex] if d3Defined else None))
                            else:
                                rimCoordinates = self.getRimCoordinates(n1, n2, n3)
                                nodeParameters.append(rimCoordinates if d3Defined else (
                                    rimCoordinates[0], rimCoordinates[1], rimCoordinates[2], None))
                            segmentNodesCount = len(junction._rimIndexToSegmentNodeList[rimIndex])
                            if segmentNodesCount == 2:
                                if e1 < elementsAlongPatchSegment // 2 + 2 or \
                                    elementsCountAround // 2 < e1 < elementsCountAround // 2 + elementsAlongPatchSegment // 2 or \
                                        e1 == elementsCountAround // 2 + elementsAlongPatchSegment // 2 or \
                                        (e1 == elementsCountAround // 2 and n1 == n1p) or \
                                        (e1 == elementsCountAround - 1 and n1 == 0):
                                    nodeLayouts.append(nodeLayoutSwapMinusD1D2)
                                elif elementsAlongPatchSegment // 2 + 2 + elementsAroundPatchSegment < e1 \
                                        < elementsCountAround // 2 + elementsAlongPatchSegment // 2 or \
                                        e1 > elementsCountAround - elementsAlongPatchSegment // 2 - 1 or \
                                        e1 == elementsAlongPatchSegment // 2 + 2 + elementsAroundPatchSegment and n1 == n1p or \
                                        e1 == elementsCountAround - elementsAlongPatchSegment // 2 - 2 and n1 == n1p:
                                    nodeLayouts.append(nodeLayoutSwapD1MinusD2)
                                else:
                                    nodeLayouts.append(None)

                    for n1 in [e1, n1p]:  # outside patch
                        rimIndex = junction._segmentNodeToRimIndex[s][n1]
                        nids.append(junction._rimNodeIds[n3][rimIndex])
                        if needParameters:
                            nodeParameters.append(
                                (junction._rimCoordinates[0][n3][rimIndex],
                                 junction._rimCoordinates[1][n3][rimIndex],
                                 junction._rimCoordinates[2][n3][rimIndex],
                                 junction._rimCoordinates[3][n3][rimIndex] if d3Defined else None))
                            if e1 == elementsAlongPatchSegment // 2 + 2 + elementsAroundPatchSegment and n1 == n1p:
                                nodeLayouts.append(nodeLayoutSwapD1MinusD2)
                            elif e1 == elementsCountAround // 2 + elementsAlongPatchSegment // 2:
                                nodeLayouts.append(nodeLayoutFlipD1D2 if n1 == e1 else nodeLayoutSwapMinusD1D2)
                            elif e1 == elementsCountAround - elementsAlongPatchSegment // 2 - 2 and n1 == n1p:
                                nodeLayouts.append(nodeLayoutSwapD1MinusD2)
                            elif e1 < elementsAlongPatchSegment // 2 + 2 or \
                                    elementsAlongPatchSegment // 2 + 2 + elementsAroundPatchSegment < e1 < \
                                    (elementsCountAround + elementsAlongPatchSegment) // 2 + 2 or \
                                    e1 > elementsCountAround - elementsAlongPatchSegment // 2 - 1:
                                nodeLayouts.append(junction.getRimIndexNodeLayout(generateData, rimIndex))
                            else:
                                nodeLayouts.append(None)

                # exploit efts being same through the rim
                eft = eftList[e1]
                scalefactors = scalefactorsList[e1]
                if not eft:
                    eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                    eftList[e1] = eft
                    scalefactorsList[e1] = scalefactors
                if lastTransition:
                    # need to generate eft again otherwise modifying object in eftList, mucking up outer layers
                    eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                    eft, scalefactors = generateData.resolveEftCoreBoundaryScaling(
                        eft, scalefactors, nodeParameters, nids, self.getCoreBoundaryScalingMode())

                elementtemplate.defineField(coordinates, -1, eft)
                element = mesh.createElement(elementIdentifier, elementtemplate)
                element.setNodesByIdentifier(eft, nids)
                # print(e1, 'Element', elementIdentifier, nids)
                if scalefactors:
                    element.setScaleFactors(eft, scalefactors)
                self.setRimElementId(e1, e2, rim_e3, elementIdentifier)
                for annotationMeshGroup in annotationMeshGroups:
                    annotationMeshGroup.addElement(element)

    def _addRimElementsToMeshGroup(self, e1Start, e1Limit, e3Start, e3Limit, meshGroup):
        """
        Add ranges of rim elements to mesh group.
        :param e1Start: Start element index around. Can be negative which supports wrapping.
        :param e1Limit: Limit element index around.
        :param e3Start: Start element index rim.
        :param e3Limit: Limit element index rim.
        :param meshGroup: Zinc MeshGroup to add elements to.
        """
        # print("Add rim elements", e1Start, e1Limit, e3Start, e3Limit, meshGroup.getName())
        elementsCountAlong = len(self._rimElementIds) - 1
        mesh = meshGroup.getMasterMesh()
        for e2 in range(elementsCountAlong):
            rimSlice = self._rimElementIds[e2]
            if rimSlice:
                for elementIdentifiersList in rimSlice[e3Start:e3Limit]:
                    partElementIdentifiersList = elementIdentifiersList[e1Start:e1Limit] if (e1Start >= 0) else (
                            elementIdentifiersList[e1Start:] + elementIdentifiersList[:e1Limit])
                    for elementIdentifier in partElementIdentifiersList:
                        if elementIdentifier is None:
                            continue
                        element = mesh.findElementByIdentifier(elementIdentifier)
                        meshGroup.addElement(element)


class TubeNetworkMeshJunction(NetworkMeshJunction):
    """
    Describes junction between multiple tube segments, some in, some out.
    """

    def __init__(self, inSegments: list, outSegments: list, useOuterTrimSurfaces):
        """
        :param inSegments: List of inward TubeNetworkMeshSegment.
        :param outSegments: List of outward TubeNetworkMeshSegment.
        :param useOuterTrimSurfaces: Set to True to use common trim surfaces calculated from outer.
        """
        super(TubeNetworkMeshJunction, self).__init__(inSegments, outSegments)
        pathsCount = self._segments[0].getPathsCount()
        self._trimSurfaces = [[None for p in range(pathsCount)] for s in range(self._segmentsCount)]
        self._useOuterTrimSurfaces = useOuterTrimSurfaces
        self._calculateTrimSurfaces()
        # rim indexes are issued for interior points connected to 2 or more segment node indexes
        # based on the outer surface, and reused through the rim
        self._rimIndexToSegmentNodeList = []  # list[rim index] giving list[(segment number, node index around)]
        self._segmentNodeToRimIndex = []  # list[segment number][node index around] to rimIndex
        # rim coordinates sampled in the junction are indexed by n3 (indexed outware through the rim) and 'rim index'
        self._rimCoordinates = None  # if set, (rx[], rd1[], rd2[], rd3[]) each over [n3][rim index]
        self._rimNodeIds = None  # if set, nodeIdentifier[n3][rim index]

        # parameters used for solid core
        self._isCore = self._segments[0].getIsCore()
        self._boxCoordinates = None  # [nAlong][nAcrossMajor][nAcrossMinor]
        self._boxNodeIds = None
        # sequence of segment indexes for bifurcation or trifurcation, proceding in increasing angle around a plane.
        # See: self._determineJunctionSequence()
        self._sequence = None
        self._boxIndexToSegmentNodeList = []
        # list[box index] giving list[(segment number, node index across major axis, node index across minor axis)]
        self._segmentNodeToBoxIndex = []
        # list[segment number][node index across major axis][node index across minor axis] to boxIndex
        self._triplePointLocationsList = []
        # list[[node Id, location], ...] used to match ETFs at triple points

    def _calculateTrimSurfaces(self):
        """
        Calculate surfaces for trimming adjacent segments so they can smoothly transition at junction.
        Algorithm gets 6 intersection points of longitudinal lines around each segment with other tubes or the end.
        These are joined to make a partial cone radiating back to the centre of the junction.
        Longitudinal lines start on the edge adjacent to the other segment most normal to it.
        """
        if self._segmentsCount < 3:
            return
        pathsCount = self._segments[0].getPathsCount()
        # get directions at end of segments' paths:
        outDirs = [[] for s in range(self._segmentsCount)]
        for s in range(self._segmentsCount):
            endIndex = -1 if self._segmentsIn[s] else 0
            for p in range(pathsCount):
                pathParameters = self._segments[s].getPathParameters(p)
                outDir = normalize(pathParameters[1][endIndex])
                if self._segmentsIn[s]:
                    outDir = [-d for d in outDir]
                outDirs[s].append(outDir)

        segmentEndPlaneTrackSurfaces = []
        for s in range(self._segmentsCount):
            endIndex = -1 if self._segmentsIn[s] else 0
            pathEndPlaneTrackSurfaces = []
            for p in range(pathsCount):
                if self._useOuterTrimSurfaces and (p > 0):
                    pathEndPlaneTrackSurfaces.append(pathEndPlaneTrackSurfaces[-1])
                    continue
                pathParameters = self._segments[s].getPathParameters(p)
                centre = pathParameters[0][endIndex]
                axis1 = pathParameters[2][endIndex]
                axis2 = pathParameters[4][endIndex]
                nx = [sub(sub(centre, axis1), axis2),
                      sub(add(centre, axis1), axis2),
                      add(sub(centre, axis1), axis2),
                      add(add(centre, axis1), axis2)]
                nd1 = [mult(axis1, 2.0)] * 4
                nd2 = [mult(axis2, 2.0)] * 4
                pathEndPlaneTrackSurfaces.append(TrackSurface(1, 1, nx, nd1, nd2))
            segmentEndPlaneTrackSurfaces.append(pathEndPlaneTrackSurfaces)

        trimPointsCountAround = 6
        trimAngle = 2.0 * math.pi / trimPointsCountAround
        for s in range(self._segmentsCount):
            endIndex = -1 if self._segmentsIn[s] else 0
            for p in range(pathsCount):
                if self._useOuterTrimSurfaces and (p > 0):
                    self._trimSurfaces[s][p] = self._trimSurfaces[s][p - 1]
                    continue
                pathParameters = self._segments[s].getPathParameters(p)
                d2End = pathParameters[2][endIndex]
                d3End = pathParameters[4][endIndex]
                endEllipseNormal = normalize(cross(d2End, d3End))
                sOutDir = outDirs[s][p]
                # get phase angles and weights of other segments
                angles = []
                weights = []
                sumWeights = 0.0
                maxWeight = 0.0
                phaseAngle = None
                for os in range(self._segmentsCount):
                    if os == s:
                        continue
                    osOutDir = outDirs[os][p]
                    dx = dot(osOutDir, d2End)
                    dy = dot(osOutDir, d3End)
                    if (dx == 0.0) and (dy == 0.0):
                        angle = 0.0
                        weight = 0.0
                    else:
                        angle = math.atan2(dy, dx)
                        if angle < 0.0:
                            angle += 2.0 * math.pi
                        dotDirs = dot(sOutDir, osOutDir)
                        # handle numerical errors which make the above outside valid range of [-1.0, 1.0]:
                        if dotDirs < -1.0:
                            dotDirs = -1.0
                        elif dotDirs > 1.0:
                            dotDirs = 1.0
                        weight = math.pi - math.acos(dotDirs)
                        if weight > maxWeight:
                            maxWeight = weight
                            phaseAngle = angle
                        sumWeights += weight
                    weights.append(weight)
                    angles.append(angle)
                # get correction to phase angle
                weightedSumDeltaAngles = 0.0
                for os in range(len(angles)):
                    angle = angles[os] - phaseAngle
                    if angle < 0.0:
                        angle += 2.0 * math.pi
                    nearestAngle = math.floor(angle / trimAngle + 0.5) * trimAngle
                    deltaAngle = nearestAngle - angle
                    weightedSumDeltaAngles += weights[os] * deltaAngle
                phaseAngle -= weightedSumDeltaAngles / sumWeights
                lx, ld1, ld2, ld12 = getPathRawTubeCoordinates(
                    pathParameters, trimPointsCountAround, radius=1.0, phaseAngle=phaseAngle)
                pointsCountAlong = len(pathParameters[0])

                # get coordinates and directions of intersection points of longitudinal lines and other track surfaces
                rx = []
                rd1 = []
                trim = False
                lowestMaxProportionFromEnd = 1.0
                for n1 in range(trimPointsCountAround):
                    cx = [lx[n2][n1] for n2 in range(pointsCountAlong)]
                    cd1 = [ld1[n2][n1] for n2 in range(pointsCountAlong)]
                    cd2 = [ld2[n2][n1] for n2 in range(pointsCountAlong)]
                    cd12 = [ld12[n2][n1] for n2 in range(pointsCountAlong)]
                    x = lx[endIndex][n1]
                    d1 = ld1[endIndex][n1]
                    maxProportionFromEnd = 0.0
                    # find intersection point with other segments which is furthest from end
                    for os in range(self._segmentsCount):
                        if os == s:
                            continue
                        otherSegment = self._segments[os]
                        for i in range(2):
                            otherTrackSurface = \
                                otherSegment.getRawTrackSurface(p) if (i == 0) else segmentEndPlaneTrackSurfaces[os][p]
                            otherSurfacePosition, curveLocation, isIntersection = \
                                otherTrackSurface.findNearestPositionOnCurve(
                                    cx, cd2, loop=False, sampleEnds=False, sampleHalf=2 if self._segmentsIn[s] else 1)
                            if isIntersection:
                                if i == 1:
                                    # must be within ellipse inside rectangular plane
                                    xi1 = otherSurfacePosition.xi1 - 0.5
                                    xi2 = otherSurfacePosition.xi2 - 0.5
                                    if (xi1 * xi1 + xi2 * xi2) > 0.25:
                                        break
                                proportion2 = (curveLocation[0] + curveLocation[1]) / (pointsCountAlong - 1)
                                proportionFromEnd = (1.0 - proportion2) if self._segmentsIn[s] else proportion2
                                if proportionFromEnd > maxProportionFromEnd:
                                    trim = True
                                    x, d2 = evaluateCoordinatesOnCurve(cx, cd2, curveLocation, loop=False, derivative=True)
                                    d1 = evaluateCoordinatesOnCurve(cd1, cd12, curveLocation, loop=False)
                                    n = cross(d1, d2)  # normal to this surface
                                    ox, od1, od2 = otherTrackSurface.evaluateCoordinates(
                                        otherSurfacePosition, derivatives=True)
                                    on = cross(od1, od2)  # normal to other surface
                                    d1 = cross(n, on)
                                    maxProportionFromEnd = proportionFromEnd
                                break
                    if maxProportionFromEnd < lowestMaxProportionFromEnd:
                        lowestMaxProportionFromEnd = maxProportionFromEnd
                    rx.append(x)
                    rd1.append(d1)

                if trim:
                    # centre of trim surfaces is at lowestMaxProportionFromEnd
                    if lowestMaxProportionFromEnd <= 0.0:
                        xCentre = pathParameters[0][endIndex]
                    else:
                        proportion = \
                            (1.0 - lowestMaxProportionFromEnd) if self._segmentsIn[s] else lowestMaxProportionFromEnd
                        eProportion = proportion * (pointsCountAlong - 1)
                        e = min(int(eProportion), (pointsCountAlong - 2))
                        curveLocation = (e, eProportion - e)
                        xCentre = evaluateCoordinatesOnCurve(pathParameters[0], pathParameters[1], curveLocation)
                    # ensure d1 directions go around in same direction as loop
                    for n1 in range(trimPointsCountAround):
                        d1 = rd1[n1]
                        if dot(endEllipseNormal, cross(sub(rx[n1], xCentre), d1)) < 0.0:
                            for c in range(3):
                                d1[c] = -d1[c]
                    rd1 = smoothCubicHermiteDerivativesLoop(rx, rd1, fixAllDirections=True,
                                                            magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
                    rd2 = [sub(rx[n1], xCentre) for n1 in range(trimPointsCountAround)]
                    rd12 = smoothCurveSideCrossDerivatives(rx, rd1, [rd2], loop=True)[0]
                    nx = []
                    nd1 = []
                    nd2 = []
                    nd12 = []
                    trimWidthFactors = (0.25, 1.75)  # if self._useOuterTrimSurfaces else (0.75, 1.25)
                    d2scale = trimWidthFactors[1] - trimWidthFactors[0]
                    for factor in trimWidthFactors:
                        for n1 in range(trimPointsCountAround):
                            d2 = sub(rx[n1], xCentre)
                            x = add(xCentre, mult(d2, factor))
                            d1 = mult(rd1[n1], factor)
                            d12 = mult(rd12[n1], factor)
                            d2 = mult(d2, d2scale)
                            nx.append(x)
                            nd1.append(d1)
                            nd2.append(d2)
                            nd12.append(d12)
                    trimSurface = TrackSurface(trimPointsCountAround, 1, nx, nd1, nd2, nd12, loop1=True)
                    self._trimSurfaces[s][p] = trimSurface

    def getTrimSurfaces(self, segment):
        """
        :param segment: TubeNetworkMeshSegment which must join at junction.
        :return: List of trim surfaces for paths of segment at junction.
        """
        return self._trimSurfaces[self._segments.index(segment)]

    def _sampleMidPoint(self, segmentsParameterLists):
        """
        Get mid-point coordinates and derivatives within junction from 2 or more segments' parameters.
        :param segmentsParameterLists: List over segment indexes s of [x, d1, d2, d3], each with 2 last parameters.
        d3 will be None for 2-D of bicubic-linear.
        :return: Mid-point x, d1, d2, d3. Derivative magnitudes will need smoothing.`
        """
        segmentsIn = [dot(sub(params[0][1], params[0][0]), params[2][1]) > 0.0 for params in segmentsParameterLists]
        segmentsCount = len(segmentsIn)
        assert segmentsCount > 1
        d3Defined = None not in segmentsParameterLists[0][3]
        # for each segment get inward parameters halfway between last 2 parameters
        xi = 0.5
        hx = []
        hd1 = []
        hd2 = []
        hn = []  # normal hd1 x hd2
        hd3 = [] if d3Defined else None

        for s in range(segmentsCount):
            params = segmentsParameterLists[s]
            hd2m = [params[2][i] if segmentsIn[s] else [-d for d in params[2][i]] for i in range(2)]
            hx.append(interpolateCubicHermite(params[0][0], hd2m[0], params[0][1], hd2m[1], xi))
            hd1.append(mult(add(params[1][0], params[1][1]), 0.5 if segmentsIn[s] else -0.5))
            hd2.append(interpolateCubicHermiteDerivative(params[0][0], hd2m[0], params[0][1], hd2m[1], xi))
            hn.append(normalize(cross(hd1[-1], hd2[-1])))
            if d3Defined:
                hd3.append(mult(add(params[3][0], params[3][1]), 0.5))

        # get lists of mid-point parameters for all segment permutations
        mx = []
        md1 = []
        md2 = []
        md3 = [] if d3Defined else None
        xi = 0.5
        sideFactor = 1.0
        outFactor = 1.0  # only used as relative proportion with non-zero sideFactor
        for s1 in range(segmentsCount - 1):
            fxs1 = segmentsParameterLists[s1][0][0]
            fd2s1 = segmentsParameterLists[s1][2][0]
            if not segmentsIn[s1]:
                fd2s1 = [-d for d in fd2s1]
            norm_fd2s1 = normalize(fd2s1)
            for s2 in range(s1 + 1, segmentsCount):
                hd2s1 = hd2[s1]
                hd2s2 = [-d for d in hd2[s2]]
                if sideFactor > 0.0:
                    # compromise with direct connection respecting surface tangents
                    sideDirection = normalize(sub(hx[s2], hx[s1]))
                    side1d1 = dot(normalize(hd1[s1]), sideDirection)
                    side1d2 = dot(normalize(hd2[s1]), sideDirection)
                    side1 = add(mult(hd1[s1], side1d1), mult(hd2[s1], side1d2))
                    side2d1 = dot(normalize(hd1[s2]), sideDirection)
                    side2d2 = dot(normalize(hd2[s2]), sideDirection)
                    side2 = add(mult(hd1[s2], side2d1), mult(hd2[s2], side2d2))
                    sideScaling = computeCubicHermiteDerivativeScaling(hx[s1], side1, hx[s2], side2)
                    hd2s1 = add(mult(hd2s1, outFactor), mult(side1, sideScaling * sideFactor))
                    hd2s2 = add(mult(hd2s2, outFactor), mult(side2, sideScaling * sideFactor))
                scaling = computeCubicHermiteDerivativeScaling(hx[s1], hd2s1, hx[s2], hd2s2)
                hd2s1 = mult(hd2s1, scaling)
                hd2s2 = mult(hd2s2, scaling)
                cx = interpolateCubicHermite(hx[s1], hd2s1, hx[s2], hd2s2, xi)
                cd2 = interpolateCubicHermiteDerivative(hx[s1], hd2s1, hx[s2], hd2s2, xi)
                mx.append(cx)
                md1.append(mult(add(hd1[s1], [-d for d in hd1[s2]]), 0.5))
                fxs2 = segmentsParameterLists[s2][0][0]
                fd2s2 = segmentsParameterLists[s2][2][0]
                if segmentsIn[s2]:
                    fd2s2 = [-d for d in fd2s2]
                norm_fd2s2 = normalize(fd2s2)
                # reduce md2 up to 50% depending on how out-of line they are
                norm_cd2 = normalize(cd2)
                # md2Factor = 0.75 + 0.25 * dot(norm_fd2s1, norm_fd2s2)
                md2Factor1 = 0.75 + 0.25 * dot(norm_fd2s1, norm_cd2)
                md2Factor2 = 0.75 + 0.25 * dot(norm_fd2s2, norm_cd2)
                md2Factor = md2Factor1 * md2Factor2
                # md2.append(mult(cd2, md2Factor))
                # smooth md2 with 2nd row from end, which it actually interpolates to
                tmd2 = smoothCubicHermiteDerivativesLine(
                    [fxs1, cx, fxs2], [fd2s1, cd2, fd2s2],
                    fixAllDirections=True, fixStartDerivative=True, fixEndDerivative=True,
                    magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
                md2.append(mult(tmd2[1], md2Factor))
                if d3Defined:
                    md3.append(mult(add(hd3[s1], hd3[s2]), 0.5))
        if segmentsCount == 2:
            if not segmentsIn[0]:
                md1[0] = [-d for d in md1[0]]
                md2[0] = [-d for d in md2[0]]
            return mx[0], md1[0], md2[0], md3[0] if d3Defined else None
        mCount = len(mx)
        # get mean x, d3 and surface normal (d1 x d2) at the centre across all segment permutations
        cx = [sum(x[c] for x in mx) / mCount for c in range(3)]
        cd3 = [sum(d3[c] for d3 in md3) / mCount for c in range(3)] if d3Defined else None
        ns12 = [0.0, 0.0, 0.0]
        for m in range(len(md1)):
            cp12 = normalize(cross(md1[m], md2[m]))
            for c in range(3):
                ns12[c] += cp12[c]
        ns12 = normalize(ns12)

        # get best fit directions with these 360/segmentsCount degrees apart

        # get preferred derivative at centre out to each segment
        rd = [interpolateLagrangeHermiteDerivative(
            cx, segmentsParameterLists[s][0][0], [-d for d in segmentsParameterLists[s][2][0]] if segmentsIn[s]
            else segmentsParameterLists[s][2][0], 0.0) for s in range(segmentsCount)]
        # get orthonormal axes with ns12, axis1 in direction of first preferred derivative
        axis1 = normalize(cross(cross(ns12, rd[0]), ns12))
        axis2 = cross(ns12, axis1)
        # get angles and sequence around normal, starting at axis1
        angles = [0.0]
        for s in range(1, segmentsCount):
            angle = math.atan2(dot(rd[s], axis2), dot(rd[s], axis1))
            angles.append((2.0 * math.pi + angle) if (angle < 0.0) else angle)
        angle = 0.0
        # get sequence of segments anticlockwise around ns12 starting at segment 0
        sequence = [0]
        for s in range(1, segmentsCount):
            nextAngle = math.pi * 4.0
            nexts = 0
            for t in range(1, segmentsCount):
                if angle < angles[t] < nextAngle:
                    nextAngle = angles[t]
                    nexts = t
            angle = nextAngle
            sequence.append(nexts)

        angleIncrement = 2.0 * math.pi / segmentsCount
        deltaAngle = 0.0
        angle = 0.0
        magSum = 0.0
        for s in range(1, segmentsCount):
            angle += angleIncrement
            deltaAngle += angles[sequence[s]] - angle
            magSum += magnitude(rd[sequence[s]])
        deltaAngle = deltaAngle / segmentsCount
        d2Mean = magSum / segmentsCount

        angle = deltaAngle
        sd = [None] * segmentsCount
        for s in range(segmentsCount):
            x1 = d2Mean * math.cos(angle)
            x2 = d2Mean * math.sin(angle)
            sd[sequence[s]] = add(mult(axis1, x1), mult(axis2, x2))
            angle += angleIncrement

        if segmentsCount == 3:
            # get through score for pairs of directions
            maxThroughScore = 0.0
            si = None
            # for s1 in range(segmentsCount - 1):
            #     dir1 = normalize(segmentsParameterLists[s1][2][1])
            #     for s2 in range(s1 + 1, segmentsCount):
            #         dir2 = normalize(segmentsParameterLists[s2][2][1])
            #         throughScore = abs(dot(dir1, dir2))
            #         if throughScore > maxThroughScore:
            #             maxThroughScore = throughScore
            #             si = (s1, s2)
            if maxThroughScore < 0.9:
                # maintain symmetry of bifurcations
                si = ((0, 1) if ((segmentsIn == [True, True, False]) or (segmentsIn == [False, False, True])) else
                      (2, 0) if ((segmentsIn == [True, False, True]) or (segmentsIn == [False, True, False])) else
                      (1, 2))

        elif segmentsCount == 4:
            si = sequence[1:3]
        else:
            print("TubeNetworkMeshJunction._sampleMidPoint not fully implemented for segmentsCount =", segmentsCount)
            si = (1, 2)

        # regular mean of d1 magnitude around midpoints
        magSum = 0.0
        for s in range(segmentsCount):
            magSum += magnitude(md1[s])
        d1Mean = magSum / segmentsCount

        # overall harmonic mean of derivatives to err on the low side for the diagonal derivatives
        dMean = 2.0 / (1.0 / d1Mean + 1.0 / d2Mean)

        if segmentsCount == 4:
            # use the equal spaced directions
            td = [mult(normalize(sd[j]), dMean) for j in si]
        else:
            # td = [sd[j] for j in si]
            # compromise between preferred directions rd and equal spaced directions sd
            td = [mult(normalize(add(rd[j], sd[j])), dMean) for j in si]
            if segmentsIn.count(True) == 2:
                # reverse so matches inward directions
                td = [[-c for c in d] for d in td]
            # td = [sub(d, mult(ns12, dot(d, ns12))) for d in td]

        cd1, cd2 = td
        if dot(cross(cd1, cd2), ns12) < 0.0:
            cd1, cd2 = cd2, cd1
        return cx, cd1, cd2, cd3

    def _determineJunctionSequence(self):
        """
        Determines the sequence of a junction, the order of segment indexes around a circle in a plane.
        Currently only works for bifurcation and trifurcation.
        Bifurcation is always sequence [0, 1, 2].
        For trifurcation, only + (plus) shape is currently supported, not tetrahedral.
        The function determines plus sequence [0, 1, 2, 3], [0, 1, 3, 2] or [0, 2, 1, 3].
        """
        assert self._segmentsCount == 3 or self._segmentsCount == 4
        outDirections = []
        for s in range(self._segmentsCount):
            d1 = self._segments[s].getPathParameters()[1][-1 if self._segmentsIn[s] else 0]
            outDirections.append(normalize([-d for d in d1] if self._segmentsIn[s] else d1))
        if self._segmentsCount == 3:
            up = cross(outDirections[0], outDirections[1])
            d3 = self._segments[0].getPathParameters()[4][-1 if self._segmentsIn[s] else 0]
            if dot(up, d3) < 0.0:
                self._sequence = [0, 2, 1]  # reverse sequence relative to d3
            else:
                self._sequence = [0, 1, 2]
        elif self._segmentsCount == 4:
            # only support plus + shape for now, not tetrahedral
            # determine plus + sequence [0, 1, 2, 3], [0, 1, 3, 2], or [0, 2, 1, 3]
            ns = None
            for s in range(1, self._segmentsCount):
                if dot(outDirections[0], outDirections[s]) > -0.9:
                    ns = cross(outDirections[0], outDirections[s])
                    break
            # get orthonormal axes with ns12, axis1 in direction of first preferred derivative
            axis1 = normalize(cross(cross(ns, outDirections[0]), ns))
            axis2 = normalize(cross(ns, axis1))
            # get angles and sequence around normal, starting at axis1
            angles = [0.0]
            for s in range(1, self._segmentsCount):
                angle = math.atan2(dot(outDirections[s], axis2), dot(outDirections[s], axis1))
                angles.append((2.0 * math.pi + angle) if (angle < 0.0) else angle)
            angle = 0.0
            sequence = [0]
            for s in range(1, self._segmentsCount):
                nextAngle = math.pi * 4.0
                nexts = 0
                for t in range(1, self._segmentsCount):
                    if angle < angles[t] < nextAngle:
                        nextAngle = angles[t]
                        nexts = t
                angle = nextAngle
                sequence.append(nexts)
            self._sequence = sequence
        else:
            return

    def _sampleBifurcation(self, aroundCounts, coreBoxMajorCounts):
        """
        Blackbox function for sampling bifurcation coordinates. The rim coordinates are first sampled, then the box
        coordinates are sampled, if Core option is enabled.
        :param aroundCounts: Number of elements around the 3 tube segments.
        :param coreBoxMajorCounts: Number of elements across core box major axis of 3 tube segments.
        :return rimIndexesCount, boxIndexesCount: Total number of rimIndexes and boxIndexes, respectively.
        """
        assert self._segmentsCount == 3

        # sample rim junction
        rimIndexesCount = 0
        # numbers of elements directly connecting pairs of segments
        # sequence = [0, 1, 2]
        connectionCounts = [(aroundCounts[s] + aroundCounts[s - 2] - aroundCounts[s - 1]) // 2 for s in range(3)]
        for s in range(3):
            if (connectionCounts[s] < 1) or (aroundCounts[s] != (connectionCounts[s - 1] + connectionCounts[s])):
                print("Can't make tube bifurcation between elements counts around", aroundCounts)
                return 0, 0

        self._rimIndexToSegmentNodeList = []  # list[rim index] giving list[(segment index, node index around)]
        self._segmentNodeToRimIndex = [[None] * aroundCounts[s] for s in range(self._segmentsCount)]
        for s1 in range(3):
            s2 = (s1 + 1) % 3
            if self._segmentsIn[s1]:
                startNodeIndex1 = (aroundCounts[s1] - connectionCounts[s1]) // 2
            else:
                startNodeIndex1 = connectionCounts[s1] // 2
            if self._segmentsIn[s2]:
                startNodeIndex2 = connectionCounts[s1] // 2
            else:
                startNodeIndex2 = (aroundCounts[s2] - connectionCounts[s1]) // 2
            for n in range(connectionCounts[s1] + 1):
                nodeIndex1 = (startNodeIndex1 + n) if self._segmentsIn[s1] else \
                    ((startNodeIndex1 - n) % aroundCounts[s1])
                if self._segmentNodeToRimIndex[s1][nodeIndex1] is None:
                    nodeIndex2 = ((startNodeIndex2 - n) % aroundCounts[s2]) if self._segmentsIn[s2] else \
                        (startNodeIndex2 + n)
                    if self._segmentNodeToRimIndex[s2][nodeIndex2] is None:
                        rimIndex = rimIndexesCount
                        # keep list in order from lowest s
                        self._rimIndexToSegmentNodeList.append(
                            [[s1, nodeIndex1], [s2, nodeIndex2]] if (s1 < s2) else
                            [[s2, nodeIndex2], [s1, nodeIndex1]])
                        self._segmentNodeToRimIndex[s2][nodeIndex2] = rimIndex
                        rimIndexesCount += 1
                    else:
                        rimIndex = self._segmentNodeToRimIndex[s2][nodeIndex2]
                        # keep list in order from lowest s
                        segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
                        index = 0
                        for i in range(len(segmentNodeList)):
                            if s1 < segmentNodeList[i][0]:
                                break
                            index += 1
                        segmentNodeList.insert(index, [s1, nodeIndex1])
                    self._segmentNodeToRimIndex[s1][nodeIndex1] = rimIndex

        # sample box junction
        boxIndexesCount = 0
        if self._isCore:
            elementsCountTransition = self._segments[0].getElementsCountTransition()
            majorConnectionCounts = [(coreBoxMajorCounts[s] + coreBoxMajorCounts[s - 2] - coreBoxMajorCounts[s - 1]) // 2 for s
                                in range(3)]
            midIndexes = [(coreBoxMajorCounts[s] - majorConnectionCounts[s]) if self._segmentsIn[s]
                          else majorConnectionCounts[s] for s in range(3)]
            # check compatible numbers of elements across major and minor directions
            lastCoreBoxMinorCount = None
            for s in range(3):
                coreBoxMinorCount = self._segments[s].getElementsCountCoreBoxMinor()
                if lastCoreBoxMinorCount:
                    if coreBoxMinorCount != lastCoreBoxMinorCount:
                        print("Can't make core bifurcation between different box minor axis element counts",
                              coreBoxMinorCount, "vs", lastCoreBoxMinorCount)
                        return 0, 0
                else:
                    lastCoreBoxMinorCount = coreBoxMinorCount
                if ((majorConnectionCounts[s] < 0) or
                        (coreBoxMajorCounts[s] != (majorConnectionCounts[s - 1] + majorConnectionCounts[s]))):
                    print("Can't make core bifurcation between box major axis element counts", coreBoxMajorCounts)
                    return 0, 0

            coreBoxMinorNodesCount = self._segments[0].getCoreBoxMinorNodesCount()
            nodesCountAcrossMajorList = [self._segments[s].getCoreBoxMajorNodesCount() for s in range(3)]
            self._segmentNodeToBoxIndex = \
                [[[None for _ in range(coreBoxMinorNodesCount)] for _ in range(nodesCountAcrossMajorList[s])]
                 for s in range(self._segmentsCount)]
            for s1 in range(3):
                s2 = (s1 + 1) % 3
                midIndex1 = midIndexes[s1]
                midIndex2 = midIndexes[s2]
                for m in range(majorConnectionCounts[s1] + 1):
                    m1 = (midIndex1 + m) if self._segmentsIn[s1] else (midIndex1 - m)
                    for n in range(coreBoxMinorNodesCount):
                        if m1 == midIndex1:
                            indexGroup = [[0, midIndexes[0], n], [1, midIndexes[1], n], [2, midIndexes[2], n]]
                        else:
                            m2 = (midIndex2 - m) if self._segmentsIn[s2] else (midIndex2 + m)
                            indexGroup = [[s1, m1, n], [s2, m2, n]] if s1 < s2 else [[s2, m2, n], [s1, m1, n]]
                        if indexGroup not in self._boxIndexToSegmentNodeList:
                            self._boxIndexToSegmentNodeList.append(indexGroup)
                            boxIndexesCount += 1

            for boxIndex in range(boxIndexesCount):
                segmentNodeList = self._boxIndexToSegmentNodeList[boxIndex]
                for segmentNode in segmentNodeList:
                    s, m, n = segmentNode
                    self._segmentNodeToBoxIndex[s][m][n] = boxIndex

        return rimIndexesCount, boxIndexesCount

    def _sampleTrifurcation(self, aroundCounts, coreBoxMajorCounts):
        """
        Blackbox function for sampling trifurcation coordinates. The rim coordinates are first sampled, then the box
        coordinates are sampled, if Core option is enabled.
        :param aroundCounts: Number of elements around the 4 tube segments.
        :param coreBoxMajorCounts: Number of elements across core box major axis of the 4 tube segments.
        :return rimIndexesCount, boxIndexesCount: Total number of rimIndexes and boxIndexes, respectively.
        """
        assert self._segmentsCount == 4

        # sample rim junction
        rimIndexesCount = 0
        sequence = self._sequence
        pairCount02 = aroundCounts[sequence[0]] + aroundCounts[sequence[2]]
        pairCount13 = aroundCounts[sequence[1]] + aroundCounts[sequence[3]]
        throughCount02 = ((pairCount02 - pairCount13) // 2) if (pairCount02 > pairCount13) else 0
        throughCount13 = ((pairCount13 - pairCount02) // 2) if (pairCount13 > pairCount02) else 0
        throughCounts = [throughCount02, throughCount13, throughCount02, throughCount13]
        # numbers of elements directly connecting pairs of segments
        freeAroundCounts = [aroundCounts[sequence[s]] - throughCounts[s] for s in range(self._segmentsCount)]
        if freeAroundCounts[0] == freeAroundCounts[2]:
            count03 = freeAroundCounts[3] // 2
            count12 = freeAroundCounts[1] // 2
            connectionCounts = [count03, count12, count12, count03]
        elif freeAroundCounts[1] == freeAroundCounts[3]:
            count03 = freeAroundCounts[0] // 2
            count12 = freeAroundCounts[2] // 2
            connectionCounts = [count03, count12, count12, count03]
        else:
            connectionCounts = [((freeAroundCounts[s] + freeAroundCounts[(s + 1) % self._segmentsCount]
                                  - freeAroundCounts[s - 1] + (s % 2)) // 2) for s in range(self._segmentsCount)]

        for s in range(self._segmentsCount):
            if ((connectionCounts[s] < 1) or
                    (aroundCounts[sequence[s]] != (connectionCounts[s - 1] + throughCounts[s] + connectionCounts[s]))):
                print("Can't make tube junction between elements counts around", aroundCounts)
                return 0, 0

        self._rimIndexToSegmentNodeList = []  # list[rim index] giving list[(segment index, node index around)]
        self._segmentNodeToRimIndex = [[None] * aroundCounts[s] for s in range(self._segmentsCount)]
        for os1 in range(self._segmentsCount):
            os2 = (os1 + 1) % self._segmentsCount
            s1 = sequence[os1]
            s2 = sequence[os2]
            s3 = sequence[(os1 + 2) % self._segmentsCount]
            halfThroughCount = throughCounts[os1] // 2
            os1ConnectionCount = connectionCounts[os1]
            os2ConnectionCount = connectionCounts[os2]
            if self._segmentsIn[s1]:
                startNodeIndex1 = (aroundCounts[s1] - os1ConnectionCount) // 2
            else:
                startNodeIndex1 = os1ConnectionCount // -2
            if self._segmentsIn[s2]:
                startNodeIndex2 = os1ConnectionCount // -2
            else:
                startNodeIndex2 = (aroundCounts[s2] - os1ConnectionCount) // 2
            if self._segmentsIn[s3]:
                startNodeIndex3l = os2ConnectionCount // -2
                startNodeIndex3h = startNodeIndex3l - (os2ConnectionCount - os1ConnectionCount)
            else:
                startNodeIndex3l = (aroundCounts[s3] - os2ConnectionCount) // 2
                startNodeIndex3h = startNodeIndex3l + (os2ConnectionCount - os1ConnectionCount)

            for n in range(-halfThroughCount, os1ConnectionCount + 1 + halfThroughCount):
                n1 = startNodeIndex1 + (n if self._segmentsIn[s1] else (os1ConnectionCount - n))
                segmentIndexes = [s1]
                nodeIndexes = [n1 % aroundCounts[s1]]
                if 0 <= n <= os1ConnectionCount:
                    n2 = startNodeIndex2 + ((os1ConnectionCount - n) if self._segmentsIn[s2] else n)
                    segmentIndexes.append(s2)
                    nodeIndexes.append(n2 % aroundCounts[s2])
                if halfThroughCount and ((n <= 0) or (n >= os1ConnectionCount)):
                    n3 = ((startNodeIndex3l if (n <= 0) else startNodeIndex3h) +
                          ((os2ConnectionCount - n) if self._segmentsIn[s3] else n))
                    segmentIndexes.append(s3)
                    nodeIndexes.append(n3 % aroundCounts[s3])

                rimIndex = None
                for i in range(len(segmentIndexes)):
                    ri = self._segmentNodeToRimIndex[segmentIndexes[i]][nodeIndexes[i]]
                    if ri is not None:
                        rimIndex = ri
                        break
                if rimIndex is None:
                    # new rim index
                    rimIndex = rimIndexesCount
                    rimIndexesCount += 1
                    segmentNodeList = []
                    self._rimIndexToSegmentNodeList.append(segmentNodeList)
                else:
                    segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
                # build maps: rim index <--> segment index, node index
                for i in range(len(segmentIndexes)):
                    segmentIndex = segmentIndexes[i]
                    nodeIndex = nodeIndexes[i]
                    if self._segmentNodeToRimIndex[segmentIndex][nodeIndex] is None:
                        # keep segment node list in order from lowest segment index
                        index = 0
                        for j in range(len(segmentNodeList)):
                            if segmentIndex < segmentNodeList[j][0]:
                                break
                            index += 1
                        segmentNodeList.insert(index, [segmentIndex, nodeIndex])
                        self._segmentNodeToRimIndex[segmentIndex][nodeIndex] = rimIndex

        # sample box junction
        boxIndexesCount = 0
        if self._isCore:
            elementsCountTransition = self._segments[0].getElementsCountTransition()
            pairCount02 = coreBoxMajorCounts[sequence[0]] + coreBoxMajorCounts[sequence[2]]
            pairCount13 = coreBoxMajorCounts[sequence[1]] + coreBoxMajorCounts[sequence[3]]
            throughCount02 = ((pairCount02 - pairCount13) // 2) if (pairCount02 > pairCount13) else 0
            throughCount13 = ((pairCount13 - pairCount02) // 2) if (pairCount13 > pairCount02) else 0
            throughCounts = [throughCount02, throughCount13, throughCount02, throughCount13]
            freeAcrossCounts = [coreBoxMajorCounts[sequence[s]] - throughCounts[s] for s in range(self._segmentsCount)]
            if freeAcrossCounts[0] == freeAcrossCounts[2]:
                count03 = freeAcrossCounts[3] // 2
                count12 = freeAcrossCounts[1] // 2
                majorConnectionCounts = [count03, count12, count12, count03]
            elif freeAcrossCounts[1] == freeAcrossCounts[3]:
                count03 = freeAcrossCounts[0] // 2
                count12 = freeAcrossCounts[2] // 2
                majorConnectionCounts = [count03, count12, count12, count03]
            else:
                majorConnectionCounts = [((freeAcrossCounts[s] + freeAcrossCounts[(s + 1) % self._segmentsCount]
                                      - freeAcrossCounts[s - 1] + (s % 2)) // 2) for s in range(self._segmentsCount)]

            # check compatible numbers of elements across major and minor directions
            lastCoreBoxMinorCount = None
            for s in range(self._segmentsCount):
                coreBoxMinorCount = self._segments[s].getElementsCountCoreBoxMinor()
                if lastCoreBoxMinorCount:
                    if coreBoxMinorCount != lastCoreBoxMinorCount:
                        print("Can't make core trifurcation between different box minor axis element counts",
                              coreBoxMinorCount, "vs", lastCoreBoxMinorCount)
                        return 0, 0
                else:
                    lastCoreBoxMinorCount = coreBoxMinorCount
                if ((majorConnectionCounts[s] < 0) or (coreBoxMajorCounts[sequence[s]] != (
                        majorConnectionCounts[s - 1] + throughCounts[s] + majorConnectionCounts[s]))):
                    print("Can't make core trifurcation between box major axis element counts", coreBoxMajorCounts)
                    return 0, 0

            coreBoxMinorNodesCount = self._segments[0].getCoreBoxMinorNodesCount()
            coreBoxMajorNodesCounts = [self._segments[s].getCoreBoxMajorNodesCount() for s in range(4)]
            # midIndexes = [coreBoxMajorCounts[s] - majorConnectionCounts[s] for s in range(3)]
            midIndexes = [nodesCountAcrossMajor // 2 for nodesCountAcrossMajor in coreBoxMajorNodesCounts]
            halfThroughCounts = [throughCounts[s] // 2 for s in range(4)]

            connectingIndexesList = [[] for s in range(4)]
            for s in range(4):
                s1 = (s - 1) % self._segmentsCount
                s2 = (s + 1) % self._segmentsCount
                shift = (midIndexes[sequence[s2]] - midIndexes[sequence[s1]]) // 2
                midIndex = midIndexes[sequence[s]]
                halfThroughCount = halfThroughCounts[s]
                connectingIndexesList[sequence[s]] = \
                    ([midIndex - halfThroughCount, midIndex + halfThroughCount] if halfThroughCount \
                         else [midIndex + shift])

            self._boxIndexToSegmentNodeList = []
            self._segmentNodeToBoxIndex = \
                [[[None for _ in range(coreBoxMinorNodesCount)] for _ in range(coreBoxMajorNodesCounts[s])]
                 for s in range(self._segmentsCount)]

            for os1 in range(self._segmentsCount):
                os2 = (os1 + 1) % self._segmentsCount
                s1 = sequence[os1]
                s2 = sequence[os2]
                s3 = sequence[(os1 + 2) % self._segmentsCount]
                s4 = sequence[(os1 + 3) % self._segmentsCount]
                aStartNodeIndex = midIndexes[s1]
                nodesCountAcrossMajor1 = coreBoxMajorNodesCounts[s1]
                nodesCountAcrossMajor2 = coreBoxMajorNodesCounts[s2]
                connectingIndexes1 = connectingIndexesList[s1]
                connectingIndexes2 = connectingIndexesList[s2]
                connectingIndexes3 = connectingIndexesList[s3]
                connectingIndexes4 = connectingIndexesList[s4]
                throughIndexes = list(range(connectingIndexes1[0] + 1, connectingIndexes1[1])) \
                    if throughCounts[os1] else [None]

                for m in range((coreBoxMajorCounts[s1] // 2) + 1):
                    m1 = (m + aStartNodeIndex) if self._segmentsIn[s1] else (aStartNodeIndex - m)
                    for n in range(coreBoxMinorNodesCount):
                        indexGroup = None
                        if m1 in throughIndexes:
                            i = m1 - midIndexes[s1]
                            if self._segmentsIn[s1] == self._segmentsIn[s3]:
                                i = -i
                            m3 = midIndexes[s3] + i
                            indexGroup = [[s1, m1, n], [s3, m3, n]] if s1 < s3 else [[s3, m3, n], [s1, m1, n]]
                        elif m1 in connectingIndexes1:
                            if all(v == 0 for v in throughCounts):
                                indexGroup = [[s1, m1, n], [s2, m1, n], [s3, m1, n], [s4, m1, n]]
                            else:
                                if throughCounts[os1]:
                                    m2 = connectingIndexes2[0]
                                    i = connectingIndexes1.index(m1)
                                    if self._segmentsIn[s1] == self._segmentsIn[s3]:
                                        i = 0 if (i == 1) else 1
                                    m3 = connectingIndexes3[i]
                                    indexGroup = [[s1, m1, n], [s2, m2, n], [s3, m3, n]]
                                else:
                                    i = 1 if (throughCounts[os2] and not self._segmentsIn[s2]) else 0
                                    m2 = connectingIndexes2[i]
                                    i = 1 if self._segmentsIn[s4] else 0
                                    m4 = connectingIndexes4[i]
                                    indexGroup = [[s1, m1, n], [s2, m2, n], [s4, m4, n]]
                        else:
                            # get index past last connecting index 1
                            i = (m1 - connectingIndexes1[-1]) if self._segmentsIn[s1] else \
                                (connectingIndexes1[0] - m1)
                            # subtract or add to matching connecting index 2
                            m2 = (connectingIndexes2[0] - i) if self._segmentsIn[s2] else \
                                (connectingIndexes2[-1] + i)
                            indexGroup = [[s1, m1, n], [s2, m2, n]] if s1 < s2 else [[s2, m2, n], [s1, m1, n]]
                        indexGroup = sorted(indexGroup, key=lambda x: (x[0], x[0]), reverse=False)
                        if indexGroup not in self._boxIndexToSegmentNodeList:
                            self._boxIndexToSegmentNodeList.append(indexGroup)
                            boxIndexesCount += 1

            for boxIndex in range(boxIndexesCount):
                segmentNodeList = self._boxIndexToSegmentNodeList[boxIndex]
                for segmentNode in segmentNodeList:
                    s, m, n = segmentNode
                    self._segmentNodeToBoxIndex[s][m][n] = boxIndex

        return rimIndexesCount, boxIndexesCount

    def _optimiseRimIndexes(self, aroundCounts, rimIndexesCount, boxIndexesCount):
        """
        Iterates through a number of permutations to find the most optimised lookup table for rim indexes.
        :param aroundCounts: Number of elements around the tubes.
        :param rimIndexesCount: Total number of rim indexes.
        :param boxIndexesCount: Total number of box indexes, if core is being used.
        """
        # get node indexes giving the lowest sum of distances between adjoining points on outer sampled tubes
        permutationCount = 1
        segmentIncrements = []
        for s in range(self._segmentsCount):
            if self._isCore:
                # core is a regular grid with 2 or 4 permutations -- latter only if same major and minor counts
                segment = self._segments[s]
                if segment.getElementsCountCoreBoxMajor() == segment.getElementsCountCoreBoxMinor():
                    count = 4
                else:
                    count = 2
            else:
                count = aroundCounts[s]
            permutationCount *= count
            segmentIncrements.append(aroundCounts[s] // count)

        minIndexes = None
        minSum = None
        indexes = [0] * self._segmentsCount
        rings = [self._segments[s].getSampledTubeCoordinatesRing(0, -1 if self._segmentsIn[s] else 0)
                 for s in range(self._segmentsCount)]
        for p in range(permutationCount):
            sum = 0.0
            for rimIndex in range(rimIndexesCount):
                segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
                sCount = len(segmentNodeList)
                for i in range(sCount - 1):
                    s1, n1 = segmentNodeList[i]
                    nodeIndex1 = (n1 + indexes[s1]) % aroundCounts[s1]
                    x1 = rings[s1][nodeIndex1]
                    for j in range(i + 1, sCount):
                        s2, n2 = segmentNodeList[j]
                        nodeIndex2 = (n2 + indexes[s2]) % aroundCounts[s2]
                        # print(nodeIndex2, aroundCounts[s2])
                        x2 = rings[s2][nodeIndex2]
                        sum += magnitude([x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]])
            if (minSum is None) or (sum < minSum):
                minIndexes = copy.copy(indexes)
                minSum = sum
            # permute through indexes:
            for s in range(self._segmentsCount):
                indexes[s] += segmentIncrements[s]
                if indexes[s] < aroundCounts[s]:
                    break
                indexes[s] = 0

        # offset rim node indexes by minIndexes
        for rimIndex in range(rimIndexesCount):
            segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
            for segmentNode in segmentNodeList:
                s, n = segmentNode
                nodeIndex = (n + minIndexes[s]) % aroundCounts[s]
                self._segmentNodeToRimIndex[s][nodeIndex] = rimIndex
                segmentNode[1] = nodeIndex

        # rotate box grid
        if self._isCore:
            segmentRotationCases = []
            segmentMajorBoxSizes = []
            segmentMinorBoxSizes = []
            elementsCountTransition = self._segments[0].getElementsCountTransition()
            for s in range(self._segmentsCount):
                segment = self._segments[s]
                segmentRotationCases.append(4 * minIndexes[s] // aroundCounts[s])
                segmentMajorBoxSizes.append(segment.getElementsCountCoreBoxMajor())
                segmentMinorBoxSizes.append(segment.getElementsCountCoreBoxMinor())
            for boxIndex in range(boxIndexesCount):
                segmentNodeList = self._boxIndexToSegmentNodeList[boxIndex]
                for segmentNode in segmentNodeList:
                    s, m, n = segmentNode
                    if segmentRotationCases[s] > 0:
                        segment = self._segments[s]
                        if segmentRotationCases[s] == 1:
                            m2 = n
                            n2 = segmentMajorBoxSizes[s] - m
                        elif segmentRotationCases[s] == 2:
                            m2 = segmentMajorBoxSizes[s] - m
                            n2 = segmentMinorBoxSizes[s] - n
                        else:  # segmentRotationCases[s] == 3:
                            m2 = segmentMinorBoxSizes[s] - n
                            n2 = m
                        self._segmentNodeToBoxIndex[s][m2][n2] = boxIndex
                        segmentNode[1] = m2
                        segmentNode[2] = n2

    def sample(self, targetElementLength):
        """
        Blend sampled d2 derivatives across 2-segment junctions with the same version.
        Sample junction coordinates between second-from-end segment coordinates.
        :param targetElementLength: Ignored here as always 2 elements across junction.
        """
        if self._segmentsCount == 1:
            return
        if self._segmentsCount == 2:
            TubeNetworkMeshSegment.blendSampledCoordinates(
                self._segments[0], -1 if self._segmentsIn[0] else 0,
                self._segments[1], -1 if self._segmentsIn[1] else 0)
            return

        aroundCounts = [segment.getElementsCountAround() for segment in self._segments]
        coreBoxMajorCounts = [segment.getElementsCountCoreBoxMajor() for segment in self._segments]

        # determine junction sequence
        self._determineJunctionSequence()

        if self._segmentsCount == 3:
            rimIndexesCount, boxIndexesCount = self._sampleBifurcation(aroundCounts, coreBoxMajorCounts)

        elif self._segmentsCount == 4:
            rimIndexesCount, boxIndexesCount = self._sampleTrifurcation(aroundCounts, coreBoxMajorCounts)

        else:
            print("Tube network mesh not implemented for", self._segmentsCount, "segments at junction")
            return

        if not rimIndexesCount:
            return

        # optimise rim indexes
        self._optimiseRimIndexes(aroundCounts, rimIndexesCount, boxIndexesCount)

        # sample rim coordinates
        elementsCountTransition = self._segments[0].getElementsCountTransition()
        nodesCountRim = self._segments[0].getNodesCountRim()
        rx, rd1, rd2, rd3 = [
            [[None] * rimIndexesCount for _ in range(nodesCountRim)] for i in range(4)]
        self._rimCoordinates = (rx, rd1, rd2, rd3)
        for n3 in range(nodesCountRim):
            n3p = n3 - (elementsCountTransition - 1) if self._isCore else n3
            for rimIndex in range(rimIndexesCount):
                segmentNodeList = self._rimIndexToSegmentNodeList[rimIndex]
                # segments have been ordered from lowest to highest s index
                segmentsParameterLists = []
                # print('segmentNodeList =', segmentNodeList)
                for s, n1 in segmentNodeList:
                    # print('s =', s, 'n1 =', n1)
                    if self._isCore and n3 < (elementsCountTransition - 1):
                        segmentsParameterLists.append(
                            self._segments[s].getTransitionCoordinatesListAlong(
                                n1, [-2, -1] if self._segmentsIn[s] else [1, 0], n3))
                    else:
                        segmentsParameterLists.append(
                            self._segments[s].getRimCoordinatesListAlong(
                                n1, [-2, -1] if self._segmentsIn[s] else [1, 0], n3p))
                rx[n3][rimIndex], rd1[n3][rimIndex], rd2[n3][rimIndex], rd3[n3][rimIndex] = \
                    self._sampleMidPoint(segmentsParameterLists)

        # sample box coordinates
        if self._isCore:
            bx, bd1, bd2, bd3 = [[None] * boxIndexesCount for _ in range(4)]
            self._boxCoordinates = (bx, bd1, bd2, bd3)
            for boxIndex in range(boxIndexesCount):
                segmentNodeList = self._boxIndexToSegmentNodeList[boxIndex]
                segmentsParameterLists = []
                for s, n3, n1 in segmentNodeList:
                    segmentsParameterLists.append(
                        self._segments[s].getBoxCoordinatesListAlong(
                            n1, [-2, -1] if self._segmentsIn[s] else [1, 0], n3))
                bx[boxIndex], bd1[boxIndex], bd2[boxIndex], bd3[boxIndex] = \
                    self._sampleMidPoint(segmentsParameterLists)

    def _createBoxBoundaryNodeIdsList(self, s):
        """
        Creates a list (in a circular format similar to other rim node id lists) of core box node ids that are
        located at the boundary of the core. This list is used to easily stitch inner rim nodes with box nodes.
        Used specifically for solid core at the junction.
        :param s: Index for identifying segments.
        :return: A list of box node ids stored in a circular format, and a lookup list that translates indexes used in
        boxBoundaryNodeIds list to indexes that can be used in boxCoordinates list.
        """
        boxBoundaryNodeIds = []
        boxBoundaryNodeToBoxId = []
        elementsCountCoreBoxMajor = self._segments[s].getElementsCountCoreBoxMajor()
        elementsCountCoreBoxMinor = self._segments[s].getElementsCountCoreBoxMinor()
        elementsCountTransition = self._segments[s].getElementsCountAcrossTransition()
        coreBoxMajorNodesCount = elementsCountCoreBoxMajor + 1
        coreBoxMinorNodesCount = elementsCountCoreBoxMinor + 1

        for n3 in range(coreBoxMajorNodesCount):
            if n3 == 0 or n3 == coreBoxMajorNodesCount - 1:
                ids = self._boxNodeIds[s][n3] if n3 == 0 else self._boxNodeIds[s][n3][::-1]
                n1List = list(range(coreBoxMinorNodesCount)) if n3 == 0 else (
                    list(range(coreBoxMinorNodesCount - 1, -1, -1)))
                boxBoundaryNodeIds += [ids[c] for c in range(coreBoxMinorNodesCount)]
                for n1 in n1List:
                    boxBoundaryNodeToBoxId.append([n3, n1])
            else:
                for n1 in [-1, 0]:
                    boxBoundaryNodeIds.append(self._boxNodeIds[s][n3][n1])
                    boxBoundaryNodeToBoxId.append([n3, n1])

        start = elementsCountCoreBoxMajor - 2
        idx = elementsCountCoreBoxMinor + 2
        for n in range(int(start), -1, -1):
            boxBoundaryNodeIds.append(boxBoundaryNodeIds.pop(idx + 2 * n))
            boxBoundaryNodeToBoxId.append(boxBoundaryNodeToBoxId.pop(idx + 2 * n))

        nloop = elementsCountCoreBoxMinor // 2
        for _ in range(nloop):
            boxBoundaryNodeIds.insert(len(boxBoundaryNodeIds), boxBoundaryNodeIds.pop(0))
            boxBoundaryNodeToBoxId.insert(len(boxBoundaryNodeToBoxId),
                                              boxBoundaryNodeToBoxId.pop(0))

        return boxBoundaryNodeIds, boxBoundaryNodeToBoxId

    def _getBoxCoordinates(self, n1):
        return (self._boxCoordinates[0][n1], self._boxCoordinates[1][n1],
                self._boxCoordinates[2][n1], self._boxCoordinates[3][n1])

    def _getRimCoordinates(self, n1):
        return (self._rimCoordinates[0][0][n1], self._rimCoordinates[1][0][n1],
                self._rimCoordinates[2][0][n1], self._rimCoordinates[3][0][n1])

    def _generateBoxElements(self, s, n2, mesh, elementtemplate, coordinates, segment, generateData):
        """
        Blackbox function for generating core box elements at a junction.
        """
        annotationMeshGroups = generateData.getAnnotationMeshGroups(segment.getAnnotationTerms())
        boxElementsCountAcrossMinor = self._segments[0].getCoreBoxMinorNodesCount() - 1
        boxElementsCountAcrossMajor = [self._segments[s].getCoreBoxMajorNodesCount() - 1
                                       for s in range(self._segmentsCount)]
        coreBoxMajorCounts = [segment.getElementsCountCoreBoxMajor() for segment in self._segments]
        is6WayTriplePoint = ((self._segmentsCount == 3) and ((max(coreBoxMajorCounts) // 2) == min(coreBoxMajorCounts)))

        eftList = [[None] * boxElementsCountAcrossMinor for _ in range(boxElementsCountAcrossMajor[s])]
        scalefactorsList = [[None] * boxElementsCountAcrossMinor for _ in range(boxElementsCountAcrossMajor[s])]

        nodeLayout6Way = generateData.getNodeLayout6Way()
        nodeLayout8Way = generateData.getNodeLayout8Way()
        nodeLayoutFlipD2 = generateData.getNodeLayoutFlipD2()

        e2 = n2 if self._segmentsIn[s] else 0
        for e3 in range(boxElementsCountAcrossMajor[s]):
            for e1 in range(boxElementsCountAcrossMinor):
                e3p = (e3 + 1)
                nids, nodeParameters, nodeLayouts = [], [], []
                # get identifier early to aid debugging
                elementIdentifier = generateData.nextElementIdentifier()
                for n1 in [e1, e1 + 1]:
                    for n3 in [e3, e3p]:
                        nids.append(segment.getBoxNodeIds(n1, n2, n3))
                        boxCoordinates = segment.getBoxCoordinates(n1, n2, n3)
                        nodeParameters.append(boxCoordinates)
                        nodeLayouts.append(None)
                    for n3 in [e3, e3p]:
                        boxIndex = self._segmentNodeToBoxIndex[s][n3][n1]
                        nids.append(self._boxNodeIds[s][n3][n1])
                        nodeParameters.append(self._getBoxCoordinates(boxIndex))
                        segmentNodesCount = len(self._boxIndexToSegmentNodeList[boxIndex])
                        if is6WayTriplePoint and (segmentNodesCount == 3) and self._segmentsCount == 3:
                            nodeLayouts.append(nodeLayout6Way)
                        elif self._segmentsIn[s] and (segmentNodesCount == 3) and self._segmentsCount == 4:
                            location = 1 if e3 < boxElementsCountAcrossMajor[s] // 2 else 2
                            nodeLayoutTrifurcation = generateData.getNodeLayoutTrifurcation(location)
                            nodeLayouts.append(nodeLayout6Way if self._sequence == [0, 1, 3, 2] else
                                               nodeLayoutTrifurcation)
                        else:
                            nodeLayouts.append(nodeLayoutFlipD2 if (segmentNodesCount == 2) else
                                               nodeLayout6Way if (segmentNodesCount == 3) else
                                               nodeLayout8Way)

                    if not self._segmentsIn[s]:
                        for a in [nids, nodeParameters, nodeLayouts]:
                            a[-4], a[-2] = a[-2], a[-4]
                            a[-3], a[-1] = a[-1], a[-3]
                eft = eftList[e3][e1]
                scalefactors = scalefactorsList[e3][e1]
                if not eft:
                    eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                    eftList[e3][e1] = eft
                    scalefactorsList[e3][e1] = scalefactors
                elementtemplate.defineField(coordinates, -1, eft)
                element = mesh.createElement(elementIdentifier, elementtemplate)
                element.setNodesByIdentifier(eft, nids)
                if scalefactors:
                    element.setScaleFactors(eft, scalefactors)
                segment.setBoxElementId(e1, e2, e3, elementIdentifier)
                for annotationMeshGroup in annotationMeshGroups:
                    annotationMeshGroup.addElement(element)

    def _generateTransitionElements(self, s, n2, mesh, elementtemplate, coordinates, segment, generateData,
                                    elementsCountAround, boxBoundaryNodeIds, boxBoundaryNodeToBoxId):
        """
        Blackbox function for generating first row of core transition elements after box at a junction.
        """
        annotationMeshGroups = generateData.getAnnotationMeshGroups(segment.getAnnotationTerms())
        coreBoxMinorNodesCount = self._segments[0].getCoreBoxMinorNodesCount()
        coreBoxMajorNodesCounts = [self._segments[s].getCoreBoxMajorNodesCount() for s in
                                 range(self._segmentsCount)]
        coreBoxMajorCounts = [segment.getElementsCountCoreBoxMajor() for segment in self._segments]

        triplePointIndexesList = segment.getTriplePointIndexes()
        triplePointIndexesList.sort()
        elementsCountTransition = self._segments[0].getElementsCountTransition()
        maxCoreBoxMajorCount = max(coreBoxMajorCounts)
        maxMajorSegment = coreBoxMajorCounts.index(maxCoreBoxMajorCount)
        # whether there are bifurcation core transition triple points on the corners of any box
        is6WayTriplePoint = ((self._segmentsCount == 3) and
                             (maxCoreBoxMajorCount == (coreBoxMajorCounts[maxMajorSegment - 1] +
                                                       coreBoxMajorCounts[maxMajorSegment - 2])))

        nodeLayout6Way = generateData.getNodeLayout6Way()
        nodeLayout8Way = generateData.getNodeLayout8Way()
        nodeLayoutFlipD2 = generateData.getNodeLayoutFlipD2()
        nodeLayoutTransition = generateData.getNodeLayoutTransition()

        e2 = n2 if self._segmentsIn[s] else 0
        for e1 in range(elementsCountAround):
            nids, nodeParameters, nodeLayouts = [], [], []
            n1p = (e1 + 1) % elementsCountAround
            oLocation = segment.getTriplePointLocation(e1)
            # get identifier early to aid debugging
            elementIdentifier = generateData.nextElementIdentifier()

            # outside of core box
            for n1 in [e1, n1p]:
                nids.append(segment.getBoxBoundaryNodeIds(n1, n2))
                n3c, n1c = segment.getBoxBoundaryNodeToBoxId(n1, n2)
                nodeParameters.append(segment.getBoxCoordinates(n1c, n2, n3c))
                nodeLayoutTransitionTriplePoint = (
                    generateData.getNodeLayoutTransitionTriplePoint(oLocation))
                nodeLayouts.append(nodeLayoutTransitionTriplePoint if n1 in triplePointIndexesList
                                   else nodeLayoutTransition)

            for n1 in [e1, n1p]:
                nid = boxBoundaryNodeIds[n1]
                nids.append(nid)
                n3c, n1c = boxBoundaryNodeToBoxId[n1]
                boxIndex = self._segmentNodeToBoxIndex[s][n3c][n1c]
                nodeParameters.append(self._getBoxCoordinates(boxIndex))
                segmentNodesCount = len(self._boxIndexToSegmentNodeList[boxIndex])
                if segmentNodesCount == 3:  # 6-way node
                    if is6WayTriplePoint:
                        top = triplePointIndexesList[0] <= n1 <= triplePointIndexesList[1]
                        bottom = triplePointIndexesList[2] <= n1 <= triplePointIndexesList[3]
                        if top or bottom:
                            nodeLayout = generateData.getNodeLayoutBifurcation6WayTriplePoint(
                                self._segmentsIn, self._sequence, maxMajorSegment, top)
                        else:
                            nodeLayout = nodeLayout6Way
                    elif self._segmentsCount == 4 and self._segmentsIn[s]:  # Trifurcation case
                        location = \
                            1 if (e1 < elementsCountAround // 4) or (e1 >= 3 * elementsCountAround // 4) else 2
                        nodeLayout = (nodeLayout6Way if self._sequence == [0, 1, 3, 2] else
                                      generateData.getNodeLayoutTrifurcation(location))
                    else:
                        nodeLayout = nodeLayout6Way
                elif segmentNodesCount == 4:  # 8-way node
                    nodeLayout = nodeLayout8Way
                elif n1 in triplePointIndexesList:  # triple-point node
                    location = oLocation
                    if ((s < segmentNodesCount - 1) and n1 == n1p or
                            not any(nid in sl for sl in self._triplePointLocationsList)):
                        self._triplePointLocationsList.append([nid, location])
                    if s > 0:
                        for sl in self._triplePointLocationsList:
                            location = sl[1] if nid == sl[0] else location
                    nodeLayout = generateData.getNodeLayoutTransitionTriplePoint(location)
                else:
                    nodeLayout = nodeLayoutTransition
                nodeLayouts.append(nodeLayout)
            if not self._segmentsIn[s]:
                for a in [nids, nodeParameters, nodeLayouts]:
                    a[-4], a[-2] = a[-2], a[-4]
                    a[-3], a[-1] = a[-1], a[-3]

            # first rim node (either transition or first shell)
            for n1 in [e1, n1p]:
                nids.append(segment.getRimNodeId(n1, n2, 0))
                nodeParameters.append(segment.getRimCoordinates(n1, n2, 0))
                nodeLayouts.append(None)
            for n1 in [e1, n1p]:
                rimIndex = self._segmentNodeToRimIndex[s][n1]
                nids.append(self._rimNodeIds[0][rimIndex])
                nodeParameters.append(self._getRimCoordinates(rimIndex))
                segmentNodesCount = len(self._rimIndexToSegmentNodeList[rimIndex])
                nodeLayouts.append(nodeLayoutFlipD2 if (segmentNodesCount == 2) else
                                   nodeLayout6Way if (segmentNodesCount == 3) else
                                   nodeLayout8Way)
            if not self._segmentsIn[s]:
                for a in [nids, nodeParameters, nodeLayouts]:
                    a[-4], a[-2] = a[-2], a[-4]
                    a[-3], a[-1] = a[-1], a[-3]

            eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
            if elementsCountTransition == 1:
                eft, scalefactors = generateData.resolveEftCoreBoundaryScaling(
                    eft, scalefactors, nodeParameters, nids, segment.getCoreBoundaryScalingMode())
            elementtemplate.defineField(coordinates, -1, eft)
            element = mesh.createElement(elementIdentifier, elementtemplate)
            element.setNodesByIdentifier(eft, nids)
            if scalefactors:
                element.setScaleFactors(eft, scalefactors)
            segment.setRimElementId(e1, e2, 0, elementIdentifier)
            for annotationMeshGroup in annotationMeshGroups:
                annotationMeshGroup.addElement(element)

    def getRimIndexNodeLayout(self, generateData, rimIndex):
        """
        Get specific node layout for rim index based on segment nodes count.
        """
        segmentNodesCount = len(self._rimIndexToSegmentNodeList[rimIndex])
        if segmentNodesCount == 2:
            return generateData.getNodeLayoutFlipD2()
        if segmentNodesCount == 4:
            return generateData.getNodeLayout8Way()
        # 3-segments
        for segment in self._segments:
            nodeLayout = segment.getRimIndexNodeLayoutSpecial(generateData, rimIndex)
            if nodeLayout:
                return nodeLayout
        return generateData.getNodeLayout6Way()

    def generateMesh(self, generateData: TubeNetworkMeshGenerateData):
        if generateData.isShowTrimSurfaces():
            dimension = generateData.getMeshDimension()
            nodeIdentifier, elementIdentifier = generateData.getNodeElementIdentifiers()
            faceIdentifier = elementIdentifier if (dimension == 2) else None
            for s in range(self._segmentsCount):
                for trimSurface in self._trimSurfaces[s]:
                    if trimSurface:
                        annotationGroup = generateData.getNewTrimAnnotationGroup()
                        nodeIdentifier, faceIdentifier = \
                            trimSurface.generateMesh(generateData.getRegion(), nodeIdentifier, faceIdentifier,
                                                     group_name=annotationGroup.getName())
                        if self._useOuterTrimSurfaces:
                            break
            if dimension == 2:
                elementIdentifier = faceIdentifier
            generateData.setNodeElementIdentifiers(nodeIdentifier, elementIdentifier)

        if self._segmentsCount < 3:
            return

        if (not self._rimCoordinates) or (self._isCore and not self._boxCoordinates):
            # incompatible number around or across core
            return

        rimIndexesCount = len(self._rimIndexToSegmentNodeList)
        nodesCountRim = self._segments[0].getNodesCountRim()
        if self._rimCoordinates:
            self._rimNodeIds = [[None] * rimIndexesCount for _ in range(nodesCountRim)]

        if self._boxCoordinates:
            coreBoxMinorNodesCount = self._segments[0].getCoreBoxMinorNodesCount()
            self._boxNodeIds = [
                [[None for n in range(coreBoxMinorNodesCount)]
                 for m in range(self._segments[s].getCoreBoxMajorNodesCount())]
                for s in range(self._segmentsCount)]

        coordinates = generateData.getCoordinates()
        fieldcache = generateData.getFieldcache()
        nodes = generateData.getNodes()
        nodetemplate = generateData.getNodetemplate()
        meshDimension = generateData.getMeshDimension()
        d3Defined = (meshDimension == 3) and not generateData.isLinearThroughShell()

        # nodes and elements are generated in order of segments
        for s in range(self._segmentsCount):
            segment = self._segments[s]
            elementsCountAlong = segment.getSampledElementsCountAlong()
            n2 = (elementsCountAlong - 1) if self._segmentsIn[s] else 1
            segment.generateMesh(generateData, n2Only=n2)
            elementsCountAround = segment.getElementsCountAround()

            # Create nodes
            if self._isCore:
                # create box nodes
                bx, bd1, bd2, bd3 = (self._boxCoordinates[0], self._boxCoordinates[1],
                                     self._boxCoordinates[2], self._boxCoordinates[3])
                coreBoxMajorNodesCount = self._segments[s].getCoreBoxMajorNodesCount()
                coreBoxMinorNodesCount = self._segments[s].getCoreBoxMinorNodesCount()
                for n3 in range(coreBoxMajorNodesCount):
                    for n1 in range(coreBoxMinorNodesCount):
                        boxIndex = self._segmentNodeToBoxIndex[s][n3][n1]
                        segmentNodeList = self._boxIndexToSegmentNodeList[boxIndex]
                        nodeIdentifiersCheck = []
                        for segmentNodes in segmentNodeList:
                            sp, n3p, n1p = segmentNodes
                            nodeIdentifiersCheck.append(self._boxNodeIds[sp][n3p][n1p])
                        if nodeIdentifiersCheck.count(None) == len(nodeIdentifiersCheck):
                            nodeIdentifier = generateData.nextNodeIdentifier()
                            node = nodes.createNode(nodeIdentifier, nodetemplate)
                            fieldcache.setNode(node)

                            for nodeValue, bValue in zip([Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                                                          Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3],
                                                         [bx, bd1, bd2, bd3]):
                                coordinates.setNodeParameters(fieldcache, -1, nodeValue, 1, bValue[boxIndex])
                        else:
                            nodeIdentifier = next(id for id in nodeIdentifiersCheck if id is not None)

                        self._boxNodeIds[s][n3][n1] = nodeIdentifier

            # create rim nodes (including core transition nodes)
            for n3 in range(nodesCountRim):
                rx = self._rimCoordinates[0][n3]
                rd1 = self._rimCoordinates[1][n3]
                rd2 = self._rimCoordinates[2][n3]
                rd3 = self._rimCoordinates[3][n3] if d3Defined else None
                layerNodeIds = self._rimNodeIds[n3]
                for n1 in range(elementsCountAround):
                    rimIndex = self._segmentNodeToRimIndex[s][n1]
                    nodeIdentifier = layerNodeIds[rimIndex]
                    if nodeIdentifier is not None:
                        continue
                    nodeIdentifier = generateData.nextNodeIdentifier()
                    node = nodes.createNode(nodeIdentifier, nodetemplate)
                    fieldcache.setNode(node)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, rx[rimIndex])
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, rd1[rimIndex])
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, rd2[rimIndex])
                    if rd3:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, rd3[rimIndex])
                    layerNodeIds[rimIndex] = nodeIdentifier

            segment.generateJunctionRimElements(self, generateData)


class TubeNetworkMeshBuilder(NetworkMeshBuilder):
    """
    Builds contiguous tube network meshes with smooth element size transitions at junctions, optionally with solid core.
    """

    def __init__(self, networkMesh: NetworkMesh, targetElementDensityAlongLongestSegment: float,
                 layoutAnnotationGroups: list=[], annotationElementsCountsAlong: list=[],
                 defaultElementsCountAround: int=8, annotationElementsCountsAround: list=[],
                 elementsCountThroughShell: int=1, isCore=False, elementsCountTransition: int=1,
                 defaultElementsCountCoreBoxMinor: int=2, annotationElementsCountsCoreBoxMinor: list=[],
                 defaultCoreBoundaryScalingMode=1, annotationCoreBoundaryScalingMode=[],
                 useOuterTrimSurfaces=True):
        """
        Builds contiguous tube network meshes with smooth element size transitions at junctions, optionally with solid
        core.
        :param networkMesh: Description of the topology of the network layout.
        :param targetElementDensityAlongLongestSegment: Real value which longest segment path in network is divided by
        to get target element length, which is used to determine numbers of elements along except when set for a segment
        through annotationElementsCountsAlong.
        :param layoutAnnotationGroups: Annotation groups defined on the layout to mirror on the final mesh.
        :param annotationElementsCountsAlong: List in same order as layoutAnnotationGroups, specifying fixed number of
        elements along segment with any elements in the annotation group. Client must ensure exclusive map from
        segments. Groups with zero value or past end of this list use the targetElementDensityAlongLongestSegment.
        :param defaultElementsCountAround: Number of elements around segments to use unless overridden by
        annotationElementsCountsAround.
        :param annotationElementsCountsAround: List in same order as layoutAnnotationGroups, specifying fixed number of
        elements around segments with any elements in the annotation group. Client must ensure exclusive map from
        segments. Groups with zero value or past end of this list use the defaultElementsCountAround.
        :param elementsCountThroughShell: Number of elements through shell >= 1.
        :param isCore: Set to True to define solid core box and transition elements.
        :param elementsCountTransition: Number of rows of elements transitioning between core box and shell >= 1.
        :param defaultElementsCountCoreBoxMinor: Number of elements across the core box in the minor/d3 direction.
        :param annotationElementsCountsCoreBoxMinor: List in same order as layoutAnnotationGroups, specifying numbers of
        elements across core box minor/d3 direction for segments with any elements in the annotation group. Client must
        ensure exclusive map from segments. Groups with zero value or past end of this list use the
        defaultElementsCountCoreBoxMinor.
        :param defaultCoreBoundaryScalingMode: Mode 1=scalefactor, 2=version.
        :param annotationCoreBoundaryScalingMode: Boundary mode 1 or 2 in order of layoutAnnotationGroups,
        or 0 to use default.
        :param useOuterTrimSurfaces: Set to False to use separate trim surfaces on inner and outer tubes. Ignored if
        no inner path.
        """
        super(TubeNetworkMeshBuilder, self).__init__(
            networkMesh, targetElementDensityAlongLongestSegment, layoutAnnotationGroups, annotationElementsCountsAlong)
        self._defaultElementsCountAround = defaultElementsCountAround
        self._annotationElementsCountsAround = annotationElementsCountsAround
        self._elementsCountThroughShell = elementsCountThroughShell
        self._isCore = isCore
        self._elementsCountTransition = elementsCountTransition
        self._defaultElementsCountCoreBoxMinor = defaultElementsCountCoreBoxMinor
        self._annotationElementsCountsCoreBoxMinor = annotationElementsCountsCoreBoxMinor
        self._defaultCoreBoundaryScalingMode = defaultCoreBoundaryScalingMode
        self._annotationCoreBoundaryScalingMode = annotationCoreBoundaryScalingMode
        layoutFieldmodule = self._layoutRegion.getFieldmodule()
        self._layoutInnerCoordinates = layoutFieldmodule.findFieldByName("inner coordinates").castFiniteElement()
        if not self._layoutInnerCoordinates.isValid():
            self._layoutInnerCoordinates = None
        self._useOuterTrimSurfaces = useOuterTrimSurfaces if self._layoutInnerCoordinates else False

    def createSegment(self, networkSegment):
        pathParametersList = [get_nodeset_path_ordered_field_parameters(
            self._layoutNodes, self._layoutCoordinates, pathValueLabels,
            networkSegment.getNodeIdentifiers(), networkSegment.getNodeVersions())]
        if self._layoutInnerCoordinates:
            pathParametersList.append(get_nodeset_path_ordered_field_parameters(
                self._layoutNodes, self._layoutInnerCoordinates, pathValueLabels,
                networkSegment.getNodeIdentifiers(), networkSegment.getNodeVersions()))
        elementsCountAround = self._defaultElementsCountAround
        elementsCountCoreBoxMinor = self._defaultElementsCountCoreBoxMinor

        coreBoundaryScalingMode = self._defaultCoreBoundaryScalingMode
        i = 0
        for layoutAnnotationGroup in self._layoutAnnotationGroups:
            if i >= len(self._annotationElementsCountsAround):
                break
            if self._annotationElementsCountsAround[i] > 0:
                if networkSegment.hasLayoutElementsInMeshGroup(layoutAnnotationGroup.getMeshGroup(self._layoutMesh)):
                    elementsCountAround = self._annotationElementsCountsAround[i]
                    break
            i += 1
        if self._isCore:
            annotationElementsCountAcrossMinor = []
            i = 0
            for layoutAnnotationGroup in self._layoutAnnotationGroups:
                if i >= len(self._annotationElementsCountsCoreBoxMinor):
                    break
                if self._annotationElementsCountsCoreBoxMinor[i] > 0:
                    if networkSegment.hasLayoutElementsInMeshGroup(
                            layoutAnnotationGroup.getMeshGroup(self._layoutMesh)):
                        elementsCountCoreBoxMinor = self._annotationElementsCountsCoreBoxMinor[i]
                        break
                i += 1
            i = 0
            for layoutAnnotationGroup in self._layoutAnnotationGroups:
                if i >= len(self._annotationCoreBoundaryScalingMode):
                    break
                if self._annotationCoreBoundaryScalingMode[i] > 0:
                    if networkSegment.hasLayoutElementsInMeshGroup(
                            layoutAnnotationGroup.getMeshGroup(self._layoutMesh)):
                        coreBoundaryScalingMode = self._annotationCoreBoundaryScalingMode[i]
                        break
                i += 1
        return TubeNetworkMeshSegment(networkSegment, pathParametersList, elementsCountAround,
                                      self._elementsCountThroughShell, self._isCore, elementsCountCoreBoxMinor,
                                      self._elementsCountTransition, coreBoundaryScalingMode)

    def createJunction(self, inSegments, outSegments):
        """
        :param inSegments: List of inward TubeNetworkMeshSegment.
        :param outSegments: List of outward TubeNetworkMeshSegment.
        :return: A TubeNetworkMeshJunction.
        """
        return TubeNetworkMeshJunction(inSegments, outSegments, self._useOuterTrimSurfaces)

    def generateMesh(self, generateData):
        super(TubeNetworkMeshBuilder, self).generateMesh(generateData)
        # build core, shell
        if self._isCore:
            coreMeshGroup = generateData.getCoreMeshGroup()
            shellMeshGroup = generateData.getShellMeshGroup()
            for networkSegment in self._networkMesh.getNetworkSegments():
                segment = self._segments[networkSegment]
                segment.addCoreElementsToMeshGroup(coreMeshGroup)
                segment.addShellElementsToMeshGroup(shellMeshGroup)


class BodyTubeNetworkMeshBuilder(TubeNetworkMeshBuilder):
    """
    Specialization of TubeNetworkMeshBuilder adding annotations for left, right, dorsal, ventral regions.
    Requires network layout to follow these conventions:
    - consistently annotates fully left or right features with names including "left" or "right", respectively.
    - along central body, +d2 direction is left, -d2 direction is right.
    - +d3 direction is ventral, -d3 is dorsal.
    """

    def generateMesh(self, generateData):
        super(BodyTubeNetworkMeshBuilder, self).generateMesh(generateData)
        # build left, right, dorsal, ventral annotation groups
        leftMeshGroup = generateData.getLeftMeshGroup()
        rightMeshGroup = generateData.getRightMeshGroup()
        dorsalMeshGroup = generateData.getDorsalMeshGroup()
        ventralMeshGroup = generateData.getVentralMeshGroup()
        for networkSegment in self._networkMesh.getNetworkSegments():
            segment = self._segments[networkSegment]
            annotationTerms = segment.getAnnotationTerms()
            for annotationTerm in annotationTerms:
                if "left" in annotationTerm[0]:
                    segment.addAllElementsToMeshGroup(leftMeshGroup)
                    break
                if "right" in annotationTerm[0]:
                    segment.addAllElementsToMeshGroup(rightMeshGroup)
                    break
            else:
                # segment on main axis
                segment.addSideD2ElementsToMeshGroup(False, leftMeshGroup)
                segment.addSideD2ElementsToMeshGroup(True, rightMeshGroup)
            segment.addSideD3ElementsToMeshGroup(False, ventralMeshGroup)
            segment.addSideD3ElementsToMeshGroup(True, dorsalMeshGroup)


class TubeEllipseGenerator:
    """
    Generates tube ellipse curves with even-sized elements with specified radius, phase angle,
    for any number of user-supplied parameters and number around.
    """

    def __init__(self, radius=1.0, phaseAngle=0.0):
        self._radius = radius
        self._phaseAngle = phaseAngle

        # sample around circle, later scaled by ellipse parameters and resampled to get even spacing in geometric space
        self._circlePointCount = 16
        self._cx = []
        self._cd = []
        aroundScale = 2.0 * math.pi / self._circlePointCount
        for q in range(self._circlePointCount):
            theta = phaseAngle + 2.0 * math.pi * q / self._circlePointCount
            x = [radius * math.cos(theta), radius * math.sin(theta)]
            self._cx.append(x)
            d = [-x[1] * aroundScale, x[0] * aroundScale]
            self._cd.append(d)

    def generate(self, px, pd1, pd2, pd12, pd3, pd13, elementsCountAround, d2Scale=1.0):
        """
        Generate a single row of 2-D ellipse parameters for the tube.
        :param px: Centre of ellipse.
        :param pd1: Derivative along centre of tube.
        :param pd2: Major axis of ellipse.
        :param pd12: Rate of change of major axis along tube.
        :param pd3: Minor axis of ellipse.
        :param pd13: Rate of change of minor axis along tube.
        :param d2Scale: Scale to apply to derivative along the tube.
        :return: 2-D tube ellipse row parameters ex, ed1, ed2, ed12
        """
        tx = []
        td1 = []
        for q in range(self._circlePointCount):
            xi2, xi3 = self._cx[q]
            tx.append([(px[c] + xi2 * pd2[c] + xi3 * pd3[c]) for c in range(3)])
            dxi2, dxi3 = self._cd[q]
            td1.append([(dxi2 * pd2[c] + dxi3 * pd3[c]) for c in range(3)])
        # smooth to get reasonable derivative magnitudes
        td1 = smoothCubicHermiteDerivativesLoop(tx, td1, fixAllDirections=True)
        # resample to get evenly spaced points around loop, temporarily adding start point to end
        ex, ed1, pe, pxi, psf = sampleCubicHermiteCurvesSmooth(tx + tx[:1], td1 + td1[:1], elementsCountAround)
        exi, edxi = interpolateSampleCubicHermite(self._cx + self._cx[:1], self._cd + self._cd[:1], pe, pxi, psf)
        ex.pop()
        ed1.pop()
        exi.pop()
        edxi.pop()

        # check closeness of x(exi[i]) to ex[i]
        # find nearest xi2, xi3 if above is finite error
        # a small but non-negligible error, but results look fine so not worrying
        # dxi = []
        # for i in range(len(ex)):
        #     xi2 = exi[i][0]
        #     xi3 = exi[i][1]
        #     xi23 = xi2 * xi3
        #     x = [(cx[c] + xi2 * cd2[c] + xi3 * cd3[c]) for c in range(3)]
        #     dxi.append(sub(x, ex[i]))
        # print("error", p, "=", [magnitude(v) for v in dxi])

        # calculate d2, d12 at exi
        mag1 = magnitude(pd1)
        mag2 = 0.5 * (magnitude(pd12) + magnitude(pd13))
        d2ScaleFinal = d2Scale / (math.sqrt(mag1 * mag1 + mag2 * mag2))
        ed2 = []
        ed12 = []
        for i in range(len(ex)):
            xi2, xi3 = exi[i]
            ed2.append([d2ScaleFinal * (pd1[c] + xi2 * pd12[c] + xi3 * pd13[c]) for c in range(3)])
            dxi2, dxi3 = edxi[i]
            ed12.append([d2ScaleFinal * (dxi2 * pd12[c] + dxi3 * pd13[c]) for c in range(3)])

        return ex, ed1, ed2, ed12


def getPathRawTubeCoordinates(pathParameters, elementsCountAround, radius=1.0, phaseAngle=0.0):
    """
    Generate coordinates around and along a tube in parametric space around the path parameters,
    at xi2^2 + xi3^2 = radius at the same density as path parameters.
    :param pathParameters: List over nodes of 6 parameters vectors [cx, cd1, cd2, cd12, cd3, cd13] giving
    coordinates cx along path centre, derivatives cd1 along path, cd2 and cd3 giving side vectors,
    and cd12, cd13 giving rate of change of side vectors. Parameters have 3 components.
    Same format as output of zinc_utils get_nodeset_path_ordered_field_parameters().
    :param elementsCountAround: Number of elements & nodes to create around tube. First location is at +d2.
    :param radius: Radius of tube in xi space.
    :param phaseAngle: Starting angle around ellipse, where 0.0 is at d2, pi/2 is at d3.
    :return: px[][], pd1[][], pd2[][], pd12[][] with first index in range(pointsCountAlong),
    second inner index in range(elementsCountAround)
    """
    assert len(pathParameters) == 6
    pointsCountAlong = len(pathParameters[0])
    assert pointsCountAlong > 1
    assert len(pathParameters[0][0]) == 3

    tubeGenerator = TubeEllipseGenerator(radius, phaseAngle)
    tx = []
    td1 = []
    td2 = []
    td12 = []
    for p in range(pointsCountAlong):
        px, pd1, pd2, pd12, pd3, pd13 = [cp[p] for cp in pathParameters]
        ex, ed1, ed2, ed12 = tubeGenerator.generate(
            px, pd1, pd2, pd12, pd3, pd13, elementsCountAround, d2Scale=magnitude(pd1))
        tx.append(ex)
        td1.append(ed1)
        td2.append(ed2)
        td12.append(ed12)

    return tx, td1, td2, td12


def resampleTubeCoordinates(rawTubeCoordinates, fixedElementsCountAlong=None,
                            targetElementLength=None, minimumElementsCountAlong=1,
                            startSurface: TrackSurface=None, endSurface: TrackSurface=None):
    """
    Generate new tube coordinates along raw tube coordinates, optionally trimmed to start/end surfaces.
    Untrimmed tube elements are even sized along each longitudinal curve.
    Trimmed tube elements adjust derivatives at trimmed ends to transition from distorted to regular spacing.
    Can specify either fixedElementsCountAlong or targetElementLength.
    :param rawTubeCoordinates: (px, pd1, pd2, pd12) returned by getPathRawTubeCoordinates().
    :param fixedElementsCountAlong: Number of elements in resampled coordinates, or None to use targetElementLength.
    :param targetElementLength: Target element length or None to use fixedElementsCountAlong.
    Length is compared with mean trimmed length to determine number along, subject to specified minimum.
    :param minimumElementsCountAlong: Minimum number along to apply regardless of fixedElementsCountAlong or number
    calculated from targetElementLength.
    :param startSurface: Optional TrackSurface specifying start of tube at intersection with it.
    :param endSurface: Optional TrackSurface specifying end of tube at intersection with it.
    :return: sx[][], sd1[][], sd2[][], sd12[][] with first index in range(elementsCountAlong + 1),
    second inner index in range(elementsCountAround)
    """
    assert fixedElementsCountAlong or targetElementLength
    px, pd1, pd2, pd12 = rawTubeCoordinates
    pointsCountAlong = len(px)
    endPointLocation = float(pointsCountAlong - 1)
    elementsCountAround = len(px[0])

    # work out lengths of longitudinal curves, raw and trimmed
    sumLengths = 0.0
    startCurveLocations = []
    startLengths = []
    meanStartLocation = 0.0
    endCurveLocations = []
    endLengths = []
    meanEndLocation = 0.0
    for q in range(elementsCountAround):
        cx = [px[p][q] for p in range(pointsCountAlong)]
        cd2 = [pd2[p][q] for p in range(pointsCountAlong)]
        startCurveLocation = None
        if startSurface:
            startSurfacePosition, startCurveLocation, startIntersects = startSurface.findNearestPositionOnCurve(cx, cd2)
            if startIntersects:
                meanStartLocation += startCurveLocation[0] + startCurveLocation[1]
            else:
                startCurveLocation = None
        startCurveLocations.append(startCurveLocation)
        endCurveLocation = None
        if endSurface:
            endSurfacePosition, endCurveLocation, endIntersects = endSurface.findNearestPositionOnCurve(cx, cd2)
            if endIntersects:
                meanEndLocation += endCurveLocation[0] + endCurveLocation[1]
            else:
                endCurveLocation = None
        if not endCurveLocation:
            meanEndLocation += endPointLocation
        endCurveLocations.append(endCurveLocation)
        startLength, length, endLength = \
            getCubicHermiteTrimmedCurvesLengths(cx, cd2, startCurveLocation, endCurveLocation)[0:3]
        sumLengths += length
        startLengths.append(startLength)
        endLengths.append(endLength)

    meanLength = sumLengths / elementsCountAround
    # small fudge factor on targetElementLength so whole numbers chosen on centroid don't go one higher:
    elementsCountAlong = max(minimumElementsCountAlong, fixedElementsCountAlong if fixedElementsCountAlong else
                             math.ceil(meanLength * 0.999 / targetElementLength))
    meanStartLocation /= elementsCountAround
    e = min(int(meanStartLocation), pointsCountAlong - 2)
    meanStartCurveLocation = (e, meanStartLocation - e)
    meanEndLocation /= elementsCountAround
    e = min(int(meanEndLocation), pointsCountAlong - 2)
    meanEndCurveLocation = (e, meanEndLocation - e)

    # resample along, with variable spacing where ends are trimmed
    sx = [[None] * elementsCountAround for _ in range(elementsCountAlong + 1)]
    sd1 = [[None] * elementsCountAround for _ in range(elementsCountAlong + 1)]
    sd2 = [[None] * elementsCountAround for _ in range(elementsCountAlong + 1)]
    sd12 = [[None] * elementsCountAround for _ in range(elementsCountAlong + 1)]
    for q in range(elementsCountAround):
        cx = [px[p][q] for p in range(pointsCountAlong)]
        cd1 = [pd1[p][q] for p in range(pointsCountAlong)]
        cd2 = [pd2[p][q] for p in range(pointsCountAlong)]
        cd12 = [pd12[p][q] for p in range(pointsCountAlong)]
        meanStartLength, meanLength, meanEndLength = \
            getCubicHermiteTrimmedCurvesLengths(cx, cd2, meanStartCurveLocation, meanEndCurveLocation)[0:3]
        derivativeMagnitudeStart = (meanLength + 2.0 * (meanStartLength - startLengths[q])) / elementsCountAlong
        derivativeMagnitudeEnd = (meanLength + 2.0 * (meanEndLength - endLengths[q])) / elementsCountAlong
        qx, qd2, pe, pxi, psf = sampleCubicHermiteCurvesSmooth(
            cx, cd2, elementsCountAlong, derivativeMagnitudeStart, derivativeMagnitudeEnd,
            startLocation=startCurveLocations[q], endLocation=endCurveLocations[q])
        qd1, qd12 = interpolateSampleCubicHermite(cd1, cd12, pe, pxi, psf)
        # swizzle
        for p in range(elementsCountAlong + 1):
            sx[p][q] = qx[p]
            sd1[p][q] = qd1[p]
            sd2[p][q] = qd2[p]
            sd12[p][q] = qd12[p]

    # recalculate d1 around intermediate rings, but still in plane
    # normally looks fine, but d1 derivatives are wavy when very distorted
    pStart = 0 if startSurface else 1
    pLimit = elementsCountAlong + 1 if endSurface else elementsCountAlong
    for p in range(pStart, pLimit):
        # first smooth to get d1 with new directions not tangential to surface
        td1 = smoothCubicHermiteDerivativesLoop(sx[p], sd1[p])
        # constraint to be tangential to surface
        td1 = [rejection(td1[q], normalize(cross(sd1[p][q], sd2[p][q]))) for q in range(elementsCountAround)]
        # smooth magnitudes only
        sd1[p] = smoothCubicHermiteDerivativesLoop(sx[p], td1, fixAllDirections=True)

    return sx, sd1, sd2, sd12
