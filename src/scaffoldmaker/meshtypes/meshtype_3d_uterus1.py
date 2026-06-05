"""
Generates a 3-D uterus mesh from a 1-D network layout, with variable
numbers of elements around, along and through wall.
"""
from cmlibs.maths.vectorops import add, cross, mult, set_magnitude, sub, normalize, magnitude, \
    axis_angle_to_rotation_matrix
from cmlibs.utils.zinc.field import findOrCreateFieldCoordinates
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, getAnnotationGroupForTerm, \
    findOrCreateAnnotationGroupForTerm
from scaffoldmaker.annotation.uterus_terms import get_uterus_term
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils.eft_utils import determineCubicHermiteSerendipityEft, HermiteNodeLayoutManager
from scaffoldmaker.utils.interpolation import smoothCurveSideCrossDerivatives, smoothCubicHermiteDerivativesLine, \
    interpolateCubicHermite, sampleCubicHermiteCurves, computeCubicHermiteDerivativeScaling, \
    interpolateLagrangeHermiteDerivative
from scaffoldmaker.utils.networkmesh import NetworkMesh, pathValueLabels
from scaffoldmaker.utils.tubenetworkmesh import TubeNetworkMeshBuilder, TubeNetworkMeshGenerateData, \
    PatchTubeNetworkMeshSegment
from scaffoldmaker.utils.zinc_utils import group_add_connected_elements, get_nodeset_path_ordered_field_parameters, \
    find_first_node_conditional, get_node_mesh_location
import copy
import math


class UterusTubeNetworkMeshGenerateData(TubeNetworkMeshGenerateData):

    def __init__(self, region, meshDimension, coordinateFieldName="coordinates",
                 startNodeIdentifier=1, startElementIdentifier=1, isLinearThroughShell=False, isShowTrimSurfaces=False):
        """
        :param isLinearThroughWall: Callers should only set if 3-D with no core.
        :param isShowTrimSurfaces: Tells junction generateMesh to make 2-D trim surfaces.
        """
        super(UterusTubeNetworkMeshGenerateData, self).__init__(
            region, meshDimension, coordinateFieldName, startNodeIdentifier, startElementIdentifier,
            isLinearThroughShell, isShowTrimSurfaces)
        self._leftOviductGroup = self.getOrCreateAnnotationGroup(get_uterus_term("left oviduct"))
        self._rightOviductGroup = self.getOrCreateAnnotationGroup(get_uterus_term("right oviduct"))
        self._fundusGroup = self.getOrCreateAnnotationGroup(get_uterus_term("fundus of uterus"))
        self._bodyGroup = self.getOrCreateAnnotationGroup(get_uterus_term("body of uterus"))
        self._bodyNotCervixGroup = self.getOrCreateAnnotationGroup(("body not cervix", ""))
        # force these annotation group names in base class
        self._leftGroup = self.getOrCreateAnnotationGroup(("left uterus", ""))
        self._rightGroup = self.getOrCreateAnnotationGroup(("right uterus", ""))
        self._dorsalGroup = self.getOrCreateAnnotationGroup(("dorsal uterus", ""))
        self._ventralGroup = self.getOrCreateAnnotationGroup(("ventral uterus", ""))

    def getLeftOviductMeshGroup(self):
        return self._leftOviductGroup.getMeshGroup(self._mesh)

    def getRightOviductMeshGroup(self):
        return self._rightOviductGroup.getMeshGroup(self._mesh)

    def getFundusMeshGroup(self):
        return self._fundusGroup.getMeshGroup(self._mesh)

    def getBodyMeshGroup(self):
        return self._bodyGroup.getMeshGroup(self._mesh)

    def getBodyNotCervixMeshGroup(self):
        return self._bodyNotCervixGroup.getMeshGroup(self._mesh)


class UterusTubeNetworkMeshBuilder(TubeNetworkMeshBuilder):
    """
    Adds left, right, dorsal, ventral, fundus, body annotations.
    Future: derive from BodyTubeNetworkMeshBuilder to get left/right/dorsal/ventral.
    """
    def createSegment(self, networkSegment):
        if networkSegment.isPatch():
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

            return PatchTubeNetworkMeshSegment(networkSegment, pathParametersList, elementsCountAround,
                                               self._elementsCountThroughShell, self._isCore, elementsCountCoreBoxMinor,
                                               self._elementsCountTransition, coreBoundaryScalingMode)

        return super(UterusTubeNetworkMeshBuilder, self).createSegment(networkSegment)

    def generateMesh(self, generateData):
        super(UterusTubeNetworkMeshBuilder, self).generateMesh(generateData)
        # build temporary left/right dorsal/ventral groups
        mesh = generateData.getMesh()
        leftOviductMeshGroup = generateData.getLeftOviductMeshGroup()
        rightOviductMeshGroup = generateData.getRightOviductMeshGroup()
        fundusMeshGroup = generateData.getFundusMeshGroup()
        bodyMeshGroup = generateData.getBodyMeshGroup()
        bodyNotCervixMeshGroup = generateData.getBodyNotCervixMeshGroup()
        leftMeshGroup = generateData.getLeftMeshGroup()
        rightMeshGroup = generateData.getRightMeshGroup()
        dorsalMeshGroup = generateData.getDorsalMeshGroup()
        ventralMeshGroup = generateData.getVentralMeshGroup()
        for networkSegment in self._networkMesh.getNetworkSegments():
            segment = self._segments[networkSegment]
            annotationTerms = segment.getAnnotationTerms()
            segment.addSideD3ElementsToMeshGroup(True, ventralMeshGroup)
            segment.addSideD3ElementsToMeshGroup(False, dorsalMeshGroup)
            for annotationTerm in annotationTerms:
                if "oviduct" in annotationTerm[0]:
                    elementsCountRim = segment.getElementsCountRim()
                    e2 = segment.getSampledElementsCountAlong() - 1
                    elementsCountAround = segment.getElementsCountAround()
                    for e1 in range(elementsCountAround):
                        for e3 in range(elementsCountRim):
                            elementIdentifier = segment.getRimElementId(e1, e2, e3)
                            if elementIdentifier is not None:
                                element = mesh.findElementByIdentifier(elementIdentifier)
                                if "left" in annotationTerm[0]:
                                    if e1 < elementsCountAround // 4 or e1 > 3 * elementsCountAround // 4 - 1:
                                        fundusMeshGroup.addElement(element)
                                    else:
                                        bodyMeshGroup.addElement(element)
                                        bodyNotCervixMeshGroup.addElement(element)
                                else:  # right
                                    if elementsCountAround // 4 <= e1 < 3 * elementsCountAround // 4:
                                        fundusMeshGroup.addElement(element)
                                    else:
                                        bodyMeshGroup.addElement(element)
                                        bodyNotCervixMeshGroup.addElement(element)
                                oviductMeshGroup = leftOviductMeshGroup if "left" in annotationTerm[0] else \
                                    rightOviductMeshGroup
                                oviductMeshGroup.removeElement(element)

                if "fundus" in annotationTerm[0]:
                    segment.addAllElementsToMeshGroup(fundusMeshGroup)
                if "body" in annotationTerm[0]:
                    segment.addAllElementsToMeshGroup(bodyMeshGroup)
                    segment.addAllElementsToMeshGroup(bodyNotCervixMeshGroup)
                if "cervix" in annotationTerm[0]:
                    segment.addAllElementsToMeshGroup(bodyMeshGroup)
                if "left" in annotationTerm[0]:
                    segment.addAllElementsToMeshGroup(leftMeshGroup)
                    break
                if "right" in annotationTerm[0]:
                    segment.addAllElementsToMeshGroup(rightMeshGroup)
                    break
            else:
                # segment on main axis
                segment.addSideD2ElementsToMeshGroup(True, leftMeshGroup)
                segment.addSideD2ElementsToMeshGroup(False, rightMeshGroup)


class Septum:
    """
    Generates a solid septum hexahedral elements for dividing the body and cervix of the rat uterus into a double
    channel structure.
    """

    def __init__(self, element_counts, nidsRim, paramRim, thickness):
        """
        :param element_counts: [elements around septum, elements along septum, elements through septum]
        :param nidsRim: node identifiers of rim nodes to be used for making septum.
        :param paramRim: parameters of rim nodes to be used for making septum.
        :param thickness: thickness of the septum at its thinnest section.
        """
        assert all((count >= 2) and (count % 2 == 0) for count in [element_counts[0], element_counts[2]])
        self._element_counts = element_counts
        self._nidsRim = nidsRim
        self._paramRim = paramRim
        self._thickness = thickness
        none_parameters = [None] * 4  # x, d1, d2, d3
        self._nx = []  # shield mesh with holes over n3, n2, n1, d
        self._nids = []

        for n2 in range(self._element_counts[1] + 1):
            nx_layer = []
            nids_layer = []
            for n3 in range(self._element_counts[2] + 1):
                nx_row = []
                nids_row = []
                # s = ""
                for n1 in range(self._element_counts[0] + 1):
                    if (n2 == 0 and n3 == 1) or (n2 == 0 and n3 == self._element_counts[2] - 1) or \
                            (n2 == 1 and n3 == 0) or (n2 == 1 and n3 == self._element_counts[2]):
                        parameters = None
                    else:
                        parameters = copy.copy(none_parameters)
                    nx_row.append(parameters)
                    nids_row.append(None)
                # print(n3, n2, nx_row)
                nx_layer.append(nx_row)
                nids_layer.append(nids_row)
            self._nx.append(nx_layer)
            self._nids.append(nids_layer)

    def build(self):
        """
        Determine coordinates and derivatives over and within septum.
        """
        half_counts = [count // 2 for count in self._element_counts]
        elementsCountAround = self._element_counts[0]
        elementsCountAlong = self._element_counts[1]
        elementsCountThrough = self._element_counts[2]

        # Populate nx and nids with rimNids
        for n1 in range(elementsCountAround + 1):
            self._nids[0][half_counts[2]][n1] = self._nidsRim[0][n1]
            self._nx[0][half_counts[2]][n1] = self._paramRim[0][n1]

            self._nids[0][0][n1] = self._nidsRim[half_counts[2] - 1][n1]
            self._nids[0][-1][elementsCountAround - n1] = \
                self._nidsRim[half_counts[2] - 1][elementsCountAround + 1 + n1]

            self._nx[0][0][n1] = self._paramRim[half_counts[2] - 1][n1]
            self._nx[0][-1][elementsCountAround - n1] = self._paramRim[half_counts[2] - 1][elementsCountAround + 1 + n1]

            for n3 in range(2, half_counts[2]):
                self._nids[0][n3][n1] = self._nidsRim[half_counts[2] - n3][n1]
                self._nids[0][-(n3 + 1)][elementsCountAround - n1] = \
                    self._nidsRim[half_counts[2] - n3][elementsCountAround + 1 + n1]
                self._nx[0][n3][n1] = self._paramRim[half_counts[2] - n3][n1]
                self._nx[0][-(n3 + 1)][elementsCountAround - n1] = \
                    self._paramRim[half_counts[2] - n3][elementsCountAround + 1 + n1]

        extraRowsCountTop = half_counts[2] - 2
        for n2 in range(2, elementsCountAlong + 1):
            for n1 in range(elementsCountAround + 1):
                self._nids[n2][0][n1] = self._nidsRim[n2 + extraRowsCountTop][n1]
                self._nx[n2][0][n1] = self._paramRim[n2 + extraRowsCountTop][n1]
                self._nids[n2][-1][elementsCountAround - n1] = \
                    self._nidsRim[n2 + extraRowsCountTop][elementsCountAround + 1 + n1]
                self._nx[n2][-1][elementsCountAround - n1] = \
                    self._paramRim[n2 + extraRowsCountTop][elementsCountAround + 1 + n1]

        # Use middle point to sample front to back into number of elements through septum
        for n2 in range(3, elementsCountAlong + 1):
            n1 = half_counts[0]
            xFront = self._nx[n2][0][n1][0]
            xBack = self._nx[n2][-1][n1][0]
            xDiffPerElement = mult(sub(xBack, xFront), 1.0 / elementsCountThrough)
            d3 = xDiffPerElement
            for n3 in range(1, elementsCountThrough):
                self._nx[n2][n3][n1][0] = add(xFront, mult(xDiffPerElement, n3))
                self._nx[n2][n3][n1][3] = [-c for c in d3]

        # populate d2
        for n3 in range(1, elementsCountThrough):
            for n2 in range(3, elementsCountAlong):
                xTop = self._nx[n2][n3][n1][0]
                xBottom = self._nx[n2 + 1][n3][n1][0]
                self._nx[n2][n3][n1][2] = sub(xBottom, xTop)
            self._nx[n2 + 1][n3][n1][2] = sub(xBottom, xTop)

        # Extrude out the middle to left and right by thickness
        for n2 in range(3, elementsCountAlong + 1):
            n1Mid = half_counts[0]
            n3Mid = half_counts[2]
            xMid = self._nx[n2][n3Mid][n1Mid][0]
            d2Mid = self._nx[n2][n3Mid][n1Mid][2]
            d3Mid = self._nx[n2][n3Mid][n1Mid][3]
            d1Mid = cross(d2Mid, d3Mid)
            dV = set_magnitude(d1Mid, self._thickness * 0.5)
            xMidLeft = add(xMid, dV)
            xMidRight = sub(xMid, dV)
            px, pd1 = sampleCubicHermiteCurves([xMidRight, xMid, xMidLeft], [dV, dV, dV], elementsCountAround,
                                               arcLengthDerivatives=True)[0:2]

            for n1 in range(elementsCountAround, -1, -1):
                self._nx[n2][n3Mid][n1][0] = px[n1]
                self._nx[n2][n3Mid][n1][1] = pd1[n1]

        # populate d2
        for n1 in range(elementsCountAround, -1, -1):
            for n2 in range(3, elementsCountAlong):
                xTop = self._nx[n2][n3Mid][n1][0]
                xBottom = self._nx[n2 + 1][n3Mid][n1][0]
                self._nx[n2][n3Mid][n1][2] = sub(xBottom, xTop)
            self._nx[n2 + 1][n3Mid][n1][2] = sub(xBottom, xTop)

        # Go from left to right and sample from front to back
        for n2 in range(3, elementsCountAlong + 1):
            for n1 in range(elementsCountAround + 1):
                if n1 == half_counts[0]:
                    continue
                xFront = self._nx[n2][0][n1][0]
                xMid = self._nx[n2][n3Mid][n1][0]
                xBack = self._nx[n2][-1][n1][0]

                px, pd3 = sampleCubicHermiteCurves(
                    [xFront, xMid, xBack],
                    [sub(xMid, xFront), add(sub(xMid, xFront), sub(xBack, xMid)), sub(xBack, xMid)],
                    elementsCountThrough, arcLengthDerivatives=True)[0:2]
                pd3 = smoothCubicHermiteDerivativesLine(px, pd3)

                for n3 in range(1, elementsCountThrough):
                    self._nx[n2][n3][n1][0] = px[n3]
                    self._nx[n2][n3][n1][3] = [-c for c in pd3[n3]]

        # populate d1 and d2
        for n3 in range(1, elementsCountThrough):
            # d1
            for n2 in range(3, elementsCountAlong + 1):
                x = []
                d1 = []
                for n1 in range(elementsCountAround):
                    x1 = self._nx[n2][n3][n1][0]
                    x2 = self._nx[n2][n3][n1 + 1][0]
                    x.append(x1)
                    d1.append(sub(x2, x1))
                x.append(x2)
                d1.append(sub(x2, x1))
                d1 = smoothCubicHermiteDerivativesLine(x, d1)
                for n1 in range(elementsCountAround + 1):
                    self._nx[n2][n3][n1][1] = d1[n1]

            # d2
            for n1 in range(elementsCountAround + 1):
                for n2 in range(3, elementsCountAlong):
                    xTop = self._nx[n2][n3][n1][0]
                    xBottom = self._nx[n2 + 1][n3][n1][0]
                    self._nx[n2][n3][n1][2] = sub(xBottom, xTop)
                self._nx[n2 + 1][n3][n1][2] = sub(xBottom, xTop)

        # Quadrants under arc
        # Sample mid-line from arc (top) to the start of regular septum elements (bottom) using scaling
        # -d3 (top) to d2 (bottom)
        for n1 in range(elementsCountAround + 1):
            ax = self._nx[0][n3Mid][n1][0]
            ad2 = [-c for c in self._nx[0][n3Mid][n1][3]]
            bx = self._nx[3][n3Mid][n1][0]
            bd2 = self._nx[3][n3Mid][n1][2]
            scaling = computeCubicHermiteDerivativeScaling(ax, ad2, bx, bd2)
            ad2 = mult(ad2, scaling)
            bd2 = mult(bd2, scaling)
            for n2 in range(1, 3):
                xi = 1.0 / 3 * n2
                self._nx[n2][n3Mid][n1][0] = interpolateCubicHermite(ax, ad2, bx, bd2, xi)

            for n2 in range(1, 3):
                xTop = self._nx[n2][n3Mid][n1][0]
                xBottom = self._nx[n2 + 1][n3Mid][n1][0]
                self._nx[n2][n3Mid][n1][2] = sub(xBottom, xTop)

            # Sample from front to back for row above regular elements
            xFront = self._nx[2][0][n1][0]
            xMid = self._nx[2][n3Mid][n1][0]
            xBack = self._nx[2][-1][n1][0]
            px, pd3 = sampleCubicHermiteCurves(
                [xFront, xMid, xBack],
                [sub(xMid, xFront), add(sub(xMid, xFront), sub(xBack, xMid)), sub(xBack, xMid)],
                elementsCountThrough)[0:2]
            pd3 = smoothCubicHermiteDerivativesLine(px, pd3)

            for n3 in range(1, elementsCountThrough):
                self._nx[2][n3][n1][0] = px[n3]
                self._nx[2][n3][n1][3] = [-c for c in pd3[n3]]

            for n3 in range(1, elementsCountThrough):
                xTop = self._nx[2][n3][n1][0]
                xBottom = self._nx[3][n3][n1][0]
                self._nx[2][n3][n1][2] = sub(xBottom, xTop)

            # Sample from row above regular elements to the top arc if there are more than 4 elements through
            for n3 in list(range(2, half_counts[2])) + list(range(half_counts[2] + 1, elementsCountThrough - 1)):
                ax = self._nx[0][n3][n1][0]
                ad2 = [-c for c in self._nx[0][n3][n1][3]]
                bx = self._nx[2][n3][n1][0]
                bd2 = self._nx[2][n3][n1][2]
                scaling = computeCubicHermiteDerivativeScaling(ax, ad2, bx, bd2)
                ad2 = mult(ad2, scaling)
                bd2 = mult(bd2, scaling)
                self._nx[1][n3][n1][0] = interpolateCubicHermite(ax, ad2, bx, bd2, 0.5)

                xTop = self._nx[1][n3][n1][0]
                xBottom = self._nx[2][n3][n1][0]
                self._nx[1][n3][n1][2] = sub(xBottom, xTop)

        # Populate d1 and d3 for mid n3 at row 1 and 2
        for n2 in [1, 2]:
            n3Start = 2 if n2 == 1 else 1
            n3End = elementsCountThrough + (-1 if n2 == 1 else 0)
            for n3 in range(n3Start, n3End):
                x = []
                d1 = []
                for n1 in range(elementsCountAround):
                    x1 = self._nx[n2][n3][n1][0]
                    x2 = self._nx[n2][n3][n1 + 1][0]
                    x.append(x1)
                    d1.append(sub(x2, x1))
                x.append(x2)
                d1.append(sub(x2, x1))
                d1 = smoothCubicHermiteDerivativesLine(x, d1)
                for n1 in range(elementsCountAround + 1):
                    self._nx[n2][n3][n1][1] = d1[n1]

        # Calculate 6 way midpoint - could be replaced with GRC's QuadTriangleMesh later
        for n1 in range(elementsCountAround + 1):
            for n3 in range(2):
                xSum = 0
                ySum = 0
                zSum = 0

                x6Way = [self._nx[0][-3 if n3 else 2][n1][0],
                         self._nx[0][-1 if n3 else 0][n1][0],
                         self._nx[2][-1 if n3 else 0][n1][0],
                         self._nx[2][-2 if n3 else 1][n1][0],
                         self._nx[2][-3 if n3 else 2][n1][0],
                         self._nx[1][-3 if n3 else 2][n1][0]]

                if n1 == 0:
                    if n3 == 0:
                        d6Way = [add([-c for c in self._nx[0][2][n1][1]], [-c for c in self._nx[0][2][n1][3]]),
                                 # -d1 - d3
                                 self._nx[0][0][n1][3],  # d3
                                 add(self._nx[2][0][n1][2], self._nx[2][0][n1][3]),  # d2 + d3
                                 self._nx[2][1][n1][2],  # d2
                                 add([-c for c in self._nx[2][2][n1][2]], self._nx[2][2][n1][3])]  # -d2 + d3
                    else:
                        d6Way = [add(self._nx[0][-3][n1][1], [-c for c in self._nx[0][-3][n1][3]]),  # d1 - d3
                                 self._nx[0][-1][n1][3],  # d3
                                 add(self._nx[2][-1][n1][1], self._nx[2][-1][n1][3]),  # d1 + d3
                                 self._nx[2][-2][n1][2],  # d2
                                 add([-c for c in self._nx[2][-3][n1][2]],
                                     [-c for c in self._nx[2][-3][n1][3]])]  # -d2 - d3

                elif n1 == elementsCountAround:
                    if n3 == 0:
                        d6Way = [add(self._nx[0][2][n1][1], [-c for c in self._nx[0][2][n1][3]]),  # d1 - d3
                                 self._nx[0][0][n1][3],  # d3
                                 add(self._nx[2][0][n1][1], self._nx[2][0][n1][3]),  # d1 + d3
                                 self._nx[2][1][n1][2],  # d2
                                 add([-c for c in self._nx[2][2][n1][2]], self._nx[2][2][n1][3])]  # -d2 + d3

                    else:
                        d6Way = [add([-c for c in self._nx[0][-3][n1][1]], [-c for c in self._nx[0][-3][n1][3]]),
                                 # -d1 - d3
                                 self._nx[0][-1][n1][3],  # d3
                                 add(self._nx[2][-1][n1][2], self._nx[2][-1][n1][3]),  # d2 + d3
                                 self._nx[2][-2][n1][2],  # d2
                                 add([-c for c in self._nx[2][-3][n1][2]],
                                     [-c for c in self._nx[2][-3][n1][3]])]  # -d2 - d3

                else:
                    if n3 == 0:
                        d6Way = [add(self._nx[0][2][n1][2], [-c for c in self._nx[0][2][n1][3]]),  # d2 - d3
                                 self._nx[0][0][n1][3],  # d3
                                 add(self._nx[2][0][n1][2], self._nx[2][0][n1][3]),  # d2 + d3
                                 self._nx[2][1][n1][2],  # d2
                                 add([-c for c in self._nx[2][2][n1][2]], self._nx[2][2][n1][3])]  # -d2 + d3

                    else:
                        d6Way = [add(self._nx[0][-3][n1][2] if elementsCountThrough // 2 > 2 else
                                     [-c for c in self._nx[0][-3][n1][2]],
                                     [-c for c in self._nx[0][-3][n1][3]]),  # -d2 - d3
                                 self._nx[0][-1][n1][3],  # d3
                                 add(self._nx[2][-1][n1][2], self._nx[2][-1][n1][3]),  # d2 + d3
                                 self._nx[2][-2][n1][2],  # d2
                                 add([-c for c in self._nx[2][-3][n1][2]],
                                     [-c for c in self._nx[2][-3][n1][3]])]  # -d2 - d3

                cd = interpolateLagrangeHermiteDerivative(x6Way[5], x6Way[2], d6Way[2], 0.0)
                d6Way.append(cd)

                pathNodes = [[0, 3], [4, 1], [5, 2]]
                for i in range(3):
                    nodeIdx = pathNodes[i]
                    xMid = interpolateCubicHermite(x6Way[nodeIdx[0]], d6Way[nodeIdx[0]],
                                                   x6Way[nodeIdx[1]], d6Way[nodeIdx[1]], 0.5)

                    xSum += xMid[0]
                    ySum += xMid[1]
                    zSum += xMid[2]

                # Calculate the averages
                x_centroid = xSum / 3
                y_centroid = ySum / 3
                z_centroid = zSum / 3

                # print(n1, n3, [x_centroid, y_centroid, z_centroid])
                x_3Way = [x_centroid, y_centroid, z_centroid]
                self._nx[1][-2 if n3 else 1][n1][0] = x_3Way

                # sample with hermite-lagrange interpolation from sides to 3-way point to get derivatives
                ad = interpolateLagrangeHermiteDerivative(x_3Way, x6Way[3], d6Way[3], 0.0)
                bd = interpolateLagrangeHermiteDerivative(x_3Way, x6Way[1], d6Way[1], 0.0)
                cd = interpolateLagrangeHermiteDerivative(x_3Way, x6Way[5], d6Way[5], 0.0)

                # use the minimum magnitude in all 3 directions
                d_factor = 0.6  # GRC revisit - try an exact triangle
                d_mag = d_factor * min(magnitude(ad), magnitude(bd), magnitude(cd))
                d2_3way = set_magnitude(bd, -d_mag)
                d3_3way = set_magnitude(cd, d_mag if n3 else -d_mag)

                self._nx[1][-2 if n3 else 1][n1][2] = d2_3way
                self._nx[1][-2 if n3 else 1][n1][3] = d3_3way

                if n3:
                    self._nx[1][-3][n1][3] = sub(self._nx[1][-3][n1][0], self._nx[1][-2][n1][0])
                else:
                    self._nx[1][2][n1][3] = sub(self._nx[1][1][n1][0], self._nx[1][2][n1][0])

        # populate derivatives at row 1
        for n3 in range(1, elementsCountThrough):
            x = []
            d1 = []
            for n1 in range(elementsCountAround):
                x1 = self._nx[1][n3][n1][0]
                x2 = self._nx[1][n3][n1 + 1][0]
                x.append(x1)
                d1.append(sub(x2, x1))
            x.append(x2)
            d1.append(sub(x2, x1))
            d1 = smoothCubicHermiteDerivativesLine(x, d1)
            for n1 in range(elementsCountAround + 1):
                self._nx[1][n3][n1][1] = d1[n1]

        for n1 in range(elementsCountAround + 1):
            x = []
            d3 = []
            for n3 in range(1, elementsCountThrough):
                x.append(self._nx[1][n3][n1][0])
                d3.append(self._nx[1][n3][n1][3])
            d3 = smoothCubicHermiteDerivativesLine(x, d3, fixStartDerivative=True, fixEndDerivative=True)
            for n3 in range(2, elementsCountThrough - 1):
                self._nx[1][n3][n1][3] = [-c for c in d3[n3 - 1]]

    def generateMesh(self, fieldmodule, coordinates, nextNodeIdentifier, nextElementIdentifier,
                     annotationGroups, region):
        """
        After build() has been called, generate nodes and elements of septum.
        Client is expected to run within ChangeManager(fieldmodule)
        :param fieldmodule: Owning fieldmodule to create mesh in.
        :param coordinates: Coordinate field to define.
        :param nextNodeIdentifier: next identifier to use for making next node.
        :param nextElementIdentifier: next identifier to use for making next element.
        :param annotationGroups: annotation groups.
        :param region: region for making septum.
        :return nextNodeIdentifer and nextElementIdentifier.
        """

        fieldcache = fieldmodule.createFieldcache()

        # create nodes
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        for value_label in [Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3]:
            nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)

        node_identifier = nextNodeIdentifier
        for n2 in range(self._element_counts[1] + 1):
            for n3 in range(self._element_counts[2] + 1):
                for n1 in range(self._element_counts[0] + 1):
                    parameters = self._nx[n2][n3][n1]
                    if not parameters or self._nids[n2][n3][n1]:
                        continue
                    x, d1, d2, d3 = parameters
                    if not x:
                        continue  # while in development
                    node = nodes.createNode(node_identifier, nodetemplate)
                    self._nids[n2][n3][n1] = node_identifier
                    fieldcache.setNode(node)
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                    if d1:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, d1)
                    if d2:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, d2)
                    if d3:
                        coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)
                    node_identifier += 1

        nextNodeIdentifier = node_identifier

        septumGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("septum", ""))
        leftGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("left", ""))
        rightGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("right", ""))
        dorsalGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("dorsal", ""))
        ventralGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("ventral", ""))

        groups = [septumGroup]

        mesh = fieldmodule.findMeshByDimension(3)

        elementtemplate_regular = mesh.createElementtemplate()
        elementtemplate_regular.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        elementbasis = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY)
        eft_regular = mesh.createElementfieldtemplate(elementbasis)
        elementtemplate_regular.defineField(coordinates, -1, eft_regular)

        elementtemplate_special = mesh.createElementtemplate()
        elementtemplate_special.setElementShapeType(Element.SHAPE_TYPE_CUBE)

        node_layout_manager = HermiteNodeLayoutManager()
        nodeLayoutD1MinusD3D2 = \
            node_layout_manager.getNodeLayoutRegularPermuted(
                d3Defined=True, limitDirections=[[[1.0, 0.0, 0.0]], [[0.0, 0.0, -1.0]], [[0.0, 1.0, 0.0]]])
        nodeLayoutMinusD1D2MinusD3 = \
            node_layout_manager.getNodeLayoutRegularPermuted(
                d3Defined=True, limitDirections=[[[-1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, -1.0]]])
        nodeLayoutMinusD1MinusD3MinusD2 = \
            node_layout_manager.getNodeLayoutRegularPermuted(
                d3Defined=True, limitDirections=[[[-1.0, 0.0, 0.0]], [[0.0, 0.0, -1.0]], [[0.0, -1.0, 0.0]]])
        nodeLayoutD2D1MinusD3 = \
            node_layout_manager.getNodeLayoutRegularPermuted(
                d3Defined=True, limitDirections=[[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 0.0, -1.0]]])
        nodeLayoutD2MinusD1D3 = \
            node_layout_manager.getNodeLayoutRegularPermuted(
                d3Defined=True, limitDirections=[[[0.0, 1.0, 0.0]], [[-1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])
        nodeLayoutD2MinusD3MinusD1 = \
            node_layout_manager.getNodeLayoutRegularPermuted(
                d3Defined=True, limitDirections=[[[0.0, 1.0, 0.0]], [[0.0, 0.0, -1.0]], [[-1.0, 0.0, 0.0]]])
        nodeLayoutMinusD2D1D3 = \
            node_layout_manager.getNodeLayoutRegularPermuted(
                d3Defined=True, limitDirections=[[[0.0, -1.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])
        nodeLayoutMinusD2MinusD1MinusD3 = \
            node_layout_manager.getNodeLayoutRegularPermuted(
                d3Defined=True, limitDirections=[[[0.0, -1.0, 0.0]], [[-1.0, 0.0, 0.0]], [[0.0, 0.0, -1.0]]])
        nodeLayoutMinusD2MinusD3D1 = \
            node_layout_manager.getNodeLayoutRegularPermuted(
                d3Defined=True, limitDirections=[[[0.0, -1.0, 0.0]], [[0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0]]])

        nodeLayout5Way = node_layout_manager.getNodeLayout5Way12(d3Defined=True)
        nodeLayoutTriplePoint23Front = node_layout_manager.getNodeLayoutTriplePoint23Front()
        nodeLayoutTriplePoint23Back = node_layout_manager.getNodeLayoutTriplePoint23Back()

        elementIdentifier = nextElementIdentifier

        # Top 4 elements
        for e3 in range(2):
            for e1 in range(self._element_counts[0]):
                e1p = e1 + 1

                n3p0 = -3 if e3 else 0
                n3p1 = -3 if e3 else 1
                n3pA = -1 if e3 else 2
                n3pB = -2 if e3 else 2

                nids = [self._nids[0][n3pA][e1], self._nids[0][n3pA][e1p],
                        self._nids[1][n3pB][e1], self._nids[1][n3pB][e1p],
                        self._nids[0][n3p0][e1], self._nids[0][n3p0][e1p],
                        self._nids[1][n3p1][e1], self._nids[1][n3p1][e1p]]

                nodeParameters = [self._nx[0][n3pA][e1], self._nx[0][n3pA][e1p],
                                  self._nx[1][n3pB][e1], self._nx[1][n3pB][e1p],
                                  self._nx[0][n3p0][e1], self._nx[0][n3p0][e1p],
                                  self._nx[1][n3p1][e1], self._nx[1][n3p1][e1p]]

                if e3 == 0 and e1 == 0:
                    nodeLayouts = [nodeLayoutD2MinusD3MinusD1, nodeLayoutD1MinusD3D2, None, None,
                                   nodeLayoutD2MinusD3MinusD1, nodeLayoutD1MinusD3D2,
                                   nodeLayoutTriplePoint23Front, nodeLayoutTriplePoint23Front]

                elif e3 == 0 and e1 < self._element_counts[0] - 1:
                    nodeLayouts = [nodeLayoutD1MinusD3D2, nodeLayoutD1MinusD3D2, None, None,
                                   nodeLayoutD1MinusD3D2, nodeLayoutD1MinusD3D2,
                                   nodeLayoutTriplePoint23Front, nodeLayoutTriplePoint23Front]

                elif e3 == 0 and e1 == self._element_counts[0] - 1:
                    nodeLayouts = [nodeLayoutD1MinusD3D2, nodeLayoutMinusD2MinusD3D1, None, None,
                                   nodeLayoutD1MinusD3D2, nodeLayoutMinusD2MinusD3D1,
                                   nodeLayoutTriplePoint23Front, nodeLayoutTriplePoint23Front]

                elif e3 == 1 and e1 == 0:
                    nodeLayouts = [nodeLayoutD2MinusD3MinusD1, nodeLayoutMinusD1MinusD3MinusD2,
                                   nodeLayoutTriplePoint23Back, nodeLayoutTriplePoint23Back,
                                   nodeLayoutD2MinusD3MinusD1, nodeLayoutD1MinusD3D2, None, None]

                elif e3 == 1 and e1 < self._element_counts[0] - 1:
                    nodeLayouts = [nodeLayoutMinusD1MinusD3MinusD2, nodeLayoutMinusD1MinusD3MinusD2,
                                   nodeLayoutTriplePoint23Back, nodeLayoutTriplePoint23Back,
                                   nodeLayoutD1MinusD3D2, nodeLayoutD1MinusD3D2, None, None]

                elif e3 == 1 and e1 == self._element_counts[0] - 1:
                    nodeLayouts = [nodeLayoutMinusD1MinusD3MinusD2, nodeLayoutMinusD2MinusD3D1,
                                   nodeLayoutTriplePoint23Back, nodeLayoutTriplePoint23Back,
                                   nodeLayoutD1MinusD3D2, nodeLayoutMinusD2MinusD3D1, None, None]

                eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                # print(elementIdentifier, nids)
                elementtemplate = mesh.createElementtemplate()
                elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
                elementtemplate.defineField(coordinates, -1, eft)
                element = mesh.createElement(elementIdentifier, elementtemplate)
                element.setNodesByIdentifier(eft, nids)
                if scalefactors:
                    element.setScaleFactors(eft, scalefactors)
                elementIdentifier += 1
                for annotationGroup in groups:
                    meshGroup = annotationGroup.getMeshGroup(mesh)
                    meshGroup.addElement(element)
                if e1 < self._element_counts[0] // 2:
                    rightMeshGroup = rightGroup.getMeshGroup(mesh)
                    rightMeshGroup.addElement(element)
                else:
                    leftMeshGroup = leftGroup.getMeshGroup(mesh)
                    leftMeshGroup.addElement(element)
                if e3:
                    ventralMeshGroup = ventralGroup.getMeshGroup(mesh)
                    ventralMeshGroup.addElement(element)
                else:
                    dorsalMeshGroup = dorsalGroup.getMeshGroup(mesh)
                    dorsalMeshGroup.addElement(element)

        # Elements connecting rows 0, 1 and 2
        for e3 in range(2):
            for e1 in range(self._element_counts[0]):
                e1p = e1 + 1

                n3p0 = -1 if e3 else 0
                n3p1 = -2 if e3 else 1

                n2p0 = 0 if e3 else 2
                n2p1 = 1 if e3 else 2
                n2pA = 2 if e3 else 0
                n2pB = 2 if e3 else 1

                if e3:
                    nids = [self._nids[n2p0][n3p0][e1], self._nids[n2p0][n3p0][e1p],
                            self._nids[n2pA][n3p0][e1], self._nids[n2pA][n3p0][e1p],
                            self._nids[n2p1][n3p1][e1], self._nids[n2p1][n3p1][e1p],
                            self._nids[n2pB][n3p1][e1], self._nids[n2pB][n3p1][e1p]]

                    nodeParameters = [self._nx[n2p0][n3p0][e1], self._nx[n2p0][n3p0][e1p],
                                      self._nx[n2pA][n3p0][e1], self._nx[n2pA][n3p0][e1p],
                                      self._nx[n2p1][n3p1][e1], self._nx[n2p1][n3p1][e1p],
                                      self._nx[n2pB][n3p1][e1], self._nx[n2pB][n3p1][e1p]]

                else:
                    nids = [self._nids[n2pB][n3p1][e1], self._nids[n2pB][n3p1][e1p],
                            self._nids[n2p1][n3p1][e1], self._nids[n2p1][n3p1][e1p],
                            self._nids[n2pA][n3p0][e1], self._nids[n2pA][n3p0][e1p],
                            self._nids[n2p0][n3p0][e1], self._nids[n2p0][n3p0][e1p]]

                    nodeParameters = [self._nx[n2pB][n3p1][e1], self._nx[n2pB][n3p1][e1p],
                                      self._nx[n2p1][n3p1][e1], self._nx[n2p1][n3p1][e1p],
                                      self._nx[n2pA][n3p0][e1], self._nx[n2pA][n3p0][e1p],
                                      self._nx[n2p0][n3p0][e1], self._nx[n2p0][n3p0][e1p]]

                if e3 == 0 and e1 == 0:
                    nodeLayouts = [nodeLayoutTriplePoint23Front, nodeLayoutTriplePoint23Front, None, None,
                                   nodeLayoutD2MinusD1D3, None, None, None]

                elif e3 == 0 and e1 < self._element_counts[0] - 1:
                    nodeLayouts = [nodeLayoutTriplePoint23Front, nodeLayoutTriplePoint23Front, None, None,
                                   None, None, None, None]

                elif e3 == 0 and e1 == self._element_counts[0] - 1:
                    nodeLayouts = [nodeLayoutTriplePoint23Front, nodeLayoutTriplePoint23Front, None, None,
                                   None, nodeLayoutMinusD2D1D3, None, nodeLayoutMinusD2D1D3]

                elif e3 == 1 and e1 == 0:
                    nodeLayouts = [nodeLayoutD2D1MinusD3, nodeLayoutMinusD1D2MinusD3,
                                   nodeLayoutD2D1MinusD3, nodeLayoutMinusD1D2MinusD3,
                                   nodeLayoutTriplePoint23Back, nodeLayoutTriplePoint23Back, None, None]

                elif e3 == 1 and e1 < self._element_counts[0] - 1:
                    nodeLayouts = [nodeLayoutMinusD1D2MinusD3, nodeLayoutMinusD1D2MinusD3,
                                   nodeLayoutMinusD1D2MinusD3, nodeLayoutMinusD1D2MinusD3,
                                   nodeLayoutTriplePoint23Back, nodeLayoutTriplePoint23Back, None, None]

                elif e3 == 1 and e1 == self._element_counts[0] - 1:
                    nodeLayouts = [nodeLayoutMinusD1D2MinusD3, nodeLayoutMinusD2MinusD1MinusD3,
                                   nodeLayoutMinusD1D2MinusD3, nodeLayoutMinusD1D2MinusD3,
                                   nodeLayoutTriplePoint23Back, nodeLayoutTriplePoint23Back, None, None]

                # print(elementIdentifier, nids)
                eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                elementtemplate = mesh.createElementtemplate()
                elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
                elementtemplate.defineField(coordinates, -1, eft)
                element = mesh.createElement(elementIdentifier, elementtemplate)
                element.setNodesByIdentifier(eft, nids)
                if scalefactors:
                    element.setScaleFactors(eft, scalefactors)
                elementIdentifier += 1
                for annotationGroup in groups:
                    meshGroup = annotationGroup.getMeshGroup(mesh)
                    meshGroup.addElement(element)
                if e1 < self._element_counts[0] // 2:
                    rightMeshGroup = rightGroup.getMeshGroup(mesh)
                    rightMeshGroup.addElement(element)
                else:
                    leftMeshGroup = leftGroup.getMeshGroup(mesh)
                    leftMeshGroup.addElement(element)
                if e3:
                    ventralMeshGroup = ventralGroup.getMeshGroup(mesh)
                    ventralMeshGroup.addElement(element)
                else:
                    dorsalMeshGroup = dorsalGroup.getMeshGroup(mesh)
                    dorsalMeshGroup.addElement(element)

        for e2 in range(1, self._element_counts[1]):
            e2p = e2 + 1
            e3Start = 1 if e2 <= 1 else 0
            e3End = self._element_counts[2] - 1 if e2 <= 1 else self._element_counts[2]
            for e3 in range(e3Start, e3End):
                lastToBackWall = (e3 == self._element_counts[2] - 1)
                for e1 in range(self._element_counts[0]):
                    e1p = e1 + 1
                    has5Way = (e2 == 2 and e3 == 0 and (e1 == 0 or e1 == self._element_counts[0] - 1)) or \
                              (e2 == 2 and e3 == self._element_counts[2] and
                               (e1 == 0 or e1 == self._element_counts[0] - 1))
                    hasTopCorners = (e2 == 1)
                    elementtemplate = elementtemplate_regular
                    eft = eft_regular
                    nids = []
                    scalefactors = []
                    for n3 in [e3 + 1, e3]:
                        nids += [self._nids[e2][n3][e1], self._nids[e2][n3][e1p],
                                 self._nids[e2p][n3][e1], self._nids[e2p][n3][e1p]]

                    if lastToBackWall or hasTopCorners or has5Way:
                        nodeParameters = []
                        nodeLayouts = []
                        # nidsCheck = []
                        # get node parameters for computing scale factors
                        for n3 in (e3 + 1, e3):
                            for n2 in (e2, e2 + 1):
                                for n1 in (e1, e1 + 1):
                                    nodeParameters.append(self._nx[n2][n3][n1])
                                    # nidsCheck.append(self._nids[n2][n3][n1])
                                    nodeLayouts.append(
                                        nodeLayout5Way if ((n2 == 2 and n3 == 0 and
                                                            (n1 == 0 or n1 == self._element_counts[0])) or
                                                           (n2 == 2 and n3 == self._element_counts[2] and
                                                            (n1 == 0 or n1 == self._element_counts[0]))) else
                                        nodeLayoutMinusD1D2MinusD3 if (n3 == self._element_counts[2]) else
                                        nodeLayoutTriplePoint23Front if (n2 == 1 and n3 == 1) else
                                        nodeLayoutTriplePoint23Back if (
                                                    n2 == 1 and n3 == self._element_counts[2] - 1) else
                                        None)

                        eft, scalefactors = determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts)
                        elementtemplate = mesh.createElementtemplate()
                        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
                        elementtemplate.defineField(coordinates, -1, eft)

                    element = mesh.createElement(elementIdentifier, elementtemplate)
                    element.setNodesByIdentifier(eft, nids)
                    if scalefactors:
                        element.setScaleFactors(eft, scalefactors)
                    elementIdentifier += 1
                    for annotationGroup in groups:
                        meshGroup = annotationGroup.getMeshGroup(mesh)
                        meshGroup.addElement(element)
                    if e1 < self._element_counts[0] // 2:
                        rightMeshGroup = rightGroup.getMeshGroup(mesh)
                        rightMeshGroup.addElement(element)
                    else:
                        leftMeshGroup = leftGroup.getMeshGroup(mesh)
                        leftMeshGroup.addElement(element)
                    if e3 < self._element_counts[2] // 2:
                        dorsalMeshGroup = dorsalGroup.getMeshGroup(mesh)
                        dorsalMeshGroup.addElement(element)
                    else:
                        ventralMeshGroup = ventralGroup.getMeshGroup(mesh)
                        ventralMeshGroup.addElement(element)

        nextElementIdentifier = elementIdentifier
        return nextNodeIdentifier, nextElementIdentifier


class MeshType_1d_uterus_network_layout1(MeshType_1d_network_layout1):
    """
    Defines uterus network layout.
    """

    @classmethod
    def getName(cls):
        return "1D Uterus Network Layout 1"

    @classmethod
    def getParameterSetNames(cls):
        return ["Default",
                "Human 1",
                "Human Pregnant 1",
                "Mouse 1",
                "Rat 1"]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {}
        options["Base parameter set"] = parameterSetName
        isHuman = "Human" in parameterSetName
        isPregnant = "Pregnant" in parameterSetName
        isHumanPregnant = isHuman and isPregnant
        isMouse = "Mouse" in parameterSetName
        isRat = "Rat" in parameterSetName
        isRodent = isMouse or isRat
        if isRodent:
            options["Structure"] = (
                "1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-36.1,"
                "17-18-19-20-21-22-23-24-25-26-27-28-29-30-31-32-36.2,"
                "#-33-34-35-36.3,"
                "36.4-37-38-39,"
                "39-40,"
                "40-41-42")
            options["Oviduct/uterine horn diameter"] = 0.5
            options["Oviduct/uterine horn length"] = 4.0
            options["Body length"] = 0.65
            options["Fundus width between oviducts/uterine horns"] = 0.8
            options["Fundus depth between oviducts/uterine horns"] = 0.5
            options["Cervical length"] = 0.25
            options["Cervical width around internal os"] = 0.8
            options["Cervical depth around internal os"] = 0.5
            options["Cervical width around external os"] = 0.8
            options["Cervical depth around external os"] = 0.5
            options["Vagina length"] = 0.5
            options["Vagina width around vagina orifice"] = 0.8
            options["Vagina depth around vagina orifice"] = 0.5
            options["Inner proportion body"] = 0.75
            options["Inner proportion cervix"] = 0.75
            options["Inner proportion vagina"] = 0.75
            options["Angle of anteversion degrees"] = 0.0
        elif isHumanPregnant:
            options["Structure"] = (
                "1-2-3-4-5-6-7-8-9-31.1,"
                "10-11-12-13-14-15-16-17-18-31.2,"
                "#-19-20-21-22-23-24-25-26-27-28-29-30-31.3,"
                "31.4-32-33-34-35-36-37-38-39-40-41-42-43,"
                "43-44,"
                "44-45-46-47-48")
            options["Oviduct/uterine horn diameter"] = 0.35
            options["Oviduct/uterine horn length"] = 10.0
            options["Body length"] = 14.0
            options["Fundus width between oviducts/uterine horns"] = 12.0
            options["Fundus depth between oviducts/uterine horns"] = 12.0
            options["Cervical length"] = 1.0
            options["Cervical width around internal os"] = 5.5
            options["Cervical depth around internal os"] = 3.8
            options["Cervical width around external os"] = 5.0
            options["Cervical depth around external os"] = 3.0
            options["Vagina length"] = 10.0
            options["Vagina width around vagina orifice"] = 1.25
            options["Vagina depth around vagina orifice"] = 1.25
            options["Inner proportion body"] = 0.95
            options["Inner proportion cervix"] = 0.15
            options["Inner proportion vagina"] = 0.8
            options["Angle of anteversion degrees"] = 70.0
        else:
            options["Structure"] = (
                "1-2-3-4-5-6-7-8-23.1,"
                "9-10-11-12-13-14-15-16-23.2,"
                "#-17-18-19-20-21-22-23.3,"
                "23.4-24-25-26-27-28-29,"
                "29-30-31,"
                "31-32-33-34-35-36-37-38")
            options["Oviduct/uterine horn diameter"] = 0.35
            options["Oviduct/uterine horn length"] = 10.0
            options["Body length"] = 7.0
            options["Fundus width between oviducts/uterine horns"] = 8.0
            options["Fundus depth between oviducts/uterine horns"] = 6.0
            options["Cervical length"] = 1.0
            options["Cervical width around internal os"] = 5.5
            options["Cervical depth around internal os"] = 3.8
            options["Cervical width around external os"] = 5.0
            options["Cervical depth around external os"] = 3.0
            options["Vagina length"] = 10.0
            options["Vagina width around vagina orifice"] = 1.25
            options["Vagina depth around vagina orifice"] = 1.25
            options["Inner proportion body"] = 0.75
            options["Inner proportion cervix"] = 0.15
            options["Inner proportion vagina"] = 0.8
            options["Angle of anteversion degrees"] = 70.0
        options["Define inner coordinates"] = True
        options["Inner proportion oviduct/uterine horn"] = 0.5

        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Oviduct/uterine horn diameter",
            "Oviduct/uterine horn length",
            "Body length",
            "Fundus width between oviducts/uterine horns",
            "Fundus depth between oviducts/uterine horns",
            "Cervical length",
            "Cervical width around internal os",
            "Cervical depth around internal os",
            "Cervical width around external os",
            "Cervical depth around external os",
            "Vagina length",
            "Vagina width around vagina orifice",
            "Vagina depth around vagina orifice",
            "Inner proportion oviduct/uterine horn",
            "Inner proportion body",
            "Inner proportion cervix",
            "Inner proportion vagina",
            "Angle of anteversion degrees"
        ]

    @classmethod
    def checkOptions(cls, options):
        dependentChanges = False
        for key in [
            "Oviduct/uterine horn diameter",
            "Oviduct/uterine horn length",
            "Body length",
            "Fundus width between oviducts/uterine horns",
            "Fundus depth between oviducts/uterine horns",
            "Cervical length",
            "Cervical width around internal os",
            "Cervical depth around internal os",
            "Cervical width around external os",
            "Cervical depth around external os",
            "Vagina length",
            "Vagina width around vagina orifice",
            "Vagina depth around vagina orifice"
        ]:
            if options[key] < 0.1:
                options[key] = 0.1
        for key in [
            "Inner proportion oviduct/uterine horn",
            "Inner proportion body",
            "Inner proportion cervix",
            "Inner proportion vagina",
        ]:
            if options[key] < 0.1:
                options[key] = 0.1
            elif options[key] > 0.95:
                options[key] = 0.95

        for key, angleRange in {
            "Angle of anteversion degrees": (-100.0, 100.0)
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
        parameterSetName = options['Base parameter set']
        structure = options["Structure"]
        oviductLength = options["Oviduct/uterine horn length"]
        oviductRadius = 0.5 * options["Oviduct/uterine horn diameter"]
        bodyLength = options["Body length"]
        halfFundusWidth = 0.5 * options["Fundus width between oviducts/uterine horns"]
        halfFundusDepth = 0.5 * options["Fundus depth between oviducts/uterine horns"]
        cervicalLength = options["Cervical length"]
        halfCervicalWidthInternalOs = 0.5 * options["Cervical width around internal os"]
        halfCervicalDepthInternalOs = 0.5 * options["Cervical depth around internal os"]
        halfCervicalWidthExternalOs = 0.5 * options["Cervical width around external os"]
        halfCervicalDepthExternalOs = 0.5 * options["Cervical depth around external os"]
        vaginaLength = options["Vagina length"]
        halfVaginaOrificeWidth = 0.5 * options["Vagina width around vagina orifice"]
        halfVaginaOrificeDepth = 0.5 * options["Vagina depth around vagina orifice"]
        anteversionAngleRad = math.radians(options["Angle of anteversion degrees"])
        innerProportionOviducts = options["Inner proportion oviduct/uterine horn"]
        innerProportionBody = options["Inner proportion body"]
        innerProportionCervix = options["Inner proportion cervix"]
        innerProportionVagina = options["Inner proportion vagina"]

        isHuman = "Human" in parameterSetName
        isPregnant = "Pregnant" in parameterSetName
        isHumanPregnant = isHuman and isPregnant
        isMouse = "Mouse" in parameterSetName
        isRat = "Rat" in parameterSetName
        isRodent = isMouse or isRat

        networkMesh = NetworkMesh(structure)
        networkMesh.create1DLayoutMesh(region)

        fieldmodule = region.getFieldmodule()
        mesh = fieldmodule.findMeshByDimension(1)

        # set up element annotations
        uterusGroup = AnnotationGroup(region, get_uterus_term("uterus"))
        if isRodent:
            leftOviductGroup = AnnotationGroup(region, get_uterus_term("left uterine horn"))
            rightOviductGroup = AnnotationGroup(region, get_uterus_term("right uterine horn"))
        else:
            leftOviductGroup = AnnotationGroup(region, get_uterus_term("left oviduct"))
            rightOviductGroup = AnnotationGroup(region, get_uterus_term("right oviduct"))
        bodyGroup = AnnotationGroup(region, get_uterus_term("body of uterus"))
        cervixGroup = AnnotationGroup(region, get_uterus_term("uterine cervix"))
        vaginaGroup = AnnotationGroup(region, get_uterus_term("vagina"))
        fundusPatchGroup = AnnotationGroup(region, get_uterus_term("fundus of uterus"))
        annotationGroups = [uterusGroup, leftOviductGroup, rightOviductGroup, fundusPatchGroup, bodyGroup,
                            cervixGroup, vaginaGroup]

        uterusMeshGroup = uterusGroup.getMeshGroup(mesh)
        elementIdentifier = 1

        left = 0
        right = 1

        if isRodent:
            oviductElementsCount = 16
            fundusPatchElementsCount = 3
            fundusPostBodyJunctionElementsCount = 3
            cervixElementsCount = 1
            vaginaElementsCount = 2
        elif isHumanPregnant:
            oviductElementsCount = 9
            fundusPatchElementsCount = 12
            fundusPostBodyJunctionElementsCount = 12
            cervixElementsCount = 1
            vaginaElementsCount = 4
        else:
            oviductElementsCount = 8
            fundusPatchElementsCount = 6
            fundusPostBodyJunctionElementsCount = 6
            cervixElementsCount = 2
            vaginaElementsCount = 7

        for side in (left, right):
            sideOviductGroup = leftOviductGroup if (side == left) else rightOviductGroup
            meshGroups = [uterusMeshGroup, sideOviductGroup.getMeshGroup(mesh)]
            for e in range(oviductElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1

        meshGroups = [uterusMeshGroup, fundusPatchGroup.getMeshGroup(mesh)]
        for e in range(fundusPatchElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1

        meshGroups = [uterusMeshGroup, bodyGroup.getMeshGroup(mesh)]
        for e in range(fundusPostBodyJunctionElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1

        meshGroups = [cervixGroup.getMeshGroup(mesh)]
        for e in range(cervixElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1

        meshGroups = [vaginaGroup.getMeshGroup(mesh)]
        for e in range(vaginaElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1

        # set coordinates (outer)
        fieldcache = fieldmodule.createFieldcache()
        coordinates = findOrCreateFieldCoordinates(fieldmodule)
        # need to ensure inner coordinates are at least defined:
        cls.defineInnerCoordinates(region, coordinates, options, networkMesh, innerProportion=0.75)
        innerCoordinates = findOrCreateFieldCoordinates(fieldmodule, "inner coordinates")
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodeIdentifier = 1

        # Calculate scales
        fundusScalePostBodyJunction = bodyLength / fundusPostBodyJunctionElementsCount
        cervicalScale = cervicalLength / cervixElementsCount
        vaginalScale = vaginaLength / vaginaElementsCount

        # Note: For rodents, the wall thickness is calculated from the uterine horn wall thickness and set to be
        # consistent from uterine horn to vagina. User-defined inner proportions of body, cervix and vagina for rat
        # and mouse will have no effect at all.
        if isRodent:
            wallThickness = oviductRadius - oviductRadius * innerProportionOviducts
            innerProportionBodyD2 = (halfFundusWidth - wallThickness) / halfFundusWidth
            innerProportionBodyD3 = (halfFundusDepth - wallThickness) / halfFundusDepth
            innerProportionCervixD2 = (halfCervicalWidthExternalOs - wallThickness) / halfCervicalWidthExternalOs
            innerProportionCervixD3 = (halfCervicalDepthExternalOs - wallThickness) / halfCervicalDepthExternalOs
            innerProportionVaginaD2 = (halfVaginaOrificeWidth - wallThickness) / halfVaginaOrificeWidth
            innerProportionVaginaD3 = (halfVaginaOrificeDepth - wallThickness) / halfVaginaOrificeDepth
        else:
            innerProportionBodyD2 = innerProportionBody
            innerProportionBodyD3 = innerProportionBody
            innerProportionCervixD2 = innerProportionCervix
            innerProportionCervixD3 = innerProportionCervix
            innerProportionVaginaD2 = innerProportionVagina
            innerProportionVaginaD3 = innerProportionVagina

        zero = [0.0, 0.0, 0.0]
        xBodyJunction = [0.0, 0.0, 0.0]

        if isHumanPregnant:
            aThetaCervicalEnd = math.asin(halfCervicalWidthInternalOs / halfFundusWidth)
            aEllipse = bodyLength / math.cos(aThetaCervicalEnd)
        cThetaCervicalEnd = math.asin(halfCervicalDepthInternalOs / halfFundusDepth)
        cEllipse = bodyLength / math.cos(cThetaCervicalEnd)

        # Oviducts
        d1BodyJunction = []
        d2BodyJunction = []
        d3BodyJunction = []
        d12BodyJunction = []
        d13BodyJunction = []
        id2BodyJunction = []
        id3BodyJunction = []
        id12BodyJunction = []
        id13BodyJunction = []

        if isRodent:
            rC = bodyLength
            thetaLimit = math.radians(55.0)
            for side in (left, right):
                rTheta = rC * thetaLimit
                straightLength = oviductLength - rTheta
                theta = thetaLimit * (-1.0 if side == left else 1.0)
                xCurveEnd = add([rC * math.cos(theta), rC * math.sin(theta), 0.0], [-rC, 0.0, 0.0])
                d1 = [(-1.0 if side == left else 1.0) * rC * math.sin(theta),
                      (1.0 if side == left else -1.0) * rC * math.cos(theta),
                      0.0]
                xStart = sub(xCurveEnd, set_magnitude(d1, straightLength))
                xEnd = xBodyJunction
                nx = [xStart, xCurveEnd, xEnd]
                nd1 = [set_magnitude(d1, straightLength), set_magnitude(d1, rTheta),
                       [0.0, (1.0 if side == left else -1.0) * rTheta, 0.0]]
                xOviduct, d1Oviduct = sampleCubicHermiteCurves(nx, nd1, oviductElementsCount)[0:2]

                for i in range(oviductElementsCount + 1):
                    x = xOviduct[i]
                    d1 = d1Oviduct[i]
                    d3 = [0.0, 0.0, oviductRadius]
                    d2Direction = cross(normalize(d3), normalize([-c for c in d1]))
                    d2 = set_magnitude(d2Direction, -oviductRadius)
                    d12 = zero
                    d13 = zero
                    id2 = mult(d2, innerProportionOviducts)
                    id3 = mult(d3, innerProportionOviducts)

                    if i < oviductElementsCount:
                        node = nodes.findNodeByIdentifier(nodeIdentifier)
                        fieldcache.setNode(node)
                        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
                        setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3)
                        nodeIdentifier += 1
                    else:
                        d1BodyJunction.append(d1)
                        d2BodyJunction.append(d2)
                        d3BodyJunction.append(d3)
                        d12BodyJunction.append(d12)
                        d13BodyJunction.append(d13)
                        id2BodyJunction.append(id2)
                        id3BodyJunction.append(id3)
                        id12BodyJunction.append(d12)
                        id13BodyJunction.append(d13)

        else:
            segment1Length = halfFundusWidth + oviductLength
            segment1LengthScale = segment1Length / oviductElementsCount
            elementsAlongHalfFundusWidth = math.ceil(halfFundusWidth / segment1LengthScale)

            for side in (left, right):
                lastTubeIdx = 0
                xStart = [0.0, segment1Length * (-1.0 if side == left else 1.0), 0.0]
                d1Oviduct = [0.0, segment1LengthScale * (1.0 if side == left else -1.0), 0.0]
                for i in range(oviductElementsCount + 1):
                    x = add(xStart, mult(d1Oviduct, i))
                    if segment1LengthScale * i < oviductLength:
                        d2 = [-oviductRadius if side == left else oviductRadius, 0.0, 0.0]
                        d3 = [0.0, 0.0, oviductRadius]
                        d12 = [0.0, 0.0, 0.0]
                        d13 = [0.0, 0.0, 0.0]
                        id2 = mult(d2, innerProportionOviducts)
                        id3 = mult(d3, innerProportionOviducts)
                        id12 = mult(d12, innerProportionOviducts)
                        id13 = mult(d13, innerProportionOviducts)
                    else:  # in ellipse zone
                        theta = math.acos(x[1] / halfFundusWidth)
                        if abs(halfFundusDepth * math.sin(theta)) < oviductRadius:
                            d2 = [-oviductRadius if side == left else oviductRadius, 0.0, 0.0]
                            d3 = [0.0, 0.0, oviductRadius]
                            d12 = [0.0, 0.0, 0.0]
                            d13 = [0.0, 0.0, 0.0]
                            id2 = mult(d2, innerProportionOviducts)
                            id3 = mult(d3, innerProportionOviducts)
                            id12 = mult(d12, innerProportionOviducts)
                            id13 = mult(d13, innerProportionOviducts)
                            lastTubeIdx = i
                        else:
                            elementsInEllipse = oviductElementsCount + 1 - lastTubeIdx
                            xiEllipse = (i - lastTubeIdx) / elementsInEllipse
                            d2 = [halfFundusDepth * math.sin(theta) * (-1.0 if side == left else 1.0), 0.0, 0.0]
                            d3 = [0.0, 0.0, halfFundusDepth * math.sin(theta)]
                            d12 = [halfFundusDepth * math.cos(theta) * (0.5 * math.pi / elementsAlongHalfFundusWidth),
                                   0.0, 0.0]
                            d13 = [0.0, 0.0,
                                   halfFundusDepth * math.cos(theta) * (0.5 * math.pi / elementsAlongHalfFundusWidth) *
                                   (-1.0 if side == left else 1.0)]
                            innerProportionFundus = \
                                interpolateCubicHermite([innerProportionOviducts, 0.0, 0.0],
                                                        [innerProportionBodyD2 - innerProportionOviducts, 0.0, 0.0],
                                                        [innerProportionBodyD2, 0.0, 0.0], [0.0, 0.0, 0.0], xiEllipse)[
                                    0]
                            id2 = mult(d2, innerProportionFundus if isHumanPregnant else innerProportionOviducts)
                            id3 = mult(d3, innerProportionFundus if isHumanPregnant else innerProportionOviducts)
                            id12 = mult(d12, innerProportionFundus if isHumanPregnant else innerProportionOviducts)
                            id13 = mult(d13, innerProportionFundus if isHumanPregnant else innerProportionOviducts)

                    if i < oviductElementsCount:
                        node = nodes.findNodeByIdentifier(nodeIdentifier)
                        fieldcache.setNode(node)
                        setNodeFieldParameters(coordinates, fieldcache, x, d1Oviduct, d2, d3, d12, d13)
                        setNodeFieldParameters(innerCoordinates, fieldcache, x, d1Oviduct, id2, id3, id12, id13)
                        nodeIdentifier += 1
                    else:
                        d1BodyJunction.append(d1Oviduct)
                        d2BodyJunction.append(d2)
                        d3BodyJunction.append(d3)
                        d12BodyJunction.append(d12)
                        d13BodyJunction.append(d13)
                        id2BodyJunction.append(id2)
                        id3BodyJunction.append(id3)
                        id12BodyJunction.append(id12)
                        id13BodyJunction.append(id13)

        xFundusPatchStart = [-bodyLength, 0.0, 0.0]
        d1FundusPatch = [bodyLength / fundusPatchElementsCount, 0.0, 0.0]
        nxPatch = []
        nd1Patch = []
        nd2Patch = []
        nd3Patch = []

        for i in range(fundusPatchElementsCount + 1):
            x = [xFundusPatchStart[0] + d1FundusPatch[0] * i, 0.0, 0.0]
            nxPatch.append(x)
            nd1Patch.append(d1FundusPatch)
            xi = i / fundusPatchElementsCount
            width = xi * halfFundusWidth + (1.0 - xi) * (halfFundusWidth * 0.01 if isRodent else
                                                         halfCervicalWidthInternalOs)
            if isHumanPregnant:
                thetaA = math.acos(x[0] / aEllipse)
                width = halfFundusWidth * math.sin(thetaA)
            thetaC = math.acos(x[0] / cEllipse)
            depth = halfFundusDepth * math.sin(thetaC)
            nd2Patch.append([0.0, width, 0.0])
            nd3Patch.append([0.0, 0.0, depth])
        nd13Patch = smoothCurveSideCrossDerivatives(nxPatch, nd1Patch, [nd3Patch])[0]
        nd12Patch = smoothCurveSideCrossDerivatives(nxPatch, nd1Patch, [nd2Patch])[0]

        for i in range(fundusPatchElementsCount):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            id3 = mult(nd3Patch[i], innerProportionBodyD3)
            id13 = mult(nd13Patch[i], innerProportionBodyD3)
            if isHumanPregnant:
                id2 = mult(nd2Patch[i], innerProportionBodyD2)
                id12 = mult(nd12Patch[i], innerProportionBodyD2)
            else:
                xi = i / fundusPatchElementsCount
                width = xi * halfFundusWidth * innerProportionBodyD2 + \
                        (1.0 - xi) * (halfFundusWidth * 0.01 * innerProportionBodyD2 if isRodent else
                                      halfCervicalWidthInternalOs * innerProportionCervixD2)

                id2 = [0.0, width, 0.0]
                id12 = [0.0,
                        (halfFundusWidth * innerProportionBodyD2 -
                         halfCervicalWidthInternalOs * innerProportionCervixD2) / fundusPatchElementsCount,
                        0.0]
            setNodeFieldParameters(coordinates, fieldcache, nxPatch[i], nd1Patch[i], nd2Patch[i], nd3Patch[i],
                                   nd12Patch[i], nd13Patch[i])
            setNodeFieldParameters(innerCoordinates, fieldcache, nxPatch[i], nd1Patch[i], id2, id3, id12, id13)
            nodeIdentifier += 1

        # body junction
        node = nodes.findNodeByIdentifier(nodeIdentifier)
        fieldcache.setNode(node)
        for field in (coordinates, innerCoordinates):
            field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, xBodyJunction)
        for side in (left, right):
            version = 1 if side == left else 2
            d1 = d1BodyJunction[version - 1]
            d2 = d2BodyJunction[version - 1]
            d3 = d3BodyJunction[version - 1]
            d12 = d12BodyJunction[version - 1]
            d13 = d13BodyJunction[version - 1]
            id2 = id2BodyJunction[version - 1]
            id3 = id3BodyJunction[version - 1]
            id12 = id12BodyJunction[version - 1]
            id13 = id13BodyJunction[version - 1]
            setNodeFieldVersionDerivatives(coordinates, fieldcache, version, d1, d2, d3, d12, d13)
            setNodeFieldVersionDerivatives(innerCoordinates, fieldcache, version, d1, id2, id3, id12, id13)

        version = 3
        id3 = mult(nd3Patch[-1], innerProportionBodyD3)
        id13 = mult(nd13Patch[-1], innerProportionBodyD3)
        if isHumanPregnant:
            id2 = mult(nd2Patch[-1], innerProportionBodyD2)
            id12 = mult(nd12Patch[-1], innerProportionBodyD2)
        else:
            id2 = [0.0, halfFundusWidth * innerProportionBodyD2, 0.0]
            id12 = [0.0,
                    (halfCervicalWidthInternalOs * innerProportionCervixD2 -
                     halfFundusWidth * innerProportionBodyD2) / fundusPatchElementsCount,
                    0.0]
        setNodeFieldVersionDerivatives(coordinates, fieldcache, version, nd1Patch[-1], nd2Patch[-1], nd3Patch[-1],
                                       nd12Patch[-1], nd13Patch[-1])
        setNodeFieldVersionDerivatives(innerCoordinates, fieldcache, version, nd1Patch[-1], id2, id3, id12, id13)

        # Post body junction
        nd12 = []
        nd13 = []
        nxBody = []
        nd1Body = []
        nd2Body = []
        nd3Body = []

        dWidth = (halfCervicalWidthInternalOs - halfFundusWidth) / fundusPostBodyJunctionElementsCount
        dDepth = (halfCervicalDepthInternalOs - halfFundusDepth) / fundusPostBodyJunctionElementsCount

        for i in range(fundusPostBodyJunctionElementsCount + 1):
            x = [fundusScalePostBodyJunction * i, 0.0, 0.0]
            d1 = [fundusScalePostBodyJunction, 0.0, 0.0]
            xi = i / fundusPostBodyJunctionElementsCount
            width = xi * halfCervicalWidthInternalOs + (1.0 - xi) * halfFundusWidth
            thetaC = math.acos(x[0] / cEllipse)
            depth = halfFundusDepth * math.sin(thetaC)
            if isHumanPregnant:
                thetaA = math.acos(x[0] / aEllipse)
                width = halfFundusWidth * math.sin(thetaA)
            d2 = [0.0, width, 0.0]
            d3 = [0.0, 0.0, depth]
            nxBody.append(x)
            nd1Body.append(d1)
            nd2Body.append(d2)
            nd3Body.append(d3)
            nd12.append([0.0, dWidth, 0.0])
            nd13.append([0.0, 0.0, dDepth])

        if dWidth == 0.0:
            nd12Body = nd12
        else:
            nd12Body = smoothCubicHermiteDerivativesLine(nd2Body, nd12)
        if dDepth == 0.0:
            nd13Body = nd13
        else:
            nd13Body = smoothCubicHermiteDerivativesLine(nd3Body, nd13)

        for i in range(fundusPostBodyJunctionElementsCount + 1):
            x = nxBody[i]
            d1 = nd1Body[i]
            d2 = nd2Body[i]
            d3 = nd3Body[i]
            d12 = nd12Body[i]
            d13 = nd13Body[i]

            if isHumanPregnant:
                id2 = mult(d2, innerProportionBodyD2)
                id12 = mult(d12, innerProportionBodyD2)
                id3 = mult(d3, innerProportionBodyD3)
                id13 = mult(d13, innerProportionBodyD3)
            else:
                xi = i / fundusPostBodyJunctionElementsCount
                width = xi * halfCervicalWidthInternalOs * innerProportionCervixD2 + \
                        (1.0 - xi) * halfFundusWidth * innerProportionBodyD2
                id2 = [0.0, width, 0.0]
                id12 = [0.0,
                        (halfCervicalWidthInternalOs * innerProportionCervixD2 -
                         halfFundusWidth * innerProportionBodyD2) / fundusPatchElementsCount,
                        0.0]
                depth = xi * halfCervicalDepthInternalOs * innerProportionCervixD3 + \
                        (1.0 - xi) * halfFundusDepth * innerProportionBodyD3
                id3 = [0.0, 0.0, depth]
                id13 = [0.0,
                        0.0,
                        (halfCervicalDepthInternalOs * innerProportionCervixD3 -
                         halfFundusDepth * innerProportionBodyD3) / fundusPatchElementsCount]
            if i:
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
            else:
                version = 4
                setNodeFieldVersionDerivatives(coordinates, fieldcache, version, d1, d2, d3, d12, d13)
                setNodeFieldVersionDerivatives(innerCoordinates, fieldcache, version, d1, id2, id3, id12, id13)
            nodeIdentifier += 1

        # Pre-calculate points in cervix and vagina
        rotMat = axis_angle_to_rotation_matrix([0.0, 1.0, 0.0], anteversionAngleRad)
        vaginalStartX = bodyLength + cervicalLength

        turningPtIdx = 1
        xTurningPt = [vaginalStartX + vaginalScale * turningPtIdx, 0.0, 0.0]
        xTranslateMat = sub(xTurningPt, xBodyJunction)

        li = list(range(2, vaginaElementsCount + 1))

        # one node before vagina
        nx = []
        nd1 = []
        nx.append([vaginalStartX - cervicalScale, 0.0, 0.0])
        nd1.append([cervicalScale, 0.0, 0.0])
        for i in li:
            x = [vaginalStartX + vaginalScale * i, 0.0, 0.0]
            d1 = [0.5 * (vaginalScale + cervicalScale) if i == 0 else vaginalScale, 0.0, 0.0]
            if i > turningPtIdx:
                xTranslate = sub(x, xTranslateMat)
                xRot = [rotMat[j][0] * xTranslate[0] + rotMat[j][1] * xTranslate[1] + rotMat[j][2] * xTranslate[2]
                        for j in range(3)]
                x = add(xRot, xTranslateMat)
                d1 = [rotMat[j][0] * d1[0] + rotMat[j][1] * d1[1] + rotMat[j][2] * d1[2] for j in range(3)]
            nx.append(x)
            nd1.append(d1)

        xSampledCurve, d1SampledCurve = sampleCubicHermiteCurves(nx, nd1, vaginaElementsCount + 1,
                                                                 arcLengthDerivatives=True)[0:2]
        d1SmoothedCurve = smoothCubicHermiteDerivativesLine(xSampledCurve, d1SampledCurve, fixStartDirection=True)

        # assign points to cervix
        nodeIdentifier -= 1
        cervixStartX = bodyLength
        nxCervix = [[cervixStartX, 0.0, 0.0]] + xSampledCurve[0:2]
        nd1Cervix = [[0.5 * (fundusScalePostBodyJunction + cervicalScale), 0.0, 0.0]] + d1SmoothedCurve[0:2]
        nxCervixSampled, nd1CervixSampled = \
            sampleCubicHermiteCurves(nxCervix, nd1Cervix, cervixElementsCount, arcLengthDerivatives=True)[0:2]
        nd1CervixSampled[0] = [0.5 * (fundusScalePostBodyJunction + cervicalScale), 0.0, 0.0]
        nd1CervixSmoothed = smoothCubicHermiteDerivativesLine(nxCervixSampled, nd1CervixSampled,
                                                              fixStartDerivative=True,
                                                              fixEndDerivative=True)
        nd2Cervix = []
        nd3Cervix = []
        for i in range(cervixElementsCount + 1):
            xi = i / cervixElementsCount
            d1 = nd1CervixSmoothed[i]
            halfCervixWidth = xi * halfCervicalWidthExternalOs + (1 - xi) * halfCervicalWidthInternalOs
            halfCervixDepth = xi * halfCervicalDepthExternalOs + (1 - xi) * halfCervicalDepthInternalOs
            d2 = [0.0, halfCervixWidth, 0.0]
            nd2Cervix.append(d2)
            d3Direction = cross(normalize(d1), normalize(d2))
            nd3Cervix.append(set_magnitude(d3Direction, halfCervixDepth))

        nxVagina = []
        nd1Vagina = []
        nd2Vagina = []
        nd3Vagina = []

        del xSampledCurve[0]
        del d1SmoothedCurve[0]

        for i in range(vaginaElementsCount + 1):
            xi = i / vaginaElementsCount
            nxVagina.append(xSampledCurve[i])
            nd1Vagina.append(d1SmoothedCurve[i])
            d2 = \
                interpolateCubicHermite(
                    [0.0, halfCervicalWidthExternalOs, 0.0],
                    [0.0, (halfCervicalWidthExternalOs - halfCervicalWidthInternalOs) / cervixElementsCount, 0.0],
                    [0.0, halfVaginaOrificeWidth, 0.0],
                    [0.0, (halfVaginaOrificeWidth - halfCervicalWidthExternalOs) / vaginaElementsCount, 0.0], xi)

            if i > turningPtIdx:
                d2 = [rotMat[j][0] * d2[0] + rotMat[j][1] * d2[1] + rotMat[j][2] * d2[2] for j in range(3)]
            nd2Vagina.append(d2)

            d3 = cross(normalize(d1SmoothedCurve[i]), normalize(d2))
            d3Interpolated = \
                interpolateCubicHermite(
                    [0.0, 0.0, halfCervicalDepthExternalOs],
                    [0.0, 0.0, (halfCervicalDepthExternalOs - halfCervicalDepthInternalOs) / cervixElementsCount],
                    [0.0, 0.0, halfVaginaOrificeDepth],
                    [0.0, 0.0, (halfVaginaOrificeDepth - halfCervicalDepthExternalOs) / vaginaElementsCount], xi)
            d3 = set_magnitude(d3, magnitude(d3Interpolated))
            nd3Vagina.append(d3)

        nd2 = nd2Cervix + nd2Vagina[1:]
        nd3 = nd3Cervix + nd3Vagina[1:]
        nd12 = [sub(nd2[c + 1], nd2[c]) for c in range(len(nd2) - 1)]
        nd12.append(nd12[-1])
        nd13 = [sub(nd3[c + 1], nd3[c]) for c in range(len(nd3) - 1)]
        nd13.append(nd13[-1])

        if halfCervicalWidthExternalOs - halfCervicalWidthInternalOs:
            nd12 = smoothCubicHermiteDerivativesLine(nd2, nd12)
        if halfCervicalDepthExternalOs - halfCervicalDepthInternalOs:
            nd13 = smoothCubicHermiteDerivativesLine(nd3, nd13)

        nd12Cervix = nd12[0:cervixElementsCount + 1]
        nd12Vagina = nd12[cervixElementsCount:]
        nd13Cervix = nd13[0:cervixElementsCount + 1]
        nd13Vagina = nd13[cervixElementsCount:]

        for i in range(cervixElementsCount):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = nxCervixSampled[i]
            d1 = nd1CervixSmoothed[i]
            d2 = nd2Cervix[i]
            d3 = nd3Cervix[i]
            if i == 0:
                d12 = mult(add(nd12Body[-1], nd12Cervix[i]), 0.5)
                d13 = mult(add(nd13Body[-1], nd13Cervix[i]), 0.5)
            else:
                d12 = nd12Cervix[i]
                d13 = nd13Cervix[i]

            id2 = mult(d2, innerProportionCervixD2)
            id3 = mult(d3, innerProportionCervixD3)
            id12 = mult(d12, innerProportionCervixD2)
            id13 = mult(d13, innerProportionCervixD3)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
            nodeIdentifier += 1

        for i in range(vaginaElementsCount + 1):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = nxVagina[i]
            d1 = nd1Vagina[i]
            d2 = nd2Vagina[i]
            d3 = nd3Vagina[i]
            d12 = nd12Vagina[i]
            d13 = nd13Vagina[i]
            if i == 0:
                innerProportionD2 = innerProportionCervixD2
                innerProportionD3 = innerProportionCervixD3
            else:
                innerProportionD2 = innerProportionVaginaD2
                innerProportionD3 = innerProportionVaginaD3
            id2 = mult(d2, innerProportionD2)
            id3 = mult(d3, innerProportionD3)
            id12 = mult(d12, innerProportionD2)
            id13 = mult(d13, innerProportionD3)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
            nodeIdentifier += 1

        return annotationGroups, networkMesh

    @classmethod
    def getInteractiveFunctions(cls):
        """
        Edit base class list to include only valid functions.
        """
        interactiveFunctions = super(MeshType_1d_uterus_network_layout1, cls).getInteractiveFunctions()
        for interactiveFunction in interactiveFunctions:
            if interactiveFunction[0] == "Edit structure...":
                interactiveFunctions.remove(interactiveFunction)
                break
        return interactiveFunctions


class MeshType_3d_uterus1(Scaffold_base):
    """
    Generates a 3-D uterus mesh from a 1-D network layout with variable numbers of elements around, along and through
    wall.
    Magnitude of D2 and D3 are the radii of the uterus in the respective directions.
    """

    @classmethod
    def getName(cls):
        return '3D Uterus 1'

    @classmethod
    def getParameterSetNames(cls):
        return [
            'Default',
            'Human 1',
            'Human Pregnant 1',
            'Mouse 1',
            'Rat 1']

    @classmethod
    def getDefaultOptions(cls, parameterSetName='Default'):
        useParameterSetName = "Human 1" if (parameterSetName == "Default") else parameterSetName
        options = {
            'Base parameter set': useParameterSetName,
            'Network layout': ScaffoldPackage(MeshType_1d_uterus_network_layout1,
                                              defaultParameterSetName=useParameterSetName),
            'Number of elements around': 20,
            'Number of elements around oviduct/uterine horn': 8,
            'Number of elements through wall': 1,
            'Number of elements along oviduct/uterine horn': 6,
            'Number of elements along body': 5,
            'Number of elements along cervix': 1,
            'Number of elements along vagina': 4,
            'Use linear through wall': True,
            'Show trim surfaces': False,
            'Refine': False,
            'Refine number of elements along': 4,
            'Refine number of elements around': 4,
            'Refine number of elements through wall': 1
        }
        if 'Mouse' in parameterSetName or 'Rat' in parameterSetName:
            options['Number of elements around'] = 12
            options['Number of elements around oviduct/uterine horn'] = 8
            options['Number of elements along oviduct/uterine horn'] = 12
            options['Number of elements along body'] = 2
            options['Number of elements along cervix'] = 1
            options['Number of elements along vagina'] = 2

        if 'Rat' in parameterSetName:
            options['Use linear through wall'] = False  # True is not implemented for rat due to septum
            options['Refine number of elements through wall'] = 4

        return options

    @classmethod
    def getOrderedOptionNames(cls):
        optionNames = [
            'Network layout',
            'Number of elements around',
            'Number of elements around oviduct/uterine horn',
            'Number of elements through wall',
            'Number of elements along oviduct/uterine horn',
            'Number of elements along body',
            'Number of elements along cervix',
            'Number of elements along vagina',
            'Use linear through wall',
            'Show trim surfaces',
            'Refine',
            'Refine number of elements along',
            'Refine number of elements around',
            'Refine number of elements through wall']
        return optionNames

    @classmethod
    def getOptionValidScaffoldTypes(cls, optionName):
        if optionName == 'Network layout':
            return [MeshType_1d_uterus_network_layout1]
        return []

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
            return ScaffoldPackage(MeshType_1d_uterus_network_layout1, defaultParameterSetName=parameterSetName)
        assert False, cls.__name__ + '.getOptionScaffoldPackage:  Option ' + optionName + ' is not a scaffold'

    @classmethod
    def checkOptions(cls, options):
        if (options["Network layout"].getScaffoldType() not in
                cls.getOptionValidScaffoldTypes("Network layout")):
            options["Network layout"] = ScaffoldPackage(MeshType_1d_uterus_network_layout1)
        for key in [
            'Number of elements through wall',
            'Refine number of elements along',
            'Refine number of elements around',
            'Refine number of elements through wall'
        ]:
            if options[key] < 1:
                options[key] = 1

        for key in [
            'Number of elements around',
            'Number of elements around oviduct/uterine horn']:
            if options[key] < 8:
                options[key] = 8
            elif (options[key] % 4) > 0:
                options[key] += options[key] % 4

        parameterSetName = options["Base parameter set"]
        dependentChanges = False
        isRat = "Rat" in parameterSetName

        if isRat and options["Use linear through wall"]:
            options["Use linear through wall"] = False
            dependentChanges = True

        # Only 8 around oviduct works for rat at the moment
        if isRat and options["Number of elements around oviduct/uterine horn"] > 8:
            options["Number of elements around oviduct/uterine horn"] = 8
            dependentChanges = True

        if options["Number of elements around oviduct/uterine horn"] >= options["Number of elements around"]:
            if isRat:
                options["Number of elements around oviduct/uterine horn"] = 8
                options["Number of elements around"] = 12
            else:
                options["Number of elements around oviduct/uterine horn"] = options["Number of elements around"] - 4
            dependentChanges = True

        return dependentChanges

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the base tricubic hermite or bicubic hermite-linear mesh. See also generateMesh().
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: None
        """
        parameterSetName = options['Base parameter set']
        isHuman = "Human" in parameterSetName
        isRat = "Rat" in parameterSetName
        isMouse = "Mouse" in parameterSetName
        isRodent = isRat or isMouse

        layoutRegion = region.createRegion()
        networkLayout = options["Network layout"]
        networkLayout.generate(layoutRegion)  # ask scaffold to generate to get user-edited parameters
        layoutAnnotationGroups = networkLayout.getAnnotationGroups()
        networkMesh = networkLayout.getConstructionObject()

        annotationElementsCountsAlong = []
        annotationElementsCountsAround = []
        for layoutAnnotationGroup in layoutAnnotationGroups:
            elementsCountAlong = 0
            elementsCountAround = 0
            name = layoutAnnotationGroup.getName()
            if "oviduct" in name or "uterine horn" in name:
                elementsCountAlong = options['Number of elements along oviduct/uterine horn']
            elif "body" in name or "fundus" in name:
                elementsCountAlong = options['Number of elements along body']
            elif "cervix" in name:
                elementsCountAlong = options['Number of elements along cervix']
            elif "vagina" in name:
                elementsCountAlong = options['Number of elements along vagina']
            annotationElementsCountsAlong.append(elementsCountAlong)

            if "oviduct" in name or "uterine horn" in name or "left fundus" in name or "right fundus" in name:
                elementsCountAround = options['Number of elements around oviduct/uterine horn']
            elif "body" in name or "fundus" in name:
                elementsCountAround = options['Number of elements around']
            annotationElementsCountsAround.append(elementsCountAround)

        uterusTubeNetworkMeshBuilder = UterusTubeNetworkMeshBuilder(
            networkMesh,
            targetElementDensityAlongLongestSegment=2.0,  # not used
            layoutAnnotationGroups=layoutAnnotationGroups,
            annotationElementsCountsAlong=annotationElementsCountsAlong,
            defaultElementsCountAround=options['Number of elements around'],
            annotationElementsCountsAround=annotationElementsCountsAround,
            elementsCountThroughShell=options["Number of elements through wall"],
            useOuterTrimSurfaces=False)
        uterusTubeNetworkMeshBuilder.build()

        generateData = UterusTubeNetworkMeshGenerateData(
            region, 3,
            isLinearThroughShell=options["Use linear through wall"],
            isShowTrimSurfaces=options["Show trim surfaces"])

        uterusTubeNetworkMeshBuilder.generateMesh(generateData)
        annotationGroups = generateData.getAnnotationGroups()
        nodeIdentifier, elementIdentifier = generateData.getNodeElementIdentifiers()

        fieldmodule = region.getFieldmodule()
        coordinates = findOrCreateFieldCoordinates(fieldmodule)
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        mesh = fieldmodule.findMeshByDimension(3)

        # add human specific annotations
        if isHuman:
            allMarkers = \
                ["junction of left round ligament with uterus",
                 "junction of right round ligament with uterus",
                 "junction of left uterosacral ligament with uterus",
                 "junction of right uterosacral ligament with uterus"]

            leftUterusGroup = getAnnotationGroupForTerm(annotationGroups, ("left uterus", ""))
            rightUterusGroup = getAnnotationGroupForTerm(annotationGroups, ("right uterus", ""))
            dorsalUterusGroup = getAnnotationGroupForTerm(annotationGroups, ("dorsal uterus", ""))
            ventralUterusGroup = getAnnotationGroupForTerm(annotationGroups, ("ventral uterus", ""))
            fundusGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("fundus of uterus"))
            leftOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("left oviduct"))
            rightOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("right oviduct"))
            bodyNotCervixGroup = getAnnotationGroupForTerm(annotationGroups, ("body not cervix", ""))
            cervixGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("uterine cervix"))

            fieldmodule.defineAllFaces()
            mesh2d = fieldmodule.findMeshByDimension(2)
            groupsFindMarkers = [dorsalUterusGroup, ventralUterusGroup,
                                 fundusGroup, leftOviductGroup, rightOviductGroup,
                                 bodyNotCervixGroup, cervixGroup]
            for group in groupsFindMarkers:
                group.addSubelements()

            is_left = leftUterusGroup.getGroup()
            is_right = rightUterusGroup.getGroup()
            is_dorsal = dorsalUterusGroup.getGroup()
            is_ventral = ventralUterusGroup.getGroup()
            is_frontal_plane = fieldmodule.createFieldAnd(is_dorsal, is_ventral)

            is_exterior = fieldmodule.createFieldIsExterior()
            is_exterior_face_xi3_1 = \
                fieldmodule.createFieldAnd(is_exterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
            is_fundus = fundusGroup.getGroup()
            is_fundus_serosa = fieldmodule.createFieldAnd(is_fundus, is_exterior_face_xi3_1)
            fundusSerosa = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("fundus serosa", ""))
            fundusSerosa.getMeshGroup(mesh2d).addElementsConditional(is_fundus_serosa)
            fundus_serosa_meshgroup = fundusSerosa.getMeshGroup(mesh2d)

            is_cervix = cervixGroup.getGroup()
            is_cervix_serosa = fieldmodule.createFieldAnd(is_cervix, is_exterior_face_xi3_1)
            cervixSerosa = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("cervix serosa", ""))
            cervixSerosa.getMeshGroup(mesh2d).addElementsConditional(is_cervix_serosa)
            cervix_serosa_meshgroup = cervixSerosa.getMeshGroup(mesh2d)

            is_round_ligament_left = \
                fieldmodule.createFieldAnd(is_frontal_plane,
                                           fieldmodule.createFieldAnd(fundusSerosa.getGroup(),
                                                                      leftOviductGroup.getGroup()))
            is_round_ligament_right = \
                fieldmodule.createFieldAnd(is_frontal_plane,
                                           fieldmodule.createFieldAnd(fundusSerosa.getGroup(),
                                                                      rightOviductGroup.getGroup()))
            is_left_frontal = fieldmodule.createFieldAnd(is_left, is_frontal_plane)
            is_uterosacral_ligament_left = \
                fieldmodule.createFieldAnd(is_left_frontal,
                                           fieldmodule.createFieldAnd(bodyNotCervixGroup.getGroup(),
                                                                      cervixSerosa.getGroup()))
            is_right_frontal = fieldmodule.createFieldAnd(is_right, is_frontal_plane)
            is_uterosacral_ligament_right = \
                fieldmodule.createFieldAnd(is_right_frontal,
                                           fieldmodule.createFieldAnd(bodyNotCervixGroup.getGroup(),
                                                                      cervixSerosa.getGroup()))

            conditional_fields = [is_round_ligament_left, is_round_ligament_right,
                                  is_uterosacral_ligament_left, is_uterosacral_ligament_right]
            for i in range(len(conditional_fields)):
                junction_node = find_first_node_conditional(nodes, conditional_fields[i])
                element_junction, xi_junction = \
                    get_node_mesh_location(junction_node, coordinates, mesh, coordinates,
                                           search_mesh=fundus_serosa_meshgroup if i < 2 else cervix_serosa_meshgroup)
                group = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                           get_uterus_term(allMarkers[i]), isMarker=True)
                markerNode = group.createMarkerNode(nodeIdentifier, element=element_junction, xi=xi_junction)
                nodeIdentifier = markerNode.getIdentifier() + 1

            annotationGroups.remove(fundusSerosa)
            annotationGroups.remove(cervixSerosa)

        uterusGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("uterus"))
        myometriumGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, get_uterus_term("myometrium"))
        myometriumGroup.getMeshGroup(mesh).addElementsConditional(uterusGroup.getGroup())

        cervixGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("uterine cervix"))
        uterusGroup.getMeshGroup(mesh).addElementsConditional(cervixGroup.getGroup())

        if isRodent:
            leftOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("left oviduct"))
            rightOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("right oviduct"))
            annotationGroups.remove(leftOviductGroup)
            annotationGroups.remove(rightOviductGroup)

        # add septum for rat
        if isRat:
            # find wall thickness
            leftUterineHornGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("left uterine horn"))
            leftUterineHornNodeset = leftUterineHornGroup.getGroup().getNodesetGroup(nodes)
            nodeIter = leftUterineHornNodeset.createNodeiterator()
            node = nodeIter.next()
            fieldcache = fieldmodule.createFieldcache()
            if node.isValid():
                firstHornNodeId = node.getIdentifier()
                fieldcache.setNode(node)
                xFirstNode = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)[1]
            matchingInnerNid = firstHornNodeId + options['Number of elements around oviduct/uterine horn'] * \
                               options['Number of elements through wall']
            node = nodes.findNodeByIdentifier(matchingInnerNid)
            fieldcache.setNode(node)
            xMatchingInnerNode = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)[1]
            septumThickness = magnitude(sub(xFirstNode, xMatchingInnerNode))

            firstSeptumElementIdentifier = elementIdentifier
            elementsCountAroundSeptum = \
                (options['Number of elements around'] - options['Number of elements around oviduct/uterine horn']) // 2
            elementsCountThroughSeptum = options['Number of elements around oviduct/uterine horn'] // 2
            elementsCountAlongSeptum = options['Number of elements along body'] + \
                                       options['Number of elements along cervix'] + 2
            elementCountsSeptum = [elementsCountAroundSeptum, elementsCountAlongSeptum, elementsCountThroughSeptum]
            elementsCountAround = options['Number of elements around']

            septumRimNids, septumRimParams = \
                getSeptumRimNodes(region, fieldmodule, annotationGroups, elementCountsSeptum, elementsCountAround)

            ratSeptum = Septum(elementCountsSeptum, septumRimNids, septumRimParams, septumThickness)
            ratSeptum.build()
            nextNodeIdentifier, nextElementIdentifier = \
                ratSeptum.generateMesh(fieldmodule, coordinates, nodeIdentifier, elementIdentifier, annotationGroups,
                                       region)

            # Make annotation groups for rat
            lastSeptumElementIdentifier = nextElementIdentifier - 1
            elementsCountSeptumCervix = elementsCountThroughSeptum * elementsCountAroundSeptum * \
                                        options['Number of elements along cervix']
            eidsSeptumBody = list(range(firstSeptumElementIdentifier,
                                        lastSeptumElementIdentifier - elementsCountSeptumCervix + 1))
            eidsSeptumCervix = list(range(lastSeptumElementIdentifier - elementsCountSeptumCervix + 1,
                                          lastSeptumElementIdentifier + 1))
            bodyGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term('body of uterus'))

            bodySeptumGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("septum body", ""))
            cervixSeptumGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("septum cervix", ""))

            for i in eidsSeptumBody:
                element = mesh.findElementByIdentifier(i)
                bodySeptumGroup.getMeshGroup(mesh).addElement(element)
                bodyGroup.getMeshGroup(mesh).addElement(element)
                uterusGroup.getMeshGroup(mesh).addElement(element)
                myometriumGroup.getMeshGroup(mesh).addElement(element)

            for i in eidsSeptumCervix:
                element = mesh.findElementByIdentifier(i)
                cervixSeptumGroup.getMeshGroup(mesh).addElement(element)

            leftSeptumGroup = getAnnotationGroupForTerm(annotationGroups, ("left", ""))
            rightSeptumGroup = getAnnotationGroupForTerm(annotationGroups, ("right", ""))
            dorsalSeptumGroup = getAnnotationGroupForTerm(annotationGroups, ("dorsal", ""))
            ventralSeptumGroup = getAnnotationGroupForTerm(annotationGroups, ("ventral", ""))
            septumGroups = [leftSeptumGroup, rightSeptumGroup, dorsalSeptumGroup, ventralSeptumGroup]

            leftUterusGroup = getAnnotationGroupForTerm(annotationGroups, ("left uterus", ""))
            rightUterusGroup = getAnnotationGroupForTerm(annotationGroups, ("right uterus", ""))
            dorsalUterusGroup = getAnnotationGroupForTerm(annotationGroups, ("dorsal uterus", ""))
            ventralUterusGroup = getAnnotationGroupForTerm(annotationGroups, ("ventral uterus", ""))
            uterusGroups = [leftUterusGroup, rightUterusGroup, dorsalUterusGroup, ventralUterusGroup]

            for i in range(len(uterusGroups)):
                uterusMeshGroup = uterusGroups[i].getMeshGroup(mesh)
                uterusMeshGroup.addElementsConditional(septumGroups[i].getGroup())
                annotationGroups.remove(septumGroups[i])

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
        """
        Add face annotation groups from the highest dimension mesh.
        Must have defined faces and added subelements for highest dimension groups.
        :param region: Zinc region containing model.
        :param options: Dict containing options. See getDefaultOptions().
        :param annotationGroups: List of annotation groups for top-level elements.
        New face annotation groups are appended to this list.
        """
        parameterSetName = options['Base parameter set']
        isHuman = "Human" in parameterSetName
        isMouse = "Mouse" in parameterSetName
        isRat = "Rat" in parameterSetName
        isRodent = isMouse or isRat

        # Create 2d surface mesh groups
        fm = region.getFieldmodule()
        mesh1d = fm.findMeshByDimension(1)
        mesh2d = fm.findMeshByDimension(2)

        uterusGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("uterus"))
        if isHuman:
            leftOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("left oviduct"))
            rightOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("right oviduct"))
        elif isRodent:
            leftOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("left uterine horn"))
            rightOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("right uterine horn"))
        bodyGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("body of uterus"))
        fundusGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("fundus of uterus"))
        cervixGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("uterine cervix"))
        vaginaGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("vagina"))

        is_exterior = fm.createFieldIsExterior()
        is_exterior_face_xi1_1 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI1_1))
        is_exterior_face_xi1_0 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI1_0))
        is_exterior_face_xi2_0 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI2_0))
        is_exterior_face_xi2_1 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI2_1))
        is_exterior_face_xi3_1 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
        is_exterior_face_xi3_0 = fm.createFieldAnd(is_exterior, fm.createFieldIsOnFace(Element.FACE_TYPE_XI3_0))

        is_uterus = uterusGroup.getGroup()
        is_uterus_outer = fm.createFieldAnd(is_uterus, is_exterior_face_xi3_1)
        is_uterus_inner = fm.createFieldAnd(is_uterus, is_exterior_face_xi3_0)

        bodyNotCervixGroup = getAnnotationGroupForTerm(annotationGroups, ("body not cervix", ""))
        isBodyNotCervix = bodyNotCervixGroup.getGroup()
        is_bodyNotCervix_outer = fm.createFieldAnd(isBodyNotCervix, is_exterior_face_xi3_1)
        is_bodyNotCervix_inner = fm.createFieldAnd(isBodyNotCervix, is_exterior_face_xi3_0)

        is_leftOviduct = leftOviductGroup.getGroup()
        is_rightOviduct = rightOviductGroup.getGroup()

        is_fundus = fundusGroup.getGroup()
        is_fundus_outer = fm.createFieldAnd(is_fundus, is_exterior_face_xi3_1)
        is_fundus_inner = fm.createFieldAnd(is_fundus, is_exterior_face_xi3_0)

        is_body = bodyGroup.getGroup()
        is_body_outer = fm.createFieldAnd(is_body, is_exterior_face_xi3_1)

        is_cervix = cervixGroup.getGroup()
        is_cervix_outer = fm.createFieldAnd(is_cervix, is_exterior_face_xi3_1)
        is_cervix_inner = fm.createFieldAnd(is_cervix, is_exterior_face_xi3_0)

        is_vagina = vaginaGroup.getGroup()
        is_vagina_outer = fm.createFieldAnd(is_vagina, is_exterior_face_xi3_1)
        is_vagina_inner = fm.createFieldAnd(is_vagina, is_exterior_face_xi3_0)
        is_vagina_xi2_0 = fm.createFieldAnd(is_vagina, is_exterior_face_xi2_0)
        is_vagina_xi2_1 = fm.createFieldAnd(is_vagina, is_exterior_face_xi2_1)
        is_vagina_xi2_01 = fm.createFieldXor(is_vagina_xi2_0, is_vagina_xi2_1)

        serosaOfUterus = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                            get_uterus_term("serosa of uterus"))
        serosaOfUterus.getMeshGroup(mesh2d).addElementsConditional(is_uterus_outer)
        serosaOfUterus.getMeshGroup(mesh2d).addElementsConditional(is_cervix_outer)

        uterineCavity = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                           get_uterus_term("uterine cavity"))
        uterineCavity.getMeshGroup(mesh2d).addElementsConditional(is_uterus_inner)
        uterineCavity.getMeshGroup(mesh2d).removeElementsConditional(is_leftOviduct)
        uterineCavity.getMeshGroup(mesh2d).removeElementsConditional(is_rightOviduct)
        uterineCavity.getMeshGroup(mesh2d).removeElementsConditional(is_cervix)

        serosaOfBody = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                          get_uterus_term("serosa of body of uterus"))
        serosaOfBody.getMeshGroup(mesh2d).addElementsConditional(is_body_outer)
        serosaOfBody.getMeshGroup(mesh2d).addElementsConditional(is_cervix_outer)

        lumenOfFundus = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                           get_uterus_term("lumen of fundus of uterus"))
        lumenOfFundus.getMeshGroup(mesh2d).addElementsConditional(is_fundus_inner)

        serosaOfFundus = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                            get_uterus_term("serosa of fundus of uterus"))
        serosaOfFundus.getMeshGroup(mesh2d).addElementsConditional(is_fundus_outer)

        lumenOfBody = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                         get_uterus_term("lumen of body of uterus"))
        lumenOfBody.getMeshGroup(mesh2d).addElementsConditional(is_bodyNotCervix_inner)

        serosaOfVagina = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                            get_uterus_term("serosa of vagina"))
        serosaOfVagina.getMeshGroup(mesh2d).addElementsConditional(is_vagina_outer)
        serosaOfVagina.getMeshGroup(mesh2d).addElementsConditional(is_cervix_outer)

        lumenOfVagina = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                           get_uterus_term("vaginal canal"))
        lumenOfVagina.getMeshGroup(mesh2d).addElementsConditional(is_vagina_inner)

        lumenOfUterusVagina = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                 get_uterus_term("lumen of uterus, cervix and vagina"))
        lumenOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(is_uterus_inner)
        lumenOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(is_cervix_inner)
        lumenOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(is_vagina_inner)

        lumenOfUterusCervix = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                 get_uterus_term("lumen of uterus and cervix"))
        lumenOfUterusCervix.getMeshGroup(mesh2d).addElementsConditional(is_uterus_inner)
        lumenOfUterusCervix.getMeshGroup(mesh2d).addElementsConditional(is_cervix_inner)

        is_lumenOfUterusCervix = lumenOfUterusCervix.getGroup()
        is_lumenOfUterusVagina = lumenOfUterusVagina.getGroup()

        serosaOfUterusVagina = \
            findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                               get_uterus_term("serosa of uterus and vagina"))
        serosaOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(is_uterus_outer)
        serosaOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(is_cervix_outer)
        serosaOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(is_vagina_outer)
        is_serosaOfUterusVagina = serosaOfUterusVagina.getGroup()

        if isRat:
            is_exterior_face_xi1 = fm.createFieldOr(is_exterior_face_xi1_0, is_exterior_face_xi1_1)
            septumBodyGroup = getAnnotationGroupForTerm(annotationGroups, ("septum body", ""))
            isSeptumBody = septumBodyGroup.getGroup()
            isSeptumBodyExterior = fm.createFieldAnd(is_exterior_face_xi1, isSeptumBody)
            lumenOfBody.getMeshGroup(mesh2d).addElementsConditional(isSeptumBodyExterior)
            uterineCavity.getMeshGroup(mesh2d).addElementsConditional(isSeptumBodyExterior)

            septumGroup = getAnnotationGroupForTerm(annotationGroups, ("septum", ""))
            isSeptum = septumGroup.getGroup()
            isSeptumExterior = fm.createFieldAnd(is_exterior, isSeptum)
            is_lumenOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(isSeptumExterior)
            isSeptumExteriorXi1 = fm.createFieldAnd(is_exterior_face_xi1, isSeptum)
            is_lumenOfUterusCervix.getMeshGroup(mesh2d).addElementsConditional(isSeptumExteriorXi1)

        leftGroup = getAnnotationGroupForTerm(annotationGroups, ("left uterus", ""))
        rightGroup = getAnnotationGroupForTerm(annotationGroups, ("right uterus", ""))
        dorsalGroup = getAnnotationGroupForTerm(annotationGroups, ("dorsal uterus", ""))
        ventralGroup = getAnnotationGroupForTerm(annotationGroups, ("ventral uterus", ""))

        isLeft = leftGroup.getGroup()
        isRight = rightGroup.getGroup()

        isLeftLumenOfUterusVagina = fm.createFieldAnd(isLeft, is_lumenOfUterusVagina)
        leftLumenOfUterusVagina = \
            findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                               get_uterus_term("left lumen of uterus, cervix and vagina"))
        leftLumenOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(isLeftLumenOfUterusVagina)

        isLeftLumenOfUterusCervix = fm.createFieldAnd(isLeft, is_lumenOfUterusCervix)
        leftLumenOfUterusCervix = \
            findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                               get_uterus_term("left lumen of uterus and cervix"))
        leftLumenOfUterusCervix.getMeshGroup(mesh2d).addElementsConditional(isLeftLumenOfUterusCervix)

        isRightLumenOfUterusVagina = fm.createFieldAnd(isRight, is_lumenOfUterusVagina)
        rightLumenOfUterusVagina = \
            findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                               get_uterus_term("right lumen of uterus, cervix and vagina"))
        rightLumenOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(isRightLumenOfUterusVagina)

        isRightLumenOfUterusCervix = fm.createFieldAnd(isRight, is_lumenOfUterusCervix)
        rightLumenOfUterusCervix = \
            findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                               get_uterus_term("right lumen of uterus and cervix"))
        rightLumenOfUterusCervix.getMeshGroup(mesh2d).addElementsConditional(isRightLumenOfUterusCervix)

        isLeftSerosaOfUterusVagina = fm.createFieldAnd(isLeft, is_serosaOfUterusVagina)
        leftSerosaOfUterusVagina = \
            findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                               get_uterus_term("left serosa of uterus and vagina"))
        leftSerosaOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(isLeftSerosaOfUterusVagina)

        isRightSerosaOfUterusVagina = fm.createFieldAnd(isRight, is_serosaOfUterusVagina)
        rightSerosaOfUterusVagina = \
            findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                               get_uterus_term("right serosa of uterus and vagina"))
        rightSerosaOfUterusVagina.getMeshGroup(mesh2d).addElementsConditional(isRightSerosaOfUterusVagina)

        if isHuman:
            leftOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("left oviduct"))
            is_leftOviduct = leftOviductGroup.getGroup()
            is_leftOviduct_outer = fm.createFieldAnd(is_leftOviduct, is_exterior_face_xi3_1)
            is_leftOviduct_inner = fm.createFieldAnd(is_leftOviduct, is_exterior_face_xi3_0)

            serosaOfLeftOviduct = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                     get_uterus_term("serosa of left oviduct"))
            serosaOfLeftOviduct.getMeshGroup(mesh2d).addElementsConditional(is_leftOviduct_outer)

            rightOviductGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("right oviduct"))
            is_rightOviduct = rightOviductGroup.getGroup()
            is_rightOviduct_outer = fm.createFieldAnd(is_rightOviduct, is_exterior_face_xi3_1)
            is_rightOviduct_inner = fm.createFieldAnd(is_rightOviduct, is_exterior_face_xi3_0)

            serosaOfRightOviduct = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                      get_uterus_term("serosa of right oviduct"))
            serosaOfRightOviduct.getMeshGroup(mesh2d).addElementsConditional(is_rightOviduct_outer)

            lumenOfLeftOviduct = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                    get_uterus_term("lumen of left oviduct"))
            lumenOfLeftOviduct.getMeshGroup(mesh2d).addElementsConditional(is_leftOviduct_inner)

            lumenOfRightOviduct = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                     get_uterus_term("lumen of right oviduct"))
            lumenOfRightOviduct.getMeshGroup(mesh2d).addElementsConditional(is_rightOviduct_inner)

            is_pubocervical = fm.createFieldAnd(is_bodyNotCervix_outer, is_cervix_outer)
            pubocervical = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                              get_uterus_term("pubocervical ligament"))
            pubocervical.getMeshGroup(mesh1d).addElementsConditional(is_pubocervical)

            is_internal_os = fm.createFieldAnd(is_bodyNotCervix_inner, is_cervix_inner)
            internalOs = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                            get_uterus_term("internal cervical os"))
            internalOs.getMeshGroup(mesh1d).addElementsConditional(is_internal_os)

            is_external_os = fm.createFieldAnd(is_vagina_inner, is_cervix_inner)
            externalOs = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                            get_uterus_term("external cervical os"))
            externalOs.getMeshGroup(mesh1d).addElementsConditional(is_external_os)

            is_vagina_orifice = fm.createFieldAnd(is_vagina_xi2_01, is_exterior_face_xi3_0)
            vaginaOrifice = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                               get_uterus_term("vagina orifice"))
            vaginaOrifice.getMeshGroup(mesh1d).addElementsConditional(is_vagina_orifice)

            # ligaments
            is_dorsalVentral = fm.createFieldAnd(dorsalGroup.getGroup(), ventralGroup.getGroup())
            is_dorsalVentralSerosa = fm.createFieldAnd(is_dorsalVentral, is_exterior_face_xi3_1)
            is_leftDorsalVentralSerosa = fm.createFieldAnd(leftGroup.getGroup(), is_dorsalVentralSerosa)
            is_rightDorsalVentralSerosa = fm.createFieldAnd(rightGroup.getGroup(), is_dorsalVentralSerosa)
            fundusGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("fundus of uterus"))
            is_bodyNotFundus = fm.createFieldAnd(bodyGroup.getGroup(), fm.createFieldNot(fundusGroup.getGroup()))

            # Broad ligament of uterus
            is_leftBroadLigament = fm.createFieldAnd(is_bodyNotFundus, is_leftDorsalVentralSerosa)
            leftBroadLigament = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_uterus_term("left broad ligament of uterus"))
            leftBroadLigament.getMeshGroup(mesh1d).addElementsConditional(is_leftBroadLigament)
            # add connected edges from left oviduct, avoiding adding dorsal-ventral edges on the superior edge
            leftBroadLigament.addSubelements()  # need current nodes in ligament for group_add_connected_elements
            tmpGroup = fm.createFieldGroup()
            tmpMeshGroup = tmpGroup.createMeshGroup(mesh1d)
            tmpMeshGroup.addElementsConditional(fm.createFieldAnd(is_leftOviduct, is_leftDorsalVentralSerosa))
            group_add_connected_elements(leftBroadLigament.getGroup(), tmpMeshGroup)
            del tmpMeshGroup
            del tmpGroup

            is_rightBroadLigament = fm.createFieldAnd(is_bodyNotFundus, is_rightDorsalVentralSerosa)
            rightBroadLigament = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_uterus_term("right broad ligament of uterus"))
            rightBroadLigament.getMeshGroup(mesh1d).addElementsConditional(is_rightBroadLigament)
            # add connected edges from right oviduct, avoiding adding dorsal-ventral edges on the superior edge
            rightBroadLigament.addSubelements()  # need current nodes in ligament for group_add_connected_elements
            tmpGroup = fm.createFieldGroup()
            tmpMeshGroup = tmpGroup.createMeshGroup(mesh1d)
            tmpMeshGroup.addElementsConditional(fm.createFieldAnd(is_rightOviduct, is_rightDorsalVentralSerosa))
            group_add_connected_elements(rightBroadLigament.getGroup(), tmpMeshGroup)
            del tmpMeshGroup
            del tmpGroup

            # Transverse cervical ligament
            is_leftTransverseCervicalLigament = fm.createFieldAnd(cervixGroup.getGroup(), is_leftDorsalVentralSerosa)
            leftTransverseCervicalLigament = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_uterus_term("left transverse cervical ligament"))
            leftTransverseCervicalLigament.getMeshGroup(mesh1d).addElementsConditional(
                is_leftTransverseCervicalLigament)

            is_rightTransverseCervicalLigament = fm.createFieldAnd(cervixGroup.getGroup(), is_rightDorsalVentralSerosa)
            rightTransverseCervicalLigament = findOrCreateAnnotationGroupForTerm(
                annotationGroups, region, get_uterus_term("right transverse cervical ligament"))
            rightTransverseCervicalLigament.getMeshGroup(mesh1d).addElementsConditional(
                is_rightTransverseCervicalLigament)

        if isRodent:
            is_rightHorn = rightOviductGroup.getGroup()
            is_rightHorn_outer = fm.createFieldAnd(is_rightHorn, is_exterior_face_xi3_1)
            is_rightHorn_inner = fm.createFieldAnd(is_rightHorn, is_exterior_face_xi3_0)

            is_leftHorn = leftOviductGroup.getGroup()
            is_leftHorn_outer = fm.createFieldAnd(is_leftHorn, is_exterior_face_xi3_1)
            is_leftHorn_inner = fm.createFieldAnd(is_leftHorn, is_exterior_face_xi3_0)

            serosaOfRightHorn = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                   get_uterus_term("serosa of right uterine horn"))
            serosaOfRightHorn.getMeshGroup(mesh2d).addElementsConditional(is_rightHorn_outer)

            lumenOfRightHorn = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                  get_uterus_term("lumen of right uterine horn"))
            lumenOfRightHorn.getMeshGroup(mesh2d).addElementsConditional(is_rightHorn_inner)

            serosaOfLeftHorn = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                  get_uterus_term("serosa of left uterine horn"))
            serosaOfLeftHorn.getMeshGroup(mesh2d).addElementsConditional(is_leftHorn_outer)

            lumenOfLeftHorn = findOrCreateAnnotationGroupForTerm(annotationGroups, region,
                                                                 get_uterus_term("lumen of left uterine horn"))
            lumenOfLeftHorn.getMeshGroup(mesh2d).addElementsConditional(is_leftHorn_inner)

        lumenOfCervix = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("lumen of uterine cervix", ""))
        lumenOfCervix.getMeshGroup(mesh2d).addElementsConditional(is_cervix_inner)
        isCervix = lumenOfCervix.getGroup()

        mesh3d = fm.findMeshByDimension(3)
        cervixGroup.getMeshGroup(mesh3d).removeAllElements()
        cervixGroup.getMeshGroup(mesh2d).addElementsConditional(isCervix)
        annotationGroups.remove(bodyNotCervixGroup)
        annotationGroups.remove(lumenOfCervix)

        if isRat:
            septumCervixGroup = getAnnotationGroupForTerm(annotationGroups, ("septum cervix", ""))
            isSeptumCervix = septumCervixGroup.getGroup()
            isSeptumCervixExterior = fm.createFieldAnd(is_exterior_face_xi1, isSeptumCervix)
            lumenOfCervix.getMeshGroup(mesh2d).addElementsConditional(isSeptumCervixExterior)
            annotationGroups.remove(septumCervixGroup)
            annotationGroups.remove(septumBodyGroup)
            annotationGroups.remove(lumenOfFundus)


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


def getSeptumRimNodes(region, fieldmodule, annotationGroups, elementCountsSeptum, elementsCountAround):
    """
    Use annotation groups for fundus, body and cervix to identify nodes from uterus lumen to be used for building
    the septum. The identified nodes are arranged in an array ([n2][n1]) from the fundus to cervix direction. At each n2
    level, the identified nodes (n1) follows the direction from d2 to d3 of the network layout defined in the body
    region. The top n2 row has 2 less nodes at the corners due to shield appearance.
    :param annotationGroups: annotation groups need to already contain groups for fundus, body and cervix.
    :param elementCountsSeptum: elements around septum, elements along septum and elements through septum.
    :param elementsCountAround: elements count around uterus scaffold.
    return nidsRim: nodeIdentifiers of rim nodes to be used for making septum.
    return paramRim: parameters of rim nodes to be used for making septum.
    """
    elementsCountAroundSeptum = elementCountsSeptum[0]
    elementsCountAlongSeptum = elementCountsSeptum[1]
    elementsCountThroughSeptum = elementCountsSeptum[2]

    fieldcache = fieldmodule.createFieldcache()
    fieldmodule.defineAllFaces()
    mesh2d = fieldmodule.findMeshByDimension(2)
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    coordinates = findOrCreateFieldCoordinates(fieldmodule)

    is_exterior = fieldmodule.createFieldIsExterior()
    is_exterior_face_xi3_0 = \
        fieldmodule.createFieldAnd(is_exterior,
                                   fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0))

    fundusGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("fundus of uterus"))
    fundusGroup.addSubelements()
    isFundus = fundusGroup.getGroup()
    isFundusInner = fieldmodule.createFieldAnd(isFundus, is_exterior_face_xi3_0)
    fundusInner = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("fundus inner", ""))
    fundusInner.getMeshGroup(mesh2d).addElementsConditional(isFundusInner)
    fundusInnerNodeset = fundusInner.getNodesetGroup(nodes)

    nodeIter = fundusInnerNodeset.createNodeiterator()
    node = nodeIter.next()
    fundusInnerNids = []
    while node.isValid():
        fundusInnerNids.append(node.getIdentifier())
        node = nodeIter.next()
    # print('fundusInner', fundusInnerNids)

    # Re-arrange inner fundus nodes
    count = 0
    nidsAroundHorns = []
    for n1 in range(2):
        nids = []
        for n3 in range(elementsCountThroughSeptum + 1):
            nids.append(fundusInnerNids[count])
            count += 1
        nidsAroundHorns.append(nids)
    nidsArc = fundusInnerNids[count:]

    # reorder left horn as index follows direction of d2 on network layout
    halfNodeCountAroundArc = len(nidsAroundHorns[0]) // 2
    nidsAroundHorns[0] = nidsAroundHorns[0][halfNodeCountAroundArc::-1] + \
                         nidsAroundHorns[0][:halfNodeCountAroundArc:-1]
    # print('nidsAroundHorns', nidsAroundHorns)

    # reorder arc nodes as they go from second front to second back, front, then last nodes now
    frontArcNodes = nidsArc[-2 * (elementsCountAroundSeptum - 1): -(elementsCountAroundSeptum - 1)]
    backArcNodes = nidsArc[-(elementsCountAroundSeptum - 1):]
    middleArcNodes = nidsArc[:-2 * (elementsCountAroundSeptum - 1)]

    nidsArc = []
    nidsArc.append(frontArcNodes)
    count = 0
    for n3 in range(elementsCountThroughSeptum - 1):
        nids = []
        for n1 in range(elementsCountAroundSeptum - 1):
            nids.append(middleArcNodes[count])
            count += 1
        nidsArc.append(nids)
    nidsArc.append(backArcNodes)
    # print('arc', nidsArc)

    # arrange into rim format
    nidsRim = []
    halfElementsCountThroughSeptum = elementsCountThroughSeptum // 2
    row = nidsArc[halfElementsCountThroughSeptum]
    row.insert(0, nidsAroundHorns[1][halfElementsCountThroughSeptum])
    row.append(nidsAroundHorns[0][halfElementsCountThroughSeptum])
    nidsRim.append(row)

    for n2 in range(elementsCountThroughSeptum // 2):
        row = nidsArc[halfElementsCountThroughSeptum - 1 - n2] + \
              nidsArc[halfElementsCountThroughSeptum + 1 + n2]
        row.insert(elementsCountAroundSeptum - 1, nidsAroundHorns[0][halfElementsCountThroughSeptum - 1 - n2])
        row.insert(elementsCountAroundSeptum, nidsAroundHorns[0][halfElementsCountThroughSeptum + 1 + n2])
        row.insert(0, nidsAroundHorns[1][halfElementsCountThroughSeptum - 1 - n2])
        row.append(nidsAroundHorns[1][halfElementsCountThroughSeptum + 1 + n2])
        nidsRim.append(row)

    bodyGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("body of uterus"))
    bodyGroup.addSubelements()
    isBody = bodyGroup.getGroup()
    isBodyInner = fieldmodule.createFieldAnd(isBody, is_exterior_face_xi3_0)

    cervixGroup = getAnnotationGroupForTerm(annotationGroups, get_uterus_term("uterine cervix"))
    cervixGroup.addSubelements()
    isCervix = cervixGroup.getGroup()
    isCervixInner = fieldmodule.createFieldAnd(isCervix, is_exterior_face_xi3_0)

    isBodyCervixInner = fieldmodule.createFieldOr(isCervixInner, isBodyInner)
    bodyCervixInterior = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("body cervix interior", ""))
    bodyCervixInterior.getMeshGroup(mesh2d).addElementsConditional(isBodyCervixInner)
    bodyCervixInnerNodeset = bodyCervixInterior.getNodesetGroup(nodes)

    nodeIter = bodyCervixInnerNodeset.createNodeiterator()
    node = nodeIter.next()
    bodyCervixInnerNids = []
    while node.isValid():
        bodyCervixInnerNids.append(node.getIdentifier())
        node = nodeIter.next()

    regularBodyCervixNids = bodyCervixInnerNids[elementsCountAround:]

    count = 0
    for n2 in range(elementsCountAlongSeptum - 2):
        nidRow = []
        for n1 in range(elementsCountAround):
            if (elementsCountAround // 4 - elementsCountAroundSeptum // 2) <= n1 <= \
                    (elementsCountAround // 4 + elementsCountAroundSeptum // 2) or \
                    (3 * elementsCountAround // 4 - elementsCountAroundSeptum // 2) <= n1 <= \
                    (3 * elementsCountAround // 4 + elementsCountAroundSeptum // 2):
                nidRow.append(regularBodyCervixNids[count])
            count += 1
        nidsRim.append(nidRow)
    # print('nidsRim', nidsRim)

    paramRim = []
    for n2 in range(len(nidsRim)):
        paramRow = []
        for n1 in range(len(nidsRim[n2])):
            node = nodes.findNodeByIdentifier(nidsRim[n2][n1])
            fieldcache.setNode(node)
            x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)[1]
            d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)[1]
            d2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)[1]
            d3 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)[1]
            param = [x, d1, d2, d3]
            paramRow.append(param)
        paramRim.append(paramRow)

    annotationGroups.remove(fundusInner)
    annotationGroups.remove(bodyCervixInterior)

    return nidsRim, paramRim
