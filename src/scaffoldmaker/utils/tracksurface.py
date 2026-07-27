"""
Utility class for representing surfaces on which features can be located.
"""

import copy
from enum import Enum
import math
from cmlibs.maths.vectorops import add, cross, distance, dot, magnitude, mult, normalize, rejection, sub, set_magnitude
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.utils.zinc.field import find_or_create_field_coordinates, find_or_create_field_group
from cmlibs.utils.zinc.finiteelement import get_maximum_element_identifier, get_maximum_node_identifier
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field, FieldGroup
from cmlibs.zinc.node import Node
from scaffoldmaker.utils.interpolation import (
    computeCubicHermiteArcLength, computeCubicHermiteDerivativeScaling, computeHermiteLagrangeDerivativeScaling,
    evaluateCoordinatesOnCurve, getCubicHermiteArcLength, getCubicHermiteBasis, getCubicHermiteBasisDerivatives,
    getCubicHermiteCurvatureSimple, incrementXiOnLine, interpolateCubicHermite, interpolateHermiteLagrangeDerivative,
    interpolateLagrangeHermiteDerivative, linearlyInterpolateVectors, sampleCubicHermiteCurves,
    sampleCubicHermiteCurvesSmooth, smoothCubicHermiteDerivativesLine, smoothCubicHermiteDerivativesLoop,
    updateCurveLocationToFaceNumber)


class TrackSurfacePosition:
    """
    Position on TrackSurface. Create with createPosition~ method of TrackSurface.
    """

    def __init__(self, e1, e2, xi1, xi2):
        """
        :param e1, e2: Element index in directions 1 and 2, starting at 0.
        :param xi1, xi2: 2-D element chart coordinates 0.0 <= xiN <= 1.0
        """
        self.e1 = e1
        self.e2 = e2
        self.xi1 = xi1
        self.xi2 = xi2

    def __str__(self):
        return 'element (' + str(self.e1) + ',' + str(self.e2) + ') xi (' + str(self.xi1) + ',' + str(self.xi2) + ')'

    def offsetXi(self, dxi1, dxi2):
        self.xi1 += dxi1
        self.xi2 += dxi2


class TrackSurface:
    """
    A surface description on which positions can be stored and tracked for
    specified directions and distances for location surface features.
    Currently represented by a lattice of elementsCount1 x elementsCount2
    square elements with bicubic Hermite interpolation but zero cross derivatives.
    """

    def __init__(self, elementsCount1, elementsCount2, nx, nd1, nd2, nd12=None, loop1=False):
        """
        Creates a TrackSurface with a lattice of elementsCount1*elementsCount2
        elements, with nodes and elements varying across direction 1 fastest.
        :param elementsCount1, elementsCount2: Number of elements in directions 1 and 2.
        :param nx: List of (elementsCount2 + 1)*(elementsCount1 + 1) node coordinates, 3 component.
        :param nd1: List of node derivatives in direction 1.
        :param nd2: List of node derivatives in direction 2.
        :param nd12: Optional list of node cross derivatives in directions 1, 2.
        coordinates, derivatives 1 and derivatives 2. Cross derivatives are zero.
        All coordinates are 3 component.
        :param loop1: Set to True if loops back to start in direction 1. Supply
        one fewer node around direction 1.
        """
        self._elementsCount1 = elementsCount1
        self._elementsCount2 = elementsCount2
        self._nx = nx
        self._nd1 = nd1
        self._nd2 = nd2
        self._nd12 = nd12
        self._loop1 = loop1
        # get max range for tolerances
        self._xMin = copy.copy(nx[0])
        self._xMax = copy.copy(nx[0])
        for x in nx:
            for c in range(3):
                s = x[c]
                if s < self._xMin[c]:
                    self._xMin[c] = s
                elif s > self._xMax[c]:
                    self._xMax[c] = s
        self._xRange = [self._xMax[c] - self._xMin[c] for c in range(3)]

    def getElementsCount1(self):
        return self._elementsCount1

    def getElementsCount2(self):
        return self._elementsCount2

    def createMirrorX(self):
        """
        Mirror track surface about x-axis by negating all x coordinates
        and rewind elements by flipping order and derivatives in direction 1.
        :return: TrackSurface
        """
        nx = []
        nd1 = []
        nd2 = []
        if self._nd12:
            print("TrackSurface.createMirrorX.  Implementation for cross derivatives not checked")
        nd12 = [] if self._nd12 else None
        nodesCount1 = self._elementsCount1 + (0 if self._loop1 else 1)
        nodesCount2 = self._elementsCount2 + 1
        for n2 in range(nodesCount2):
            for n1 in range(nodesCount1):
                oi = n2*nodesCount1 + self._elementsCount1 - n1
                ox = copy.deepcopy(self._nx[oi])
                od1 = copy.deepcopy(self._nd1[oi])
                od2 = copy.deepcopy(self._nd2[oi])
                nx.append([-ox[0],  ox[1],  ox[2]])
                nd1.append([od1[0], -od1[1], -od1[2]])
                nd2.append([-od2[0], od2[1], od2[2]])
                if nd12:
                    od12 = copy.deepcopy(self._nd12[oi])
                    nd12.append([od12[0], od12[1], od12[2]])  # need to check
        return TrackSurface(self._elementsCount1, self._elementsCount2, nx, nd1, nd2, nd12, loop1=self._loop1)

    def createPositionProportion(self, proportion1, proportion2):
        """
        Return position on surface for proportions across directions 1 and 2.
        :param proportion1, proportion2: Proportions across directions 1 and 2,
        each varying from 0.0 to 1.0 (or 2.0 if loop1), with elements equal sized.
        :return: TrackSurfacePosition
        """
        maxProportion1 = 2.0 if self._loop1 else 1.0
        assert (proportion1 >= 0.0) and (proportion1 <= maxProportion1), \
            'createPositionProportion:  Proportion 1 (' + str(proportion1) + ') out of range'
        assert (proportion2 >= 0.0) and (proportion2 <= 1.0), \
            'createPositionProportion:  Proportion 2 (' + str(proportion2) + ') out of range'

        pe1 = proportion1 * self._elementsCount1
        max_e1 = 2 * self._elementsCount1 if self._loop1 else self._elementsCount1
        if pe1 < max_e1:
            e1 = int(pe1)
            xi1 = pe1 - e1
        else:
            e1 = max_e1 - 1
            xi1 = 1.0
        pe2 = proportion2 * self._elementsCount2
        if pe2 < self._elementsCount2:
            e2 = int(pe2)
            xi2 = pe2 - e2
        else:
            e2 = self._elementsCount2 - 1
            xi2 = 1.0
        return TrackSurfacePosition(e1, e2, xi1, xi2)

    def getProportion(self, position):
        """
        From a position on this track surface, return proportions.
        :return: proportion1, proportion2
        """
        return [(position.e1 + position.xi1) / self._elementsCount1,
                (position.e2 + position.xi2) / self._elementsCount2]

    def evaluateCoordinates(self, position: TrackSurfacePosition, derivatives=False):
        """
        Evaluate coordinates on surface at position, and optionally
        derivative w.r.t. xi1 and xi2.
        :param position: A valid TrackSurfacePosition.
        :param derivatives: Set to True to calculate and return derivatives w.r.t. element xi.
        :return: If derivatives is False: coordinates [x, y, z].
        If derivatives is True: coordinates, derivative1, derivative2.
        """
        nodesCount1 = self._elementsCount1 if self._loop1 else self._elementsCount1 + 1
        e1 = position.e1 % self._elementsCount1  # to handle loop1
        nx = self._nx
        nd1 = self._nd1
        nd2 = self._nd2
        nd12 = self._nd12
        nid0 = position.e2 * nodesCount1
        n1 = nid0 + e1
        n2 = nid0 if (self._loop1 and ((e1 + 1) == self._elementsCount1)) else n1 + 1
        n3 = n1 + nodesCount1
        n4 = n2 + nodesCount1
        nid = [n1, n2, n3, n4]
        f1x1, f1d1, f1x2, f1d2 = getCubicHermiteBasis(position.xi1)
        f2x1, f2d1, f2x2, f2d2 = getCubicHermiteBasis(position.xi2)
        fx = [f1x1*f2x1, f1x2*f2x1, f1x1*f2x2, f1x2*f2x2]
        fd1 = [f1d1*f2x1, f1d2*f2x1, f1d1*f2x2, f1d2*f2x2]
        fd2 = [f1x1*f2d1, f1x2*f2d1, f1x1*f2d2, f1x2*f2d2]
        fd12 = [f1d1*f2d1, f1d2*f2d1, f1d1*f2d2, f1d2*f2d2] if nd12 else None
        coordinates = []
        for c in range(3):
            x = 0.0
            for ln in range(4):
                gn = nid[ln]
                x += fx[ln]*nx[gn][c] + fd1[ln]*nd1[gn][c] + fd2[ln]*nd2[gn][c]
                if nd12:
                    x += fd12[ln] * nd12[gn][c]
            coordinates.append(x)
        if not derivatives:
            return coordinates
        df1x1, df1d1, df1x2, df1d2 = getCubicHermiteBasisDerivatives(position.xi1)
        d1fx = [df1x1*f2x1, df1x2*f2x1, df1x1*f2x2, df1x2*f2x2]
        d1fd1 = [df1d1*f2x1, df1d2*f2x1, df1d1*f2x2, df1d2*f2x2]
        d1fd2 = [df1x1*f2d1, df1x2*f2d1, df1x1*f2d2, df1x2*f2d2]
        d1fd12 = [df1d1*f2d1, df1d2*f2d1, df1d1*f2d2, df1d2*f2d2] if nd12 else None
        df2x1, df2d1, df2x2, df2d2 = getCubicHermiteBasisDerivatives(position.xi2)
        d2fx = [f1x1*df2x1, f1x2*df2x1, f1x1*df2x2, f1x2*df2x2]
        d2fd1 = [f1d1*df2x1, f1d2*df2x1, f1d1*df2x2, f1d2*df2x2]
        d2fd2 = [f1x1*df2d1, f1x2*df2d1, f1x1*df2d2, f1x2*df2d2]
        d2fd12 = [f1d1*df2d1, f1d2*df2d1, f1d1*df2d2, f1d2*df2d2] if nd12 else None
        derivative1 = []
        derivative2 = []
        for c in range(3):
            d1 = 0.0
            d2 = 0.0
            for ln in range(4):
                gn = nid[ln]
                d1 += d1fx[ln]*nx[gn][c] + d1fd1[ln]*nd1[gn][c] + d1fd2[ln]*nd2[gn][c]
                d2 += d2fx[ln]*nx[gn][c] + d2fd1[ln]*nd1[gn][c] + d2fd2[ln]*nd2[gn][c]
                if nd12:
                    d1 += d1fd12[ln]*nd12[gn][c]
                    d2 += d2fd12[ln]*nd12[gn][c]
            derivative1.append(d1)
            derivative2.append(d2)
        return coordinates, derivative1, derivative2

    class HermiteCurveMode(Enum):
        SMOOTH = 1    # smooth variation of element size between end derivatives
        TRANSITION_END = 2  # transition from start derivative then even size
        TRANSITION_START = 3  # even size then transition to end derivative
        TRANSITION_START_AND_END = 3  # transition to/from start and end derivatives and even size in between
        UNIFORM_SIZE = 4  # regardless of initial derivatives, sample in even sizes

    def createHermiteCurvePoints(self, aProportion1, aProportion2, bProportion1, bProportion2, elementsCount,
                                 derivativeStart=None, derivativeEnd=None, curveMode=HermiteCurveMode.SMOOTH):
        """
        Create hermite curve points between two points a and b on the surface, each defined
        by their proportions over the surface in directions 1 and 2.
        Also returns cross direction 2 in plane of surface with similar magnitude to curve derivative 1,
        and unit surface normals.
        :param derivativeStart, derivativeEnd: Optional derivative vectors in 3-D world coordinates
        to match at the start and end of the curves. If omitted, fits in with other derivative or is
        in a straight line from a to b.
        :param elementsCount:  Number of elements out.
        :return: nx[], nd1[], nd2[], nd3[], nProportions[]
        """
        # print('createHermiteCurvePoints', aProportion1, aProportion2, bProportion1, bProportion2, elementsCount,
        #       derivativeStart, derivativeEnd)
        if derivativeStart:
            position = self.createPositionProportion(aProportion1, aProportion2)
            _, sd1, sd2 = self.evaluateCoordinates(position, derivatives=True)
            delta_xi1, delta_xi2 = calculate_surface_delta_xi(sd1, sd2, derivativeStart)
            dp1Start = delta_xi1 / self._elementsCount1
            dp2Start = delta_xi2 / self._elementsCount2
            derivativeMagnitudeStart = math.sqrt(dp1Start*dp1Start + dp2Start*dp2Start)
            dp1Start *= elementsCount
            dp2Start *= elementsCount
            # print('start delta_xi1', delta_xi1, 'delta_xi2', delta_xi2)
            # print('dp1Start', dp1Start, 'dp2Start', dp2Start)
        if derivativeEnd:
            position = self.createPositionProportion(bProportion1, bProportion2)
            _, sd1, sd2 = self.evaluateCoordinates(position, derivatives=True)
            delta_xi1, delta_xi2 = calculate_surface_delta_xi(sd1, sd2, derivativeEnd)
            dp1End = delta_xi1 / self._elementsCount1
            dp2End = delta_xi2 / self._elementsCount2
            derivativeMagnitudeEnd = math.sqrt(dp1End*dp1End + dp2End*dp2End)
            dp1End *= elementsCount
            dp2End *= elementsCount
            # print('end delta_xi1', delta_xi1, 'delta_xi2', delta_xi2)
            # print('dp1End', dp1End, 'dp2End', dp2End)
        if not derivativeStart:
            if derivativeEnd:
                dp1Start, dp2Start = interpolateLagrangeHermiteDerivative(
                    [aProportion1, aProportion2], [bProportion1, bProportion2], [dp1End, dp2End], 0.0)
            else:
                dp1Start = bProportion1 - aProportion1
                dp2Start = bProportion2 - aProportion2
            derivativeMagnitudeStart = math.sqrt(dp1Start*dp1Start + dp2Start*dp2Start)/elementsCount
        if not derivativeEnd:
            if derivativeStart:
                dp1End, dp2End = interpolateHermiteLagrangeDerivative(
                    [aProportion1, aProportion2], [dp1Start, dp2Start], [bProportion1, bProportion2], 1.0)
            else:
                dp1End = bProportion1 - aProportion1
                dp2End = bProportion2 - aProportion2
            derivativeMagnitudeEnd = math.sqrt(dp1End*dp1End + dp2End*dp2End)/elementsCount
        # print('derivativeMagnitudeStart', derivativeMagnitudeStart, 'derivativeMagnitudeEnd', derivativeMagnitudeEnd)
        proportions, dproportions = sampleCubicHermiteCurvesSmooth(
            [[aProportion1, aProportion2], [bProportion1, bProportion2]],
            [[dp1Start, dp2Start], [dp1End, dp2End]],
            elementsCount, derivativeMagnitudeStart, derivativeMagnitudeEnd)[0:2]
        if curveMode != self.HermiteCurveMode.SMOOTH:
            if derivativeStart and (curveMode in [self.HermiteCurveMode.TRANSITION_START,
                                                  self.HermiteCurveMode.TRANSITION_START_AND_END]):
                addLengthStart = 0.5*derivativeMagnitudeStart
                lengthFractionStart = 0.5
            else:
                addLengthStart = 0.0
                lengthFractionStart = 1.0
            if derivativeEnd and (curveMode in [self.HermiteCurveMode.TRANSITION_END,
                                                self.HermiteCurveMode.TRANSITION_START_AND_END]):
                addLengthEnd = 0.5*derivativeMagnitudeEnd
                lengthFractionEnd = 0.5
            else:
                addLengthEnd = 0.0
                lengthFractionEnd = 1.0
            proportions, dproportions = sampleCubicHermiteCurves(
                proportions, dproportions, elementsCount,
                addLengthStart, addLengthEnd, lengthFractionStart, lengthFractionEnd)[0:2]
        # print(' proportions', proportions)
        # print('dproportions', dproportions)
        nx = []
        nd1 = []
        nd2 = []
        nd3 = []
        for n in range(0, elementsCount + 1):
            position = self.createPositionProportion(proportions[n][0], proportions[n][1])
            x, sd1, sd2 = self.evaluateCoordinates(position, derivatives=True)
            f1 = dproportions[n][0] * self._elementsCount1
            f2 = dproportions[n][1] * self._elementsCount2
            d1 = [(f1*sd1[c] + f2*sd2[c]) for c in range(3)]
            d3 = cross(sd1, sd2)
            # handle zero magnitude of d3
            mag = math.sqrt(sum(d3[c]*d3[c] for c in range(3)))
            if mag > 0.0:
                d3 = [(d3[c]/mag) for c in range(3)]
            d2 = cross(d3, d1)
            nx .append(x)
            nd2.append(d2)
            nd1.append(d1)
            nd3.append(d3)
        # print('createHermiteCurvePoints end \n nx', nx,'\nnd1',nd1,'\nnd2',nd2,'\nnd3',nd3)
        return nx, nd1, nd2, nd3, proportions

    def resampleHermiteCurvePointsSmooth(self, nx, nd1, nd2, nd3, nProportions,
                                         derivativeMagnitudeStart=None, derivativeMagnitudeEnd=None):
        """
        Call sampleCubicHermiteCurvesSmooth on nx, nd1 and recalculate positions, nd2, nd3 for points.
        :return: nx[], nd1[], nd2[], nd3[], nProportions[]
        """
        elementsCount = len(nx) - 1
        # print(nx, nd1, elementsCount, derivativeMagnitudeStart, derivativeMagnitudeEnd)
        nx, nd1 = sampleCubicHermiteCurvesSmooth(
            nx, nd1, elementsCount, derivativeMagnitudeStart, derivativeMagnitudeEnd)[0:2]
        mag2 = magnitude(nd2[0])
        if mag2 > 0.0:
            nd2[0] = set_magnitude(nd2[0], magnitude(nd1[0]))
        for n in range(1, elementsCount):
            p = self.findNearestPosition(nx[n], self.createPositionProportion(*nProportions[n]))
            nProportions[n] = self.getProportion(p)
            _, sd1, sd2 = self.evaluateCoordinates(p, derivatives=True)
            _, d2, d3 = calculate_surface_axes(sd1, sd2, normalize(nd1[n]))
            nd2[n] = set_magnitude(d2, magnitude(nd1[n]))
            nd3[n] = d3
        mag2 = magnitude(nd2[-1])
        if mag2 > 0.0:
            nd2[-1] = set_magnitude(nd2[-1], magnitude(nd1[-1]))
        return nx, nd1, nd2, nd3, nProportions

    def getSurfaceNormal(self, start_x):
        """
        Get surface normal on the surface at or nearest to start_x.
        :param start_x: Coordinates on or near surface.
        :return: Unit surface normal at nearest point on surface to start_x.
        """
        position = self.findNearestPosition(start_x)
        x, d1, d2 = self.evaluateCoordinates(position, derivatives=True)
        return normalize(cross(d1, d2))

    def makeCoordinatesOnSurface(self, start_x):
        """
        Return nearest coordinate to start_x on the surface.
        :param start_x: Coordinates on or near surface.
        :return: Nearest coordinates to start_x on surface.
        """
        position = self.findNearestPosition(start_x, startPosition=self.findNearestPositionSample(start_x)[0])
        return self.evaluateCoordinates(position)

    def makeCoordinatesAndDerivativeOnSurface(self, start_x, start_d):
        """
        Finds nearest position on TrackSurface to start_x and returns both coordinates there and start_d converted
        to a surface tangent at that position.
        :param start_x: Coordinates on or near surface of TrackSurface.
        :param start_d: Derivative near tangential to TrackSurface at nearest surface point to x.
        :return: Coordinates at nearest point on surface, derivative made tangential to surface there with same
        magnitude as start_d.
        """
        position = self.findNearestPosition(start_x, startPosition=self.findNearestPositionSample(start_x)[0])
        x, d1, d2 = self.evaluateCoordinates(position, derivatives=True)
        normal = normalize(cross(d1, d2))
        d = rejection(start_d, normal)
        return x, d

    def sampleCurve(self, start_x, start_d1, start_d2, end_x, end_d1, end_d2, elements_count,
                    start_weight=None, end_weight=None, overweighting=1.0, end_transition=False):
        """
        Samples an even-spaced Hermite curve from start point and direction to end point and direction.
        Also interpolates side derivatives. Compatible with QuadTriangleMesh.
        :param start_x: start coordinate on surface of TrackSurface.
        :param start_d1: start direction on surface of TrackSurface.
        :param start_d2: optional start side direction on TrackSurface. Must be supplied if end_d2 is
        supplied.
        :param end_x: end coordinate on surface of TrackSurface.
        :param end_d1: end direction on surface of TrackSurface, or None to estimate from start_d1.
        Must be supplied with end_transition.
        :param end_d2: optional end side direction on surface of TrackSurface, or None to use surface tangent normal to
        end direction on same side as start_d2.
        :param elements_count: Number of elements to sample.
        :param start_weight, end_weight: Optional relative weights for start/end d1. If not supplied, weighting is
        by distance from the other end.
        :param overweighting: Multiplier of arc length to use with initial curve to exaggerate end derivatives.
        :param end_transition: If supplied with end_d1, modify size of last element to fit end_d1.
        :return: x[], d1[]{, d2[] if start_d2 supplied}
        """
        assert (not end_transition) or end_d1
        end_d1_mag = magnitude(end_d1) if end_d1 else None
        length = distance(start_x, end_x)
        # initial cubic interpolation, weighting derivatives by distance to other end
        start_d = start_d1
        end_d = end_d1
        if not end_d:
            start_d = normalize(start_d)
            scaling = computeHermiteLagrangeDerivativeScaling(start_x, start_d, end_x)
            start_d = [d * scaling for d in start_d]
            end_d = interpolateHermiteLagrangeDerivative(start_x, start_d, end_x, 1.0)
            end_d = self.makeCoordinatesAndDerivativeOnSurface(end_x, end_d)[1]
        use_start_weight = start_weight if start_weight else 1.0
        use_end_weight = end_weight if end_weight else 1.0
        start_d = set_magnitude(start_d, 2.0 * length * use_start_weight / (use_start_weight + use_end_weight))
        end_d = set_magnitude(end_d, 2.0 * length * use_end_weight / (use_start_weight + use_end_weight))
        scaling = computeCubicHermiteDerivativeScaling(start_x, start_d, end_x, end_d) * overweighting
        start_d = mult(start_d, scaling)
        end_d = mult(end_d, scaling)
        px, pd1 = sampleCubicHermiteCurvesSmooth([start_x, end_x], [start_d, end_d], elements_count)[0:2]
        iter_count = 2
        for iter in range(iter_count):
            for n in range(1, elements_count):
                px[n], pd1[n] = self.makeCoordinatesAndDerivativeOnSurface(px[n], pd1[n])
            if end_transition:
                if elements_count == 1:
                    # sampleCubicHermiteCurves doesn't work properly with 1 element...
                    arc_length = getCubicHermiteArcLength(px[0], pd1[0], px[1], pd1[1])
                    pd1[0] = set_magnitude(pd1[0], 2.0 * arc_length - end_d1_mag)
                    pd1[1] = set_magnitude(pd1[1], end_d1_mag)
                else:
                    px, pd1 = sampleCubicHermiteCurves(
                        px, pd1, elements_count, addLengthEnd=0.5 * end_d1_mag, lengthFractionEnd=0.5)[0:2]
            else:
                px, pd1 = sampleCubicHermiteCurves(px, pd1, elements_count)[0:2]
        if start_d2:
            if not end_d2:
                end_d2 = self.makeCoordinatesAndDerivativeOnSurface(px[-1], start_d2)[1]
                normal = normalize(cross(pd1[-1], end_d2))
                end_d2 = cross(normal, pd1[-1])
            pd2 = [start_d2]
            if end_transition:
                # get xi scale giving the appropriate end element size for magnitude of end_d1 vs. regular derivatives
                reg_d1_mag = magnitude(pd1[-2])
                end_d1_mag = magnitude(end_d1)
                xi_scale = reg_d1_mag / ((elements_count - 0.5) * reg_d1_mag + 0.5 * end_d1_mag)
            else:
                xi_scale = 1.0 / elements_count
            for n in range(1, elements_count):
                xi = n * xi_scale
                d2 = linearlyInterpolateVectors(start_d2, end_d2, xi)
                pd2.append(self.makeCoordinatesAndDerivativeOnSurface(px[n], d2)[1])
            pd2.append(end_d2)
            return px, pd1, pd2
        return px, pd1

    def positionOnBoundary(self, position: TrackSurfacePosition):
        """
        Determines if position is within tolerance of boundary and which boundary it is on.
        :param position: A position on the track surface.
        :return: 1 if on xi1 boundary, 2 if on xi2 boundary, 0 if not on a boundary.
        """
        LOWER_P_LIMIT = 1.0E-8
        UPPER_P_LIMIT = 1.0 - LOWER_P_LIMIT
        if not self._loop1:
            proportion1 = (position.e1 + position.xi1) / self._elementsCount1
            if proportion1 < LOWER_P_LIMIT:
                position.xi1 = 0.0
                return 1
            elif proportion1 > UPPER_P_LIMIT:
                position.xi1 = 1.0
                return 1
        proportion2 = (position.e2 + position.xi2) / self._elementsCount2
        if proportion2 < LOWER_P_LIMIT:
            position.xi2 = 0.0
            return 2
        elif proportion2 > UPPER_P_LIMIT:
            position.xi2 = 1.0
            return 2
        return 0

    def _boundaryDirection(self, position: TrackSurfacePosition):
        """
        Determine if position is on boundary, with tolerance, and return outward direction in xi space.
        :param position: TrackSurfacePosition.
        :return: None if not on boundary otherwise components of d1, d2 pointing outward.
        e.g. [-1.0, 0.0] if on xi1 == 0 boundary.
        """
        LOWER_P_LIMIT = 1.0E-5
        UPPER_P_LIMIT = 1.0 - LOWER_P_LIMIT
        if not self._loop1:
            proportion1 = (position.e1 + position.xi1) / self._elementsCount1
            if proportion1 < LOWER_P_LIMIT:
                return [-1.0, 0.0]
            elif proportion1 > UPPER_P_LIMIT:
                return [1.0, 0.0]
        proportion2 = (position.e2 + position.xi2) / self._elementsCount2
        if proportion2 < LOWER_P_LIMIT:
            return [0.0, -1.0]
        elif proportion2 > UPPER_P_LIMIT:
            return [0.0, 1.0]
        return None

    def _advancePosition(self, startPosition, dxi1, dxi2, MAX_MAG_DXI=0.5):
        """
        Advance position by element delta xi to maximum change of xi coordinates or boundary.
        :param startPosition: Start position.
        :param dxi1: Increment in xi1.
        :param dxi2: Increment in xi2.
        :param MAX_MAG_DXI: Maximum magnitude of dxi to keep increments reasonable for a cubic curve.
        :return: Advanced position, onBoundary (0/1=xi1/2=xi2), actual dxi1, actual dxi2
        """
        startProportion1, startProportion2 = self.getProportion(startPosition)
        adxi1 = dxi1
        adxi2 = dxi2
        magDxi = magnitude([dxi1, dxi2])
        if magDxi > MAX_MAG_DXI:
            factor = MAX_MAG_DXI / magDxi
            adxi1 *= factor
            adxi2 *= factor
        proportion1 = startProportion1 + adxi1 / self._elementsCount1
        proportion2 = startProportion2 + adxi2 / self._elementsCount2
        onBoundary = 0
        if self._loop1:
            if proportion1 < 0.0:
                proportion1 += 2.0
            elif proportion1 > 2.0:
                proportion1 -= 2.0
        else:
            if proportion1 < 0.0:
                proportion1 = 0.0
                onBoundary = 1
            elif proportion1 > 1.0:
                proportion1 = 1.0
                onBoundary = 1
        if proportion2 < 0.0:
            proportion2 = 0.0
            onBoundary = 2
        elif proportion2 > 1.0:
            proportion2 = 1.0
            onBoundary = 2
        if onBoundary:
            if not self._loop1:
                adxi1 = (proportion1 - startProportion1) * self._elementsCount1
            adxi2 = (proportion2 - startProportion2) * self._elementsCount2
        return self.createPositionProportion(proportion1, proportion2), onBoundary, adxi1, adxi2

    def _getIntersectionDelta(self, position, otherTrackSurface, otherPosition, stickyBoundaryCount):
        """
        Calculate delta (rTangent) vector directed in surface toward nearest/intersection point.
        Extracted as called multiple times in findIntersectionPoint().
        :param position: Current position on surface.
        :param otherTrackSurface: Other surface to intersect with.
        :param otherPosition: Current/prior nearest position on otherTrackSurface, updated before use.
        :param stickyBoundaryCount: A positive number forces sticking onto boundary of this surface.
        :return: coordinate on surface (x, d1, d2), surface normal, increment normal (different if tracking on
        boundary), onBoundary, otherPosition, coordinates of nearest point on other surface (ox, od1, od2),
        onOtherBoundary, projection vector r to other position, projection r normal vector, projection r tangent vector.
        """
        onBoundary = self.positionOnBoundary(position)
        x, d1, d2 = self.evaluateCoordinates(position, derivatives=True)
        otherPosition = otherTrackSurface.findNearestPosition(x, otherPosition)
        onOtherBoundary = otherTrackSurface.positionOnBoundary(otherPosition)
        if onOtherBoundary and ((not onBoundary) or (stickyBoundaryCount <= 0)):
            ox = otherTrackSurface.evaluateCoordinates(otherPosition)
            position2 = self.findNearestPosition(ox, position)
            position.e1 = position2.e1
            position.e2 = position2.e2
            position.xi1 = position2.xi1
            position.xi2 = position2.xi2
            onBoundary = self.positionOnBoundary(position)
            x, d1, d2 = self.evaluateCoordinates(position, derivatives=True)
            otherPosition = otherTrackSurface.findNearestPosition(x, otherPosition)
            onOtherBoundary = otherTrackSurface.positionOnBoundary(otherPosition)
        ox, od1, od2 = otherTrackSurface.evaluateCoordinates(otherPosition, derivatives=True)
        r = sub(ox, x)
        n1 = normalize(cross(d1, d2))
        n = n1
        r_dot_n = dot(r, n)
        if r_dot_n < 0:
            # flip normal to be towards other x
            n = [-s for s in n]
            r_dot_n = -r_dot_n
        rNormal = mult(n, r_dot_n)
        rTangent = sub(r, rNormal)
        if onBoundary and (magnitude(rTangent) > 0.0):
            # stick on boundary unless strongly inward
            xiBoundaryDirection = self._boundaryDirection(position)
            outward = normalize(add(mult(d1, xiBoundaryDirection[0]), mult(d2, xiBoundaryDirection[1])))
            rTangent_dot_outward = dot(outward, normalize(rTangent))
            # print("    rTangent_dot_outward", rTangent_dot_outward)
            if (rTangent_dot_outward > -0.01) or (stickyBoundaryCount > 0):  # was -0.95: then -0.5
                d = d1 if (onBoundary == 2) else d2
                n = cross(cross(d, r), d)
                if dot(n, n1) < 0.0:
                    n = [-s for s in n]
                n = normalize(n)
                r_dot_n = dot(r, n)
                if r_dot_n < 0:
                    # flip normal to be towards other x
                    n = [-s for s in n]
                    r_dot_n = -r_dot_n
                rNormal = mult(n, r_dot_n)
                rTangent = sub(r, rNormal)
        return (x, d1, d2), n1, n, onBoundary, otherPosition, (ox, od1, od2), onOtherBoundary, r, rNormal, rTangent

    def _getDirectionalCurvature(self, position, direction):
        """
        Get positive scalar curvature (1 / R) in direction.
        :param position: Position on surface.
        :param direction: Direction [dxi1, dxi2], not necessary to be unit size.
        :return: Positive scalar curvature, kappa = 1/R, tangent dx/dxi, dTangent d2x/dxi2
        """
        DELTA_XI = 1.0E-5
        mag_direction = magnitude(direction)
        dxi1 = DELTA_XI * direction[0] / mag_direction
        dxi2 = DELTA_XI * direction[1] / mag_direction
        tmpPosition = copy.copy(position)
        tmpPosition.offsetXi(-0.5 * dxi1, -0.5 * dxi2)
        xa, d1, d2 = self.evaluateCoordinates(tmpPosition, derivatives=True)
        da = add(mult(d1, dxi1), mult(d2, dxi2))
        tmpPosition.offsetXi(dxi1, dxi2)
        xb, d1, d2 = self.evaluateCoordinates(tmpPosition, derivatives=True)
        db = add(mult(d1, dxi1), mult(d2, dxi2))
        # print(" ** directional curvature", sub(xb, xa), da, db)
        curvature, tangent, dTangent = getCubicHermiteCurvatureSimple(xa, da, xb, db, 0.5)
        tangent = mult(tangent, 1.0 / DELTA_XI)
        dTangent = mult(dTangent, 1.0 / (DELTA_XI * DELTA_XI))
        return curvature, tangent, dTangent

    def findIntersectionPoint(self, otherTrackSurface,
                              startPosition: TrackSurfacePosition,
                              otherStartPosition: TrackSurfacePosition,
                              instrument=False):
        """
        Find an intersection point on both self and otherTrackSurface from an initial guess.
        :param otherTrackSurface: Other TrackSurface to find intersection with.
        :param startPosition: Start position to get intersection point near to.
        Needs to be a good guess if surface has complex curvature.
        :param otherStartPosition: Initial estimate of nearest position on other TrackSurface.
        Use findNearestPositionParameter or findNearestPositionSample if a better guess is not available.
        :param instrument: Set to True to print debug messages.
        :return: TrackSurfacePosition, OtherTrackSurfacePosition, x, tangent, onBoundary (0/1=xi1/2=xi2)
        or None, None, None, None, 0  if no intersection found.
        The tangent is normalized n1 x n2, a unit vector along the intersection curve.
        """
        if instrument:
            print("findIntersectionPoint startPosition", startPosition, "otherPosition", otherStartPosition)
        position = copy.deepcopy(startPosition)
        otherPosition = copy.deepcopy(otherStartPosition)
        MAX_MAG_DXI = 0.5  # target/maximum magnitude of xi increment
        MINIMUM_DXI = 1.0E-10
        X_TOL = 1.0E-6 * max(self._xRange)
        MAG_JOLT_DXI = 0.05  # magnitude of xi change when jolting out of local minimum
        mag_dxi = 0.0
        jolt_index = 0
        jolt_it = -2  # iter number of last jolt
        STICKY_BOUNDARY_ITERATIONS = 4
        MAX_SLOPE_FACTOR = 10000.0
        oldOnBoundary = self.positionOnBoundary(position)
        stickyBoundaryCount = STICKY_BOUNDARY_ITERATIONS if oldOnBoundary else 0
        joltBoundaryCount = 0
        # XI_TOL = 1.0E-6
        # lowDxiCount = 0
        for it in range(100):
            coords, n1, n, onBoundary, otherPosition, otherCoords, onOtherBoundary, r, rNormal, rTangent = \
                self._getIntersectionDelta(position, otherTrackSurface, otherPosition, stickyBoundaryCount)
            if onBoundary:
                if onBoundary != oldOnBoundary:
                    stickyBoundaryCount = STICKY_BOUNDARY_ITERATIONS
            else:
                stickyBoundaryCount = 0
            oldOnBoundary = onBoundary
            mag_r = magnitude(r)
            if instrument:
                print("iter", it + 1, "pos", position, onBoundary, "other", otherPosition, onOtherBoundary,
                      "mag_r", mag_r)
            if mag_r < X_TOL:
                n2 = normalize(cross(otherCoords[1], otherCoords[2]))
                # print("dot(n1, n2)", dot(n1, n2))
                if abs(dot(n1, n2)) < 0.9999:
                    tangent = cross(n1, n2)
                elif (onBoundary == 2) and (dot(normalize(coords[1]), normalize(otherCoords[1])) > 0.9999):
                    tangent = coords[1]
                elif (onBoundary == 1) and (dot(normalize(coords[2]), normalize(otherCoords[2])) > 0.9999):
                    tangent = coords[2]
                else:
                    tangent = None  # caller needs to estimate
                if tangent:
                    tangent = normalize(tangent)
                if instrument:
                    print("TrackSurface.findIntersectionPoint found intersection: "
                          "pos", position, "other", otherPosition, "mag_r", mag_r, "tangent", tangent, "iter", it + 1)
                return position, otherPosition, coords[0], tangent, onBoundary
            mag_ri = magnitude(rTangent)
            slope_factor = MAX_SLOPE_FACTOR
            if mag_ri == 0.0:
                u = [0.0, 0.0, 0.0]
            else:
                slope_factor = mag_r * mag_r / (mag_ri * mag_ri)
                if instrument:
                    print("    slope_factor", slope_factor, "rTangent", rTangent)
                if slope_factor > MAX_SLOPE_FACTOR:
                    slope_factor = MAX_SLOPE_FACTOR
                u = mult(rTangent, slope_factor)
            dxi1, dxi2 = calculate_surface_delta_xi(coords[1], coords[2], u)
            # dxi = [dxi1, dxi2]
            # mag_dxi = magnitude(dxi)
            if instrument:
                print("    initial dxi", dxi1, dxi2, jolt_it, stickyBoundaryCount)
            if (it > (jolt_it + 1)) and (slope_factor >= MAX_SLOPE_FACTOR) \
                    and (stickyBoundaryCount <= 0):  # or (lowDxiCount > 2))
                # slow progress: may be a local minimum; jolt along boundary edge or cardinal direction
                if onBoundary and (joltBoundaryCount < 2) and ((not onOtherBoundary) or (jolt_index % 4 < 2)):
                    # maximum 2 boundary jolts to allow return to interior after sticky boundary
                    jolt_case = jolt_index % 2
                    dxi1 = 0.0 if (onBoundary == 1) else -MAG_JOLT_DXI if (jolt_case == 0) else MAG_JOLT_DXI
                    dxi2 = 0.0 if (onBoundary == 2) else -MAG_JOLT_DXI if (jolt_case == 0) else MAG_JOLT_DXI
                    if instrument:
                        print("JOLT boundary")
                    joltBoundaryCount += 1
                elif onOtherBoundary:
                    jolt_case = jolt_index % 2
                    u = otherCoords[1] if (onOtherBoundary == 2) else otherCoords[2]
                    scale = (-MAG_JOLT_DXI if (jolt_case == 0) else MAG_JOLT_DXI) / magnitude(u)
                    u = [scale * s for s in u]
                    dxi1, dxi2 = calculate_surface_delta_xi(coords[1], coords[2], u)
                    if instrument:
                        print("JOLT other boundary")  # GRC
                else:
                    jolt_case = jolt_index % 8
                    dxi1 = -MAG_JOLT_DXI if (jolt_case in [0, 4, 6]) else \
                        MAG_JOLT_DXI if (jolt_case in [2, 5, 7]) else 0.0
                    dxi2 = -MAG_JOLT_DXI if (jolt_case in [1, 4, 5]) else \
                        MAG_JOLT_DXI if (jolt_case in [3, 6, 7]) else 0.0
                    if instrument:
                        print("JOLT interior")
                if instrument:
                    print("    jolt dxi", dxi1, dxi2)
                jolt_it = it
                jolt_index += 1
            if instrument:
                print("    1st modified dxi", dxi1, dxi2)
            # sample at  offset and halfway between to get quadratic curve to minimise
            r1 = mag_r
            positionPlus, crossBoundary, dxi1, dxi2 = self._advancePosition(
                position, dxi1, dxi2, MAX_MAG_DXI=MAX_MAG_DXI)
            rPlus = self._getIntersectionDelta(positionPlus, otherTrackSurface, otherPosition, stickyBoundaryCount)[7]
            r3 = magnitude(rPlus)
            if dot(rPlus, n) < 0.0:
                r3 = -r3
            positionHalf = self._advancePosition(position, 0.5 * dxi1, 0.5 * dxi2, MAX_MAG_DXI=MAX_MAG_DXI)[0]
            rHalf = self._getIntersectionDelta(positionHalf, otherTrackSurface, otherPosition, stickyBoundaryCount)[7]
            r2 = magnitude(rHalf)
            if dot(rHalf, n) < 0.0:
                r2 = -r2
            # quadratic equation
            a = 2.0 * r1 - 4.0 * r2 + 2.0 * r3
            b = -3.0 * r1 + 4.0 * r2 - r3
            c = r1
            discr = b * b - 4.0 * a * c
            xiFactor = 0.0
            minima = False
            if instrument:
                print("    quadratic a", a, "b", b, "c", c, "discr", discr)
            if (discr >= 0.0) and (abs(a) > 0.0):
                sqrt_discr = math.sqrt(discr)
                root1 = (-b - sqrt_discr) / (2.0 * a)
                root2 = (-b + sqrt_discr) / (2.0 * a)
                if instrument:
                    print("    root1", root1, "root2", root2)
                if 0.0 < root1 <= 2.0:
                    xiFactor = root1
                elif 0.0 < root1 <= 2.0:
                    xiFactor = root2
            if (xiFactor <= 0.0) or (xiFactor > 2.0):
                if (a > 0.0) and (b < 0):
                    # minimum at zero slope
                    xiFactor = -b / (2.0 * a)
                    minima = True
                    if instrument:
                        print("    MINIMA", xiFactor)
                    if xiFactor < -1.0:
                        xiFactor = -1.0
                    elif xiFactor > 2.0:
                        xiFactor = 2.0
                else:
                    abs_r1 = abs(r1)
                    abs_r2 = abs(r2)
                    abs_r3 = abs(r3)
                    xiFactor = 1.0 if ((abs_r3 < abs_r1) and (abs_r3 < abs_r2)) else 0.5 if (abs_r2 < abs_r1) else 0.0
                    # xiFactor = 1.0 if ((abs_r3 < abs_r1) and (abs_r3 < abs_r2)) else 0.5 if (abs_r2 < abs_r1) else \
                    #     0.1 if (it == jolt_it) else 0.0
            dxi1 = dxi1 * xiFactor
            dxi2 = dxi2 * xiFactor
            if instrument:
                print("    quadratic xi factor", xiFactor, "MIN" if minima else "REG", "r1", r1, "r2", r2, "r3", r3)
            position, onBoundary, dxi1, dxi2 = self._advancePosition(position, dxi1, dxi2, MAX_MAG_DXI=MAX_MAG_DXI)
            if instrument:
                print("    final dxi", dxi1, dxi2)
            dxi = [dxi1, dxi2]
            mag_dxi = magnitude(dxi)
            if (it >= 50) and (mag_dxi < MINIMUM_DXI):
                if instrument:
                    print("TrackSurface.findIntersectionPoint low increment after", it + 1, "iterations, dxi", dxi)
                break
            stickyBoundaryCount -= 1
        else:
            print('TrackSurface.findIntersectionPoint failed:  Reached max iterations', it + 1,
                  'last increment', mag_dxi)
            # if not instrument:
            #     return self.findIntersectionPoint(
            #         otherTrackSurface, startPosition, otherStartPosition, instrument=True)
        return None, None, None, None, 0

    def findIntersectionCurve(self, otherTrackSurface, startPosition: TrackSurfacePosition = None,
                              MAX_MAG_DXI=0.5, curveElementsCount: int = 8, instrument=False):
        """
        Find curve of intersection of self and otherTrackSurface starting from intersection point nearest
        to start position. The curve is resampled to have equal sized elements along/around it.
        :param otherTrackSurface: Other TrackSurface to find intersection with.
        :param startPosition: Optional start position to get intersection point near to.
        Needs to be a good guess if surface has complex curvature. If not supplied, gets nearest positions
        from element centres.
        :param MAX_MAG_DXI: Maximum increment in xi for each iteration.
        :param curveElementsCount: Number of elements to return in resampled curve.
        :param instrument: Set to True to print debug messages.
        :return: cx[], cd1[], cProportions (on this surface), loop (True if curve is a closed loop),
        or None, None, None, False if no intersection.
        """
        if instrument:
            print("findIntersectionCurve", startPosition)
        if startPosition:
            nextPosition = startPosition
            startX = self.evaluateCoordinates(startPosition, derivatives=False)
            otherPosition = otherStartPosition = self.findNearestPositionSample(startX)[0]
        else:
            nearestDistance = None
            startPosition = None
            otherStartPosition = None
            for n2 in range(self._elementsCount2):
                for n1 in range(self._elementsCount1):
                    position = TrackSurfacePosition(n1, n2, 0.5, 0.5)
                    targetx = self.evaluateCoordinates(position)
                    otherPosition, distance = otherTrackSurface.findNearestPositionSample(targetx)
                    if (nearestDistance is None) or (distance < nearestDistance):
                        nearestDistance = distance
                        startPosition = position
                        otherStartPosition = otherPosition
            nextPosition = startPosition
            otherPosition = otherStartPosition
        START_MAX_MAG_DXI = MAX_MAG_DXI
        X_TOL = 1.0E-6 * max(self._xRange)
        px = []
        pd1 = []
        boundaryCount = 0
        loop = False
        xiLoopSamples = [0.25, 0.5, 0.75, 1.0]
        pointCount = 0
        crossBoundary = 0
        lastx = None
        dirn = None
        if instrument:
            print("TrackSurface.findIntersectionCurve.  nextPosition", nextPosition, "otherPosition", otherPosition)
        while True:
            position, otherPosition, x, t, onBoundary = \
                self.findIntersectionPoint(otherTrackSurface, nextPosition, otherPosition, instrument=False)
            if not position:
                if pointCount > 0:
                    print("TrackSurface.findIntersectionCurve.  Stopping as lost intersection")
                    position, otherPosition, x, t, onBoundary = \
                        self.findIntersectionPoint(otherTrackSurface, nextPosition, otherPosition, instrument=True)
                    break
                if instrument:
                    print("TrackSurface.findIntersectionCurve.  No intersection")
                return None, None, None, False
            onOtherBoundary = otherTrackSurface.positionOnBoundary(otherPosition)
            noProgressBoundary = False
            if (pointCount > 0) and (dot(dirn, sub(x, lastx)) < X_TOL):
                MAX_MAG_DXI *= 0.5
                if MAX_MAG_DXI < 1.0E-6:
                    print("TrackSurface.findIntersectionCurve.  No progress boundary")
                    noProgressBoundary = True
                else:
                    t = pd1[-1] if (boundaryCount == 0) else pd1[0]
            else:
                if boundaryCount == 0:
                    if pointCount == 0:
                        startPosition = position
                        otherStartPosition = otherPosition
                    if not t:
                        if pointCount > 0:
                            prev_d1 = mult(pd1[-1], magnitude(sub(x, px[-1])))
                            next_d1 = interpolateHermiteLagrangeDerivative(px[-1], prev_d1, x, 1.0)
                        else:
                            tmpx, tmpd1, tmpd2 = self.evaluateCoordinates(position, derivatives=True)
                            next_d1 = tmpd2 if (onBoundary == 1) else tmpd1
                        t = normalize(next_d1)
                    px.append(x)
                    pd1.append(t)
                else:
                    if not t:
                        prev_d1 = mult(pd1[0], magnitude(sub(x, px[0])))
                        next_d1 = interpolateLagrangeHermiteDerivative(x, px[0], prev_d1, 0.0)
                        t = normalize(next_d1)
                    px.insert(0, x)
                    pd1.insert(0, t)
                pointCount += 1
                if instrument:
                    print("- curve points", pointCount, "pos", position, "boundary", onBoundary,
                          "onOtherBoundary", onOtherBoundary)
                if pointCount > 100:
                    print("TrackSurface.findIntersectionCurve.  Stopping as too many points")
                    # stop infinite loop for problem cases
                    break
            if noProgressBoundary or ((pointCount > 1) and (onBoundary or onOtherBoundary) and
                                      (crossBoundary or not (onBoundary and onOtherBoundary))):
                boundaryCount += 1
                MAX_MAG_DXI = START_MAX_MAG_DXI
                if instrument:
                    print("- add boundary", boundaryCount)
                if boundaryCount == 2:
                    break
                # go in reverse from start positions
                position = startPosition
                otherPosition = otherStartPosition
                t = pd1[0]
                if instrument:
                    print("- go back to position", position, "otherPosition", otherPosition, "t", pd1[0])
            # check for loop; can't happen after a boundary has been found
            if (boundaryCount == 0) and (pointCount > 2):
                x1, d1 = px[-2], pd1[-2]
                x2, d2 = px[-1], pd1[-1]
                dscale = computeCubicHermiteArcLength(x1, d1, x2, d2, rescaleDerivatives=True)
                d1 = mult(d1, dscale)
                d2 = mult(d2, dscale)
                for xi in xiLoopSamples:
                    tx = interpolateCubicHermite(x1, d1, x2, d2, xi)
                    delta = sub(tx, px[0])
                    distance = magnitude(delta)
                    if distance < (0.2 * dscale):
                        loop = True
                        break
                if loop:
                    # remove the last point as close to or past the start
                    px.pop()
                    pd1.pop()
                    pointCount -= 1
                    break
            x, d1, d2 = self.evaluateCoordinates(position, derivatives=True)
            lastx = x
            dirn = mult(t, -1.0) if (boundaryCount == 1) else t
            dxi1, dxi2 = calculate_surface_delta_xi(d1, d2, dirn)
            nextPosition, crossBoundary, adxi1, adxi2 = \
                self._advancePosition(position, dxi1, dxi2, MAX_MAG_DXI=MAX_MAG_DXI)
            if instrument:
                print("  dirn", dirn, "next", nextPosition, "crossBoundary", crossBoundary)

        # resample and re-find positions and tangents
        if pointCount == 1:
            nx, nd1 = px, pd1
        else:
            if loop:
                px.append(px[0])
                pd1.append(pd1[0])
            nx, nd1 = sampleCubicHermiteCurves(px, pd1, curveElementsCount, arcLengthDerivatives=True)[0:2]
            if loop:
                nx.pop()
                nd1.pop()
        cx = []
        cd1 = []
        cProportions = []
        otherPosition = otherTrackSurface.findNearestPositionSample(nx[0])[0]
        assert otherPosition is not None
        for n in range(len(nx)):
            position = self.findNearestPosition(nx[n], position)
            assert position is not None
            # tmpPosition = position
            # tmpOtherPosition = otherPosition
            position, otherPosition, x, t, onBoundary =\
                self.findIntersectionPoint(otherTrackSurface, position, otherPosition)
            if position is None:
                print("\nDid not re-find position!\n")
                # position, otherPosition, x, t, onBoundary = \
                #     self.findIntersectionPoint(otherTrackSurface, tmpPosition, tmpOtherPosition, instrument=True)
            cx.append(x)
            cd1.append(t)
            proportion = self.getProportion(position)
            if self._loop1 and (n > 0) and (abs(proportion[0] - cProportions[-1][0]) > 1.0):
                # wrapped around 0.0 or 2.0: displace proportions to be within (0, 2)
                if proportion[0] < 1.0:
                    proportion[0] += 1.0
                    deltaProportion1 = -1.0
                else:
                    proportion[0] -= 1.0
                    deltaProportion1 = 1.0
                for cProportion in cProportions:
                    cProportion[0] += deltaProportion1
                position = self.createPositionProportion(proportion[0], proportion[1])
            cProportions.append(proportion)
        if pointCount > 1:
            if loop:
                cd1 = smoothCubicHermiteDerivativesLoop(cx, cd1, fixAllDirections=True)
            else:
                # fix indeterminate d1 at either end:
                for n in range(len(nx)):
                    if cd1[n] is not None:
                        if n > 0:
                            cd1[:n + 1] = smoothCubicHermiteDerivativesLine(
                                cx[:n + 1], [[0.0, 0.0, 0.0]] * n + [cd1[n]], fixEndDirection=True)
                        break
                for n in range(len(nx) - 1, -1, -1):
                    if cd1[n] is not None:
                        if n < (len(nx) - 1):
                            cd1[n:] = smoothCubicHermiteDerivativesLine(
                                cx[n:], [cd1[n]] + [[0.0, 0.0, 0.0]] * (len(nx) - 1 - n), fixStartDirection=True)
                        break
                for n in range(1, len(cd1) - 1):
                    if cd1[n] is None:
                        print("Fix missing derivative")
                        cd1[n] = [0.5 * (cd1[n - 1][c] + cd1[n + 1][c]) for c in range(3)]
                cd1 = smoothCubicHermiteDerivativesLine(cx, cd1, fixAllDirections=True)
        return cx, cd1, cProportions, loop

    def findNearestPositionParameter(self, targetx: list):
        """
        Get position of x parameter nearest to targetx.
        Use to set a good starting point for findNearestPosition and findIntersectionPoint.
        :param targetx: Coordinates of point to find nearest to.
        :return: nearest TrackSurfacePosition, nearest distance
        """
        if self._loop1:
            # future: loop option to limit to between [0.5, 1.5]
            # n1Start = self._elementsCount1 // 2
            # n1Limit = n1Start + self._elementsCount1
            n1Start = 0
            n1Limit = self._elementsCount1
        else:
            n1Start = 0
            n1Limit = self._elementsCount1 + 1
        p = 0
        nearest_distance = None
        nearest_n1 = None
        nearest_n2 = None
        for n2 in range(self._elementsCount2 + 1):
            for n1 in range(n1Start, n1Limit):
                distance = magnitude(sub(self._nx[p], targetx))
                if (nearest_distance is None) or (distance < nearest_distance):
                    nearest_distance = distance
                    nearest_n1 = n1
                    nearest_n2 = n2
                p += 1
        return self.createPositionProportion(nearest_n1 / self._elementsCount1, nearest_n2 / self._elementsCount2), \
            nearest_distance

    def findNearestPositionSample(self, targetx: list):
        """
        Get position of nearest element centre to targetx.
        Use to set a good starting point for findNearestPosition and findIntersectionPoint.
        :param targetx: Coordinates of point to find nearest to.
        :return: nearest TrackSurfacePosition, nearest distance
        """
        e1Start = 0
        # future: loop option to limit to between [0.5, 1.5]
        # e1Start = (self._elementsCount1 // 2) if self._loop1 else 0
        e1Limit = e1Start + self._elementsCount1
        nearestDistance = None
        nearestPosition = None
        for e2 in range(self._elementsCount2):
            for e1 in range(e1Start, e1Limit):
                position = TrackSurfacePosition(e1, e2, 0.5, 0.5)
                x = self.evaluateCoordinates(position)
                distance = magnitude(sub(x, targetx))
                if (nearestDistance is None) or (distance < nearestDistance):
                    nearestDistance = distance
                    nearestPosition = position
        return nearestPosition, nearestDistance

    def findNearestPosition(self, targetx: list, startPosition: TrackSurfacePosition = None, instrument=False) \
            -> TrackSurfacePosition:
        """
        Find the nearest point to targetx on the track surface, with optional start position.
        Only works if track surface is simply shaped; use a close startPosition if not.
        :param targetx:  Coordinates of point to find nearest to.
        :param startPosition: Optional initial track surface position.
        :param instrument: Set to True to print debug messages.
        :return: Nearest TrackSurfacePosition
        """
        if instrument:
            print("> findNearestPosition target", targetx, "startPosition", startPosition)
        if not startPosition:
            startPosition = self.createPositionProportion(0.5, 0.5)
        position = copy.deepcopy(startPosition)
        MAX_MAG_DXI = 0.5  # target/maximum magnitude of xi increment
        XI_TOL = 1.0E-7
        MIN_CURVATURE = 0.1 / max(self._xRange)  # minimum to consider
        MAX_CURVATURE_FACTOR = 100.0
        r_mag = 0.0
        mag_adxi = 0
        MAX_ITERS = 100
        for it in range(MAX_ITERS):
            x, d1, d2 = self.evaluateCoordinates(position, derivatives=True)
            onBoundary = self.positionOnBoundary(position)
            r = sub(targetx, x)
            r_mag = magnitude(r)
            if instrument:
                print('> iter', it + 1, 'position', position, 'bdy', onBoundary, 'r', r, 'dist', r_mag)
            # get linear increment
            dxi1, dxi2 = calculate_surface_delta_xi(d1, d2, r)
            dxi = [dxi1, dxi2]
            if instrument:
                print("    initial dxi", dxi)
            if onBoundary:
                # is increment outward from boundary?
                xiBoundaryDirection = self._boundaryDirection(position)
                outward = normalize(add(mult(d1, xiBoundaryDirection[0]), mult(d2, xiBoundaryDirection[1])))
                r_dot_outward = dot(r, outward)
                if r_dot_outward > 0.0:
                    # track along boundary edge - recalculate increment with d1 or d2, as may not be orthogonal
                    if onBoundary == 1:
                        dxi1 = 0.0
                        mag_d = magnitude(d2)
                        dxi2 = dot(d2, r) / (mag_d * mag_d)
                    else:  # if onBoundary == 2:
                        mag_d = magnitude(d1)
                        dxi1 = dot(d1, r) / (mag_d * mag_d)
                        dxi2 = 0.0
                    dxi = [dxi1, dxi2]
                    if instrument:
                        print("    track boundary", onBoundary, "dxi", dxi)
            mag_dxi = magnitude(dxi)
            if mag_dxi == 0.0:
                if instrument:
                    print("TrackSurface.findNearestPosition:  converged in", it + 1, "iterations, mag_dxi", mag_dxi)
                break
            curvature, tangent, dTangent = self._getDirectionalCurvature(position, dxi)
            if curvature > MIN_CURVATURE:
                # get non-linear increment using radius of curvature
                radius = 1.0 / curvature
                jVector = normalize(tangent)
                iVector = normalize(cross(tangent, cross(tangent, dTangent)))
                centre = sub(x, mult(iVector, radius))
                delta = sub(targetx, centre)
                dj = dot(delta, jVector)
                di = dot(delta, iVector)
                angle = math.atan2(dj, di)
                if (it < 10) and (abs(angle) > 0.1):  # radians ~ 6 degrees
                    arcLength = radius * angle
                    originalLength = magnitude(add(mult(d1, dxi1), mult(d2, dxi2)))
                    curvatureFactor = arcLength / originalLength
                else:
                    curvatureFactor = radius / di
                    if curvatureFactor > MAX_CURVATURE_FACTOR:
                         curvatureFactor = MAX_CURVATURE_FACTOR
                dxi1 *= curvatureFactor
                dxi2 *= curvatureFactor
                dxi = [dxi1, dxi2]
                if instrument:
                    print("    curvature", curvature, "curvature factor", curvatureFactor, "angle",
                          math.degrees(angle), "degrees")
                    print("    centre", centre, "delta", delta)
                    print("    iVector", iVector, "di", di)
                    print("    jVector", jVector, "dj", dj)
                    print("    curved dxi", dxi)
            position, onBoundary, adxi1, adxi2 = self._advancePosition(position, dxi1, dxi2, MAX_MAG_DXI=MAX_MAG_DXI)
            adxi = [adxi1, adxi2]
            mag_adxi = magnitude(adxi)
            if instrument:
                print("    final dxi", adxi)
            if mag_adxi < XI_TOL:
                if instrument:
                    print("TrackSurface.findNearestPosition:  converged in", it + 1, "iterations, dxi", mag_adxi)
                break
        else:
            print('TrackSurface.findNearestPosition:  Reached max iterations', it + 1, 'dxi', mag_adxi, 'dist', r_mag)
            # if not instrument:
            #     self.findNearestPosition(targetx, startPosition, instrument=True)
        if instrument:
            print("    final position", position)
        return position

    def findNearestPositionOnCurve(self, cx, cd1, loop=False, startCurveLocation=None, curveSamples: int = 4,
                                   sampleEnds=True, sampleHalf=0, instrument=False):
        """
        Find nearest/intersection point on curve to this surface.
        :param cx: Coordinates along curve.
        :param cd1: Derivatives along curve.
        :param loop: True if curve loops back to first point, False if not.
        :param startCurveLocation: Optional initial location (element index, xi) to search from.
        If not supplied, samples curve element coordinates to get nearest initial curve location.
        :param curveSamples: If startLocation not supplied, sets number of curve xi locations to evaluate when finding
        initial nearest curve location.
        :param sampleEnds: If not loop: set False to remove start/end points from search for initial curve location.
        :param sampleHalf: Region of curve to search for initial curve location: 0=all, 1=first half, 2=last half.
        :param instrument: Set to True to print debug messages.
        :return: Nearest TrackSurfacePosition on self, nearest/intersection point on curve (element index, xi),
        isIntersection (True/False).
        """
        if instrument:
            print("findNearestPositionOnCurve", cx, cd1, loop, startCurveLocation, curveSamples, sampleEnds)
        nCount = len(cx)
        assert nCount > 1
        eCount = nCount if loop else nCount - 1
        curveLocation = copy.copy(startCurveLocation) if startCurveLocation else None
        surfacePosition = None
        if curveLocation:
            targetx = evaluateCoordinatesOnCurve(cx, cd1, curveLocation, loop)
            surfacePosition = self.findNearestPositionSample(targetx)[0]
        else:
            nearestDistance = None
            sCount = eCount * curveSamples
            sStart = 0 if (loop or sampleEnds) else 1
            sLimit = sCount if (loop or not sampleEnds) else sCount + 1
            if sampleHalf == 1:
                sLimit = (sCount + 1) // 2  # first half
            elif sampleHalf == 2:
                sStart = (sCount - 1) // 2  # last half
            for s in range(sStart, sLimit):
                tmpCurveLocation = (s // curveSamples, (s % curveSamples) / curveSamples)
                if not loop and (s == sCount):
                    tmpCurveLocation = (tmpCurveLocation[0] - 1, 1.0)
                targetx = evaluateCoordinatesOnCurve(cx, cd1, tmpCurveLocation, loop)
                tmpSurfacePosition, tmpDistance = self.findNearestPositionSample(targetx)
                if (nearestDistance is None) or (tmpDistance < nearestDistance):
                    nearestDistance = tmpDistance
                    curveLocation = tmpCurveLocation
                    surfacePosition = tmpSurfacePosition
        MAX_MAG_DXI = 0.5  # target/maximum magnitude of xi increment
        XI_TOL = 1.0E-7
        X_TOL = 1.0E-6 * max(self._xRange)
        MAX_SLOPE_FACTOR = 10000.0
        lastOnBoundary = False
        last_dxi = None
        for it in range(100):
            x, d = evaluateCoordinatesOnCurve(cx, cd1, curveLocation, loop, derivative=True)
            surfacePosition = self.findNearestPosition(x, surfacePosition, instrument=False)  # instrument=instrument
            onOtherBoundary = self.positionOnBoundary(surfacePosition)
            ox, od1, od2 = self.evaluateCoordinates(surfacePosition, derivatives=True)
            r = sub(ox, x)
            mag_r = magnitude(r)
            if instrument:
                print("iter", it, "curve location", curveLocation, "surface position", surfacePosition, "mag_r", mag_r)
            if mag_r < X_TOL:
                if instrument:
                    print("TrackSurface.findNearestPositionOnCurve:  Found intersection: ", curveLocation, "on iter", it + 1)
                return surfacePosition, curveLocation, True
            cross_dr = cross(d, r)
            if magnitude(cross_dr) < (1.0E-6 * magnitude(d)):
                rNormal = [0.0, 0.0, 0.0]
                r_dot_n = 0.0
            else:
                n = normalize(cross(cross_dr, d))
                r_dot_n = dot(r, n)
                if r_dot_n < 0.0:
                    # flip normal to be towards other x
                    n = [-s for s in n]
                    r_dot_n = -r_dot_n
                rNormal = mult(n, r_dot_n)
            rTangent = sub(r, rNormal)
            mag_ri = magnitude(rTangent)
            mag_ro = magnitude(rNormal)
            # get tangential displacement u
            if onOtherBoundary:
                u = rTangent
            else:
                # add out-of-plane slope component
                if it < 10:
                    if mag_ri == 0:
                        slope_factor = MAX_SLOPE_FACTOR
                    else:
                        slope_factor = mag_r * mag_r / (mag_ri * mag_ri)
                else:
                    slope_factor = 1.0 + r_dot_n / mag_r  # wrong, but more reliable
                if instrument:
                    print("    slope_factor", slope_factor, "rTangent", rTangent)
                if slope_factor > MAX_SLOPE_FACTOR:
                    slope_factor = MAX_SLOPE_FACTOR
                u = mult(rTangent, slope_factor)
            # limit by curvature and distance to ox
            nm = curveLocation[0]
            np = (nm + 1) % nCount
            curveCurvature = getCubicHermiteCurvatureSimple(cx[nm], cd1[nm], cx[np], cd1[np], curveLocation[1])[0]
            curveSurfaceDirection = calculate_surface_delta_xi(od1, od2, u)
            surfaceCurvature = (
                self._getDirectionalCurvature(surfacePosition, direction=normalize(curveSurfaceDirection))[0]
                if any(curveSurfaceDirection) else 0.0)
            curvature = curveCurvature + surfaceCurvature
            uNormal = sub(r, u)
            un = magnitude(uNormal)
            # GRC check:
            curvatureFactor = 1.0 / (un * curvature + 1.0)
            mag_u = magnitude(u) * curvatureFactor
            # never go further than parallel, based on curvature from initial angle
            parallelFactor = 1.0
            if (mag_ro > 0.0) and (curvature > 0.0):
                max_u = math.atan(mag_ri / mag_ro) / curvature
                if mag_u > max_u:
                    parallelFactor = max_u / mag_u
                    mag_u = max_u
            if instrument:
                print("--> curvature", curvature, "curvature factor", curvatureFactor, "parallel factor",
                      parallelFactor)
            mag_dxi = mag_u / magnitude(d)
            if mag_dxi > MAX_MAG_DXI:
                mag_dxi = MAX_MAG_DXI
            dxi = mag_dxi
            if dot(u, d) < 0.0:
                dxi = -dxi
            if instrument:
                print("    dxi", dxi)
            # control oscillations
            if (it > 0) and ((dxi * last_dxi) < -0.5 * (last_dxi * last_dxi)):
                osc_factor = mag_dxi / (mag_dxi + abs(last_dxi))
                if instrument:
                    print("    osc factor", osc_factor)
                dxi *= osc_factor
                mag_dxi *= osc_factor
            if (it % 20) == 19:
                MAX_MAG_DXI *= 0.5
                if instrument:
                    print("    reduce MAX_MAG_DXI to", MAX_MAG_DXI)
            last_dxi = dxi
            if instrument:
                print("    final dxi", dxi)
            bxi, faceNumber = incrementXiOnLine(curveLocation[1], dxi)
            curveLocation = (curveLocation[0], bxi)
            if faceNumber:
                curveLocation, onBoundary = updateCurveLocationToFaceNumber(curveLocation, faceNumber, eCount, loop)
                if onBoundary and lastOnBoundary:
                    if instrument:
                        print("TrackSurface.findNearestPositionOnCurve:  Found nearest on boundary in",
                              it + 1, "iterations")
                    break
                lastOnBoundary = onBoundary
            else:
                lastOnBoundary = False
            if mag_dxi < XI_TOL:
                if instrument:
                    print("TrackSurface.findNearestPositionOnCurve:  Found nearest in",
                          it + 1, "iterations, dxi", mag_dxi)
                break
        else:
            print('TrackSurface.findNearestPositionOnCurve did not converge:  Reached max iterations', it + 1,
                  'closeness in xi', mag_dxi, 'distance tangent', mag_ri, 'normal', mag_ro)
        return surfacePosition, curveLocation, False

    def trackVector(self, startPosition, direction, trackDistance):
        """
        Track from startPosition the given distance in the vector direction.
        Approximate, uses improved Euler method (mean of original & final gradient).
        :param startPosition: TrackSurfacePosition
        :param direction: 3-D vector (x, y, z) to track along. Projected onto surface.
        :param trackDistance: Distance to track along. Can be negative.
        :return: Final TrackSurfacePosition
        """
        # print("TrackSurface.trackVector  start position", startPosition, "direction", direction,
        #       "distance", trackDistance)
        useDirection = direction
        useTrackDistance = trackDistance
        if trackDistance < 0.0:
            useDirection = [-d for d in direction]
            useTrackDistance = -trackDistance
        position = copy.deepcopy(startPosition)
        distance = 0.0
        distanceLimit = 0.9999*useTrackDistance
        MAX_MAG_DXI = 0.02  # target/maximum magnitude of xi increment

        while distance < useTrackDistance:
            xi1 = position.xi1
            xi2 = position.xi2
            ax, ad1, ad2 = self.evaluateCoordinates(position, derivatives=True)
            adelta_xi1, adelta_xi2 = calculate_surface_delta_xi(ad1, ad2, useDirection)
            # print('adelta_xi', adelta_xi1, adelta_xi2)
            scale = MAX_MAG_DXI/math.sqrt(adelta_xi1*adelta_xi1 + adelta_xi2*adelta_xi2)
            adxi1 = dxi1 = scale*adelta_xi1
            adxi2 = dxi2 = scale*adelta_xi2
            # print('adxi', adxi1, adxi2)
            for i in range(1):  # can try more in future
                # can go slightly outside element to get predictor/correct position
                position.xi1 = xi1 + dxi1
                position.xi2 = xi2 + dxi2
                bx, bd1, bd2 = self.evaluateCoordinates(position, derivatives=True)
                bdelta_xi1, bdelta_xi2 = calculate_surface_delta_xi(bd1, bd2, useDirection)
                # use mean of start and end derivatives
                delta_xi1 = 0.5*(adelta_xi1 + bdelta_xi1)
                delta_xi2 = 0.5*(adelta_xi2 + bdelta_xi2)
                scale = MAX_MAG_DXI/math.sqrt(delta_xi1*delta_xi1 + delta_xi2*delta_xi2)
                dxi1 = scale*delta_xi1
                dxi2 = scale*delta_xi2
            bxi1, bxi2, proportion, faceNumber = increment_xi_on_square(xi1, xi2, dxi1, dxi2)
            position.xi1 = bxi1
            position.xi2 = bxi2
            # print(distance, '-->', position)
            bx, bd1, bd2 = self.evaluateCoordinates(position, derivatives=True)
            bdelta_xi1, bdelta_xi2 = calculate_surface_delta_xi(bd1, bd2, useDirection)
            scale = MAX_MAG_DXI/math.sqrt(bdelta_xi1*bdelta_xi1 + bdelta_xi2*bdelta_xi2)
            bdxi1 = scale*bdelta_xi1
            bdxi2 = scale*bdelta_xi2
            ad = [proportion*(adxi1*ad1[c] + adxi2*ad2[c]) for c in range(3)]
            bd = [proportion*(bdxi1*bd1[c] + bdxi2*bd2[c]) for c in range(3)]
            arcLength = getCubicHermiteArcLength(ax, ad, bx, bd)
            # print('scales', magnitude([ (bx[c] - ax[c]) for c in range(3) ]),
            #       magnitude(ad), magnitude(bd), 'arc length', arcLength)
            if (distance + arcLength) >= distanceLimit:
                # limit to useTrackDistance, approximately, and finish
                r = proportion*(useTrackDistance - distance)/arcLength
                position.xi1 = xi1 + r*dxi1
                position.xi2 = xi2 + r*dxi2
                # print(distance, '-->', position, '(final)')
                break
            if (arcLength == 0.0) and (not faceNumber):
                print('TrackSurface.trackVector. No increment at', position, 'final distance', distance, 'of',
                      useTrackDistance)
                break
            distance += arcLength
            if faceNumber:
                onBoundary = self.updatePositionTofaceNumber(position, faceNumber)
                if onBoundary:
                    print('TrackSurface.trackVector:  End on boundary at', position)
                    break
                # print('  cross face', faceNumber, 'new position', position)
        return position

    def updatePositionTofaceNumber(self, position, faceNumber):
        """
        Update coordinates of TrackSurfacePosition position to cross
        the given face number, and either clamp to range if reached boundary,
        or loop around depending on mode.
        :return: 1 if on xi1 boundary, 2 if on xi2 boundary, 0 if not on a boundary.
        """
        onBoundary = 0
        if faceNumber == 1:  # xi1 == 0.0
            if position.e1 > 0:
                position.e1 -= 1
                position.xi1 = 1.0
            elif self._loop1:
                position.e1 = self._elementsCount1 - 1
                position.xi1 = 1.0
            else:
                position.xi1 = 0.0
                onBoundary = 1
        elif faceNumber == 2:  # xi1 == 1.0
            if position.e1 < (self._elementsCount1 - 1):
                position.e1 += 1
                position.xi1 = 0.0
            elif self._loop1:
                position.e1 = 0
                position.xi1 = 0.0
            else:
                position.xi1 = 1.0
                onBoundary = 1
        elif faceNumber == 3:  # xi2 == 0.0
            if position.e2 > 0:
                position.e2 -= 1
                position.xi2 = 1.0
            else:
                position.xi2 = 0.0
                onBoundary = 2
        elif faceNumber == 4:  # xi2 == 1.0
            if position.e2 < (self._elementsCount2 - 1):
                position.e2 += 1
                position.xi2 = 0.0
            else:
                position.xi2 = 1.0
                onBoundary = 2
        # if onBoundary:
        #     print('!!! Reached boundary of face', faceNumber, 'position', position)
        return onBoundary

    def generateMesh(self, region, startNodeIdentifier: int = None, startElementIdentifier: int = None,
                     serendipity=False, group_name=None):
        """
        Generate nodes and surface elements in region to show track surface.
        Client is required to define all faces.
        :param region: Zinc Region.
        :param startNodeIdentifier: Optional first node identifier to use.
        :param startElementIdentifier: Optional first 2D element identifier to use.
        :param serendipity: Set to True to use Hermite serendipity basis.
        :param group_name: Optional name of group to put new nodes and elements in.
        :return: next node identifier, next 2D element identifier
        """
        fieldmodule = region.getFieldmodule()
        with ChangeManager(fieldmodule):
            coordinates = find_or_create_field_coordinates(fieldmodule)
            group = find_or_create_field_group(fieldmodule, group_name) if group_name else None

            nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            firstNodeIdentifier = startNodeIdentifier if startNodeIdentifier is not None else \
                max(get_maximum_node_identifier(nodes), 0) + 1
            nodeIdentifier = firstNodeIdentifier

            nodetemplate = nodes.createNodetemplate()
            nodetemplate.defineField(coordinates)
            nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
            nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
            if not serendipity:
                nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D2_DS1DS2, 1)

            mesh = fieldmodule.findMeshByDimension(2)
            firstElementIdentifier = startElementIdentifier if startElementIdentifier is not None else \
                max(get_maximum_element_identifier(mesh), 0) + 1
            elementIdentifier = firstElementIdentifier
            elementtemplate = mesh.createElementtemplate()
            elementtemplate.setElementShapeType(Element.SHAPE_TYPE_SQUARE)
            bicubicHermiteBasis = fieldmodule.createElementbasis(
                2, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY if (serendipity and not self._nd12)
                else Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
            eft = mesh.createElementfieldtemplate(bicubicHermiteBasis)
            # keep cross derivatives so can experiment with effect
            # if not serendipity:
            #     # remove cross derivative terms for regular Hermite
            #     for n in range(4):
            #         eft.setFunctionNumberOfTerms(n * 4 + 4, 0)
            elementtemplate.defineField(coordinates, -1, eft)

            fieldcache = fieldmodule.createFieldcache()

            nodeset_group = group.getOrCreateNodesetGroup(nodes) if group else None
            for n1 in range(len(self._nx)):
                node = nodes.createNode(nodeIdentifier, nodetemplate)
                fieldcache.setNode(node)
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, self._nx[n1])
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, self._nd1[n1])
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, self._nd2[n1])
                if self._nd12:
                    coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, self._nd12[n1])
                if nodeset_group:
                    nodeset_group.addNode(node)
                nodeIdentifier += 1
            del n1

            mesh_group = group.getOrCreateMeshGroup(mesh) if group else None
            nodesCount1 = self._elementsCount1 if self._loop1 else self._elementsCount1 + 1
            for e2 in range(self._elementsCount2):
                for e1 in range(self._elementsCount1):
                    nid0 = firstNodeIdentifier + e2 * nodesCount1
                    n1 = nid0 + e1
                    n2 = nid0 if (self._loop1 and ((e1 + 1) == self._elementsCount1)) else n1 + 1
                    n3 = n1 + nodesCount1
                    n4 = n2 + nodesCount1
                    nids = [n1, n2, n3, n4]
                    element = mesh.createElement(elementIdentifier, elementtemplate)
                    element.setNodesByIdentifier(eft, nids)
                    # print(elementIdentifier, element.isValid(), nids)
                    if mesh_group:
                        mesh_group.addElement(element)
                    elementIdentifier += 1

            if group:
                # ensure all lines are in group
                group.setSubelementHandlingMode(FieldGroup.SUBELEMENT_HANDLING_MODE_FULL)
                mesh_group.addElementsConditional(group)

        return nodeIdentifier, elementIdentifier


def calculate_surface_delta_xi(d1, d2, direction):
    """
    Calculate dxi1, dxi2 in 3-D vector direction.
    :param d1: Vector surface derivative of coordinate w.r.t. xi1.
    :param d2: Vector surface derivative of coordinate w.r.t. xi2.
    :param direction: 3-D vector.
    :return: delta_xi1, delta_xi2
    """
    # overdetermined system (3-D vector, 2-D surface), solve least squares
    # A transpose A x = A transpose b
    a = [[0.0, 0.0], [0.0, 0.0]]
    b = [0.0, 0.0]
    dx_dxi = [d1, d2]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a[i][j] += dx_dxi[i][k] * dx_dxi[j][k]
        for k in range(3):
            b[i] += dx_dxi[i][k] * direction[k]
    # 2x2 matrix inverse
    deta = a[0][0]*a[1][1] - a[0][1]*a[1][0]
    if deta > 0.0:
        inva = [[a[1][1] / deta, -a[0][1] / deta], [-a[1][0] / deta, a[0][0] / deta]]
        delta_xi1 = inva[0][0]*b[0] + inva[0][1]*b[1]
        delta_xi2 = inva[1][0]*b[0] + inva[1][1]*b[1]
    else:
        # at pole: assume direction is inline with d1 or d2 and other is zero
        delta_xi2 = dot(d2, direction)
        if math.fabs(delta_xi2) > 0.0:
            delta_xi1 = 0.0
            delta_xi2 = (1.0 if (delta_xi2 > 0.0) else -1.0)*magnitude(direction)/magnitude(d2)
        else:
            delta_xi1 = dot(d1, direction)
            if math.fabs(delta_xi1) > 0.0:
                delta_xi1 = (1.0 if (delta_xi1 > 0.0) else -1.0)*magnitude(direction)/magnitude(d1)
                delta_xi2 = 0.0
    # delx = [ (delta_xi1*d1[c] + delta_xi2*d2[c]) for c in range(3) ]
    # print('delx', delx, 'dir', direction, 'diff', magnitude([ (delx[c] - direction[c]) for c in range(3) ]))
    return delta_xi1, delta_xi2


def calculate_surface_axes(d1, d2, direction):
    """
    :return: Vectors ax1, ax2, ax3: ax1 in-plane in 3-D vector direction,
    ax2 in-plane normal to ax1 and ax3 normal to the surface plane.
    Vectors all have unit magnitude.
    """
    delta_xi1, delta_xi2 = calculate_surface_delta_xi(d1, d2, direction)
    ax1 = normalize([delta_xi1*d1[c] + delta_xi2*d2[c] for c in range(3)])
    ax3 = cross(d1, d2)
    mag3 = magnitude(ax3)
    if mag3 > 0.0:
        ax3 = [s/mag3 for s in ax3]
        ax2 = normalize(cross(ax3, ax1))
    else:
        ax3 = [0.0, 0.0, 0.0]
        ax2 = [0.0, 0.0, 0.0]
    return ax1, ax2, ax3


def increment_xi_on_square(xi1, xi2, dxi1, dxi2):
    """
    Increment xi1, xi2 by dxi1, dxi2 limited to square element bounds on [0,1].
    Works out face crossed first and limits that xi to 0.0 or 1.0 and other xi
    in proportion.
    :return: New xi1, xi2, proportion, face number 1-4 or None if within boundary.
    Proportion is 1.0 if increment is within element, otherwise equal to proportion
    of dxi1, dxi2 incremented to hit the boundary. If so limited to the boundary,
    the face number is returned as an integer:
    1 is xi1==0.0, 2 is xi1==1.0, 3 is xi2==0.0, 4 is xi2==1.0
    """
    onxi1 = nxi1 = xi1 + dxi1
    onxi2 = nxi2 = xi2 + dxi2
    proportion = 1.0
    faceNumber = None
    if (nxi1 < 0.0) or (nxi1 > 1.0) or (nxi2 < 0.0) or (nxi2 > 1.0):
        # come back in direction of dxi1, dxi2 to first boundary
        if (onxi1 < 0.0) and (dxi1 < 0.0):
            thisProportion = -xi1/dxi1
            if thisProportion < proportion:
                proportion = thisProportion
                faceNumber = 1
                nxi1 = 0.0
                nxi2 = xi2 + proportion*dxi2
        elif (onxi1 > 1.0) and (dxi1 > 0.0):
            thisProportion = (1.0 - xi1)/dxi1
            if thisProportion < proportion:
                proportion = thisProportion
                faceNumber = 2
                nxi1 = 1.0
                nxi2 = xi2 + proportion*dxi2
        if (onxi2 < 0.0) and (dxi2 < 0.0):
            thisProportion = -xi2/dxi2
            if thisProportion < proportion:
                proportion = thisProportion
                faceNumber = 3
                nxi1 = xi1 + proportion*dxi1
                nxi2 = 0.0
        elif (onxi2 > 1.0) and (dxi2 > 0.0):
            thisProportion = (1.0 - xi2)/dxi2
            if thisProportion < proportion:
                proportion = thisProportion
                faceNumber = 4
                nxi1 = xi1 + proportion*dxi1
                nxi2 = 1.0
        # print('  increment to face', faceNumber, 'xi (' + str(xi1) + ',' + str(xi2) + ')', 'nxi (' + str(nxi1)
        #       + ',' + str(nxi2) + ')')
    return nxi1, nxi2, proportion, faceNumber
