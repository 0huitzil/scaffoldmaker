"""
Class for generating a shield-shaped mesh, with a regular flat top but
a rounded bottom formed by having two points where 3 square elements
merge to form a triangle.
"""

from __future__ import division
import copy
from enum import Enum

from cmlibs.maths.vectorops import cross, normalize
from cmlibs.zinc.element import Element
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from scaffoldmaker.utils.eft_utils import remapEftNodeValueLabel, setEftScaleFactorIds
from scaffoldmaker.utils.eftfactory_tricubichermite import eftfactory_tricubichermite
from scaffoldmaker.utils.interpolation import DerivativeScalingMode, sampleCubicHermiteCurves, \
    smoothCubicHermiteDerivativesLine
from scaffoldmaker.utils.mirror import Mirror
from scaffoldmaker.utils.tracksurface import TrackSurface, calculate_surface_axes


class ShieldShape2D(Enum):
    SHIELD_SHAPE_FULL = 1
    SHIELD_SHAPE_LOWER_HALF = 2


class ShieldShape3D(Enum):
    SHIELD_SHAPE_FULL = 1
    SHIELD_SHAPE_HALF_AAP = 2    # AAP is a hemisphere where x_a3>=0
    SHIELD_SHAPE_OCTANT_PPP = 3   # PPP means positive axis1, positive axis2, positive axis3.
    SHIELD_SHAPE_OCTANT_PNP = 4
    SHIELD_SHAPE_OCTANT_NNP = 5
    SHIELD_SHAPE_OCTANT_NPP = 6
    SHIELD_SHAPE_OCTANT_PPN = 7
    SHIELD_SHAPE_OCTANT_PNN = 8
    SHIELD_SHAPE_OCTANT_NNN = 9
    SHIELD_SHAPE_OCTANT_NPN = 10


class ShieldRimDerivativeMode(Enum):
    SHIELD_RIM_DERIVATIVE_MODE_AROUND = 1  # rim derivatives are d1 anticlockwise around shield, d3 outward
    SHIELD_RIM_DERIVATIVE_MODE_REGULAR = 2  # rim derivatives d1, d2 match interior nodes for regular elements


class ShieldMesh2D:
    """
    Shield mesh generator.
    """

    def __init__(self, elementsCountAcross, elementsCountUpFull, elementsCountRim, trackSurface: TrackSurface=None,
                 elementsCountAlong=1, shieldMode=ShieldShape2D.SHIELD_SHAPE_LOWER_HALF,
                 shieldType=ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR):
        r"""
        Data structure for defining a shield-shaped mesh which is flat on the top and rounded around the bottom
        and/or the same mirror mirrored on top.
        It is represented as a regular box of elementsCountAcross x elementsCountUp
        but with strips of elements absent on the bottom left and right.
        Example showing elements for 4 across, 2 up and 0 rim, and how nodes/coordinates are stored in ShieldMesh:

        N___N___N___N___N           N___N___N___N___N
        |   |   |   |   |               |   |   |
        |   |   |   |   |               |   |   |
        \   T---N---T   /    -->        T---N---T
         \ /    |    \ /                |   |   |
          N     |     N                 |   |   |
           \----N----/                  N---N---N

        The removal of 2 corners of the box is achieved by triple points T where 3 square elements
        join in a triangle.
        Extra rim elements go around the sides and bottom curve. Rim elements add nodes on the sides above the
        triple points, and across the shorter bottom row.
        Extra elements up add regular rows of nodes/elements on top, and extra non-rim elements across
        add regular columns of nodes/elements up the centre.
        :param elementsCountAcross: Number of elements across top of shield. Must be at least  4 + 2*elementsCountRim.
        :param elementsCountUpFull: Number of elements up central axis of shield. Must be at least 2 + elementsCountRim
         if half and 4 + 2*elementsCountRim if full.
        :param elementsCountAlong: Number of elements through wall for ventricle case (only 1 element is supported) and
        along cylinder axis in cylinder case.
        :param elementsCountRim: Number of elements around bottom rim (not top) outside of 'triple points'.
        :param trackSurface: Optional trackSurface to store or restrict points to.
        :param shieldMode: It determines if the shield is full or just part of it.
        :param shieldType: To distinguish between cylinder and ventricle type. Derivatives and directions are chosen
         differently for two cases.
        """
        assert elementsCountRim >= 0
        assert elementsCountAlong >= 1
        assert elementsCountAcross >= (elementsCountRim + 4)
        assert elementsCountUpFull >= (elementsCountRim + 2)
        self.elementsCountAcross = elementsCountAcross
        self.elementsCountUpFull = elementsCountUpFull
        elementsCountUp = elementsCountUpFull//2 if shieldMode == ShieldShape2D.SHIELD_SHAPE_FULL\
            else elementsCountUpFull
        self.elementsCountUp = elementsCountUp
        self.elementsCountRim = elementsCountRim
        self.elementsCountAlong = elementsCountAlong
        self.elementsCountUpRegular = elementsCountUp - 2 - elementsCountRim
        elementsCountAcrossNonRim = self.elementsCountAcross - 2*elementsCountRim
        self.elementsCountAroundFull = 2*self.elementsCountUpRegular + elementsCountAcrossNonRim
        self.trackSurface = trackSurface
        self._mode = shieldMode
        self._type = shieldType
        self.px = [[] for _ in range(elementsCountAlong+1)]
        self.pd1 = [[] for _ in range(elementsCountAlong+1)]
        self.pd2 = [[] for _ in range(elementsCountAlong+1)]
        self.pd3 = [[] for _ in range(elementsCountAlong+1)]
        self.nodeId = [[] for _ in range(elementsCountAlong+1)]
        for n3 in range(elementsCountAlong+1):
            for n2 in range(elementsCountUpFull + 1):
                for p in [self.px[n3], self.pd1[n3], self.pd2[n3], self.pd3[n3], self.nodeId[n3]]:
                    p.append([None]*(elementsCountAcross + 1))
        if trackSurface:
            self.pProportions = [[None]*(elementsCountAcross + 1) for n2 in range(elementsCountUp + 1)]
        if shieldType == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
            self.elementId = [[[None]*elementsCountAcross for n2 in range(elementsCountUpFull)]
                              for e3 in range(elementsCountAlong)]
        else:
            self.elementId = [[None] * elementsCountAcross for n2 in range(elementsCountUpFull)]

    def convertRimIndex(self, ix, rx=0):
        """
        Convert point index around the lower rim to n1, n2 across and up box.
        :param ix: index around from 0 to self.elementsCountAroundFull
        :param rx: rim index from 0 (around outside) to self.elementsCountRim
        :return: n1, n2
        """
        assert 0 <= ix <= self.elementsCountAroundFull
        assert 0 <= rx <= self.elementsCountRim
        if ix <= self.elementsCountUpRegular:
            return rx, self.elementsCountUp - ix
        mx = self.elementsCountAroundFull - ix
        if mx <= self.elementsCountUpRegular:
            return self.elementsCountAcross - rx, self.elementsCountUp - mx
        return self.elementsCountRim + ix - self.elementsCountUpRegular, rx

    def getTriplePoints(self, n3):
        """
        Compute coordinates and derivatives of points where 3 square elements merge.
        :param n3: Index of through-wall coordinates to use.
        """
        n1a = self.elementsCountRim
        n1b = n1a + 1
        n1c = n1a + 2
        m1a = self.elementsCountAcross - self.elementsCountRim
        m1b = m1a - 1
        m1c = m1a - 2
        n2a = self.elementsCountRim
        n2b = n2a + 1
        n2c = n2a + 2
        # left
        ltx = []

        if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2a][n1c], self.px[n3][n2c][n1b]],
                [[(-self.pd1[n3][n2a][n1c][c] - self.pd3[n3][n2a][n1c][c]) for c in range(3)], self.pd1[n3][n2c][n1b]],
                2, arcLengthDerivatives=True)[0:2]
            ltx.append(tx[1])
            # tx, td1 = sampleCubicHermiteCurves(
            #     [ self.px[n3][n2a][n1b], self.px[n3][n2c][n1c] ], [ [-self.pd3[n3][n2a][n1b][c] for c in range(3)],
            #     [ (self.pd1[n3][n2c][n1c][c] + self.pd3[n3][n2c][n1c][c]) for c in range(3) ] ],
            #     2, arcLengthDerivatives = True)[0:2]
            ltx.append(tx[1])
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2c][n1a], self.px[n3][n2b][n1c]],
                [[(self.pd1[n3][n2c][n1a][c] - self.pd3[n3][n2c][n1a][c]) for c in range(3)],
                 self.pd3[n3][n2b][n1c]], 2, arcLengthDerivatives=True)[0:2]
            ltx.append(tx[1])
        elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2a][n1c], self.px[n3][n2c][n1b]],
                [[(-self.pd1[n3][n2a][n1c][c] + self.pd2[n3][n2a][n1c][c]) for c in range(3)],
                 self.pd2[n3][n2c][n1b]], 2, arcLengthDerivatives=True)[0: 2]
            ltx.append(tx[1])
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2a][n1b], self.px[n3][n2c][n1c]],
                [self.pd2[n3][n2a][n1b], [(self.pd1[n3][n2c][n1c][c] + self.pd2[n3][n2c][n1c][c]) for c in range(3)]],
                2, arcLengthDerivatives=True)[0: 2]
            ltx.append(tx[1])
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2c][n1a], self.px[n3][n2b][n1c]],
                [[(self.pd1[n3][n2c][n1a][c] - self.pd2[n3][n2c][n1a][c]) for c in range(3)], self.pd1[n3][n2b][n1c]],
                2, arcLengthDerivatives=True)[0: 2]
            ltx.append(tx[1])
        #  x = [ (ltx[0][c] + ltx[1][c] + ltx[2][c])/3.0 for c in range(3) ]
        x = [(ltx[0][c] + ltx[2][c])/2.0 for c in range(3)]
        if self.trackSurface:
            p = self.trackSurface.findNearestPosition(x, startPosition=self.trackSurface.createPositionProportion(*(
                self.pProportions[n2b][n1c])))
            self.pProportions[n2b][n1b] = self.trackSurface.getProportion(p)
            x, sd1, sd2 = self.trackSurface.evaluateCoordinates(p, derivatives=True)
            d1, d2, d3 = calculate_surface_axes(sd1, sd2, normalize(sd1))
            self.pd3[n3][n2b][n1b] = d3
        self.px[n3][n2b][n1b] = x
        if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
            self.pd3[n3][n2b][n1b] = [(self.px[n3][n2b][n1c][c] - self.px[n3][n2b][n1b][c]) for c in range(3)]
            self.pd1[n3][n2b][n1b] = [(self.px[n3][n2c][n1b][c] - self.px[n3][n2b][n1b][c]) for c in range(3)]
        elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
            self.pd1[n3][n2b][n1b] = [(self.px[n3][n2b][n1c][c] - self.px[n3][n2b][n1b][c]) for c in range(3)]
            self.pd2[n3][n2b][n1b] = [(self.px[n3][n2c][n1b][c] - self.px[n3][n2b][n1b][c]) for c in range(3)]
        if not self.trackSurface:
            if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                self.pd2[n3][n2b][n1b] = normalize(cross(self.pd3[n3][n2b][n1b],
                                                                               self.pd1[n3][n2b][n1b]))
            elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                self.pd3[n3][n2b][n1b] = normalize(cross(self.pd1[n3][n2b][n1b],
                                                                               self.pd2[n3][n2b][n1b]))
        # right
        rtx = []
        if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2a][m1c], self.px[n3][n2c][m1b]],
                [[(self.pd1[n3][n2a][m1c][c] - self.pd3[n3][n2a][m1c][c]) for c in range(3)], self.pd1[n3][n2c][m1b]],
                2, arcLengthDerivatives=True)[0:2]
            rtx.append(tx[1])
            # tx, td1 = sampleCubicHermiteCurves(
            #     [ self.px[n3][n2a][m1b], self.px[n3][n2c][m1c] ], [ [-self.pd3[n3][n2a][m1b][c] for c in range(3)],
            #     [ (-self.pd3[n3][n2c][m1c][c] + self.pd1[n3][n2c][m1c][c]) for c in range(3) ] ],
            #     2, arcLengthDerivatives = True)[0:2]
            rtx.append(tx[1])
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2c][m1a], self.px[n3][n2b][m1c]],
                [[(-self.pd1[n3][n2c][m1a][c] - self.pd3[n3][n2c][m1a][c]) for c in range(3)],
                 [-d for d in self.pd3[n3][n2b][m1c]]], 2, arcLengthDerivatives=True)[0:2]
            rtx.append(tx[1])
        elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2a][m1c], self.px[n3][n2c][m1b]],
                [[(self.pd1[n3][n2a][m1c][c] + self.pd2[n3][n2a][m1c][c]) for c in range(3)], self.pd2[n3][n2c][m1b]],
                2, arcLengthDerivatives=True)[0: 2]
            rtx.append(tx[1])
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2a][m1b], self.px[n3][n2c][m1c]],
                [self.pd2[n3][n2a][m1b], [(-self.pd1[n3][n2c][m1c][c] + self.pd2[n3][n2c][m1c][c]) for c in range(3)]],
                2, arcLengthDerivatives=True)[0: 2]
            rtx.append(tx[1])
            tx, td1 = sampleCubicHermiteCurves(
                [self.px[n3][n2c][m1a], self.px[n3][n2b][m1c]],
                [[(-self.pd1[n3][n2c][m1a][c] - self.pd2[n3][n2c][m1a][c]) for c in range(3)],
                 [-d for d in self.pd1[n3][n2b][m1c]]], 2, arcLengthDerivatives=True)[0: 2]
            rtx.append(tx[1])
        #  x = [ (rtx[0][c] + rtx[1][c] + rtx[2][c])/3.0 for c in range(3) ]
        x = [(rtx[0][c] + rtx[2][c])/2.0 for c in range(3)]
        if self.trackSurface:
            p = self.trackSurface.findNearestPosition(x, startPosition=self.trackSurface.createPositionProportion(*(
                self.pProportions[n2b][m1c])))
            self.pProportions[n2b][m1b] = self.trackSurface.getProportion(p)
            x, sd1, sd2 = self.trackSurface.evaluateCoordinates(p, derivatives=True)
            d1, d2, d3 = calculate_surface_axes(sd1, sd2, normalize(sd1))
            self.pd3[n3][n2b][m1b] = d3
        self.px[n3][n2b][m1b] = x
        if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
            self.pd3[n3][n2b][m1b] = [(self.px[n3][n2b][m1b][c] - self.px[n3][n2b][m1c][c]) for c in range(3)]
            self.pd1[n3][n2b][m1b] = [(self.px[n3][n2c][m1b][c] - self.px[n3][n2b][m1b][c]) for c in range(3)]
        elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
            self.pd1[n3][n2b][m1b] = [(self.px[n3][n2b][m1b][c] - self.px[n3][n2b][m1c][c]) for c in range(3)]
            self.pd2[n3][n2b][m1b] = [(self.px[n3][n2c][m1b][c] - self.px[n3][n2b][m1b][c]) for c in range(3)]
        if not self.trackSurface:
            if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                self.pd2[n3][n2b][m1b] = normalize(cross(self.pd3[n3][n2b][m1b],
                                                                               self.pd1[n3][n2b][m1b]))
            elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                self.pd3[n3][n2b][m1b] = normalize(cross(self.pd1[n3][n2b][m1b],
                                                                               self.pd2[n3][n2b][m1b]))

    def smoothDerivativesToTriplePoints(self, n3, fixAllDirections=False):
        """
        Smooth derivatives leading to triple points where 3 square elements merge.
        :param n3: Index of through-wall coordinates to use.
        """
        n1a = self.elementsCountRim
        n1b = n1a + 1
        n1z = self.elementsCountAcross - self.elementsCountRim
        n1y = n1z - 1
        m1a = self.elementsCountAcross - self.elementsCountRim
        m1b = m1a - 1
        n2a = self.elementsCountRim
        n2b = n2a + 1
        n2c = n2a + 2
        if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
            # left
            tx = []
            td3 = []
            for n2 in range(n2c):
                tx.append(self.px[n3][n2][n1b])
                if n2 < n2b:
                    td3.append([-self.pd3[n3][n2][n1b][c] for c in range(3)])
                else:
                    td3.append([(self.pd1[n3][n2b][n1b][c] + self.pd3[n3][n2b][n1b][c]) for c in range(3)])

            td3 = smoothCubicHermiteDerivativesLine(tx, td3, fixStartDirection=True, fixEndDirection=True)

            for n2 in range(n2b):
                self.pd3[n3][n2][n1b] = [-td3[n2][c] for c in range(3)]

            # right
            tx = []
            td3 = []
            for n2 in range(n2c):
                tx.append(self.px[n3][n2][n1y])
                if n2 < n2b:
                    td3.append([-self.pd3[n3][n2][n1y][c] for c in range(3)])
                else:
                    td3.append([(self.pd1[n3][n2b][n1y][c] - self.pd3[n3][n2b][n1y][c]) for c in range(3)])

            td3 = smoothCubicHermiteDerivativesLine(tx, td3, fixStartDirection=True, fixEndDirection=True)

            for n2 in range(n2b):
                self.pd3[n3][n2][n1y] = [-td3[n2][c] for c in range(3)]

        elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
            # left
            tx = []
            td2 = []
            for n2 in range(0, n2c):
                tx .append(self.px[n3][n2][n1b])
                td2.append(self.pd2[n3][n2][n1b] if (n2 < n2b) else
                           [(self.pd1[n3][n2][n1b][c] + self.pd2[n3][n2][n1b][c]) for c in range(3)])
            td2 = smoothCubicHermiteDerivativesLine(tx, td2, fixAllDirections=fixAllDirections, fixEndDerivative=True,
                                                    magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
            for n2 in range(0, n2b):
                self.pd2[n3][n2][n1b] = td2[n2]
            # right
            tx = []
            td2 = []
            for n2 in range(0, n2c):
                tx .append(self.px[n3][n2][m1b])
                td2.append(self.pd2[n3][n2][m1b] if (n2 < n2b) else
                           [(-self.pd1[n3][n2][m1b][c] + self.pd2[n3][n2][m1b][c]) for c in range(3)])
            td2 = smoothCubicHermiteDerivativesLine(tx, td2, fixAllDirections=fixAllDirections, fixEndDerivative=True,
                                                    magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
            for n2 in range(0, n2b):
                self.pd2[n3][n2][m1b] = td2[n2]

    def smoothDerivativesAroundRim(self, n3, n3d=None, rx=0):
        """
        Smooth derivatives around rim.
        :param n3: Index of through-wall coordinates to use.
        :param n3d: Which n3 index to copy initial derivatives from. If None, use n3.
        :param rx: rim index from 0 (around outside) to self.elementsCountRim
        """
        assert 0 <= rx <= self.elementsCountRim
        if not n3d:
            n3d = n3
        tx = []
        td1 = []
        for ix in range(self.elementsCountAroundFull + 1):
            n1, n2 = self.convertRimIndex(ix, rx)
            tx.append(self.px[n3][n2][n1])
            if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                td1.append(self.pd1[n3d][n2][n1])
            elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                if n2 > self.elementsCountRim:  # regular rows
                    if n1 <= self.elementsCountRim:
                        td1.append([-d for d in self.pd2[n3d][n2][n1]])
                    else:
                        td1.append(self.pd2[n3d][n2][n1])
                else:
                    td1.append(self.pd1[n3d][n2][n1])

        if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
            td1 = smoothCubicHermiteDerivativesLine(tx, td1, fixStartDirection=True, fixEndDirection=True)
        elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
            td1 = smoothCubicHermiteDerivativesLine(tx, td1)

        for ix in range(self.elementsCountAroundFull + 1):
            n1, n2 = self.convertRimIndex(ix, rx)
            if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                self.pd1[n3][n2][n1] = td1[ix]
            elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                if n2 > self.elementsCountRim:  # regular rows
                    if n1 <= self.elementsCountRim:
                        self.pd2[n3][n2][n1] = [-d for d in td1[ix]]
                    else:
                        self.pd2[n3][n2][n1] = td1[ix]
                else:
                    self.pd1[n3][n2][n1] = td1[ix]

    def remap_derivatives(self, squareMapping, circleMapping=None):
        """
        It remaps the derivatives as indicated by squareMapping and circleMapping.
        Limited to SHIELD_RIM_DERIVATIVE_MODE_AROUND.
        :param squareMapping: List[up, right]. Determines what derivatives should be in the up and right directions in
         the square part. Their values are in [1,2,3] range which 1, 2, 3 means d1, d2 and d3 respectively.
          The negative sign reverses the direction. e.g. [-3,2] means d3 is down and d2 is right. The third derivative
           is determined by the other two. RH rule applied. Assumes [1,3] initially.
        :param circleMapping: List[circumferential, radial]. Determines what derivatives should be used for
         circumferential and radial directions around the circle.
         [-1, 3] means d1 -> clockwise and around. d3 -> outward and radial. Assumes [1,3] initially.
        :return:
        """
        dct = {1: self.pd1, 2: self.pd2, 3: self.pd3}
        perm = {(1, 2): -3, (2, 3): -1, (3, 1): -2, (3, 2): 1, (1, 3): 2, (2, 1): 3}

        square = True
        tripleRow = [self.elementsCountRim + 1, self.elementsCountUpFull - (self.elementsCountRim + 1)]
        ellipseMapping = [squareMapping, circleMapping]
        for mapping in ellipseMapping:
            if mapping:
                signs = []
                for c in mapping:
                    sign = 1 if c > 0 else -1
                    signs.append(sign)
                derv = (abs(mapping[0]), abs(mapping[1]))
                sign = 1 if perm[derv] > 0 else -1
                signs.append(signs[0]*signs[1]*sign)
                dervMapping = (derv[0], derv[1], abs(perm[derv]))
                temp1 = copy.deepcopy(self.pd3)
                temp2 = copy.deepcopy(self.pd2)
                for n2 in range(self.elementsCountUpFull + 1):
                    for n3 in range(self.elementsCountAlong+1):
                        for n1 in range(self.elementsCountAcross + 1):
                            if self.px[n3][n2][n1]:
                                is_on_square = ((self.px[n3][n2][0] and self.px[n3][0][n1]) or n2 in tripleRow)
                                if (is_on_square and square) or (not is_on_square and not square):
                                    dct[dervMapping[0]][n3][n2][n1] = [signs[0]*c for c in self.pd1[n3][n2][n1]]
                                    dct[dervMapping[1]][n3][n2][n1] = [signs[1]*c for c in temp1[n3][n2][n1]]
                                    dct[dervMapping[2]][n3][n2][n1] = [signs[2]*c for c in temp2[n3][n2][n1]]
            square = False

    def generateNodesForOtherHalf(self, mirrorPlane):
        """
        Generates coordinates and derivatives for the other half by mirroring them. It keeps the d1 direction.
        :param mirrorPlane: plane ax+by+cz=d in form of [a,b,c,d]
        :return:
        """
        mirror = Mirror(mirrorPlane)
        for n2 in range(self.elementsCountUp):
            for n3 in range(self.elementsCountAlong + 1):
                for n1 in range(self.elementsCountAcross + 1):
                    if self.px[n3][n2][n1]:
                        self.px[n3][2*self.elementsCountUp-n2][n1] = mirror.mirrorImageOfPoint(self.px[n3][n2][n1])
                        self.pd1[n3][2*self.elementsCountUp-n2][n1] = mirror.reverseMirrorVector(self.pd1[n3][n2][n1])
                        self.pd2[n3][2*self.elementsCountUp-n2][n1] = mirror.mirrorVector(self.pd2[n3][n2][n1])
                        self.pd3[n3][2*self.elementsCountUp-n2][n1] = mirror.mirrorVector(self.pd3[n3][n2][n1])

    def generateNodes(self, fieldmodule, coordinates, startNodeIdentifier, rangeOfRequiredElements=None,
                      mirrorPlane=None):
        """
        Create shield nodes from coordinates.
        :param fieldmodule: Zinc fieldmodule to create nodes in. Uses DOMAIN_TYPE_NODES.
        :param coordinates: Coordinate field to define.
        :param startNodeIdentifier: First node identifier to use.
        :param rangeOfRequiredElements: Only the elements and nodes for the given range is generated.
        Specify None to create all.
        :param mirrorPlane: mirror plane ax+by+cz=d in form of [a,b,c,d]
        :return: next nodeIdentifier.
         """
        nodeIdentifier = startNodeIdentifier
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)
        cache = fieldmodule.createFieldcache()

        # for n2 in range(self.elementsCountUp, -1, -1):
        #    s = ""
        #    for n1 in range(self.elementsCountAcross + 1):
        #        s += str(n1) if self.px[1][n2][n1] else " "
        #    print(n2, s, n2 - self.elementsCountUp - 1)

        if self._mode == ShieldShape2D.SHIELD_SHAPE_FULL and mirrorPlane:
            self.generateNodesForOtherHalf(mirrorPlane)

        for n2 in range(self.elementsCountUpFull + 1):
            for n3 in range(self.elementsCountAlong+1):
                for n1 in range(self.elementsCountAcross + 1):
                    if rangeOfRequiredElements:
                        if n3 > rangeOfRequiredElements[2][1] or n3 < rangeOfRequiredElements[2][0]\
                                or n2 > rangeOfRequiredElements[0][1] or n2 < rangeOfRequiredElements[0][0]\
                                or n1 > rangeOfRequiredElements[1][1] or n1 < rangeOfRequiredElements[1][0]:
                            continue
                    if self.px[n3][n2][n1]:
                        node = nodes.createNode(nodeIdentifier, nodetemplate)
                        self.nodeId[n3][n2][n1] = nodeIdentifier
                        cache.setNode(node)
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, self.px[n3][n2][n1])
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, self.pd1[n3][n2][n1])
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, self.pd2[n3][n2][n1])
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, self.pd3[n3][n2][n1])
                        nodeIdentifier += 1

        return nodeIdentifier

    def generateElements(self, fieldmodule, coordinates, startElementIdentifier, rangeOfRequiredElements=None,
                         meshGroupsElementsAlong=[], meshGroups=[]):
        """
        Create shield elements from nodes.
        :param fieldmodule: Zinc fieldmodule to create elements in.
        :param coordinates: Coordinate field to define.
        :param startElementIdentifier: First element identifier to use.
        :param rangeOfRequiredElements: Only the elements and nodes for the given range is generated.
        Specify None to create all.
        :param meshGroups, meshGroupsElementsAlong: Zinc mesh groups to add elements to. meshGroupsElementsAlong is a
        list that specifies number of elements along the cylinder to be added to the same meshGroup.
        :return: next elementIdentifier.
         """
        assert len(meshGroupsElementsAlong) == len(meshGroups), 'meshGroups and meshGroupsElementsAlong should have ' \
                                                                'the same length'
        elementIdentifier = startElementIdentifier
        useCrossDerivatives = False
        mesh = fieldmodule.findMeshByDimension(3)

        tricubichermite = eftfactory_tricubichermite(mesh, useCrossDerivatives)
        eft = tricubichermite.createEftNoCrossDerivatives()
        elementtemplate = mesh.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        elementtemplate.defineField(coordinates, -1, eft)

        elementtemplate1 = mesh.createElementtemplate()
        elementtemplate1.setElementShapeType(Element.SHAPE_TYPE_CUBE)

        elementEnd = []
        count = 0
        for c in meshGroupsElementsAlong:
            count += c
            elementEnd.append(count)

        isEven = (self.elementsCountAcross % 2) == 0
        e1a = self.elementsCountRim
        e1b = e1a + 1
        e1z = self.elementsCountAcross - 1 - self.elementsCountRim
        e1y = e1z - 1
        e2a = self.elementsCountRim
        e2b = self.elementsCountRim + 1
        e2c = self.elementsCountRim + 2
        e2z = 2*self.elementsCountUp-1-self.elementsCountRim
        e2y = e2z - 1
        e2x = e2z - 2
        for e3 in range(self.elementsCountAlong):
            for e2 in range(self.elementsCountUpFull):
                for e1 in range(self.elementsCountAcross):
                    if rangeOfRequiredElements:
                        if e3 >= rangeOfRequiredElements[2][1] or e3 < rangeOfRequiredElements[2][0] or\
                                e2 >= rangeOfRequiredElements[0][1] or e2 < rangeOfRequiredElements[0][0]\
                                or e1 >= rangeOfRequiredElements[1][1] or e1 < rangeOfRequiredElements[1][0]:
                            continue
                    eft1 = eft
                    scalefactors = None
                    if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                        nids = [self.nodeId[e3][e2][e1], self.nodeId[e3][e2 + 1][e1],
                                self.nodeId[e3+1][e2][e1], self.nodeId[e3+1][e2 + 1][e1],
                                self.nodeId[e3][e2][e1 + 1], self.nodeId[e3][e2 + 1][e1 + 1],
                                self.nodeId[e3+1][e2][e1 + 1], self.nodeId[e3+1][e2 + 1][e1 + 1]]
                    elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                        nids = [self.nodeId[0][e2][e1], self.nodeId[0][e2][e1 + 1],
                                self.nodeId[0][e2 + 1][e1], self.nodeId[0][e2 + 1][e1 + 1],
                                self.nodeId[1][e2][e1], self.nodeId[1][e2][e1 + 1],
                                self.nodeId[1][e2 + 1][e1], self.nodeId[1][e2 + 1][e1 + 1]]
                    if (e2 < e2b) or (e2 > e2y):
                        if (e1 < e1b) or (e1 > e1y):
                            continue  # no element due to triple point closure
                        if (e2 < e2a) or (e2 > e2z):
                            if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                                if e2 < e2a:
                                    nids = [self.nodeId[e3][e2+1][e1], self.nodeId[e3][e2+1][e1+1],
                                            self.nodeId[e3+1][e2+1][e1], self.nodeId[e3+1][e2+1][e1+1],
                                            self.nodeId[e3][e2][e1], self.nodeId[e3][e2][e1+1],
                                            self.nodeId[e3+1][e2][e1],  self.nodeId[e3+1][e2][e1+1]]
                                elif e2 > e2z:
                                    nids = [self.nodeId[e3][e2][e1+1], self.nodeId[e3][e2][e1],
                                            self.nodeId[e3+1][e2][e1+1], self.nodeId[e3+1][e2][e1],
                                            self.nodeId[e3][e2+1][e1+1], self.nodeId[e3][e2+1][e1],
                                            self.nodeId[e3+1][e2+1][e1+1], self.nodeId[e3+1][e2+1][e1]]
                        elif (e2 == e2a) or (e2 == e2z):
                            # bottom and top row elements
                            if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                                if e2 == e2a:
                                    eft1 = tricubichermite.createEftNoCrossDerivatives()
                                    setEftScaleFactorIds(eft1, [1], [])
                                    scalefactors = [-1.0]
                                    remapEftNodeValueLabel(eft1, [1, 3, 5, 7],
                                                           Node.VALUE_LABEL_D_DS1, [(Node.VALUE_LABEL_D_DS3, [1])])
                                    remapEftNodeValueLabel(eft1, [1, 3, 5, 7],
                                                           Node.VALUE_LABEL_D_DS3, [(Node.VALUE_LABEL_D_DS1, [])])
                                    if (e1 == e1b) or (e1 == e1y):
                                        # map bottom triple point element
                                        if e1 == e1b:
                                            remapEftNodeValueLabel(eft1, [2, 4], Node.VALUE_LABEL_D_DS1,
                                                                   [(Node.VALUE_LABEL_D_DS1, []),
                                                                    (Node.VALUE_LABEL_D_DS3, [])])
                                        else:
                                            remapEftNodeValueLabel(eft1, [6, 8], Node.VALUE_LABEL_D_DS1,
                                                                   [(Node.VALUE_LABEL_D_DS1, []),
                                                                    (Node.VALUE_LABEL_D_DS3, [1])])
                                elif e2 == e2z:
                                    eft1 = tricubichermite.createEftNoCrossDerivatives()
                                    setEftScaleFactorIds(eft1, [1], [])
                                    scalefactors = [-1.0]
                                    remapEftNodeValueLabel(eft1, [2, 4, 6, 8], Node.VALUE_LABEL_D_DS3,
                                                           [(Node.VALUE_LABEL_D_DS1, [1])])
                                    remapEftNodeValueLabel(eft1, [2, 4, 6, 8], Node.VALUE_LABEL_D_DS1,
                                                           [(Node.VALUE_LABEL_D_DS3, [])])
                                    if (e1 == e1b) or (e1 == e1y):
                                        # map top triple point element
                                        if e1 == e1b:
                                            remapEftNodeValueLabel(eft1, [1, 3], Node.VALUE_LABEL_D_DS1,
                                                                   [(Node.VALUE_LABEL_D_DS1, []),
                                                                    (Node.VALUE_LABEL_D_DS3, [1])])
                                        else:
                                            remapEftNodeValueLabel(eft1, [5, 7], Node.VALUE_LABEL_D_DS1,
                                                                   [(Node.VALUE_LABEL_D_DS1, []),
                                                                    (Node.VALUE_LABEL_D_DS3, [])])
                            elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                                if (e1 == e1b) or (e1 == e1y):
                                    # map bottom triple point element
                                    eft1 = tricubichermite.createEftNoCrossDerivatives()
                                    if e1 == e1b:
                                        remapEftNodeValueLabel(eft1, [3, 7], Node.VALUE_LABEL_D_DS2,
                                                               [(Node.VALUE_LABEL_D_DS1, []),
                                                                (Node.VALUE_LABEL_D_DS2, [])])
                                    else:
                                        setEftScaleFactorIds(eft1, [1], [])
                                        scalefactors = [-1.0]
                                        remapEftNodeValueLabel(eft1, [4, 8], Node.VALUE_LABEL_D_DS2,
                                                               [(Node.VALUE_LABEL_D_DS1, [1]),
                                                                (Node.VALUE_LABEL_D_DS2, [])])

                    elif (e2 == e2b) or (e2 == e2y):
                        if (e1 <= e1a) or (e1 >= e1z):
                            if e1 < e1a:
                                e2r = e1
                                if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                                    if e2 == e2b:
                                        nids = [self.nodeId[e3][e2c][e1+1], self.nodeId[e3][e2r+1][e1b],
                                                self.nodeId[e3+1][e2c][e1+1], self.nodeId[e3+1][e2r+1][e1b],
                                                self.nodeId[e3][e2c][e1], self.nodeId[e3][e2r][e1b],
                                                self.nodeId[e3+1][e2c][e1], self.nodeId[e3+1][e2r][e1b]]
                                    if e2 == e2y:
                                        e2r = 2*self.elementsCountUp - e1-1
                                        nids = [self.nodeId[e3][e2r][e1b], self.nodeId[e3][e2y][e1+1],
                                                self.nodeId[e3+1][e2r][e1b], self.nodeId[e3+1][e2y][e1+1],
                                                self.nodeId[e3][e2r+1][e1b], self.nodeId[e3][e2y][e1],
                                                self.nodeId[e3+1][e2r+1][e1b], self.nodeId[e3+1][e2y][e1]]
                                elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                                    eft1 = tricubichermite.createEftNoCrossDerivatives()
                                    setEftScaleFactorIds(eft1, [1], [])
                                    scalefactors = [-1.0]
                                    nids[0] = self.nodeId[0][e2r][e1b]
                                    nids[1] = self.nodeId[0][e2r + 1][e1b]
                                    nids[4] = self.nodeId[1][e2r][e1b]
                                    nids[5] = self.nodeId[1][e2r + 1][e1b]
                                    remapEftNodeValueLabel(eft1, [1, 2, 5, 6], Node.VALUE_LABEL_D_DS2,
                                                           [(Node.VALUE_LABEL_D_DS1, [1])])
                                    remapEftNodeValueLabel(eft1, [1, 2, 5, 6], Node.VALUE_LABEL_D_DS1,
                                                           [(Node.VALUE_LABEL_D_DS2, [])])
                            elif e1 == e1a:
                                eft1 = tricubichermite.createEftNoCrossDerivatives()
                                setEftScaleFactorIds(eft1, [1], [])
                                scalefactors = [-1.0]
                                if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                                    setEftScaleFactorIds(eft1, [1], [])
                                    scalefactors = [-1.0]
                                    if e2 == e2b:
                                        nids[0] = self.nodeId[e3][e2a][e1b]
                                        nids[2] = self.nodeId[e3+1][e2a][e1b]
                                        tripleN = [5, 7]
                                        remapEftNodeValueLabel(eft1, tripleN, Node.VALUE_LABEL_D_DS3,
                                                               [(Node.VALUE_LABEL_D_DS1, []),
                                                                (Node.VALUE_LABEL_D_DS3, [])])
                                    elif e2 == e2y:
                                        nids[1] = self.nodeId[e3][e2z+1][e1b]
                                        nids[3] = self.nodeId[e3+1][e2z+1][e1b]
                                        tripleN = [6, 8]
                                        remapEftNodeValueLabel(eft1, tripleN, Node.VALUE_LABEL_D_DS3,
                                                               [(Node.VALUE_LABEL_D_DS1, [1]),
                                                                (Node.VALUE_LABEL_D_DS3, [])])
                                    remapEftNodeValueLabel(eft1, [1, 2, 3, 4], Node.VALUE_LABEL_D_DS1,
                                                           [(Node.VALUE_LABEL_D_DS1, [1])])
                                    remapEftNodeValueLabel(eft1, [1, 2, 3, 4], Node.VALUE_LABEL_D_DS3,
                                                           [(Node.VALUE_LABEL_D_DS3, [1])])
                                elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                                    setEftScaleFactorIds(eft1, [1], [])
                                    scalefactors = [-1.0]
                                    nids[0] = self.nodeId[0][e2a][e1b]
                                    nids[4] = self.nodeId[1][e2a][e1b]
                                    remapEftNodeValueLabel(eft1, [1, 5], Node.VALUE_LABEL_D_DS2,
                                                           [(Node.VALUE_LABEL_D_DS1, [1])])
                                    remapEftNodeValueLabel(eft1, [1, 5], Node.VALUE_LABEL_D_DS1,
                                                           [(Node.VALUE_LABEL_D_DS2, [])])
                                    remapEftNodeValueLabel(eft1, [2, 6], Node.VALUE_LABEL_D_DS1,
                                                           [(Node.VALUE_LABEL_D_DS1, []), (Node.VALUE_LABEL_D_DS2, [])])
                            elif e1 == e1z:
                                eft1 = tricubichermite.createEftNoCrossDerivatives()
                                if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                                    if e2 == e2b:
                                        setEftScaleFactorIds(eft1, [1], [])
                                        scalefactors = [-1.0]
                                        nids[4] = self.nodeId[e3][e2a][e1z]
                                        nids[6] = self.nodeId[e3+1][e2a][e1z]
                                        remapEftNodeValueLabel(eft1, [1, 3], Node.VALUE_LABEL_D_DS3,
                                                               [(Node.VALUE_LABEL_D_DS1, [1]),
                                                                (Node.VALUE_LABEL_D_DS3, [])])
                                    elif e2 == e2y:
                                        nids[5] = self.nodeId[e3][e2z+1][e1z]
                                        nids[7] = self.nodeId[e3+1][e2z+1][e1z]
                                        remapEftNodeValueLabel(eft1, [2, 4], Node.VALUE_LABEL_D_DS3,
                                                               [(Node.VALUE_LABEL_D_DS1, []),
                                                                (Node.VALUE_LABEL_D_DS3, [])])
                                elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                                    setEftScaleFactorIds(eft1, [1], [])
                                    scalefactors = [-1.0]
                                    nids[1] = self.nodeId[0][e2a][e1z]
                                    nids[5] = self.nodeId[1][e2a][e1z]
                                    remapEftNodeValueLabel(eft1, [1, 5], Node.VALUE_LABEL_D_DS1,
                                                           [(Node.VALUE_LABEL_D_DS1, []),
                                                            (Node.VALUE_LABEL_D_DS2, [1])])
                                    remapEftNodeValueLabel(eft1, [2, 6], Node.VALUE_LABEL_D_DS1,
                                                           [(Node.VALUE_LABEL_D_DS2, [1])])
                                    remapEftNodeValueLabel(eft1, [2, 6], Node.VALUE_LABEL_D_DS2,
                                                           [(Node.VALUE_LABEL_D_DS1, [])])
                            elif e1 > e1z:
                                e2r = self.elementsCountAcross - e1
                                if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                                    if e2 == e2b:
                                        nids = [self.nodeId[e3][e2r][e1z], self.nodeId[e3][e2c][e1],
                                                self.nodeId[e3+1][e2r][e1z], self.nodeId[e3+1][e2c][e1],
                                                self.nodeId[e3][e2r-1][e1z], self.nodeId[e3][e2c][e1+1],
                                                self.nodeId[e3+1][e2r-1][e1z], self.nodeId[e3+1][e2c][e1+1]]
                                    elif e2 == e2y:
                                        e2r = e2z+e1-e1z
                                        nids[1] = self.nodeId[e3][e2r][e1z]
                                        nids[3] = self.nodeId[e3+1][e2r][e1z]
                                        nids[5] = self.nodeId[e3][e2r+1][e1z]
                                        nids[7] = self.nodeId[e3+1][e2r+1][e1z]
                                elif self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_REGULAR:
                                    eft1 = tricubichermite.createEftNoCrossDerivatives()
                                    setEftScaleFactorIds(eft1, [1], [])
                                    scalefactors = [-1.0]
                                    nids[0] = self.nodeId[0][e2r][e1z]
                                    nids[1] = self.nodeId[0][e2r - 1][e1z]
                                    nids[4] = self.nodeId[1][e2r][e1z]
                                    nids[5] = self.nodeId[1][e2r - 1][e1z]
                                    remapEftNodeValueLabel(eft1, [1, 2, 5, 6], Node.VALUE_LABEL_D_DS1,
                                                           [(Node.VALUE_LABEL_D_DS2, [1])])
                                    remapEftNodeValueLabel(eft1, [1, 2, 5, 6], Node.VALUE_LABEL_D_DS2,
                                                           [(Node.VALUE_LABEL_D_DS1, [])])
                    else:
                        if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                            if e1 < e1a:
                                nids = [self.nodeId[e3][e2 + 1][e1 + 1], self.nodeId[e3][e2][e1 + 1],
                                        self.nodeId[e3+1][e2 + 1][e1 + 1], self.nodeId[e3+1][e2][e1 + 1],
                                        self.nodeId[e3][e2 + 1][e1], self.nodeId[e3][e2][e1],
                                        self.nodeId[e3+1][e2 + 1][e1], self.nodeId[e3+1][e2][e1]]
                            elif e1 == e1a:
                                # map left column elements
                                eft1 = tricubichermite.createEftNoCrossDerivatives()
                                setEftScaleFactorIds(eft1, [1], [])
                                scalefactors = [-1.0]
                                remapEftNodeValueLabel(eft1, [1, 2, 3, 4], Node.VALUE_LABEL_D_DS1,
                                                       [(Node.VALUE_LABEL_D_DS1, [1])])
                                remapEftNodeValueLabel(eft1, [1, 2, 3, 4], Node.VALUE_LABEL_D_DS3,
                                                       [(Node.VALUE_LABEL_D_DS3, [1])])

                    if eft1 is not eft:
                        elementtemplate1.defineField(coordinates, -1, eft1)
                        element = mesh.createElement(elementIdentifier, elementtemplate1)
                    else:
                        element = mesh.createElement(elementIdentifier, elementtemplate)
                    result2 = element.setNodesByIdentifier(eft1, nids)
                    if scalefactors:
                        result3 = element.setScaleFactors(eft1, scalefactors)
                    else:
                        result3 = 7
                    # print('create element shield', elementIdentifier, result2, result3, nids)
                    if self._type == ShieldRimDerivativeMode.SHIELD_RIM_DERIVATIVE_MODE_AROUND:
                        self.elementId[e3][e2][e1] = elementIdentifier
                    else:
                        self.elementId[e2][e1] = elementIdentifier
                    elementIdentifier += 1

                    for c, meshGroup in enumerate(meshGroups):
                        if e3 < elementEnd[c]:
                            meshGroup.addElement(element)
                            break


        return elementIdentifier


class ShieldMesh3D:
    """
        Generates a 3D shield mesh.
    """

    def __init__(self, elementsCountAcross, elementsCountRim, box_derivatives=None):
        """
        3D shield structure can be used for a sphere mesh. 3 hex mesh merges to one box in the corner located in the
         centre.
        The structure is a 3D version of the 2D shield structure. It has a 'quadruple point', a unique node on the
        surface where the elements merge. 'Triple curves' meet at the quadruple point and a fourth curve that connects
        to another quadruple point which is inside.
         The structure is like a elementsCountAcross[0] X elementsCountAcross[1] X elementsCountAcross[2] box where
        some nodes does not exist and stored as None. The quadruple point is stored at [n3z][0][n1z] (top, front and
         right most node) where n3z is top most and n1z is the right most indexes.
         Triple curves and surfaces connecting to the quadruple point divide the exterior surface and interior region
          into 3 regions (top, left and right). Triple curve 1 connects the quadruple point to the plane 2-3.
          Similarly, triple curves 2 and 3 connect the quadruple point to the planes 1-3 and 1-2, respectively.
          It is the same for the inside quadruple. Any point on region left has n2 = 0. Similarly on region
           right -> n1 = n1z and on top -> n3 = n3z.
        There is a gap between node indexes of a node on a triple curve and the next nodes on the surface.

        :param elementsCountAcross: number of elements as
         a list [elementsCountAcrossAxis1, elementsCountAcrossAxis2, elementsCountAcrossAxis3]
        :param elementsCountRim:
        :param box_derivatives: It is a list of [deriv1,deriv2,deriv3]. It is used to set the derivative directions.
        default is [1, 3, 2] which means it makes -d1, d3 and d2 in direction of axis1, axis2 and axis3. To make
        d1, d2 and d3 directions in direction of axis1, axis2 and axis3 use [-1, 2, 3].
        """
        assert elementsCountRim >= 0
        # assert elementsCountAcross >= (elementsCountRim + 4)
        # assert elementsCountUpFull >= (elementsCountRim + 2)
        self.elementsCountAcross = elementsCountAcross
        self.elementsCountRim = elementsCountRim
        # self.elementsCountUpRegular = elementsCountUp - 2 - elementsCountRim
        self.elementsCountAcrossNonRim = [ne - 2*elementsCountRim for ne in elementsCountAcross]
        # self.elementsCountAroundFull = 2*self.elementsCountUpRegular + elementsCountAcrossNonRim
        self._boxDerivatives = box_derivatives
        self._boxMapping = None
        self._element_needs_scale_factor = False
        self._xi_mapping = None
        self._xi_signs = None
        self._box_deriv_mapping = None
        self._box_deriv_signs = None

        self.px = [[] for _ in range(elementsCountAcross[2] + 1)]
        self.pd1 = [[] for _ in range(elementsCountAcross[2] + 1)]
        self.pd2 = [[] for _ in range(elementsCountAcross[2] + 1)]
        self.pd3 = [[] for _ in range(elementsCountAcross[2] + 1)]
        self.nodeId = [[] for _ in range(elementsCountAcross[2] + 1)]
        for n3 in range(elementsCountAcross[2] + 1):
            for n2 in range(elementsCountAcross[0] + 1):
                for p in [self.px[n3], self.pd1[n3], self.pd2[n3], self.pd3[n3], self.nodeId[n3]]:
                    p.append([None]*(elementsCountAcross[1] + 1))

        self.elementId = [[[None]*elementsCountAcross[1] for n2 in range(elementsCountAcross[0])]
                          for e3 in range(elementsCountAcross[2])]

    # node types
    CORNER_1 = 1
    CORNER_2 = 2
    CORNER_3 = 3

    QUADRUPLE_DOWN_LEFT = 4
    QUADRUPLE_RIGHT = 5
    QUADRUPLE_UP = 6
    QUADRUPLE0_DOWN_LEFT = 7
    QUADRUPLE0_RIGHT = 8
    QUADRUPLE0_UP = 9

    TRIPLE_12_LEFT = 10
    TRIPLE_12_RIGHT = 11
    TRIPLE_13_DOWN = 12
    TRIPLE_13_UP = 13
    TRIPLE_23_UP = 14
    TRIPLE_23_DOWN = 15
    TRIPLE0_12_LEFT = 16
    TRIPLE0_12_RIGHT = 17
    TRIPLE0_13_DOWN = 18
    TRIPLE0_13_Up = 19
    TRIPLE0_23_DOWN = 20
    TRIPLE0_23_UP = 21

    BOUNDARY_12_LEFT = 22
    BOUNDARY_12_RIGHT = 23
    BOUNDARY_13_DOWN = 24
    BOUNDARY_13_UP = 25
    BOUNDARY_23_UP = 26
    BOUNDARY_23_DOWN = 27

    TRIPLE_CURVE_1_DOWN = 28
    TRIPLE_CURVE_1_UP = 29
    TRIPLE_CURVE_2_DOWN = 30
    TRIPLE_CURVE_2_UP = 31
    TRIPLE_CURVE_3_LEFT = 32
    TRIPLE_CURVE_3_RIGHT = 33
    TRIPLE_CURVE0_1_UP = 34
    TRIPLE_CURVE0_1_DOWN = 35
    TRIPLE_CURVE0_2_DOWN = 36
    TRIPLE_CURVE0_2_UP = 37
    TRIPLE_CURVE0_3_LEFT = 38
    TRIPLE_CURVE0_3_RIGHT = 39

    SURFACE_REGULAR_DOWN_LEFT = 40
    SURFACE_REGULAR_DOWN_RIGHT = 41
    SURFACE_REGULAR_UP = 42
    REGULAR = 43

    # element types
    ELEMENT_REGULAR = 1
    ELEMENT_QUADRUPLE_DOWN_LEFT = 2
    ELEMENT_QUADRUPLE_DOWN_RIGHT = 3
    ELEMENT_QUADRUPLE_UP_LEFT = 4
    ELEMENT_QUADRUPLE_DOWN = 5
    ELEMENT_QUADRUPLE_UP = 6
    ELEMENT_QUADRUPLE_LEFT = 7
    ELEMENT_QUADRUPLE_RIGHT = 8
    ELEMENT_DOWN_RIGHT = 9
    ELEMENT_DOWN_LEFT = 10

    def generateNodes(self, fieldmodule, coordinates, startNodeIdentifier, rangeOfRequiredElements):
        """
        Create shield nodes from coordinates.
        :param fieldmodule: Zinc fieldmodule to create nodes in. Uses DOMAIN_TYPE_NODES.
        :param coordinates: Coordinate field to define.
        :param startNodeIdentifier: First node identifier to use.
        :param rangeOfRequiredElements: Only the elements and nodes for the given range is generated.
        :return: next nodeIdentifier.
         """
        nodeIdentifier = startNodeIdentifier
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)
        # In case we swap derivatives in the when we use remap function.
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D2_DS2DS3, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D2_DS1DS3, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D2_DS1DS2, 1)
        cache = fieldmodule.createFieldcache()

        for n2 in range(self.elementsCountAcross[0] + 1):
            for n3 in range(self.elementsCountAcross[2] + 1):
                for n1 in range(self.elementsCountAcross[1] + 1):
                    if n3 > rangeOfRequiredElements[2][1] or n3 < rangeOfRequiredElements[2][0]\
                            or n2 > rangeOfRequiredElements[0][1] or n2 < rangeOfRequiredElements[0][0]\
                            or n1 > rangeOfRequiredElements[1][1] or n1 < rangeOfRequiredElements[1][0]:
                        continue
                    if self.px[n3][n2][n1]:
                        node = nodes.createNode(nodeIdentifier, nodetemplate)
                        self.nodeId[n3][n2][n1] = nodeIdentifier
                        cache.setNode(node)
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, self.px[n3][n2][n1])
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, self.pd1[n3][n2][n1])
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, self.pd2[n3][n2][n1])
                        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, self.pd3[n3][n2][n1])
                        nodeIdentifier += 1

        return nodeIdentifier

    def set_derivatives(self, box_derivatives):
        """
        Set the derivatives as specified by box_derivatives.
        :param box_derivatives: List[d_axis1_negative, d_axis2, d_axis3]. Determines what derivatives should be
        for back, right and up directions in the box region. Their values are in [1,2,3] range which 1, 2, 3 means
         d1, d2 and d3 respectively.
        The negative sign reverses the direction. e.g. [1, -3,2] means for the right direction use -d3.
         Default is [1,3, 2].
        :return:
        """
        self._boxMapping = box_derivatives

        dct = {1: self.pd1, 2: self.pd2, 3: self.pd3}
        perm = {(1, 2): -3, (2, 3): -1, (3, 1): -2, (3, 2): 1, (1, 3): 2, (2, 1): 3}

        if box_derivatives:
            signs = [1 if c > 0 else -1 for c in box_derivatives]
            dervMapping = (abs(box_derivatives[0]), abs(box_derivatives[1]), abs(box_derivatives[2]))

            temp1 = copy.deepcopy(self.pd3)
            temp2 = copy.deepcopy(self.pd2)
            for n3 in range(self.elementsCountAcross[2] + 1):
                for n2 in range(self.elementsCountAcross[0] + 1):
                    for n1 in range(self.elementsCountAcross[1] + 1):
                        if self.px[n3][n2][n1]:
                            if self.is_interior_regular_nodes(n3, n2, n1):
                                dct[dervMapping[0]][n3][n2][n1] = [signs[0] * c for c in self.pd1[n3][n2][n1]]
                                dct[dervMapping[1]][n3][n2][n1] = [signs[1] * c for c in temp1[n3][n2][n1]]
                                dct[dervMapping[2]][n3][n2][n1] = [signs[2] * c for c in temp2[n3][n2][n1]]

    def is_interior_regular_nodes(self, n3, n2, n1):
        """
        Determine if a node indicated by [n3,n2,n1], is inside the sphere.
        :return: True, if the node is inside.
        """
        n3z = self.elementsCountAcross[2]
        n1z = self.elementsCountAcross[1]
        n2z = self.elementsCountAcross[0]
        n3y = n3z - 1
        n1y = n1z - 1

        return n3 <= n3y and n2 >= 1 and n1 <= n1y

    def set_derivatives_for_irregular_nodes(self, octant_number):
        """
        For irregular nodes such as corners that are common between octants, change local derivatives to be the same
        directions of the octant_PPP. This is because the remapping is done for octant_PPP only and for others we need
        to make them similar, so similar remapping can be performed. Therefore, always same vectors is used for
         remapping.
         e.g. [1, 0, 0] generates curve 1 in octant1. For one octant d1 is [1,0,0] and for another octant -d2.
        :param octant_number: see get_element_type
        :return: Derivatives for irregular nodes.
        """
        default = [1, 2, 3]

        corner1derivs = default
        corner2derivs = default
        corner3derivs = default
        boundary12leftderivs = default
        boundary12rightderivs = default
        triple12leftderivs = default
        triple12rightderivs = default

        if octant_number == 1:
            corner3derivs = [-2, 1, 3]
        elif octant_number == 2:
            corner3derivs = [-1, -2, 3]
        elif octant_number == 3:
            corner3derivs = [2, -1, 3]
        elif octant_number == 4:
            corner3derivs = [1, 2, 3]
        elif octant_number == 5:
            corner3derivs = [-1, -2, 3]
        elif octant_number == 6:
            corner3derivs = [-2, 1, 3]
        elif octant_number == 7:
            corner3derivs = [1, 2, 3]
        elif octant_number == 8:
            corner3derivs = [2, -1, 3]
        else:
            corner3derivs = [1, 2, 3]

        if octant_number in [5, 6, 7, 8]:
            corner1derivs = [-1, -2, 3]
            corner2derivs = [-1, -2, 3]
            boundary12leftderivs = [-1, -2, 3]
            boundary12rightderivs = [-1, -2, 3]
            triple12leftderivs = [-1, -2, 3]
            triple12rightderivs = [-1, -2, 3]

        return corner1derivs, corner2derivs, corner3derivs, boundary12leftderivs, boundary12rightderivs, \
               triple12leftderivs, triple12rightderivs

    def get_box_mapping_for_other_octants(self, octant_number, octant1_box_derivatives=None):
        """
        Find mapping for box nodes derivatives to make each octant similar to octant1.
        :param octant1_box_derivatives: default is [1, 3, 2] for octant 1 box derivatives.
        :return: boxMapping
        """
        if not octant1_box_derivatives:
            octant1_box_derivatives = [1, 3, 2]

        swap12 = [octant1_box_derivatives[1], octant1_box_derivatives[0], octant1_box_derivatives[2]]

        signs = self.get_octant_signs(octant_number)

        if octant_number in [2, 4, 5, 7]:
            boxMapping = [swap12[c] * signs[c] for c in range(3)]
        else:
            boxMapping = [octant1_box_derivatives[c] * signs[c] for c in range(3)]

        return boxMapping

    def get_octant_signs(self, octant_number):
        """
        Box mapping for each octant is different from octant1. for some of them 1 and 2 swaps. Also, the sign of them
        might change.
        :return:
        """
        if octant_number == 1:
            signs = [1, 1, 1]
        elif octant_number == 2:
            # y<0
            signs = [1, -1, 1]
        elif octant_number == 3:
            # x<0, y<0
            signs = [-1, -1, 1]
        elif octant_number == 4:
            # x<0
            signs = [-1, 1, 1]
        elif octant_number == 5:
            # z<0
            signs = [-1, -1, -1]
        elif octant_number == 6:
            # y<0, z<0
            signs = [1, -1, -1]
        elif octant_number == 7:
            # x<0, y<0, z<0
            signs = [1, 1, -1]
        elif octant_number == 8:
            # x<0, z<0
            signs = [-1, 1, -1]
        else:
            signs = [1, 1, 1]

        return signs

    def get_element_node_identifiers(self, octant_number, e3, e2, e1, e3zo, e1zo):
        """
        Find node identifiers for given element represented by its indexes (e3,e2,e1). It uses default order of nodes
         for [1,3,2] default element axes and finds the mapping between octant default local node numbers and
          the sphere.
        :param octant_number:
        :return: nids, node s for given element represented by its indexes (e3,e2,e1).
        """
        nids_default = [self.getNodeId(octant_number, e3, e2, e1, e3zo, e1zo),
                        self.getNodeId(octant_number, e3, e2+1, e1, e3zo, e1zo),
                        self.getNodeId(octant_number, e3+1, e2, e1, e3zo, e1zo),
                        self.getNodeId(octant_number, e3+1, e2+1, e1, e3zo, e1zo),
                        self.getNodeId(octant_number, e3, e2, e1+1, e3zo, e1zo),
                        self.getNodeId(octant_number, e3, e2+1, e1+1, e3zo, e1zo),
                        self.getNodeId(octant_number, e3+1, e2, e1+1, e3zo, e1zo),
                        self.getNodeId(octant_number, e3+1, e2+1, e1+1, e3zo, e1zo)]

        # change the order of nodes according to the box derivatives.
        boxMappingOctant1 = self.get_box_mapping_for_other_octants(1, octant1_box_derivatives=self._boxDerivatives)
        lnm_octant1 = self.local_node_mapping(boxMappingOctant1)
        nids = [1] * 8
        for ln in range(1, 9):
            nids[lnm_octant1[ln] - 1] = nids_default[ln - 1]

        return nids

    def generateElements(self, fieldmodule, coordinates, startElementIdentifier, rangeOfRequiredElements,
                         meshGroups=[]):
        """
        Create shield elements from nodes.
        :param fieldmodule: Zinc fieldmodule to create elements in.
        :param coordinates: Coordinate field to define.
        :param startElementIdentifier: First element identifier to use.
        :param rangeOfRequiredElements: Only the elements and nodes for the given range is generated.
        :param meshGroups: Zinc mesh groups to add elements to.
        :return: next elementIdentifier.
         """
        elementIdentifier = startElementIdentifier
        useCrossDerivatives = False
        mesh = fieldmodule.findMeshByDimension(3)

        tricubichermite = eftfactory_tricubichermite(mesh, useCrossDerivatives)
        eft = tricubichermite.createEftNoCrossDerivatives()
        elementtemplate = mesh.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        elementtemplate.defineField(coordinates, -1, eft)

        elementtemplate1 = mesh.createElementtemplate()
        elementtemplate1.setElementShapeType(Element.SHAPE_TYPE_CUBE)

        for e3 in range(self.elementsCountAcross[2]):
            for e2 in range(self.elementsCountAcross[0]):
                for e1 in range(self.elementsCountAcross[1]):
                    if e3 >= rangeOfRequiredElements[2][1] or e3 < rangeOfRequiredElements[2][0] or\
                            e2 >= rangeOfRequiredElements[0][1] or e2 < rangeOfRequiredElements[0][0]\
                            or e1 >= rangeOfRequiredElements[1][1] or e1 < rangeOfRequiredElements[1][0]:
                        continue

                    scalefactors = None
                    self._element_needs_scale_factor = False

                    def shell_element(e3o, e2o, e1o, e3zo, e1zo):
                        return e3o > e3zo or e2o < self.elementsCountRim or e1o > e1zo

                    octant_number, element_type, e3zo, e2zo, e1zo = self.get_element_type(e3, e2, e1)
                    e3o, e2o, e1o = self.get_local_element_index(octant_number, e3, e2, e1)
                    element_shell = shell_element(e3o, e2o, e1o, e3zo, e1zo)
                    e3yo, e2bo, e1yo = e3zo - 1, 1 + self.elementsCountRim, e1zo - 1

                    eft1 = eft if element_type == self.ELEMENT_REGULAR else\
                        tricubichermite.createEftNoCrossDerivatives()

                    nids = self.get_element_node_identifiers(octant_number, e3, e2, e1, e3zo, e1zo)

                    boxMapping = self.get_box_mapping_for_other_octants(octant_number,
                                                                        octant1_box_derivatives=self._boxDerivatives)
                    lnm = self.local_node_mapping(boxMapping)
                    corner1derivs, corner2derivs, corner3derivs, boundary12leftderivs, boundary12rightderivs,\
                        triple12leftderivs, triple12rightderivs =\
                        self.set_derivatives_for_irregular_nodes(octant_number)

                    if element_type == self.ELEMENT_REGULAR:
                        pass
                    elif element_type == self.ELEMENT_QUADRUPLE_DOWN_LEFT:
                        self.remap_eft_node_value_label(eft1, [lnm[7]], self.QUADRUPLE_DOWN_LEFT)
                        if element_shell:
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.QUADRUPLE_DOWN_LEFT)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.QUADRUPLE0_DOWN_LEFT)
                        if e3o == 0:
                            self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_12_LEFT,
                                                            derivatives=triple12leftderivs)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_12_LEFT,
                                                                derivatives=triple12leftderivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE0_12_LEFT)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_CURVE_3_LEFT)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_CURVE_3_LEFT)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_CURVE0_3_LEFT)
                        if e1yo == 0:
                            self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_13_DOWN)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE_13_DOWN)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE0_13_DOWN)

                            if e3yo == 0:
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_1,
                                                                    derivatives=corner1derivs)
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.CORNER_1,
                                                                derivatives=corner1derivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_13_DOWN)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_13_DOWN)
                        else:
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE_CURVE_2_DOWN)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE_CURVE0_2_DOWN)
                            self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE_2_DOWN)
                            if e3o == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_12_LEFT,
                                                                derivatives=boundary12leftderivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_12_LEFT,
                                                                    derivatives=boundary12leftderivs)
                            else:
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.SURFACE_REGULAR_DOWN_LEFT)
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.SURFACE_REGULAR_DOWN_LEFT)

                    elif element_type == self.ELEMENT_QUADRUPLE_DOWN:
                        if not element_shell:
                            self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE0_13_DOWN)
                        self.remap_eft_node_value_label(eft1, [lnm[7]], self.TRIPLE_CURVE_2_DOWN)
                        if element_shell:
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_CURVE_2_DOWN)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_CURVE0_2_DOWN)
                        if e1o == 0:
                            self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_13_DOWN)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE_13_DOWN)
                            if e3yo == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.CORNER_1,
                                                                derivatives=corner1derivs)
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.BOUNDARY_12_LEFT,
                                                                derivatives=boundary12leftderivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_1,
                                                                    derivatives=corner1derivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_12_LEFT,
                                                                    derivatives=boundary12leftderivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_13_DOWN)
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.SURFACE_REGULAR_DOWN_LEFT)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_13_DOWN)
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.SURFACE_REGULAR_DOWN_LEFT)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE_2_DOWN)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE_CURVE_2_DOWN)
                            if e3yo == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_12_LEFT,
                                                                derivatives=boundary12leftderivs)
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.BOUNDARY_12_LEFT,
                                                                derivatives=boundary12leftderivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_12_LEFT,
                                                                    derivatives=boundary12leftderivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_12_LEFT,
                                                                    derivatives=boundary12leftderivs)
                            else:
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2], lnm[6]],
                                                                    self.SURFACE_REGULAR_DOWN_LEFT)
                                self.remap_eft_node_value_label(eft1, [lnm[1], lnm[5]], self.SURFACE_REGULAR_DOWN_LEFT)

                    elif element_type == self.ELEMENT_QUADRUPLE_DOWN_RIGHT:
                        if e2o == e2bo:
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.QUADRUPLE_RIGHT)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.QUADRUPLE0_RIGHT)
                            self.remap_eft_node_value_label(eft1, [lnm[7]], self.QUADRUPLE_RIGHT)
                            if e3o == 0:
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_12_RIGHT,
                                                                    derivatives=triple12rightderivs)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE0_12_RIGHT)
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_12_RIGHT,
                                                                derivatives=triple12rightderivs)
                            else:
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE_3_RIGHT)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE0_3_RIGHT)
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_CURVE_3_RIGHT)

                            if e2bo == e2zo:
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE_23_DOWN)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE0_23_DOWN)
                                self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_23_DOWN)
                                if e3yo == 0:
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.CORNER_2,
                                                                    derivatives=corner2derivs)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_2,
                                                                        derivatives=corner2derivs)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_23_DOWN)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_23_DOWN)
                            else:
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE_CURVE_1_DOWN)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE_CURVE0_1_DOWN)
                                self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_CURVE_1_DOWN)
                                if e3yo == 0:
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_12_RIGHT,
                                                                    derivatives=boundary12rightderivs)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_12_RIGHT,
                                                                        derivatives=boundary12rightderivs)
                                else:
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.SURFACE_REGULAR_DOWN_RIGHT)

                        elif e2o == e2zo:
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE_1_DOWN)
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE_23_DOWN)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE0_1_DOWN)
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.TRIPLE0_23_DOWN)
                            self.remap_eft_node_value_label(eft1, [lnm[7]], self.TRIPLE_CURVE_1_DOWN)
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_23_DOWN)
                            if e3yo == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.BOUNDARY_12_RIGHT,
                                                                derivatives=boundary12rightderivs)
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.CORNER_2,
                                                                derivatives=corner2derivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_12_RIGHT,
                                                                    derivatives=boundary12rightderivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_2,
                                                                    derivatives=corner2derivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_23_DOWN)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_23_DOWN)
                        else:
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[3], lnm[4]], self.TRIPLE_CURVE_1_DOWN)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[3], lnm[4]], self.TRIPLE_CURVE0_1_DOWN)
                            if e3yo == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[5], lnm[6]], self.BOUNDARY_12_RIGHT,
                                                                derivatives=boundary12rightderivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1], lnm[2]], self.BOUNDARY_12_RIGHT,
                                                                    derivatives=boundary12rightderivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[5], lnm[6]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1], lnm[2]],
                                                                    self.SURFACE_REGULAR_DOWN_RIGHT)
                            self.remap_eft_node_value_label(eft1, [lnm[7], lnm[8]], self.TRIPLE_CURVE_1_DOWN)

                    elif element_type == self.ELEMENT_QUADRUPLE_UP_LEFT:
                        if e2o == e2bo:
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.QUADRUPLE_UP)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE0_2_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.QUADRUPLE0_UP)
                            self.remap_eft_node_value_label(eft1, [lnm[7]], self.QUADRUPLE_UP)
                            if e2bo == e2zo:
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_23_UP)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE0_23_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_23_UP)
                                if e1yo == 0:
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.CORNER_3,
                                                                    derivatives=corner3derivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_13_UP)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_3,
                                                                        derivatives=corner3derivs)
                                        self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_13_UP)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_23_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE_2_UP)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_23_UP)
                                        self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE_2_UP)
                            else:
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_CURVE_1_UP)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_CURVE0_1_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_CURVE_1_UP)
                                if e1yo == 0:
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_13_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_13_UP)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_13_UP)
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_13_UP)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE_2_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.SURFACE_REGULAR_UP)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE_2_UP)
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.SURFACE_REGULAR_UP)
                        elif e2o == e2zo:
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_23_UP)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_CURVE_1_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_23_UP)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_CURVE0_1_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE0_23_UP)
                            self.remap_eft_node_value_label(eft1, [lnm[7]], self.TRIPLE_CURVE_1_UP)
                            if e1yo == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.CORNER_3,
                                                                derivatives=corner3derivs)
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.BOUNDARY_13_UP)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_3,
                                                                    derivatives=corner3derivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_13_UP)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_23_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.SURFACE_REGULAR_UP)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_23_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.SURFACE_REGULAR_UP)
                        else:
                            if e1yo == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[3], lnm[4]], self.BOUNDARY_13_UP)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1], lnm[2]], self.BOUNDARY_13_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[5], lnm[6]], self.TRIPLE_CURVE_1_UP)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[5], lnm[6]], self.TRIPLE_CURVE0_1_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[7], lnm[8]], self.TRIPLE_CURVE_1_UP)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[3], lnm[4]], self.SURFACE_REGULAR_UP)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1], lnm[2]], self.SURFACE_REGULAR_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[5], lnm[6]], self.TRIPLE_CURVE_1_UP)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[5], lnm[6]], self.TRIPLE_CURVE0_1_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[7], lnm[8]], self.TRIPLE_CURVE_1_UP)

                    elif element_type == self.ELEMENT_QUADRUPLE_UP:
                        if e2o == e2bo:
                            if e1o == 0:
                                if not element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE0_13_Up)
                            else:
                                if not element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE0_2_UP)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_CURVE_2_UP)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_CURVE0_2_UP)
                            self.remap_eft_node_value_label(eft1, [lnm[7]], self.TRIPLE_CURVE_2_UP)
                            if e2bo == e2zo:
                                self.remap_eft_node_value_label(eft1, [lnm[8]], self.BOUNDARY_23_UP)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_23_UP)
                                if e1o == 0:
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_13_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.CORNER_3,
                                                                    derivatives=corner3derivs)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_13_UP)
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_3,
                                                                        derivatives=corner3derivs)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE_2_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_23_UP)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE_2_UP)
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_23_UP)

                            else:
                                if e1o == 0:
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_13_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_13_UP)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_13_UP)
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_13_UP)
                                else:
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE_2_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.SURFACE_REGULAR_UP)
                                    if element_shell:
                                        self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE_2_UP)
                                        self.remap_eft_node_value_label(eft1, [lnm[2]], self.SURFACE_REGULAR_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[8]], self.SURFACE_REGULAR_UP)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[6]], self.SURFACE_REGULAR_UP)
                        elif e2o == e2zo:
                            if e1o == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.BOUNDARY_13_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.CORNER_3,
                                                                derivatives=corner3derivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_13_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_3,
                                                                    derivatives=corner3derivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.SURFACE_REGULAR_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_23_UP)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.SURFACE_REGULAR_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_23_UP)
                            self.remap_eft_node_value_label(eft1, [lnm[7]], self.SURFACE_REGULAR_UP)
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.BOUNDARY_23_UP)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.SURFACE_REGULAR_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_23_UP)
                        else:
                            if e1o == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[3], lnm[4]], self.BOUNDARY_13_UP)
                                self.remap_eft_node_value_label(eft1, [lnm[7], lnm[8]], self.SURFACE_REGULAR_UP)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1], lnm[2]], self.BOUNDARY_13_UP)
                                    self.remap_eft_node_value_label(eft1, [lnm[5], lnm[6]], self.SURFACE_REGULAR_UP)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[3], lnm[4], lnm[7], lnm[8]],
                                                                self.SURFACE_REGULAR_UP)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[1], lnm[2], lnm[5], lnm[6]],
                                                                    self.SURFACE_REGULAR_UP)

                    elif element_type == self.ELEMENT_QUADRUPLE_LEFT:
                        self.remap_eft_node_value_label(eft1, [lnm[7]], self.TRIPLE_CURVE_3_LEFT)
                        if element_shell:
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_CURVE_3_LEFT)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.TRIPLE_CURVE0_3_LEFT)
                        if e3o == 0:
                            self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_12_LEFT,
                                                            derivatives=triple12leftderivs)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_12_LEFT,
                                                                derivatives=triple12leftderivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE0_12_LEFT)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_CURVE_3_LEFT)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_CURVE_3_LEFT)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.TRIPLE_CURVE0_3_LEFT)
                        if e1yo == 0:
                            self.remap_eft_node_value_label(eft1, [lnm[3]], self.BOUNDARY_13_DOWN)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_13_DOWN)
                            if e3o == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.CORNER_1,
                                                                derivatives=corner1derivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_1,
                                                                    derivatives=corner1derivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_13_DOWN)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_13_DOWN)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[3]], self.SURFACE_REGULAR_DOWN_LEFT)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.SURFACE_REGULAR_DOWN_LEFT)
                            if e3o == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_12_LEFT,
                                                                derivatives=boundary12leftderivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_12_LEFT,
                                                                    derivatives=boundary12leftderivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.SURFACE_REGULAR_DOWN_LEFT)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.SURFACE_REGULAR_DOWN_LEFT)

                    elif element_type == self.ELEMENT_QUADRUPLE_RIGHT:
                        if element_shell:
                            self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE_3_RIGHT)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[3]], self.TRIPLE_CURVE0_3_RIGHT)
                        self.remap_eft_node_value_label(eft1, [lnm[7]], self.TRIPLE_CURVE_3_RIGHT)
                        if e3o == 0:
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_12_RIGHT,
                                                                derivatives=triple12rightderivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE0_12_RIGHT)
                            self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_12_RIGHT,
                                                            derivatives=triple12rightderivs)
                        else:
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE_3_RIGHT)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.TRIPLE_CURVE0_3_RIGHT)
                            self.remap_eft_node_value_label(eft1, [lnm[5]], self.TRIPLE_CURVE_3_RIGHT)
                        if e2bo == e2zo:
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.BOUNDARY_23_DOWN)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_23_DOWN)
                            if e3o == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.CORNER_2,
                                                                derivatives=corner2derivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_2,
                                                                    derivatives=corner2derivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_23_DOWN)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_23_DOWN)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[8]], self.SURFACE_REGULAR_DOWN_RIGHT)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[4]], self.SURFACE_REGULAR_DOWN_RIGHT)
                            if e3o == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_12_RIGHT,
                                                                derivatives=boundary12rightderivs)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_12_RIGHT,
                                                                    derivatives=boundary12rightderivs)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.SURFACE_REGULAR_DOWN_RIGHT)

                    elif element_type == self.ELEMENT_DOWN_RIGHT:
                        if e3o == 0:
                            if e2o == e2zo:
                                self.remap_eft_node_value_label(eft1, [lnm[7]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.BOUNDARY_12_RIGHT,
                                                                derivatives=boundary12rightderivs)
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.CORNER_2,
                                                                derivatives=corner2derivs)
                                self.remap_eft_node_value_label(eft1, [lnm[8]], self.BOUNDARY_23_DOWN)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_12_RIGHT,
                                                                    derivatives=boundary12rightderivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_2,
                                                                    derivatives=corner2derivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_23_DOWN)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[7]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                self.remap_eft_node_value_label(eft1, [lnm[5]], self.BOUNDARY_12_RIGHT,
                                                                derivatives=boundary12rightderivs)
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_12_RIGHT,
                                                                derivatives=boundary12rightderivs)
                                self.remap_eft_node_value_label(eft1, [lnm[8]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[3]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                    self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_12_RIGHT,
                                                                    derivatives=boundary12rightderivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_12_RIGHT,
                                                                    derivatives=boundary12rightderivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.SURFACE_REGULAR_DOWN_RIGHT)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[7]], self.SURFACE_REGULAR_DOWN_RIGHT)
                            self.remap_eft_node_value_label(eft1, [lnm[5]], self.SURFACE_REGULAR_DOWN_RIGHT)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.SURFACE_REGULAR_DOWN_RIGHT)
                            if e2o == e2zo:
                                self.remap_eft_node_value_label(eft1, [lnm[6], lnm[8]], self.BOUNDARY_23_DOWN)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2], lnm[4]], self.BOUNDARY_23_DOWN)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[6], lnm[8]], self.SURFACE_REGULAR_DOWN_RIGHT)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2], lnm[4]],
                                                                    self.SURFACE_REGULAR_DOWN_RIGHT)

                    elif element_type == self.ELEMENT_DOWN_LEFT:
                        if e3o == 0:
                            self.remap_eft_node_value_label(eft1, [lnm[5]], self.BOUNDARY_12_LEFT,
                                                            derivatives=boundary12rightderivs)
                            self.remap_eft_node_value_label(eft1, [lnm[7]], self.SURFACE_REGULAR_DOWN_LEFT)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[6]], self.BOUNDARY_12_LEFT,
                                                                derivatives=boundary12rightderivs)
                                self.remap_eft_node_value_label(eft1, [lnm[8]], self.SURFACE_REGULAR_DOWN_LEFT)
                            if e1o == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.CORNER_1,
                                                                derivatives=corner1derivs)
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.BOUNDARY_13_DOWN)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.CORNER_1,
                                                                    derivatives=corner1derivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.BOUNDARY_13_DOWN)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[1]], self.BOUNDARY_12_LEFT,
                                                                derivatives=boundary12rightderivs)
                                self.remap_eft_node_value_label(eft1, [lnm[3]], self.SURFACE_REGULAR_DOWN_LEFT)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2]], self.BOUNDARY_12_LEFT,
                                                                    derivatives=boundary12rightderivs)
                                    self.remap_eft_node_value_label(eft1, [lnm[4]], self.SURFACE_REGULAR_DOWN_LEFT)
                        else:
                            self.remap_eft_node_value_label(eft1, [lnm[5], lnm[7]], self.SURFACE_REGULAR_DOWN_LEFT)
                            if element_shell:
                                self.remap_eft_node_value_label(eft1, [lnm[6], lnm[8]], self.SURFACE_REGULAR_DOWN_LEFT)
                            if e1o == 0:
                                self.remap_eft_node_value_label(eft1, [lnm[1], lnm[3]], self.BOUNDARY_13_DOWN)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2], lnm[4]], self.BOUNDARY_13_DOWN)
                            else:
                                self.remap_eft_node_value_label(eft1, [lnm[1], lnm[3]], self.SURFACE_REGULAR_DOWN_LEFT)
                                if element_shell:
                                    self.remap_eft_node_value_label(eft1, [lnm[2], lnm[4]],
                                                                    self.SURFACE_REGULAR_DOWN_LEFT)

                    else:
                        continue

                    if self._element_needs_scale_factor:
                        scalefactors = [-1.0]

                    if eft1 is not eft:
                        elementtemplate1.defineField(coordinates, -1, eft1)
                        element = mesh.createElement(elementIdentifier, elementtemplate1)
                    else:
                        element = mesh.createElement(elementIdentifier, elementtemplate)
                    result2 = element.setNodesByIdentifier(eft1, nids)
                    if scalefactors:
                        result3 = element.setScaleFactors(eft1, scalefactors)
                    else:
                        result3 = 7

                    self.elementId[e3][e2][e1] = elementIdentifier

                    elementIdentifier += 1

                    if element_type == self.ELEMENT_REGULAR:
                        for meshGroup in meshGroups[:1]:
                            meshGroup.addElement(element)
                    else:
                        for meshGroup in meshGroups[1:2]:
                            meshGroup.addElement(element)

        return elementIdentifier

    def local_node_mapping(self, boxMapping):
        """
        Find local node mapping between default local node number of octant and global local node number of sphere.
         Obtain xi mapping and box irregular derivatives mapping. they are used for elements in octant that locally
         might have different xi directions from their global sphere xi directions. Then they are used in
          remap_eft_node_value_label.
        :return: lnm, local node mapping between local node number of octant and local node number of sphere.
        """
        boxMapping_default = [1, 3, 2]

        xi_node_map = {(0, 0, 0): 1, (1, 0, 0): 2, (0, 1, 0): 3, (1, 1, 0): 4,
                       (0, 0, 1): 5, (1, 0, 1): 6, (0, 1, 1): 7, (1, 1, 1): 8}
        node_xi_map = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (1, 1, 0),
                       5: (0, 0, 1), 6: (1, 0, 1), 7: (0, 1, 1), 8: (1, 1, 1)}

        xi_dm = {1: 'xi1', 2: 'xi2', 3: 'xi3'}
        xi_default_to_global_map = {xi_dm[abs(boxMapping_default[0])]: abs(boxMapping[0]),
                                    xi_dm[abs(boxMapping_default[1])]: abs(boxMapping[1]),
                                    xi_dm[abs(boxMapping_default[2])]: abs(boxMapping[2])}
        xi_global_to_default_map = {xi_dm[abs(boxMapping[0])]: abs(boxMapping_default[0]),
                                    xi_dm[abs(boxMapping[1])]: abs(boxMapping_default[1]),
                                    xi_dm[abs(boxMapping[2])]: abs(boxMapping_default[2])}
        signs = [1 if boxMapping[c]*boxMapping_default[c] > 0 else -1 for c in range(3)]
        xi_sign_change = [signs[abs(boxMapping_default[0])-1], signs[abs(boxMapping_default[1])-1],
                          signs[abs(boxMapping_default[2])-1]]

        lnm = {}
        signs = [xi_sign_change[xi_global_to_default_map['xi1'] - 1],
                 xi_sign_change[xi_global_to_default_map['xi2'] - 1],
                 xi_sign_change[xi_global_to_default_map['xi3'] - 1]]
        for ln in range(1, 9):
            xi_l = []
            xi_default = node_xi_map[ln]
            xi = [xi_default[xi_global_to_default_map['xi1'] - 1],
                  xi_default[xi_global_to_default_map['xi2'] - 1],
                  xi_default[xi_global_to_default_map['xi3'] - 1]]

            for i in range(3):
                if signs[i] > 0:
                    xi_l.append(xi[i])
                else:
                    xi_l.append(1 - xi[i])
            lnm[ln] = xi_node_map[tuple(xi_l)]

        deriv_default_to_global_map = {xi_dm[abs(boxMapping_default[0])]: abs(boxMapping[0]),
                                       xi_dm[abs(boxMapping_default[1])]: abs(boxMapping[1]),
                                       xi_dm[abs(boxMapping_default[2])]: abs(boxMapping[2])}
        signs = [1 if boxMapping[c]*boxMapping_default[c] > 0 else -1 for c in range(3)]
        deriv_sign_change = [signs[abs(boxMapping_default[0])-1], signs[abs(boxMapping_default[1])-1],
                             signs[abs(boxMapping_default[2])-1]]

        self._xi_mapping = {1: xi_default_to_global_map['xi1'], 2: xi_default_to_global_map['xi2'],
                            3: xi_default_to_global_map['xi3']}
        self._xi_signs = {1: xi_sign_change[0], 2: xi_sign_change[1], 3: xi_sign_change[2]}
        self._box_deriv_mapping = {1: deriv_default_to_global_map['xi1'],
                                   2: deriv_default_to_global_map['xi2'],
                                   3: deriv_default_to_global_map['xi3']}
        self._box_deriv_signs = {1: deriv_sign_change[0], 2: deriv_sign_change[1], 3: deriv_sign_change[2]}

        return lnm

    def get_element_type(self, e3, e2, e1):
        """
        Find element type.
        :return:octant number, element type and extends for the octant.
        """
        if e3 >= self.elementsCountAcross[2] // 2:
            if e2 < self.elementsCountAcross[0] // 2:
                if e1 >= self.elementsCountAcross[1] // 2:
                    octant_number = 1
                else:
                    octant_number = 2
            else:
                if e1 < self.elementsCountAcross[1] // 2:
                    octant_number = 3
                else:
                    octant_number = 4
        else:
            if e2 < self.elementsCountAcross[0] // 2:
                if e1 >= self.elementsCountAcross[1] // 2:
                    octant_number = 5
                else:
                    octant_number = 6
            else:
                if e1 < self.elementsCountAcross[1] // 2:
                    octant_number = 7
                else:
                    octant_number = 8

        if octant_number in [2, 4, 5, 7]:
            e3zo = self.elementsCountAcross[2] // 2 - 1 - self.elementsCountRim
            e2zo = self.elementsCountAcross[1] // 2 - 1
            e1zo = self.elementsCountAcross[0] // 2 - 1 - self.elementsCountRim
        else:
            e3zo = self.elementsCountAcross[2] // 2 - 1 - self.elementsCountRim
            e2zo = self.elementsCountAcross[0] // 2 - 1
            e1zo = self.elementsCountAcross[1] // 2 - 1 - self.elementsCountRim

        e3yo, e2bo, e1yo = e3zo - 1, 1 + self.elementsCountRim, e1zo - 1
        e3o, e2o, e1o = self.get_local_element_index(octant_number, e3, e2, e1)

        if e3o <= e3yo and e2o >= e2bo and e1o <= e1yo:
            element_type = self.ELEMENT_REGULAR
        elif e3o == e3yo and e2o <= self.elementsCountRim and e1o == e1yo:
            element_type = self.ELEMENT_QUADRUPLE_DOWN_LEFT
        elif e3o == e3yo and e2o >= e2bo and e1o >= e1zo:
            element_type = self.ELEMENT_QUADRUPLE_DOWN_RIGHT
        elif e3o >= e3zo and e2o >= e2bo and e1o == e1yo:
            element_type = self.ELEMENT_QUADRUPLE_UP_LEFT
        elif e3o == e3yo and e2o <= self.elementsCountRim and e1o < e1yo:
            element_type = self.ELEMENT_QUADRUPLE_DOWN
        elif e3o >= e3zo and e2o >= e2bo and e1o < e1yo:
            element_type = self.ELEMENT_QUADRUPLE_UP
        elif e3o < e3yo and e2o <= self.elementsCountRim and e1o == e1yo:
            element_type = self.ELEMENT_QUADRUPLE_LEFT
        elif e3o < e3yo and e2o == e2bo and e1o >= e1zo:
            element_type = self.ELEMENT_QUADRUPLE_RIGHT
        elif e3o < e3yo and e2o > e2bo and e1o >= e1zo:
            element_type = self.ELEMENT_DOWN_RIGHT
        elif e3o < e3yo and e2o <= self.elementsCountRim and e1o < e1yo:
            element_type = self.ELEMENT_DOWN_LEFT
        else:
            element_type = 0

        return octant_number, element_type, e3zo, e2zo, e1zo

    def get_local_element_index(self, octant_number, e3, e2, e1):
        """
        Get octant element index from sphere element index.
        :return:e3o, e2o, e1o, local octant element indexes.
        """
        if octant_number == 1:
            e3o, e2o, e1o = e3 - self.elementsCountAcross[2] // 2, e2, e1 - self.elementsCountAcross[1] // 2
        elif octant_number == 2:
            e3o, e2o, e1o = e3 - self.elementsCountAcross[2] // 2, e1, self.elementsCountAcross[0] // 2 - 1 - e2
        elif octant_number == 3:
            e3o, e2o, e1o = e3 - self.elementsCountAcross[2] // 2, self.elementsCountAcross[0] - 1 - e2,\
                            self.elementsCountAcross[1] // 2 - 1 - e1
        elif octant_number == 4:
            e3o, e2o, e1o = e3 - self.elementsCountAcross[2] // 2, self.elementsCountAcross[1] - 1 - e1,\
                            e2 - self.elementsCountAcross[0] // 2
        elif octant_number == 5:
            e3o, e2o, e1o = self.elementsCountAcross[2]//2 - 1 - e3, self.elementsCountAcross[1] - 1 - e1,\
                            self.elementsCountAcross[0]//2 - 1 - e2
        elif octant_number == 6:
            e3o, e2o, e1o = self.elementsCountAcross[2]//2 - 1 - e3, e2, self.elementsCountAcross[1] // 2 - 1 - e1
        elif octant_number == 7:
            e3o, e2o, e1o = self.elementsCountAcross[2]//2 - 1 - e3, e1, e2 - self.elementsCountAcross[0]//2
        elif octant_number == 8:
            e3o, e2o, e1o = self.elementsCountAcross[2]//2 - 1 - e3, self.elementsCountAcross[0] - 1 - e2,\
                            e1 - self.elementsCountAcross[1]//2

        return e3o, e2o, e1o

    def get_global_node_index(self, octant_number, n3o, n2o, n1o):
        """
        Get octant element index from sphere element index.
        :return: n3, n2, n1, sphere element indexes.
        """
        if octant_number == 1:
            n3, n2, n1 = n3o + self.elementsCountAcross[2]//2, n2o, n1o + self.elementsCountAcross[1] // 2
        elif octant_number == 2:
            n3, n2, n1 = n3o + self.elementsCountAcross[2]//2, self.elementsCountAcross[0] // 2 - n1o, n2o
        elif octant_number == 3:
            n3, n2, n1 = n3o + self.elementsCountAcross[2]//2, self.elementsCountAcross[0] - n2o,\
                         self.elementsCountAcross[1] // 2 - n1o
        elif octant_number == 4:
            n3, n2, n1 = n3o + self.elementsCountAcross[2]//2, self.elementsCountAcross[0] // 2 + n1o,\
                         self.elementsCountAcross[1] - n2o
        elif octant_number == 5:
            n3, n2, n1 = self.elementsCountAcross[2] // 2 - n3o, self.elementsCountAcross[0]//2 - n1o,\
                         self.elementsCountAcross[1] - n2o
        elif octant_number == 6:
            n3, n2, n1 = self.elementsCountAcross[2] // 2 - n3o, n2o, self.elementsCountAcross[1]//2 - n1o
        elif octant_number == 7:
            n3, n2, n1 = self.elementsCountAcross[2] // 2 - n3o, n1o + self.elementsCountAcross[0]//2, n2o
        elif octant_number == 8:
            n3, n2, n1 = self.elementsCountAcross[2] // 2 - n3o, self.elementsCountAcross[0] - n2o,\
                         self.elementsCountAcross[1]//2 + n1o

        return n3, n2, n1

    def get_octant_node_index(self, octant_number, n3, n2, n1):
        """
        Get local octant node index from sphere node index.
        :return: n3o, n2o, n1o, local octant node indexes.
        """
        if octant_number == 1:
            n3o, n2o, n1o = n3 - self.elementsCountAcross[2] // 2, n2, n1 - self.elementsCountAcross[1] // 2
        elif octant_number == 2:
            n3o, n2o, n1o = n3 - self.elementsCountAcross[2] // 2, n1, self.elementsCountAcross[0] // 2 - n2
        elif octant_number == 3:
            n3o, n2o, n1o = n3 - self.elementsCountAcross[2] // 2, self.elementsCountAcross[0] - n2,\
                            self.elementsCountAcross[1] // 2 - n1
        elif octant_number == 4:
            n3o, n2o, n1o = n3 - self.elementsCountAcross[2] // 2, self.elementsCountAcross[1] - n1,\
                            n2 - self.elementsCountAcross[0]//2
        elif octant_number == 5:
            n3o, n2o, n1o = self.elementsCountAcross[2] // 2 - n3, self.elementsCountAcross[1] - n1,\
                            self.elementsCountAcross[0]//2 - n2
        elif octant_number == 6:
            n3o, n2o, n1o = self.elementsCountAcross[2] // 2 - n3, n2, self.elementsCountAcross[1]//2 - n1
        elif octant_number == 7:
            n3o, n2o, n1o = self.elementsCountAcross[2] // 2 - n3, n1, n2 - self.elementsCountAcross[0]//2
        elif octant_number == 8:
            n3o, n2o, n1o = self.elementsCountAcross[2] // 2 - n3, self.elementsCountAcross[0] - n2,\
                            n1 - self.elementsCountAcross[1]//2

        return n3o, n2o, n1o

    def getNodeId(self, octant_number, n3, n2, n1, n3yo, n1yo):
        """
        Get node identifier. Use node indexes n3, n2, n1 to find it. If the indexes represent removed nodes in the
         lattice, find the right indexes first.
        :return: Node identifier.
        """
        n1zo = n1yo + 1
        n3zo = n3yo + 1
        n3o, n2o, n1o = self.get_octant_node_index(octant_number, n3, n2, n1)

        n3_skip, n2_skip, n1_skip = 0, 0, 0

        if n1o == n1yo:
            if n2o <= self.elementsCountRim:
                n1_skip = 1 + self.elementsCountRim - n2o
            elif n3o >= n3zo:
                n1_skip = 1 + n3o - n3zo

        if n2o == self.elementsCountRim + 1:
            if n1o >= n1zo:
                n2_skip = -(1 + n1o - n1zo)
            elif n3o >= n3zo:
                n2_skip = -(1 + n3o - n3zo)

        if n3o == n3yo:
            if n2o <= self.elementsCountRim:
                n3_skip = 1 + self.elementsCountRim - n2o
            elif n1o >= n1zo:
                n3_skip = 1 + n1o - n1zo

        n3r, n2r, n1r = self.get_global_node_index(octant_number, n3o + n3_skip, n2o + n2_skip, n1o + n1_skip)

        return self.nodeId[n3r][n2r][n1r]

    def remap_eft_node_value_label(self, eft, localNodeIndexes, node_type, derivatives=None):
        """
        Remaps derivatives for common types of nodes. It also checks if each node needs scale factors. If one of the
         nodes of an element needs it, then flags it. It assumes the directions of derivatives in a certain way
          (based on default [1, 3, 2] xi directions).
          If directions are different, then we can use the 'derivatives' argument to fix this. This part is done for
          irregular nodes, in the box and for other nodes, derivatives should be given.
        :param node_type: e.g. corner 1 represents an end node on the axis1.
        :param derivatives: derivatives to be changed to.
        :return:
        """
        label = {1: Node.VALUE_LABEL_D2_DS2DS3, 2: Node.VALUE_LABEL_D2_DS1DS3, 3: Node.VALUE_LABEL_D2_DS1DS2}
        expressionLabel = {1: Node.VALUE_LABEL_D_DS1, 2: Node.VALUE_LABEL_D_DS2, 3: Node.VALUE_LABEL_D_DS3}
        sf = {1: [], -1: [1]}

        xim = self._xi_mapping
        xis = self._xi_signs
        if derivatives:
            dm = {1: abs(derivatives[0]), 2: abs(derivatives[1]), 3: abs(derivatives[2])}
            signs = [1 if c > 0 else -1 for c in derivatives]
            ds = {1: signs[0], 2: signs[1], 3: signs[2]}
        else:
            dm = self._box_deriv_mapping
            ds = self._box_deriv_signs

        # use cross derivatives for swapping derivatives.
        remapEftNodeValueLabel(eft, localNodeIndexes, Node.VALUE_LABEL_D_DS1, [(Node.VALUE_LABEL_D2_DS2DS3, [])])
        remapEftNodeValueLabel(eft, localNodeIndexes, Node.VALUE_LABEL_D_DS2, [(Node.VALUE_LABEL_D2_DS1DS3, [])])
        remapEftNodeValueLabel(eft, localNodeIndexes, Node.VALUE_LABEL_D_DS3, [(Node.VALUE_LABEL_D2_DS1DS2, [])])

        if node_type == self.CORNER_1:
            if any(c != 1 for c in [-1 * xis[1] * ds[3], 1 * xis[2] * ds[2], 1 * xis[3] * ds[1]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[3]],
                                                                           sf[-1 * xis[1] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[3] * ds[1]])])
        elif node_type == self.CORNER_2:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[2] * ds[2], 1 * xis[3] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
        elif node_type == self.CORNER_3:
            if any(c != 1 for c in [-1 * xis[3] * ds[1], -1 * xis[1] * ds[2], 1 * xis[2] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[2]],
                                                                           sf[-1 * xis[1] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[2] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[1]],
                                                                           sf[-1 * xis[3] * ds[1]])])

        elif node_type == self.QUADRUPLE_DOWN_LEFT:
            if any(c != 1 for c in [-1 * xis[1], 1 * xis[2], 1 * xis[2], -1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS3, sf[-1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]],
                                   [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[2]]), (Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS2, sf[-1 * xis[3]])])
        elif node_type == self.QUADRUPLE_RIGHT:
            if any(c != 1 for c in [1 * xis[1], 1 * xis[3], 1 * xis[2], 1 * xis[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]],
                                   [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[2]]), (Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
        elif node_type == self.QUADRUPLE_UP:
            if any(c != 1 for c in [1 * xis[1], -1 * xis[3], 1 * xis[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS2, sf[-1 * xis[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[2]])])
        elif node_type == self.QUADRUPLE0_DOWN_LEFT:
            if any(c != 1 for c in [1 * xis[3] * ds[3], 1 * xis[2] * ds[2], 1 * xis[1] * ds[1], -1 * xis[1] * ds[2],
                                    -1 * xis[1] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]],
                                   [(expressionLabel[dm[1]], sf[1 * xis[1] * ds[1]]),
                                    (expressionLabel[dm[2]], sf[-1 * xis[1] * ds[2]]),
                                    (expressionLabel[dm[3]], sf[-1 * xis[1] * ds[3]])])
        elif node_type == self.QUADRUPLE0_RIGHT:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[2] * ds[2], -1 * xis[3] * ds[1], 1 * xis[3] * ds[2],
                                    1 * xis[3] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]],
                                   [(expressionLabel[dm[1]], sf[-1 * xis[3] * ds[1]]),
                                    (expressionLabel[dm[2]], sf[1 * xis[3] * ds[2]]),
                                    (expressionLabel[dm[3]], sf[1 * xis[3] * ds[3]])])
        elif node_type == self.QUADRUPLE0_UP:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[3] * ds[3], -1 * xis[2] * ds[1], 1 * xis[2] * ds[2],
                                    1 * xis[2] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]],
                                   [(expressionLabel[dm[1]], sf[-1 * xis[2] * ds[1]]),
                                    (expressionLabel[dm[2]], sf[1 * xis[2] * ds[2]]),
                                    (expressionLabel[dm[3]], sf[1 * xis[2] * ds[3]])])

        elif node_type == self.TRIPLE_12_LEFT:
            if any(c != 1 for c in [1 * xis[2] * ds[2], -1 * xis[1] * ds[3], 1 * xis[3] * ds[1]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[3]],
                                                                           sf[-1 * xis[1] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[3] * ds[1]])])
        elif node_type == self.TRIPLE_12_RIGHT:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[2] * ds[2], 1 * xis[3] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
        elif node_type == self.TRIPLE_13_DOWN:
            if any(c != 1 for c in [-1 * xis[1], 1 * xis[2], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS3, sf[-1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[3]])])
        elif node_type == self.TRIPLE_13_UP:
            if any(c != 1 for c in [1 * xis[1], 1 * xis[2], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[1]])])
            # remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D2_DS2DS3, [])])
            # temporary to enable swap
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[3]])])
            # remapEftNodeValueLabel(eft, localNodeIndexes, Node.VALUE_LABEL_D2_DS2DS3, [(Node.VALUE_LABEL_D_DS2, [])])
            # finish swap
        elif node_type == self.TRIPLE_23_DOWN:
            if any(c != 1 for c in [1 * xis[1], 1 * xis[2], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[3]])])
        elif node_type == self.TRIPLE_23_UP:
            if any(c != 1 for c in [1 * xis[1], -1 * xis[3], 1 * xis[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS2, sf[-1 * xis[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[2]])])
        elif node_type == self.TRIPLE0_12_LEFT:
            if any(c != 1 for c in [1 * xis[3] * ds[3], 1 * xis[2] * ds[2], 1 * xis[1] * ds[1], -1 * xis[1] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]],
                                   [(expressionLabel[dm[1]], sf[1 * xis[1] * ds[1]]),
                                    (expressionLabel[dm[3]], sf[-1 * xis[1] * ds[3]])])
        elif node_type == self.TRIPLE0_12_RIGHT:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[2] * ds[2], -1 * xis[3] * ds[1], 1 * xis[3] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]],
                                   [(expressionLabel[dm[1]], sf[-1 * xis[3] * ds[1]]), (expressionLabel[dm[3]],
                                                                                        sf[1 * xis[3] * ds[3]])])
        elif node_type == self.TRIPLE0_13_DOWN:
            if any(c != 1 for c in [1 * xis[3] * ds[3], 1 * xis[2] * ds[2], 1 * xis[1] * ds[1], -1 * xis[1] * ds[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]],
                                   [(expressionLabel[dm[1]], sf[1 * xis[1] * ds[1]]), (expressionLabel[dm[2]],
                                                                                       sf[-1 * xis[1] * ds[2]])])
        elif node_type == self.TRIPLE0_13_Up:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[3] * ds[3], -1 * xis[2] * ds[1], 1 * xis[2] * ds[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]],
                                   [(expressionLabel[dm[1]], sf[-1 * xis[2] * ds[1]]),
                                    (expressionLabel[dm[2]], sf[1 * xis[2] * ds[2]])])
        elif node_type == self.TRIPLE0_23_DOWN:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[2] * ds[2], 1 * xis[3] * ds[2], 1 * xis[3] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]],
                                   [(expressionLabel[dm[2]], sf[1 * xis[3] * ds[2]]),
                                    (expressionLabel[dm[3]], sf[1 * xis[3] * ds[3]])])
        elif node_type == self.TRIPLE0_23_UP:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[3] * ds[3], 1 * xis[2] * ds[2], 1 * xis[2] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]],
                                   [(expressionLabel[dm[2]], sf[1 * xis[2] * ds[2]]),
                                    (expressionLabel[dm[3]], sf[1 * xis[2] * ds[3]])])

        elif node_type == self.BOUNDARY_12_LEFT:
            if any(c != 1 for c in [1 * xis[2] * ds[2], -1 * xis[1] * ds[3], 1 * xis[3] * ds[1]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[3]],
                                                                           sf[-1 * xis[1] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[3] * ds[1]])])
        elif node_type == self.BOUNDARY_12_RIGHT:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[2] * ds[2], 1 * xis[3] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
        elif node_type == self.BOUNDARY_13_DOWN:
            if any(c != 1 for c in [-1 * xis[1], 1 * xis[2], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS3, sf[-1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[3]])])
        elif node_type == self.BOUNDARY_13_UP:
            if any(c != 1 for c in [1 * xis[1], 1 * xis[2], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[1]])])
            # remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D2_DS2DS3, [])])
            # temporary to enable swap
            # remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS3, [])])
            # remapEftNodeValueLabel(eft, localNodeIndexes, Node.VALUE_LABEL_D2_DS2DS3, [(Node.VALUE_LABEL_D_DS2, [])])
            # finish swaps
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[3]])])
        elif node_type == self.BOUNDARY_23_DOWN:
            if any(c != 1 for c in [1 * xis[3], 1 * xis[1], 1 * xis[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
        elif node_type == self.BOUNDARY_23_UP:
            if any(c != 1 for c in [1 * xis[1], -1 * xis[3], 1 * xis[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS2, sf[-1 * xis[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[2]])])

        elif node_type == self.TRIPLE_CURVE_1_DOWN:
            if any(c != 1 for c in [1 * xis[1], 1 * xis[2], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[3]])])
        elif node_type == self.TRIPLE_CURVE_1_UP:
            if any(c != 1 for c in [1 * xis[1], -1 * xis[3], 1 * xis[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS2, sf[-1 * xis[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[2]])])
        elif node_type == self.TRIPLE_CURVE_2_DOWN:
            if any(c != 1 for c in [-1 * xis[1], 1 * xis[2], -1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS3, sf[-1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS2, sf[-1 * xis[3]])])
        elif node_type == self.TRIPLE_CURVE_2_UP:
            if any(c != 1 for c in [1 * xis[1], -1 * xis[3], 1 * xis[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS2, sf[-1 * xis[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[2]])])
        elif node_type == self.TRIPLE_CURVE_3_LEFT:
            if any(c != 1 for c in [1 * xis[2], -1 * xis[1], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS3, sf[-1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[3]])])
        elif node_type == self.TRIPLE_CURVE_3_RIGHT:
            if any(c != 1 for c in [1 * xis[1], 1 * xis[2], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[3]])])
        elif node_type == self.TRIPLE_CURVE0_1_UP:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[3] * ds[3], 1 * xis[2] * ds[2], 1 * xis[2] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]],
                                   [(expressionLabel[dm[2]], sf[1 * xis[2] * ds[2]]), (expressionLabel[dm[3]],
                                                                                       sf[1 * xis[2] * ds[3]])])
        elif node_type == self.TRIPLE_CURVE0_1_DOWN:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[2] * ds[2], 1 * xis[3] * ds[2], 1 * xis[3] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]],
                                   [(expressionLabel[dm[2]], sf[1 * xis[3] * ds[2]]), (expressionLabel[dm[3]],
                                                                                       sf[1 * xis[3] * ds[3]])])
        elif node_type == self.TRIPLE_CURVE0_2_DOWN:
            if any(c != 1 for c in [1 * xis[2] * ds[2], 1 * xis[3] * ds[3], 1 * xis[1] * ds[1], -1 * xis[1] * ds[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]],
                                   [(expressionLabel[dm[1]], sf[1 * xis[1] * ds[1]]), (expressionLabel[dm[2]],
                                                                                       sf[-1 * xis[1] * ds[2]])])
        elif node_type == self.TRIPLE_CURVE0_2_UP:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[3] * ds[3], -1 * xis[2] * ds[1], 1 * xis[2] * ds[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]],
                                   [(expressionLabel[dm[1]], sf[-1 * xis[2] * ds[1]]), (expressionLabel[dm[2]],
                                                                                        sf[1 * xis[2] * ds[2]])])
        elif node_type == self.TRIPLE_CURVE0_3_LEFT:
            if any(c != 1 for c in [1 * xis[2] * ds[2], 1 * xis[3] * ds[3], 1 * xis[1] * ds[1], -1 * xis[1] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(expressionLabel[dm[3]],
                                                                           sf[1 * xis[3] * ds[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]],
                                   [(expressionLabel[dm[1]], sf[1 * xis[1] * ds[1]]), (expressionLabel[dm[3]],
                                                                                       sf[-1 * xis[1] * ds[3]])])
        elif node_type == self.TRIPLE_CURVE0_3_RIGHT:
            if any(c != 1 for c in [1 * xis[1] * ds[1], 1 * xis[2] * ds[2], -1 * xis[3] * ds[1], 1 * xis[3] * ds[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(expressionLabel[dm[1]],
                                                                           sf[1 * xis[1] * ds[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(expressionLabel[dm[2]],
                                                                           sf[1 * xis[2] * ds[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]],
                                   [(expressionLabel[dm[1]], sf[-1 * xis[3] * ds[1]]), (expressionLabel[dm[3]],
                                                                                        sf[1 * xis[3] * ds[3]])])

        elif node_type == self.SURFACE_REGULAR_DOWN_LEFT:
            if any(c != 1 for c in [1 * xis[2], -1 * xis[1], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2,
                                                                           sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS3,
                                                                           sf[-1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[3]])])
        elif node_type == self.SURFACE_REGULAR_DOWN_RIGHT:
            if any(c != 1 for c in [1 * xis[1], 1 * xis[2], 1 * xis[3]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[3]])])
        elif node_type == self.SURFACE_REGULAR_UP:
            if any(c != 1 for c in [1 * xis[1], -1 * xis[3], 1 * xis[2]]):
                if not self._element_needs_scale_factor:
                    self._element_needs_scale_factor = True
                    setEftScaleFactorIds(eft, [1], [])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS2, sf[-1 * xis[3]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[2]])])
        elif node_type == self.REGULAR:
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[1]], [(Node.VALUE_LABEL_D_DS1, sf[1 * xis[1]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[2]], [(Node.VALUE_LABEL_D_DS2, sf[1 * xis[2]])])
            remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[3]], [(Node.VALUE_LABEL_D_DS3, sf[1 * xis[3]])])
        else:
            raise ValueError("Remapping for derivatives of this 'node type' is not implemented")

        # for ci in range(1, 4):
        #     expressionTerms = []
        #     for di in range(1, 4):
        #         if default_sign[ci][di-1]:
        #             expressionTerms.append((expressionLabel[dm[di]], sf[default_sign[ci][di-1] * xis[ci] * ds[di]]))
        #     remapEftNodeValueLabel(eft, localNodeIndexes, label[xim[ci]], expressionTerms)
