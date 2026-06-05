"""
Interpolation functions shared by mesh generators.
"""
from cmlibs.maths.vectorops import (
    add, axis_angle_to_rotation_matrix, cross, distance, div, dot, euler_to_rotation_matrix, matrix_inv, magnitude,
    matrix_vector_mult, mult, normalize, sub, set_magnitude)
from scipy.optimize import minimize
import copy
from collections.abc import Sequence
from enum import Enum
import math


gaussXi3 = ( (-math.sqrt(0.6)+1.0)/2.0, 0.5, (+math.sqrt(0.6)+1.0)/2.0 )
gaussWt3 = ( 5.0/18.0, 4.0/9.0, 5.0/18.0 )

gaussXi4 = (
    (-math.sqrt((3.0+2.0*math.sqrt(6.0/5.0))/7.0)+1.0)/2.0,
    (-math.sqrt((3.0-2.0*math.sqrt(6.0/5.0))/7.0)+1.0)/2.0,
    (+math.sqrt((3.0-2.0*math.sqrt(6.0/5.0))/7.0)+1.0)/2.0,
    (+math.sqrt((3.0+2.0*math.sqrt(6.0/5.0))/7.0)+1.0)/2.0 )
gaussWt4 = (
    (18.0-math.sqrt(30.0))/72.0,
    (18.0+math.sqrt(30.0))/72.0,
    (18.0+math.sqrt(30.0))/72.0,
    (18.0-math.sqrt(30.0))/72.0 )

def getCubicHermiteBasis(xi):
    """
    :return: 4 cubic Hermite basis function values for x1, d1, x2, d2 at xi.
    """
    xi2 = xi*xi
    xi3 = xi2*xi
    f1 = 1.0 - 3.0*xi2 + 2.0*xi3
    f2 = xi - 2.0*xi2 + xi3
    f3 = 3.0*xi2 - 2.0*xi3
    f4 = -xi2 + xi3
    return f1, f2, f3, f4

def interpolateCubicHermite(v1, d1, v2, d2, xi):
    """
    Get values of cubic Hermite interpolated from v1, d1 to v2, d2.
    :param v1, v2: Values at xi = 0.0 and xi = 1.0, respectively.
    :param d1, d2: Derivatives w.r.t. xi at xi = 0.0 and xi = 1.0, respectively.
    :param xi: Position in curve, nominally in [0.0, 1.0].
    :return: List of interpolated values at xi.
    """
    xi2 = xi*xi
    xi3 = xi2*xi
    f1 = 1.0 - 3.0*xi2 + 2.0*xi3
    f2 = xi - 2.0*xi2 + xi3
    f3 = 3.0*xi2 - 2.0*xi3
    f4 = -xi2 + xi3
    return [ (f1*v1[i] + f2*d1[i] + f3*v2[i] + f4*d2[i]) for i in range(len(v1)) ]

def getCubicHermiteBasisDerivatives(xi):
    """
    :return: 4 cubic Hermite basis function first derivative values for x1, d1, x2, d2 at xi.
    """
    xi2 = xi*xi
    df1 = -6.0*xi + 6.0*xi2
    df2 = 1.0 - 4.0*xi + 3.0*xi2
    df3 = 6.0*xi - 6.0*xi2
    df4 = -2.0*xi + 3.0*xi2
    return df1, df2, df3, df4

def interpolateCubicHermiteDerivative(v1, d1, v2, d2, xi):
    """
    Get derivatives of cubic Hermite interpolated from v1, d1 to v2, d2.
    :param v1, v2: Values at xi = 0.0 and xi = 1.0, respectively.
    :param d1, d2: Derivatives w.r.t. xi at xi = 0.0 and xi = 1.0, respectively.
    :param xi: Position in curve, nominally in [0.0, 1.0].
    :return: List of interpolated derivatives at xi.
    """
    xi2 = xi*xi
    f1 = -6.0*xi + 6.0*xi2
    f2 = 1.0 - 4.0*xi + 3.0*xi2
    f3 = 6.0*xi - 6.0*xi2
    f4 = -2.0*xi + 3.0*xi2
    return [ (f1*v1[i] + f2*d1[i] + f3*v2[i] + f4*d2[i]) for i in range(len(v1)) ]

def interpolateCubicHermiteSecondDerivative(v1, d1, v2, d2, xi):
    """
    Get second derivatives of cubic Hermite interpolated from v1, d1 to v2, d2.
    :param v1, v2: Values at xi = 0.0 and xi = 1.0, respectively.
    :param d1, d2: Derivatives w.r.t. xi at xi = 0.0 and xi = 1.0, respectively.
    :param xi: Position in curve, nominally in [0.0, 1.0].
    :return: List of interpolated second derivatives at xi.
    """
    f1 = -6.0 + 12.0*xi
    f2 = -4.0 +  6.0*xi
    f3 =  6.0 - 12.0*xi
    f4 = -2.0 +  6.0*xi
    return [ (f1*v1[i] + f2*d1[i] + f3*v2[i] + f4*d2[i]) for i in range(len(v1)) ]

def computeCubicHermiteArcLength(v1, d1, v2, d2, rescaleDerivatives):
    """
    Compute arc length between v1 and v2, scaling unit d1 and d2.
    Iterative; not optimised.
    :param d1: Initial derivative at v1.
    :param d2: Initial derivative at v2.
    :param rescaleDerivatives: If True, rescale initial d1 and d2 to |v2 - v|
    :return: Arc length.
    """
    if rescaleDerivatives:
        lastArcLength = math.sqrt(sum((v2[i] - v1[i])*(v2[i] - v1[i]) for i in range(len(v1))))
    else:
        lastArcLength = getCubicHermiteArcLength(v1, d1, v2, d2)
    d1 = normalize(d1)
    d2 = normalize(d2)
    tol = 1.0E-6
    for iters in range(100):
        #print('iter',iters,'=',lastArcLength)
        d1s = [lastArcLength*d for d in d1]
        d2s = [lastArcLength*d for d in d2]
        arcLength = getCubicHermiteArcLength(v1, d1s, v2, d2s)
        if iters > 9:
            arcLength = 0.8*arcLength + 0.2*lastArcLength
        if math.fabs(arcLength - lastArcLength) < tol*arcLength:
            #print('computeCubicHermiteArcLength converged at iter',iters,'=',arcLength,', closeness', math.fabs(arcLength - lastArcLength))
            return arcLength
        lastArcLength = arcLength
    print('computeCubicHermiteArcLength:  Max iters reached:',iters,'=',arcLength,', closeness', math.fabs(arcLength - lastArcLength))
    return arcLength

def computeCubicHermiteDerivativeScaling(v1, d1, v2, d2):
    """
    Compute scaling for d1, d2 which makes their sum twice the arc length.
    :return: Scale factor to multiply d1, d2
    """
    origMag = 0.5*(magnitude(d1) + magnitude(d2))
    scaling = 1.0
    for iters in range(100):
        mag = origMag * scaling
        arcLength = getCubicHermiteArcLength(v1, [d * scaling for d in d1], v2, [d * scaling for d in d2])
        if math.fabs(arcLength - mag) < 1.0E-6 * arcLength:
            #print('compute scaling', v1, d1, v2, d2, '\n  --> scaling',scaling)
            return scaling
        scaling *= arcLength / mag
    print('computeCubicHermiteDerivativeScaling:  Max iters reached:', iters, ' mag', mag, 'arc', arcLength)
    return scaling

def computeHermiteLagrangeDerivativeScaling(v1, d1, v2):
    """
    Compute scaling for d1 to equal arc length.
    :param d1: Seems more reliable to pass a unit vector direction here.
    :return: Scale factor to multiply d1.
    """
    origMag = magnitude(d1)
    # start with scaling which makes this equal to distance from v1 to v2
    scaling = distance(v1, v2) / origMag
    scaling = 1.0
    for iters in range(100):
        mag = origMag * scaling
        scaled_d1 = [d * scaling for d in d1]
        d2 = interpolateHermiteLagrangeDerivative(v1, scaled_d1, v2, 1.0)
        arcLength = getCubicHermiteArcLength(v1, scaled_d1, v2, d2)
        if math.fabs(arcLength - mag) < 1.0E-6 * arcLength:
            return scaling
        scaling *= arcLength / mag
    print('computeHermiteLagrangeDerivativeScaling:  Max iters reached:', iters, ' mag', mag, 'arc', arcLength)
    return scaling

def computeCubicHermiteStartDerivative(v1, d1_in, v2, d2):
    """
    Compute scaled d1 which makes their sum of d1 and d2 magnitudes twice the arc length.
    :param d1_in: Original start derivative.
    :return: Scaled d1
    """
    d2_mag = magnitude(d2)
    d1 = set_magnitude(d1_in, 0.5 * d2_mag)
    for iters in range(100):
        arcLength = getCubicHermiteArcLength(v1, d1, v2, d2)
        d1_mag = 2.0 * arcLength - d2_mag
        d1 = set_magnitude(d1_in, d1_mag)
        if math.fabs(2.0 * arcLength - d1_mag - d2_mag) < (1.0E-6 * arcLength):
            break
    else:
        print('computeCubicHermiteStartDerivative:  Max iters reached:', iters, ' mag d1', d1_mag, 'mag d2', d2_mag,
              'arc length', arcLength)
    return d1

def computeCubicHermiteEndDerivative(v1, d1, v2, d2_in):
    """
    Compute scaled d2 which makes their sum of d1 and d2 magnitudes twice the arc length.
    :param d2_in: Original end derivative.
    :return: Scaled d2
    """
    d1_mag = magnitude(d1)
    d2 = set_magnitude(d2_in, 0.5 * d1_mag)
    for iters in range(100):
        arcLength = getCubicHermiteArcLength(v1, d1, v2, d2)
        d2_mag = 2.0 * arcLength - d1_mag
        d2 = set_magnitude(d2_in, d2_mag)
        if math.fabs(2.0 * arcLength - d1_mag - d2_mag) < (1.0E-6 * arcLength):
            break
    else:
        print('computeCubicHermiteEndDerivative:  Max iters reached:', iters, ' mag d1', d1_mag, 'mag d2', d2_mag,
              'arc length', arcLength)
    return d2

def getCubicHermiteArcLength(v1, d1, v2, d2):
    """
    Note this is approximate.
    :return: Arc length of cubic curve using 4 point Gaussian quadrature.
    """
    arcLength = 0.0
    for i in range(4):
        dm = interpolateCubicHermiteDerivative(v1, d1, v2, d2, gaussXi4[i])
        arcLength += gaussWt4[i]*math.sqrt(sum(d*d for d in dm))
    return arcLength

def getCubicHermiteArcLengthToXi(v1, d1, v2, d2, xi):
    """
    Note this is approximate.
    :return: Arc length of cubic curve up to given xi coordinate.
    """
    d1m = [d * xi for d in d1]
    v2m = interpolateCubicHermite(v1, d1, v2, d2, xi)
    d2m = interpolateCubicHermiteDerivative(v1, d1, v2, d2, xi)
    d2m = [d * xi for d in d2m]
    return getCubicHermiteArcLength(v1, d1m, v2m, d2m)

def getCubicHermiteCurvesLength(cx, cd1, loop=False):
    """
    Calculate total length of a curve.
    :param cx: coordinates along the curve.
    :param cd1: d1 derivatives.
    :param loop: True if curve loops back to first point, False if not.
    :return: Length
    """
    totalLength = 0.0
    elementsCount = len(cx)
    if not loop:
        elementsCount -= 1
    for e in range(elementsCount):
        ep = e + 1
        if loop and (ep == elementsCount):
            ep = 0
        arcLength = getCubicHermiteArcLength(cx[e], cd1[e], cx[ep], cd1[ep])
        totalLength += arcLength
    return totalLength

def getCubicHermiteCurvesLengthLoop(cx, cd1):
    """
    Calculate total length of a loop curve, which repeats back to first point.
    :param cx: coordinates along the curve.
    :param cd1: d1 derivatives.
    :return: Length
    """
    totalLength = 0.0
    elementsCount = len(cx)
    for e in range(elementsCount):
        ep = e - elementsCount + 1
        arcLength = getCubicHermiteArcLength(cx[e], cd1[e], cx[ep], cd1[ep])
        totalLength += arcLength
    return totalLength


def getCubicHermiteTrimmedCurvesLengths(cx, cd1, startLocation=None, endLocation=None):
    """
    Get trimmed lengths of curve: before start, between start and end, and after end, lengths to nodes
    :param cx: coordinates along the curve.
    :param cd1: d1 derivatives.
    :param startLocation: Optional tuple of 'in' (element, xi) to start curve at.
    :param endLocation: Optional tuple of 'out' (element, xi) to end curve at.
    :return: Length before start, length between start and end, length after end, list of lengths to nodes.
    """
    elementsCount = len(cx) - 1
    lengthToNode = [0.0]
    length = 0.0
    for e in range(elementsCount):
        length += getCubicHermiteArcLength(cx[e], cd1[e], cx[e + 1], cd1[e + 1])
        lengthToNode.append(length)
    startLength = 0.0
    if startLocation:
        e = startLocation[0]
        startLength = (lengthToNode[e] +
                       getCubicHermiteArcLengthToXi(cx[e], cd1[e], cx[e + 1], cd1[e + 1], startLocation[1]))
        length -= startLength
    endLength = 0.0
    if endLocation:
        e = endLocation[0]
        endLength = (lengthToNode[-1] - lengthToNode[e] -
                     getCubicHermiteArcLengthToXi(cx[e], cd1[e], cx[e + 1], cd1[e + 1], endLocation[1]))
        length -= endLength
    return startLength, length, endLength, lengthToNode


def getCubicHermiteCurvature(v1, d1, v2, d2, radialVector, xi):
    """
    :param v1, v2: Values at xi = 0.0 and xi = 1.0, respectively.
    :param d1, d2: Derivatives w.r.t. xi at xi = 0.0 and xi = 1.0, respectively.
    :param radialVector: Radial direction, assumed unit normal to curve tangent at point.
    :param xi: Position in curve, nominally in [0.0, 1.0].
    :return: Scalar curvature (1/R) of the 1-D cubic Hermite curve.
    """
    tangent = interpolateCubicHermiteDerivative(v1, d1, v2, d2, xi)
    dTangent = interpolateCubicHermiteSecondDerivative(v1, d1, v2, d2, xi)
    #tangentVector = normalize(tangent)
    #tangentCurvature = dot(dTangent, tangentVector)
    radialCurvature = dot(dTangent, radialVector)
    magTangent = magnitude(tangent)
    curvature = radialCurvature/(magTangent*magTangent)

    return curvature

def getCubicHermiteCurvatureSimple(v1, d1, v2, d2, xi):
    """
    :param v1, v2: Values at xi = 0.0 and xi = 1.0, respectively.
    :param d1, d2: Derivatives w.r.t. xi at xi = 0.0 and xi = 1.0, respectively.
    :param xi: Position in curve, nominally in [0.0, 1.0].
    :return: Scalar curvature (1/R) of the 1-D cubic Hermite curve, tangent, dTangent
    """
    tangent = interpolateCubicHermiteDerivative(v1, d1, v2, d2, xi)
    mag_tangent = magnitude(tangent)
    if mag_tangent > 0.0:
        dTangent = interpolateCubicHermiteSecondDerivative(v1, d1, v2, d2, xi)
        componentsCount = len(v1)
        if componentsCount > 1:
            if componentsCount == 3:
                cp = cross(tangent, dTangent)
                mag_cp = magnitude(cp)
            else:
                mag_cp = tangent[0] * dTangent[1] - tangent[1] * dTangent[0]
            curvature = mag_cp / (mag_tangent * mag_tangent * mag_tangent)
        else:
            curvature = 0.0
    else:
        curvature = 0.0
        dTangent = [0.0, 0.0, 0.0]
    return curvature, tangent, dTangent


def interpolateHermiteLagrange(v1, d1, v2, xi):
    """
    Get value at xi for quadratic Hermite-Lagrange interpolation from v1, d1 to v2.
    :return: List of values at xi.
    """
    f1 = 1 - xi*xi
    f2 = xi - xi*xi
    f3 = xi*xi
    return [ (v1[c]*f1 + d1[c]*f2 + v2[c]*f3) for c in range(len(v1)) ]


def interpolateHermiteLagrangeDerivative(v1, d1, v2, xi):
    """
    Get derivative at xi for quadratic Hermite-Lagrange interpolation from v1, d1 to v2.
    :return: List of derivatives w.r.t. xi at xi.
    """
    df1 = -2.0*xi
    df2 = 1 - 2.0*xi
    df3 = 2.0*xi
    return [ (v1[c]*df1 + d1[c]*df2 + v2[c]*df3) for c in range(len(v1)) ]


def interpolateLagrangeHermite(v1, v2, d2, xi):
    """
    Get value at xi for quadratic Lagrange-Hermite interpolation from v1 to v2, d2.
    :return: List of values at xi.
    """
    f1 = 1 - 2.0*xi + xi*xi
    f2 = 2.0*xi - xi*xi
    f3 = -xi + xi*xi
    return [ (v1[c]*f1 + v2[c]*f2 + d2[c]*f3) for c in range(len(v1)) ]


def interpolateLagrangeHermiteDerivative(v1, v2, d2, xi):
    """
    Get derivative at xi for quadratic Lagrange-Hermite interpolation to from v1 to v2, d2.
    :return: List of derivatives w.r.t. xi at xi.
    """
    df1 = -2.0 + 2.0*xi
    df2 = 2.0 - 2.0*xi
    df3 = -1.0 + 2.0*xi
    return [ (v1[c]*df1 + v2[c]*df2 + d2[c]*df3) for c in range(len(v1)) ]


def computeLagrangeHermiteDerivative(v1, v2, d2_in):
    """
    Compute scaled d2 which makes it equal to arc length.
    :param d2_in: Direction of d2; initial magnitude is ignored.
    :return: Scaled d2
    """
    d2_mag = magnitude(sub(v2, v1))
    if d2_mag == 0.0:
        return set_magnitude(d2_in, 0.0)
    TOL = 1.0E-6 * d2_mag if (d2_mag > 0.0) else 1.0
    d2 = set_magnitude(d2_in, d2_mag)
    for iters in range(100):
        d1 = interpolateLagrangeHermiteDerivative(v1, v2, d2, 0.0)
        arc_length = getCubicHermiteArcLength(v1, d1, v2, d2)
        d2 = set_magnitude(d2_in, arc_length)
        if math.fabs(arc_length - d2_mag) < TOL:
            break
        d2_mag = arc_length
    else:
        print('computeLagrangeHermiteDerivative:  Max iters reached:', iters, ' v', v1, 'c2', v2,
              'd2_in', d2_in, 'arc length', arc_length)
    return d2


def getNearestPointIndex(nx, x):
    """
    :return: index of point in nx at shortest distance from x.
    """
    pointCount = len(nx)
    assert pointCount > 0
    components = len(x)
    minDistance = math.sqrt(sum( ((nx[0][c] - x[c])*(nx[0][c] - x[c])) for c in range(components)))
    index = 0
    for n in range(1, pointCount):
        distance = math.sqrt(sum( ((nx[n][c] - x[c])*(nx[n][c] - x[c])) for c in range(components)))
        if distance < minDistance:
            minDistance = distance
            index = n
    return index

def projectHermiteCurvesThroughWall(nx, nd1, nd2, n, wallThickness, loop=False):
    """
    From Hermite curve nx, nd1 with cross direction nd2, project normal to wall
    by wall thickness to get coordinates, d1 affected by curvature etc.
    Assumes 3 components.
    :param n: Index into nx, nd1, nd2 of where to project.
    :param wallThickness: Use positive from in to out, negative from outside to in.
    :param loop: True if curve loops back to first point, False if not.
    :return: x, d1, d2, d3
    """
    maxPointIndex = len(nx) - 1
    assert (0 <= n <= maxPointIndex), 'projectHermiteCurvesThroughWall.  Invalid index'
    unitNormal = normalize(cross(nd1[n], nd2[n]))
    x  = [ (nx[n][c] + wallThickness*unitNormal[c]) for c in range(3) ]
    # calculate inner d1 from curvature around
    curvature = 0.0
    count = 0
    if loop or (n > 0) and (nx[n - 1]):
        curvature += getCubicHermiteCurvature(nx[n - 1], nd1[n - 1], nx[n], nd1[n], unitNormal, 1.0)
        count += 1
    if loop or (n < maxPointIndex) and (nx[n - maxPointIndex]):
        curvature += getCubicHermiteCurvature(nx[n], nd1[n], nx[n - maxPointIndex], nd1[n - maxPointIndex], unitNormal, 0.0)
        count += 1
    curvature /= count
    factor = 1.0 - curvature*wallThickness
    d1 = [ factor*c for c in nd1[n] ]
    d2 = copy.deepcopy(nd2[n])  # magnitude can't be determined here
    d3 = set_magnitude(unitNormal, math.fabs(wallThickness))
    return x, d1, d2, d3


def sampleCubicHermiteCurves(nx, nd1, elementsCountOut,
    addLengthStart = 0.0, addLengthEnd = 0.0,
    lengthFractionStart = 1.0, lengthFractionEnd = 1.0,
    elementLengthStartEndRatio = 1.0, arcLengthDerivatives = False):
    """
    Get systematically spaced points and derivatives over cubic Hermite interpolated
    curves with nodes nx and derivatives nd1. The first element uses the first two nodes.
    :param nx: Coordinates of nodes along curves.
    :param nd1: Derivatives of nodes along curves.
    :param addLengthStart, addLengthEnd: Extra length to add to start and end elements.
    :param lengthFractionStart, lengthFractionEnd: Fraction of mid element length for
        start and end elements. Can use in addition to AddLengths: If LengthFraction
        is 0.5 and AddLength is derivative/2.0 can blend into known derivative at start
        or end.
    :param elementLengthStartEndRatio: Start/end element length ratio, with lengths
        smoothly varying in between. Requires at least 2 elements. Applied in proportion
        to lengthFractionStart, lengthFractionEnd.
    :param arcLengthDerivatives: If True each cubic section is rescaled to arc length.
    If False (default), derivatives and distances are used as supplied.
    :return: px[], pd1[], pe[], pxi[], psf[], where pe[] and pxi[] are lists of element indices and
    and xi locations in the 'in' elements to pass to partner interpolateSample functions. psf[] is
    a list of scale factors for converting derivatives from old to new xi coordinates: dxi(old)/dxi(new).
    """
    elementsCountIn = len(nx) - 1
    assert (elementsCountIn > 0) and (len(nd1) == (elementsCountIn + 1)) and \
        (elementsCountOut > 0), 'sampleCubicHermiteCurves.  Invalid arguments'
    lengths = [ 0.0 ]
    nd1a = []
    nd1b = []
    length = 0.0
    for e in range(elementsCountIn):
        if arcLengthDerivatives:
            arcLength = computeCubicHermiteArcLength(nx[e], nd1[e], nx[e + 1], nd1[e + 1], rescaleDerivatives = True)
            nd1a.append(set_magnitude(nd1[e], arcLength))
            nd1b.append(set_magnitude(nd1[e + 1], arcLength))
        else:
            arcLength = getCubicHermiteArcLength(nx[e], nd1[e], nx[e + 1], nd1[e + 1])
        length += arcLength
        lengths.append(length)
    proportionEnd = 2.0/(elementLengthStartEndRatio + 1)
    proportionStart = elementLengthStartEndRatio*proportionEnd
    if elementsCountOut == 1:
        elementLengthMid = length
    else:
        elementLengthMid = (length - addLengthStart - addLengthEnd) / \
            (elementsCountOut - 2.0 + proportionStart*lengthFractionStart + proportionEnd*lengthFractionEnd)
    elementLengthProportionStart = proportionStart*lengthFractionStart*elementLengthMid
    elementLengthProportionEnd = proportionEnd*lengthFractionEnd*elementLengthMid
    # get smoothly varying element lengths, not accounting for start and end
    if (elementsCountOut == 1) or (elementLengthStartEndRatio == 1.0):
        elementLengths = [ elementLengthMid ]*elementsCountOut
    else:
        elementLengths = []
        for eOut in range(elementsCountOut):
            xi = eOut/(elementsCountOut - 1)
            elementLengths.append(((1.0 - xi)*proportionStart + xi*proportionEnd)*elementLengthMid)
    # get middle derivative magnitudes
    nodeDerivativeMagnitudes = [ None ]*(elementsCountOut + 1)  # start and end determined below
    for n in range(1, elementsCountOut):
        nodeDerivativeMagnitudes[n] = 0.5*(elementLengths[n - 1] + elementLengths[n])
    # fix end lengths:
    elementLengths[ 0] = addLengthStart + elementLengthProportionStart
    elementLengths[-1] = addLengthEnd + elementLengthProportionEnd
    #print('\nsampleCubicHermiteCurves:')
    #print('  elementLengths', elementLengths, 'addLengthStart', addLengthStart, 'addLengthEnd', addLengthEnd)
    #print('  sum lengths', sum(elementLengths), 'vs. length', length, 'diff', sum(elementLengths) - length)
    # set end derivatives:
    if elementsCountOut == 1:
        nodeDerivativeMagnitudes[0] = nodeDerivativeMagnitudes[1] = elementLengths[0]
    else:
        nodeDerivativeMagnitudes[ 0] = 2.0 * elementLengths[ 0] - nodeDerivativeMagnitudes[ 1]
        nodeDerivativeMagnitudes[-1] = 2.0 * elementLengths[-1] - nodeDerivativeMagnitudes[-2]

    px = []
    pd1 = []
    pe = []
    pxi = []
    psf = []
    distance = 0.0
    e = 0
    for eOut in range(elementsCountOut):
        while e < elementsCountIn:
            if distance < lengths[e + 1]:
                partDistance = distance - lengths[e]
                if arcLengthDerivatives:
                    xi = partDistance/(lengths[e + 1] - lengths[e])
                    x = interpolateCubicHermite(nx[e], nd1a[e], nx[e + 1], nd1b[e], xi)
                    d1 = interpolateCubicHermiteDerivative(nx[e], nd1a[e], nx[e + 1], nd1b[e], xi)
                else:
                    x, d1, _eIn, xi = getCubicHermiteCurvesPointAtArcDistance(nx[e:e + 2], nd1[e:e + 2], partDistance)
                sf = nodeDerivativeMagnitudes[eOut]/magnitude(d1)
                px.append(x)
                pd1.append([ sf*d for d in d1 ])
                pe.append(e)
                pxi.append(xi)
                psf.append(sf)
                break
            e += 1
        distance += elementLengths[eOut]
    e = elementsCountIn
    eOut = elementsCountOut
    xi = 1.0
    d1 = nd1[e]
    sf = nodeDerivativeMagnitudes[eOut]/magnitude(d1)
    px.append(nx[e])
    pd1.append([ sf*d for d in d1 ])
    pe.append(e - 1)
    pxi.append(xi)
    psf.append(sf)
    return px, pd1, pe, pxi, psf


def sampleCubicHermiteCurvesSmooth(nx, nd1, elementsCountOut,
        derivativeMagnitudeStart=None, derivativeMagnitudeEnd=None,
        startLocation=None, endLocation=None):
    """
    Get smoothly spaced points and derivatives over cubic Hermite interpolated
    curves with nodes nx and derivatives nd1. The first element uses the first two nodes.
    Gives smooth variation of element size to fit the supplied start and end derivative
    magnitudes.
    :param nx: Coordinates of nodes along curves.
    :param nd1: Derivatives of nodes along curves.
    :param derivativeMagnitudeStart, derivativeMagnitudeEnd: Optional magnitudes of start and end
    derivatives appropriate for elementsCountOut. If unspecified these are calculated from the other
    end or set to be equal for even spaced elements. 0.0 is a valid derivative magnitude.
    :param startLocation: Optional tuple of 'in' (element, xi) to start curve at.
    :param endLocation: Optional tuple of 'out' (element, xi) to end curve at.
    :return: px[], pd1[], pe[], pxi[], psf[], where pe[] and pxi[] are lists of element indices and
    xi locations in the 'in' elements to pass to partner interpolateSample functions. psf[] is
    a list of scale factors for converting derivatives from old to new xi coordinates: dxi(old)/dxi(new).
    """
    elementsCountIn = len(nx) - 1
    assert (elementsCountIn > 0) and (len(nd1) == (elementsCountIn + 1)) and (elementsCountOut > 0), \
        "sampleCubicHermiteCurvesSmooth.  Invalid arguments"
    startLength, length, endLength, lengthToNodeIn =\
        getCubicHermiteTrimmedCurvesLengths(nx, nd1, startLocation, endLocation)
    hasStartDerivative = derivativeMagnitudeStart is not None
    hasEndDerivative = derivativeMagnitudeEnd is not None
    if hasStartDerivative and hasEndDerivative:
        pass
    elif hasEndDerivative:
        derivativeMagnitudeStart = (2.0 * length - elementsCountOut * derivativeMagnitudeEnd) / elementsCountOut
    elif hasStartDerivative:
        derivativeMagnitudeEnd = (2.0 * length - elementsCountOut * derivativeMagnitudeStart) / elementsCountOut
    else:
        derivativeMagnitudeStart = derivativeMagnitudeEnd = length / elementsCountOut
    # sample over length to get distances to elements boundaries
    x1 = startLength
    d1 = derivativeMagnitudeStart * elementsCountOut
    x2 = startLength + length
    d2 = derivativeMagnitudeEnd * elementsCountOut
    nodeDistances = []
    nodeDerivativeMagnitudes = []
    nodesCountOut = elementsCountOut + 1
    for n in range(nodesCountOut):
        xi = n / elementsCountOut
        f1, f2, f3, f4 = getCubicHermiteBasis(xi)
        distance = f1*x1 + f2*d1 + f3*x2 + f4*d2
        nodeDistances.append(distance)
        f1, f2, f3, f4 = getCubicHermiteBasisDerivatives(xi)
        derivative = f1*x1 + f2*d1 + f3*x2 + f4*d2
        nodeDerivativeMagnitudes.append(derivative/elementsCountOut)
    px = []
    pd1 = []
    pe = []
    pxi = []
    psf = []
    e = 0
    lastElementIn = elementsCountIn - 1
    for nOut in range(nodesCountOut):
        distance = nodeDistances[nOut]
        while (e < lastElementIn) and (distance >= lengthToNodeIn[e + 1]):
            e += 1
        partDistance = distance - lengthToNodeIn[e]
        x, d1, _, xi = getCubicHermiteCurvesPointAtArcDistance(nx[e:e + 2], nd1[e:e + 2], partDistance)
        sf = nodeDerivativeMagnitudes[nOut] / magnitude(d1)
        px.append(x)
        pd1.append([sf * d for d in d1])
        pe.append(e)
        pxi.append(xi)
        psf.append(sf)
    return px, pd1, pe, pxi, psf


def interpolateSampleCubicHermite(v, d, pe, pxi, psf):
    """
    Partner function to sampleCubicHermiteCurves for interpolating additional variables with
    cubic Hermite basis, at the element indexes, xi coordinates and xi scaling returned from that function.
    Note: this does not work for sampleCubicHermiteCurves with arcLengthDerivatives = False.
    :param v, d: List of values and derivatives to interpolate, either scalar or sequence-of-scalar.
    len(v) == len(d) == number of elements in + 1.
    :param pe, pxi: List if integer element indexes and real xi coordinates giving sample positions into v to
    interpolate linearly. len(pe) == len(pxi) == number of values out.
    Indexes in pe start at 0, and are not checked; sampleCubicHermiteCurves() guarantees these are valid for
    the number of elements passed to it.
    :param psf: List of scale factors dxi(old)/dxi(new). Length same as pe, pxi. Used to convert derivatives
    from old to new xi spacing.
    :return: List of interpolated values, list of interpolated derivatives; scalar or vector as for v, d.
    """
    assert (len(v) > 1) and (len(d) == len(v)), 'interpolateSampleCubicHermite. Invalid values v, d'
    valuesCountOut = len(pe)
    assert (valuesCountOut > 0) and (len(pxi) == valuesCountOut), 'interpolateSampleCubicHermite. Invalid element, xi'
    vOut = []
    dOut = []
    if isinstance(v[0], Sequence):
        for n in range(valuesCountOut):
            e = pe[n]
            v1 = v[e]
            d1 = d[e]
            v2 = v[e + 1]
            d2 = d[e + 1]
            vOut.append(interpolateCubicHermite(v1, d1, v2, d2, pxi[n]))
            dOut.append([ psf[n]*d for d in interpolateCubicHermiteDerivative(v1, d1, v2, d2, pxi[n]) ])
    else:
        for n in range(valuesCountOut):
            e = pe[n]
            v1 = [ v[e] ]
            d1 = [ d[e] ]
            v2 = [ v[e + 1] ]
            d2 = [ d[e + 1] ]
            vOut.append(interpolateCubicHermite(v1, d1, v2, d2, pxi[n])[0])
            dOut.append(psf[n]*interpolateCubicHermiteDerivative(v1, d1, v2, d2, pxi[n])[0])
    return vOut, dOut


def interpolateSampleLinear(v, pe, pxi):
    """
    Partner function to sampleCubicHermiteCurves for linearly interpolating additional variables based on the 
    element indexes and element xi coordinates returned from that function.
    :param v: List of scalar values or sequence-of-values to interpolate. len(v) == number of elements in + 1.
    :param pe, pxi: List if integer element indexes and real xi coordinates giving sample positions into v to
    interpolate linearly. len(pe) == len(pxi) == number of values out.
    Indexes in pe start at 0, and are not checked; sampleCubicHermiteCurves() guarantees these are valid for
    the number of elements passed to it.
    :return: List of interpolated values, scalar or vector as for v.
    """
    assert len(v) > 1, 'interpolateSampleLinear. Invalid values v: not enough data'
    valuesCountOut = len(pe)
    assert (valuesCountOut > 0) and (len(pxi) == valuesCountOut), 'interpolateSampleLinear. Invalid element, xi'
    vOut = []
    if isinstance(v[0], Sequence):
        vLen = len(v[0])
        for n in range(valuesCountOut):
            wp = pxi[n]
            wm = 1.0 - wp
            vp = v[pe[n] + 1]
            vm = v[pe[n]]
            vOut.append([ (wm*vm[c] + wp*vp[c]) for c in range(vLen) ])
    else:
        for n in range(valuesCountOut):
            wp = pxi[n]
            wm = 1.0 - wp
            vOut.append(wm*v[pe[n]] + wp*v[pe[n] + 1])
    return vOut


def sampleCubicElementLengths(length, elementsCount, startDerivative = None, endDerivative = None):
    """
    Get lengths of elements gradually changing over length, satisfying the start and end derivatives.
    :param startDerivative, endDerivative: Magnitudes of end derivatives to use with elementsCount,
    or None to choose a natural size.
    :return: List of elementsCount element lengths
    """
    assert (elementsCount > 0), 'interpolation sampleCubicElementLengths:  Invalid number of elements'
    x1 = 0.0
    x2 = length
    d1 = startDerivative*elementsCount if startDerivative else None
    d2 = endDerivative*elementsCount if endDerivative else None
    if not (d1 and d2):
        d1 = d2 = length
    elif not d2:
        d2 = 2.0*length - d1
    elif not d1:
        d1 = 2.0*length - d2
    elementLengths = []
    lastx = 0.0
    for n in range(1, elementsCount + 1):
        xi = n/elementsCount
        f1, f2, f3, f4 = getCubicHermiteBasis(xi)
        x = f1*x1 + f2*d1 + f3*x2 + f4*d2
        elementLengths.append(x - lastx)
        lastx = x
    return elementLengths


def getCubicHermiteCurvesPointAtArcDistance(nx, nd, arcDistance):
    """
    Get the coordinates, derivatives at distance along cubic Hermite curves.
    Supplied derivatives are used i.e. not rescaled to arc length.
    Note this is approximate.
    :param nx: Coordinates of nodes along curves.
    :param nd: Derivatives of nodes along curves.
    :param distance: Distance along curves.
    :return: coordinates, derivatives, element index, xi; clamped to first or last nx if distance is beyond curves
    """
    elementsCount = len(nx) - 1
    assert elementsCount > 0, 'getCubicHermiteCurvesPointAtArcDistance.  Invalid number of points'
    if arcDistance < 0.0:
        return nx[0], nd[0], 0, 0.0
    length = 0.0
    xiDelta = 1.0E-6
    xiTol = 1.0E-6
    #print('elementsCount',elementsCount,'arcDistance',arcDistance)
    for e in range(elementsCount):
        partDistance = arcDistance - length
        v1 = nx[e]
        d1 = nd[e]
        v2 = nx[e + 1]
        d2 = nd[e + 1]
        arcLength = getCubicHermiteArcLength(v1, d1, v2, d2)
        #print('e',e,'partDistance',partDistance,'arcLength',arcLength)
        if partDistance <= arcLength:
            xiLast = 100.0
            xi = partDistance/arcLength
            dxiLimit = 0.1
            for iter in range(100):
                xiLast = xi
                dist = getCubicHermiteArcLengthToXi(v1, d1, v2, d2, xi)
                distp = getCubicHermiteArcLengthToXi(v1, d1, v2, d2, xi + xiDelta)
                distm = getCubicHermiteArcLengthToXi(v1, d1, v2, d2, xi - xiDelta)
                if (xi - xiDelta) < 0.0:
                    distm = -distm
                dxi_ddist = 2.0*xiDelta/(distp - distm)
                dxi = dxi_ddist*(partDistance - dist)
                #print('iter',iter,'xi',xi,'--> dist',dist,'dxi',dxi,'dxiLimit',dxiLimit)
                if dxi > dxiLimit:
                    dxi = dxiLimit
                elif dxi < -dxiLimit:
                    dxi = -dxiLimit
                xi += dxi
                if math.fabs(xi - xiLast) <= xiTol:
                    #print('converged xi',xi)
                    return interpolateCubicHermite(v1, d1, v2, d2, xi), interpolateCubicHermiteDerivative(v1, d1, v2, d2, xi), e, xi
                if iter in [ 4, 10, 25, 62 ]:
                    dxiLimit *= 0.5
            print('getCubicHermiteCurvesPointAtArcDistance Max iters reached:',iter,': e', e, ', xi',xi,', closeness', math.fabs(dist - partDistance))
            return v2, d2, e, xi
        length += arcLength
    return nx[-1], nd[-1], elementsCount - 1, 1.0

class DerivativeScalingMode(Enum):
    ARITHMETIC_MEAN = 1  # derivative is half of sum of arclengths on either side
    HARMONIC_MEAN   = 2  # derivative is reciprocal of arithmetic mean of reciprocals of arclengths = arc lengths weighted by proportion from other side

def smoothCubicHermiteDerivativesLine(nx, nd1,
        fixAllDirections = False,
        fixStartDerivative = False, fixEndDerivative = False,
        fixStartDirection = False, fixEndDirection = False,
        magnitudeScalingMode = DerivativeScalingMode.ARITHMETIC_MEAN, instrument=False):
    """
    Modifies derivatives nd1 to be smoothly varying and near arc length.
    Values are treated as being in a line.
    Assumes initial derivatives are zero or reasonable.
    Where directions are smoothed the weighted/harmonic mean is used.
    :param nx: List of coordinates of nodes along curves.
    :param nd1: List of derivatives of nodes along curves.
    :param fixAllDirections: Set to True to only smooth magnitudes, otherwise both direction and magnitude are adjusted.
    :param fixStartDerivative, fixEndDerivative: Set to True to fix derivative direction and magnitude at respective end.
    :param fixStartDirection, fixEndDirection: Set to True to fix direction at respective end.
    Redundant if fixAllDirections or respective fixStart/EndDerivative is True.
    :param magnitudeScalingMode: A value from enum DerivativeScalingMode specifying
    expression used to get derivative magnitude from adjacent arc lengths.
    :return: Modified nd1
    """
    nodesCount = len(nx)
    elementsCount = nodesCount - 1
    assert elementsCount > 0, 'smoothCubicHermiteDerivativesLine.  Too few nodes/elements'
    assert len(nd1) == nodesCount, 'smoothCubicHermiteDerivativesLine.  Mismatched number of derivatives'
    arithmeticMeanMagnitude = magnitudeScalingMode is DerivativeScalingMode.ARITHMETIC_MEAN
    assert arithmeticMeanMagnitude or (magnitudeScalingMode is DerivativeScalingMode.HARMONIC_MEAN), \
        'smoothCubicHermiteDerivativesLine. Invalid magnitude scaling mode'
    md1 = copy.copy(nd1)
    componentsCount = len(nx[0])
    componentRange = range(componentsCount)
    if elementsCount == 1:
        # special cases for one element
        if fixStartDerivative and fixEndDerivative:
            return nd1
        if not (fixStartDerivative or fixEndDerivative or fixStartDirection or fixEndDirection or fixAllDirections):
            # straight line
            delta = [ (nx[1][c] - nx[0][c]) for c in componentRange ]
            return [ delta, copy.deepcopy(delta) ]
        if fixAllDirections or ((fixStartDirection or fixStartDerivative) and (fixEndDirection or fixEndDerivative)):
            if fixStartDerivative:
                return [nd1[0], computeCubicHermiteEndDerivative(nx[0], nd1[0], nx[1], nd1[1])]
            if fixEndDerivative:
                return [computeCubicHermiteStartDerivative(nx[0], nd1[0], nx[1], nd1[1]), nd1[1]]
            # fixed directions, equal magnitude
            arcLength = computeCubicHermiteArcLength(nx[0], nd1[0], nx[1], nd1[1], rescaleDerivatives=True)
            return [ set_magnitude(nd1[0], arcLength), set_magnitude(nd1[1], arcLength) ]
    # fixStartMag = magnitude(md1[0]) if fixStartDerivative else 0.0
    # fixEndMag = magnitude(md1[-1]) if fixEndDerivative else 0.0
    tol = 1.0E-6
    if instrument:
        print('iter 0', md1)
    for iter in range(100):
        lastmd1 = copy.copy(md1)
        arcLengths = [ getCubicHermiteArcLength(nx[e], md1[e], nx[e + 1], md1[e + 1]) for e in range(elementsCount) ]
        # start
        if not fixStartDerivative:
            if fixAllDirections or fixStartDirection:
                mag = 2.0*arcLengths[0] - magnitude(lastmd1[1])
                md1[0] = set_magnitude(nd1[0], mag) if (mag > 0.0) else [ 0.0, 0.0, 0.0 ]
            else:
                md1[0] = interpolateLagrangeHermiteDerivative(nx[0], nx[1], lastmd1[1], 0.0)
        # middle
        for n in range(1, nodesCount - 1):
            nm = n - 1
            if not fixAllDirections:
                # get mean of directions from point n to points (n - 1) and (n + 1)
                np = n + 1
                dirm = [ (nx[n ][c] - nx[nm][c]) for c in componentRange ]
                dirp = [ (nx[np][c] - nx[n ][c]) for c in componentRange ]
                # mean weighted by fraction towards that end, equivalent to harmonic mean
                arcLengthmp = arcLengths[nm] + arcLengths[n]
                wm = arcLengths[n ]/arcLengthmp
                wp = arcLengths[nm]/arcLengthmp
                md1[n] = [ (wm*dirm[c] + wp*dirp[c]) for c in componentRange ]
            dnm = arcLengths[nm]
            dn = arcLengths[n]
            # future: 2nd nodes from end need to work in with fixed end derivatives
            # if fixStartDerivative and (n == 1):
            #     dnm = 2.0 * dnm - fixStartMag
            # if fixEndDerivative and (n == (nodesCount - 2)):
            #     dn = 2.0 * dn - fixEndMag
            if arithmeticMeanMagnitude:
                mag = 0.5 * (dnm + dn)
            else: # harmonicMeanMagnitude
                mag = 2.0 / (1.0 / dnm + 1.0 / dn)
            md1[n] = set_magnitude(md1[n], mag)
        # end
        if not fixEndDerivative:
            if fixAllDirections or fixEndDirection:
                mag = 2.0*arcLengths[-1] - magnitude(lastmd1[-2])
                md1[-1] = set_magnitude(nd1[-1], mag) if (mag > 0.0) else [ 0.0, 0.0, 0.0 ]
            else:
                md1[-1] = interpolateHermiteLagrangeDerivative(nx[-2], lastmd1[-2], nx[-1], 1.0)
        if instrument:
            print('iter', iter + 1, md1)
        dtol = tol*sum(arcLengths)/len(arcLengths)
        for n in range(nodesCount):
            for c in componentRange:
                if math.fabs(md1[n][c] - lastmd1[n][c]) > dtol:
                    break
            else:
                continue
            break
        else:
            if instrument:
                print('smoothCubicHermiteDerivativesLine converged after iter:', iter + 1)
            return md1

    cmax = 0.0
    for n in range(nodesCount):
        for c in componentRange:
            cmax = max(cmax, math.fabs(md1[n][c] - lastmd1[n][c]))
    closeness = cmax / dtol
    print('smoothCubicHermiteDerivativesLine max iters reached:', iter + 1, ', cmax = ', round(closeness,2), 'x tolerance')
    return md1


def computeCubicHermiteSideCrossDerivatives(ax, ad1, bx, bd1, asd_list, bsd_list,
                                            ascd_fixed=None, bscd_fixed=None):
    """
    Compute 'side cross derivatives' which smoothly transition between side vectors at start and end of hermite
    curve, handling curvature of the curve and rotations of side vectors about its axis.
    :param ax: Coordinates at start of curve.
    :param ad1: Derivative at start of curve.
    :param bx: Coordinates at end of curve.
    :param bd1: Derivative at end of curve.
    :param asd_list: List of start side vectors e.g. [ad2, ad3].
    :param bsd_list: List of end side vectors e.g. [bd2, bd3].
    :param ascd_fixed: Optional fixed values of start side cross derivatives.
    :param bscd_fixed: Optional fixed values of end side cross derivatives.
    :return: List of side cross derivatives at start, list of side cross derivatives at end.
    """
    sideVectorCount = len(asd_list)
    assert len(bsd_list) == sideVectorCount
    assert not (ascd_fixed and bscd_fixed)
    aTangent = normalize(ad1)
    bTangent = normalize(bd1)
    # get hermite basis functions at xi = 1/3, 2/3
    aaXi = 1.0 / 3.0
    bbXi = 2.0 / 3.0
    aaPhi1, aaPhi2, aaPhi3, aaPhi4 = getCubicHermiteBasis(aaXi)
    bbPhi1, bbPhi2, bbPhi3, bbPhi4 = getCubicHermiteBasis(bbXi)
    aaTangent = normalize(interpolateCubicHermiteDerivative(ax, ad1, bx, bd1, aaXi))
    bbTangent = normalize(interpolateCubicHermiteDerivative(ax, ad1, bx, bd1, bbXi))
    ascd_list = []
    bscd_list = []
    for s in range(sideVectorCount):
        asd = asd_list[s]
        bsd = bsd_list[s]
        aNormal = normalize(cross(cross(ad1, asd), ad1))
        bNormal = normalize(cross(cross(bd1, bsd), bd1))
        asdNormalComp = dot(asd, aNormal)
        bsdNormalComp = dot(bsd, bNormal)
        asdTangentComp = dot(asd, aTangent)
        bsdTangentComp = dot(bsd, bTangent)
        # initial guess of side derivatives at xi = 1/3, 2/3
        if ascd_fixed:
            # hermite-lagrange
            aasdGuess = interpolateHermiteLagrange(asd, ascd_fixed[s], bsd, aaXi)
            bbsdGuess = interpolateHermiteLagrange(asd, ascd_fixed[s], bsd, bbXi)
        elif bscd_fixed:
            # lagrange-hermite
            aasdGuess = interpolateLagrangeHermite(asd, bsd, bscd_fixed[s], aaXi)
            bbsdGuess = interpolateLagrangeHermite(asd, bsd, bscd_fixed[s], bbXi)
        else:
            # linear
            aasdGuess = add(mult(asd, bbXi), mult(bsd, aaXi))
            bbsdGuess = add(mult(asd, aaXi), mult(bsd, bbXi))
        aaNormal = normalize(cross(cross(aaTangent, aasdGuess), aaTangent))
        bbNormal = normalize(cross(cross(bbTangent, bbsdGuess), bbTangent))
        # make target side derivatives at xi = 1/3, 2/3
        if ascd_fixed:
            ascdNormalTangent = [dot(ascd_fixed[s], aNormal), dot(ascd_fixed[s], aTangent)]
            aaNormalComp, aaTangentComp = interpolateHermiteLagrange(
                [asdNormalComp, asdTangentComp], ascdNormalTangent, [bsdNormalComp, bsdTangentComp], aaXi)
            bbNormalComp, bbTangentComp = interpolateHermiteLagrange(
                [asdNormalComp, asdTangentComp], ascdNormalTangent, [bsdNormalComp, bsdTangentComp], bbXi)
        elif bscd_fixed:
            bscdNormalTangent = [dot(bscd_fixed[s], bNormal), dot(bscd_fixed[s], bTangent)]
            aaNormalComp, aaTangentComp = interpolateLagrangeHermite(
                [asdNormalComp, asdTangentComp], [bsdNormalComp, bsdTangentComp], bscdNormalTangent, aaXi)
            bbNormalComp, bbTangentComp = interpolateLagrangeHermite(
                [asdNormalComp, asdTangentComp], [bsdNormalComp, bsdTangentComp], bscdNormalTangent, bbXi)
        else:
            aaNormalComp = bbXi * asdNormalComp + aaXi * bsdNormalComp
            aaTangentComp = bbXi * asdTangentComp + aaXi * bsdTangentComp
            bbNormalComp = aaXi * asdNormalComp + bbXi * bsdNormalComp
            bbTangentComp = aaXi * asdTangentComp + bbXi * bsdTangentComp
        aasdTarget = add(mult(aaNormal, aaNormalComp), mult(aaTangent, aaTangentComp))
        bbsdTarget = add(mult(bbNormal, bbNormalComp), mult(bbTangent, bbTangentComp))
        # print("aasdTarget", aasdTarget, "mag", magnitude(aasdTarget))
        # print("bbsdTarget", bbsdTarget, "mag", magnitude(bbsdTarget))
        # subtract known contributions from interpolated start and end side derivatives
        # these are the fractions that the side cross derivative contributions must sum to
        aasdCross = sub(sub(aasdTarget, mult(asd, aaPhi1)), mult(bsd, aaPhi3))
        bbsdCross = sub(sub(bbsdTarget, mult(asd, bbPhi1)), mult(bsd, bbPhi3))
        betaFact = aaPhi4 / bbPhi4
        alphaFact = aaPhi2 - betaFact * bbPhi2
        ascd = []
        bscd = []
        for c in range(3):
            ascdc = (aasdCross[c] - betaFact * bbsdCross[c]) / alphaFact
            bscdc = (aasdCross[c] - aaPhi2 * ascdc) / aaPhi4
            # print("(1)", aaPhi2 * ascdc + aaPhi4 * bscdc, "=", aasdCross[c])
            # print("(2)", bbPhi2 * ascdc + bbPhi4 * bscdc, "=", bbsdCross[c])
            ascd.append(ascdc)
            bscd.append(bscdc)
        aasdCrossActual = add(mult(ascd, aaPhi2), mult(bscd, aaPhi4))
        bbsdCrossActual = add(mult(ascd, bbPhi2), mult(bscd, bbPhi4))
        # print("aasd cross", aasdCross, "actual", aasdCrossActual)
        # print("bbsd cross", bbsdCross, "actual", bbsdCrossActual)
        aasdActual = interpolateCubicHermiteDerivative(asd, ascd, bsd, bscd, aaXi)
        bbsdActual = interpolateCubicHermiteDerivative(asd, ascd, bsd, bscd, bbXi)
        # print("aasd target", aasdTarget, "actual", aasdActual)
        # print("bbsd target", bbsdTarget, "actual", bbsdActual)
        ascd_list.append(ascd)
        bscd_list.append(bscd)
    return ascd_list, bscd_list


def smoothCurveSideCrossDerivatives(nx, nd1, nsv, loop=False):
    """
    Smooth cross derivatives for rate of change of side directions of hermite curves.
    Uses arithmetic mean of side cross derivatives between elements.
    :param nx: List of coordinates of nodes along curve.
    :param nd1: List of derivatives of nodes along curve.
    :param nsv: List over S indexes of lateral direction vectors of nodes along curves.
    :param loop: True if curve loops back to first index.
    :return: List over S indexes of rate of change of nsv w.r.t. d1.
    """
    nodesCount = len(nx)
    elementsCount = nodesCount if loop else nodesCount - 1
    sideVectorCount = len(nsv)
    assert sideVectorCount > 0, 'smoothCurveSideCrossDerivatives.  No side vectors supplied'
    assert len(nsv[0]) == len(nd1) == nodesCount, 'smoothCurveSideCrossDerivatives.  Mismatched number of derivatives'
    assert elementsCount > 0, 'smoothCurveSideCrossDerivatives.  Too few elements'
    assert not loop or (elementsCount > 2), 'smoothCurveSideCrossDerivatives.  Too few elements for loop'
    dnsv = [[None] * nodesCount for s in range(sideVectorCount)]
    for e in range(elementsCount):
        nm = e
        np = e + 1
        if loop and (np == elementsCount):
            np = 0
        dnsvm, dnsvp = computeCubicHermiteSideCrossDerivatives(
            nx[nm], nd1[nm], nx[np], nd1[np],
            [nsv[s][nm] for s in range(sideVectorCount)],
            [nsv[s][np] for s in range(sideVectorCount)])
        for s in range(sideVectorCount):
            dnsv[s][nm] = mult(add(dnsv[s][nm], dnsvm[s]), 0.5) if dnsv[s][nm] else dnsvm[s]
            dnsv[s][np] = mult(add(dnsv[s][np], dnsvp[s]), 0.5) if dnsv[s][np] else dnsvp[s]
    if (not loop) and (elementsCount > 1):
        # get start and end derivatives by recomputing with fixed nearest inter-element derivative
        dnsvm = computeCubicHermiteSideCrossDerivatives(
            nx[0], nd1[0], nx[1], nd1[1],
            [nsv[s][0] for s in range(sideVectorCount)],
            [nsv[s][1] for s in range(sideVectorCount)],
            bscd_fixed=[dnsv[s][1] for s in range(sideVectorCount)])[0]
        dnsvp = computeCubicHermiteSideCrossDerivatives(
            nx[-2], nd1[-2], nx[-1], nd1[-1],
            [nsv[s][-2] for s in range(sideVectorCount)],
            [nsv[s][-1] for s in range(sideVectorCount)],
            ascd_fixed=[dnsv[s][-2] for s in range(sideVectorCount)])[1]
        for s in range(sideVectorCount):
            dnsv[s][0] = dnsvm[s]
            dnsv[s][-1] = dnsvp[s]
    return dnsv


def smoothCubicHermiteDerivativesLoop(nx, nd1,
        fixAllDirections = False,
        magnitudeScalingMode = DerivativeScalingMode.ARITHMETIC_MEAN, instrument=False):
    """
    Modifies derivatives nd1 to be smoothly varying and near arc length.
    Values are treated as being in a loop, so the first point follows the last.
    Assumes initial derivatives are zero or reasonable.
    Where directions are smoothed the weighted/harmonic mean is used.
    :param nx: List of coordinates of nodes along curves.
    :param nd1: List of derivatives of nodes along curves.
    :param fixAllDirections: Set to True to only smooth magnitudes, otherwise both direction and magnitude are adjusted.
    :param magnitudeScalingMode: A value from enum DerivativeScalingMode specifying
    expression used to get derivative magnitude from adjacent arc lengths.
    :return: Modified nd1
    """
    nodesCount = elementsCount = len(nx)
    assert elementsCount > 1, 'smoothCubicHermiteDerivativesLoop.  Too few nodes/elements'
    assert len(nd1) == elementsCount, 'smoothCubicHermiteDerivativesLoop.  Mismatched number of derivatives'
    arithmeticMeanMagnitude = magnitudeScalingMode is DerivativeScalingMode.ARITHMETIC_MEAN
    assert arithmeticMeanMagnitude or (magnitudeScalingMode is DerivativeScalingMode.HARMONIC_MEAN), \
        'smoothCubicHermiteDerivativesLoop. Invalid magnitude scaling mode'
    md1 = copy.copy(nd1)
    componentsCount = len(nx[0])
    componentRange = range(componentsCount)
    tol = 1.0E-6
    if instrument:
        print('iter 0', md1)
    for iter in range(100):
        lastmd1 = copy.copy(md1)
        arcLengths = [ getCubicHermiteArcLength(nx[e], md1[e], nx[(e + 1)%elementsCount], md1[(e + 1)%elementsCount]) for e in range(elementsCount) ]
        for n in range(nodesCount):
            nm = n - 1
            if not fixAllDirections:
                # get mean of directions from point n to points (n - 1) and (n + 1)
                np = (n + 1)%nodesCount
                dirm = [ (nx[n ][c] - nx[nm][c]) for c in componentRange ]
                dirp = [ (nx[np][c] - nx[n ][c]) for c in componentRange ]
                # mean weighted by fraction towards that end, equivalent to harmonic mean
                arcLengthmp = arcLengths[nm] + arcLengths[n]
                wm = arcLengths[n ]/arcLengthmp
                wp = arcLengths[nm]/arcLengthmp
                md1[n] = [ (wm*dirm[c] + wp*dirp[c]) for c in componentRange ]
            if arithmeticMeanMagnitude:
                mag = 0.5*(arcLengths[nm] + arcLengths[n])
            else: # harmonicMeanMagnitude
                mag = 2.0/(1.0/arcLengths[nm] + 1.0/arcLengths[n])
            md1[n] = set_magnitude(md1[n], mag)
        if instrument:
            print('iter', iter + 1, md1)
        dtol = tol*sum(arcLengths)/len(arcLengths)
        for n in range(nodesCount):
            for c in componentRange:
                if math.fabs(md1[n][c] - lastmd1[n][c]) > dtol:
                    break
            else:
                continue
            break
        else:
            if instrument:
                print('smoothCubicHermiteDerivativesLoop converged after iter:',iter)
            return md1

    cmax = 0.0
    for n in range(nodesCount):
        for c in componentRange:
            cmax = max(cmax, math.fabs(md1[n][c] - lastmd1[n][c]))
    closeness = cmax / dtol
    print('smoothCubicHermiteDerivativesLoop max iters reached:', iter + 1, ', cmax = ', round(closeness, 2), '* TOL')
    return md1

def getDoubleCubicHermiteCurvesMidDerivative(ax, ad1, mx, bx, bd1):
    """
    Get derivative at centre of two cubic curves.
    :return: Derivative at mx to balance ax, ad1 with bx, bd1.
    """
    md1 = [ (bx[c] - ax[c]) for c in range(3) ]
    arcLengtha = computeCubicHermiteArcLength(ax, ad1, mx, md1, rescaleDerivatives = True)
    arcLengthb = computeCubicHermiteArcLength(mx, md1, bx, bd1, rescaleDerivatives = True)
    maga = magnitude(ad1)
    magb = magnitude(bd1)
    magm = arcLengtha + arcLengthb - 0.5*(maga + magb)
    return set_magnitude(md1, magm)

def sampleParameterAlongLine(lengthList, paramList, elementsCountOut):
    """
    Smooths derivative of parameter with linearly varying lengths, and
    samples smoothed parameter at equal distances along the lengths
    without including parameter as a component of coordinates.
    :param lengthList: List of length locations along a line.
    :param paramList: List of parameter values at length locations specified
    in lengthList.
    :param elementsCountOut: Number of output elements along length.
    :return sP, sdP: Parameter values and rate of change at each sampled point.
    """
    assert len(lengthList) == len(paramList), 'sampleParameterAlongLine.  Mismatched number of lengths and parameters'
    nodesCount = len(lengthList)

    md1 = []
    sP = []
    sdP = []

    # Find smoothed parameter derivatives
    # Middle
    for n in range(1, nodesCount - 1):
        # get mean of directions from point n to points (n - 1) and (n + 1)
        nm = n - 1
        np = n + 1
        dirm = paramList[n ] - paramList[nm]
        dirp = paramList[np] - paramList[n ]
        # mean weighted by fraction towards that end, equivalent to harmonic mean
        arcLengthm = lengthList[n ] - lengthList[nm]
        arcLengthn = lengthList[np] - lengthList[n ]
        arcLengthmp = arcLengthm + arcLengthn
        wm = arcLengthn/arcLengthmp
        wp = arcLengthm/arcLengthmp
        md1.append(wm*dirm + wp*dirp)

    # Start
    md1Start = interpolateLagrangeHermiteDerivative([paramList[0]], [paramList[1]], [md1[0]], 0.0)

    # End
    md1End = interpolateHermiteLagrangeDerivative([paramList[-2]], [md1[-1]], [paramList[-1]], 1.0)
    md1All = md1Start + md1 + md1End

    # Sample into equally spaced elements along line
    distance = 0.0
    e = 0
    totalLength = lengthList[-1]
    lengthPerElementOut = totalLength / elementsCountOut
    dLength = []

    for n in range(1, nodesCount):
        dLength.append(lengthList[n] - lengthList[n-1])
    dLength.append(dLength[-1])

    for eOut in range(elementsCountOut):
        while e < nodesCount - 1:
            if distance < lengthList[e + 1]:
                partDistance = distance - lengthList[e]
                xi = partDistance/(lengthList[e+1] - lengthList[e])
                p = interpolateCubicHermite([paramList[e]], [md1All[e]], [paramList[e+1]], [md1All[e+1]], xi)[0]
                dpdxi = interpolateCubicHermiteDerivative([paramList[e]], [md1All[e]], [paramList[e+1]], [md1All[e+1]], xi)[0]
                dxdxi = interpolateCubicHermiteDerivative([lengthList[e]], [dLength[e]], [lengthList[e+1]], [dLength[e+1]], xi)[0]
                dpdx = dpdxi*1.0/dxdxi
                dp = dpdx*lengthPerElementOut
                sP.append(p)
                sdP.append(dp)
                break
            e += 1
        distance += lengthPerElementOut

    # Last node
    sP.append(paramList[-1])
    dpdx = md1All[-1] * 1.0/dLength[-1]
    sdP.append(dpdx*lengthPerElementOut)

    return sP, sdP


def getCoordinatesRange(nx):
    """
    :param nx: list of coordinates.
    :return: xMin[], xMax[]
    """
    xMin = copy.copy(nx[0])
    xMax = copy.copy(nx[0])
    componentsCount = len(xMin)
    for x in nx:
        for c in range(componentsCount):
            s = x[c]
            if s < xMin[c]:
                xMin[c] = s
            elif s > xMax[c]:
                xMax[c] = s
    return xMin, xMax


def incrementXiOnLine(xi, dxi):
    """
    Increment xi, dxi limited to line element bounds on [0,1].
    :return: New xi, face number 1-2 or None if within boundary.
    Face 1 is xi==0.0, 2 is xi==1.0.
    """
    new_xi = xi + dxi
    if new_xi >= 1.0:
        return 1.0, 2
    elif new_xi < 0.0:
        return 0.0, 1
    return new_xi, None


def advanceCurveLocation(startLocation, dxi, elementsCount, loop=False, MAX_MAG_DXI=0.5):
    """
    Advance location by element delta xi, subject to maximum.
    :param startLocation: Start location on curve.
    :param dxi: Increment in xi.
    :param elementsCount: Number of elements in curve.
    :param loop: True if curve loops back to start.
    :param MAX_MAG_DXI: Maximum magnitude of dxi to keep increments reasonable for a cubic curve.
    :return: Advanced location, actual dxi, onBoundary (0/1=xi_0/2=xi_1)
    """
    startProportion = (startLocation[0] + startLocation[1]) / elementsCount
    adxi = dxi
    magDxi = abs(dxi)
    if magDxi > MAX_MAG_DXI:
        factor = MAX_MAG_DXI / magDxi
        adxi *= factor
    proportion = startProportion + adxi / elementsCount
    onBoundary = 0
    if loop:
        if proportion < 0.0:
            proportion += 1.0
        elif proportion >= 1.0:
            proportion -= 1.0
    else:
        if proportion < 0.0:
            proportion = 0.0
            onBoundary = 1
        elif proportion > 1.0:
            proportion = 1.0
            onBoundary = 2
        adxi = (proportion - startProportion) * elementsCount
    scaledProportion = proportion * elementsCount
    elementIndex = min(int(scaledProportion), elementsCount - 1)
    location = (elementIndex, scaledProportion - elementIndex)
    return location, adxi, onBoundary


def evaluateCoordinatesOnCurve(nx, nd1, location, loop=False, derivative=False):
    """
    :param nx: Coordinates along curve.
    :param nd1: Derivatives along curve.
    :param location: Location (element index, xi) to get coordinates for.
    :param loop: True if curve loops back to first point, False if not.
    :param derivative: Set to True to calculate and return derivative w.r.t. element xi.
    :return: If derivative is False: x; if derivative is True: x, d1.
    """
    e1 = location[0]
    e2 = 0 if (loop and (e1 == (len(nx) - 1))) else e1 + 1
    xi = location[1]
    x = interpolateCubicHermite(nx[e1], nd1[e1], nx[e2], nd1[e2], xi)
    if not derivative:
        return x
    d = interpolateCubicHermiteDerivative(nx[e1], nd1[e1], nx[e2], nd1[e2], xi)
    return x, d


def evaluateScalarOnCurve(sv, sd, location, loop=False, derivative=False):
    """
    Evaluate scalar interpolated along curve.
    :param sv: List of scalar values at nodes along curve.
    :param sd: List of scalar derivatives w.r.t. xi at nodes along curve.
    :param location: Location (element index, xi) to get scalar value for.
    :param loop: True if curve loops back to first point, False if not.
    :param derivative: Set to True to calculate and return derivative w.r.t. element xi.
    :return: If derivative is False: scalar s; if derivative is True: s, ds/dxi.
    """
    e1 = location[0]
    e2 = 0 if (loop and (e1 == (len(sv) - 1))) else e1 + 1
    xi = location[1]
    s = interpolateCubicHermite([sv[e1]], [sd[e1]], [sv[e2]], [sd[e2]], xi)[0]
    if not derivative:
        return s
    d = interpolateCubicHermiteDerivative([sv[e1]], [sd[e1]], [sv[e2]], [sd[e2]], xi)[0]
    return s, d


def isLocationOnCurveBoundary(location, elementsCount):
    """
    For non-loop curves, query if location is on one of the ends
    :param location: Curve location (element index, xi).
    :param elementsCount: Number of elements in curve.
    :return: True if on boundary.
    """
    XI_TOL = 1.0E-8
    e = location[0]
    xi = location[1]
    return ((e == 0) and (xi < XI_TOL)) or ((e == (elementsCount - 1)) and (xi > (1.0 - XI_TOL)))


def updateCurveLocationToFaceNumber(location, faceNumber, elementsCount, loop=False):
    """
    Determine curve location after crossing given face number, and either clamp to range if reached boundary,
    or loop around depending on mode.
    :param location: Location tuple (element index, xi).
    :param faceNumber: Face number 1 (xi==0.0) or 2 (xi==1.0).
    :param elementsCount: Number of elements in curve.
    :param loop: True if curve loops back to first point.
    :return: New location, onBoundary (True or False).
    """
    e = location[0]
    xi = location[1]
    onBoundary = False
    if faceNumber == 1:
        if e == 0:
            if loop:
                e = elementsCount - 1
                xi = 1.0
            else:
                onBoundary = True
        else:
            e -= 1
            xi = 1.0
    elif faceNumber == 2:
        if e == (elementsCount - 1):
            if loop:
                e = 0
                xi = 0.0
            else:
                onBoundary = True
        else:
            e += 1
            xi = 0.0
    return (e, xi), onBoundary


def getNearestParameterLocationOnCurve(nx, targetx, loop=False):
    """
    Get location of nx parameter nearest to targetx.
    :param nx: Coordinates along curve.
    :param targetx: Coordinates of point to find nearest to.
    :param loop: True if curve loops back to first point, False if not.
    :return: nearest parameter location tuple (element index, xi), nearest distance, or None, None if not found.
    """
    nCount = len(nx)
    assert nCount > 1
    nearest_distance = None
    nearest_n = None
    for n in range(nCount):
        distance = magnitude(sub(nx[n], targetx))
        if (nearest_distance is None) or (distance < nearest_distance):
            nearest_distance = distance
            nearest_n = n
    nearest_e = nearest_n
    xi = 0.0
    if not loop and (nearest_n == (nCount - 1)):
        nearest_e -= 1
        xi = 1.0
    return (nearest_e, xi), nearest_distance


def getNearestLocationOnCurve(nx, nd1, targetx, loop=False, startLocation=None, instrument=False):
    """
    Get location on a piecewise Hermite curve which is closest to target coordinates.
    Can be a local minimum depending on start location.
    :param nx: Coordinates along curve.
    :param nd1: Derivatives along curve.
    :param targetx: Coordinates to get nearest point on curve to.
    :param loop: True if curve loops back to first point, False if not.
    :param startLocation: Optional initial location (element index, xi) to search from.
    If not supplied, uses element location at the nearest node coordinates.
    :param instrument: Set to True to print debug messages.
    :return: nearest location tuple (element index, xi), nearest x.
    """
    if instrument:
        print("getNearestLocationOnCurve targetx", targetx)
    location = copy.copy(startLocation) if startLocation else getNearestParameterLocationOnCurve(nx, targetx, loop)[0]
    nodesCount = len(nx)
    assert nodesCount > 1
    elementsCount = nodesCount if loop else nodesCount - 1
    MAX_MAG_DXI = 0.5  # target/maximum magnitude of xi increment
    XI_TOL = 1.0E-7
    xMin, xMax = getCoordinatesRange(nx)
    xRange = [xMax[c] - xMin[c] for c in range(len(xMin))]
    MIN_CURVATURE = 0.1 / max(xRange)  # minimum to consider
    MAX_CURVATURE_FACTOR = 100.0
    x = None
    mag_adxi = -1
    it = MAX_ITERS = 100
    componentsCount = len(targetx)
    for it in range(MAX_ITERS):
        x, d = evaluateCoordinatesOnCurve(nx, nd1, location, loop, derivative=True)
        mag_d = magnitude(d)
        if mag_d == 0.0:
            print("getNearestLocationOnCurve:  Zero derivative at iteration", it + 1)
            break
        deltax = sub(targetx, x)
        norm_d = normalize(d)
        ut = dot(deltax, norm_d)  # tangential deltax, scalar
        dxi = ut / mag_d
        if instrument:
            print("iter", it + 1, "location", location, "deltax", deltax, "displacement", ut)
            print("    initial dxi", dxi)
        nm = location[0]
        np = (nm + 1) % nodesCount
        curvature, tangent, dTangent = getCubicHermiteCurvatureSimple(nx[nm], nd1[nm], nx[np], nd1[np], location[1])
        if curvature > MIN_CURVATURE:
            # get non-linear increment using radius of curvature
            radius = 1.0 / curvature
            jVector = normalize(tangent)
            if dxi < 0.0:
                jVector = [-d for d in jVector]
            if componentsCount == 3:
                iVector = normalize(cross(tangent, cross(tangent, dTangent)))
            else:
                iVector = [-jVector[1], jVector[0]]
            centre = sub(x, mult(iVector, radius))
            delta = sub(targetx, centre)
            dj = dot(delta, jVector)
            di = dot(delta, iVector)
            angle = math.atan2(dj, di)
            if (it < 10) and (abs(angle) > 0.1):  # radians ~ 6 degrees
                arcLength = radius * angle
                originalLength = abs(ut)
                curvatureFactor = arcLength / originalLength
            else:
                curvatureFactor = radius / di
                if curvatureFactor > MAX_CURVATURE_FACTOR:
                    curvatureFactor = MAX_CURVATURE_FACTOR
            dxi *= curvatureFactor
            if instrument:
                print("    curvature", curvature, "curvature factor", curvatureFactor, "angle",
                      math.degrees(angle), "degrees")
                print("    centre", centre, "delta", delta)
                print("    iVector", iVector, "di", di)
                print("    jVector", jVector, "dj", dj)
                print("    curved dxi", dxi)

        location, adxi, onBoundary = advanceCurveLocation(location, dxi, elementsCount, loop, MAX_MAG_DXI)
        mag_adxi = abs(adxi)
        if instrument:
            print("    final dxi", adxi)
        if mag_adxi < XI_TOL:
            if instrument:
                print("getNearestLocationOnCurve:  Converged in", it + 1, "iterations, dxi", mag_adxi)
            break
    else:
        print("getNearestLocationOnCurve:  Reached max iterations", it + 1, "closeness in xi", mag_adxi)
    return location, x


def getNearestLocationBetweenCurves(nx, nd1, ox, od1, nLoop=False, oLoop=False, startLocation=None, instrument=False):
    """
    Get the closest locations on two piecewise Hermite curves. Can be a local minimum depending on start location.
    :param nx: Coordinates along curve.
    :param nd1: Derivatives along curve.
    :param ox: Coordinates along other curve.
    :param od1: Derivatives along other curve.
    :param nLoop: True if curve loops back to first point, False if not.
    :param oLoop: True if other curve loops back to first point, False if not.
    :param startLocation: Optional initial location (element index, xi) to search from.
    If not supplied, uses element location at the nearest parameter location to any parameter location on other curve.
    :param instrument: Set to True to print debug messages.
    :return: Nearest/intersection location (element index, xi), other curve location (element index, xi),
    isIntersection (True/False).
    """
    if instrument:
        print("getNearestLocationBetweenCurves")
    nnCount = len(nx)
    assert nnCount > 1
    neCount = nnCount if nLoop else nnCount - 1
    onCount = len(ox)
    assert onCount > 1
    oeCount = onCount if oLoop else onCount - 1
    location = copy.copy(startLocation) if startLocation else None
    if not location:
        nearestDistance = None
        for targetx in ox:
            tmpLocation, tmpDistance = getNearestParameterLocationOnCurve(nx, targetx, nLoop)
            if (nearestDistance is None) or (tmpDistance < nearestDistance):
                nearestDistance = tmpDistance
                location = tmpLocation
    targetx = evaluateCoordinatesOnCurve(nx, nd1, location, nLoop)
    otherLocation = getNearestParameterLocationOnCurve(ox, targetx, oLoop)[0]
    MAX_MAG_DXI = 0.5  # target/maximum magnitude of xi increment
    XI_TOL = 1.0E-7
    # get max range for tolerances
    xMin, xMax = getCoordinatesRange(nx)
    xRange = [xMax[c] - xMin[c] for c in range(len(xMin))]
    MAX_SLOPE_FACTOR = 1000.0
    x_tol = 1.0E-6 * max(xRange)
    lastOnBoundary = False
    last_dxi = None
    for it in range(100):
        x, d = evaluateCoordinatesOnCurve(nx, nd1, location, nLoop, derivative=True)
        otherLocation = getNearestLocationOnCurve(ox, od1, x, oLoop, otherLocation)[0]
        onOtherBoundary = not oLoop and isLocationOnCurveBoundary(otherLocation, oeCount)
        other_x = evaluateCoordinatesOnCurve(ox, od1, otherLocation, oLoop)
        r = sub(other_x, x)
        mag_r = magnitude(r)
        if instrument:
            print("iter", it, "pos", location, "other", otherLocation, "mag_r", mag_r, "x", x)
        if mag_r < x_tol:
            # print("getNearestLocationBetweenCurves:  Found intersection: ", location, "on iter", iter + 1)
            return location, otherLocation, True
        n = normalize(cross(cross(d, r), d))
        r_dot_n = dot(r, n)
        if r_dot_n < 0:
            # flip normal to be towards other x
            n = [-s for s in n]
            r_dot_n = -r_dot_n
        rNormal = mult(n, r_dot_n)
        rTangent = sub(r, rNormal)
        mag_ri = magnitude(rTangent)
        mag_ro = magnitude(rNormal)
        # get tangential displacement u
        if onOtherBoundary:
            slope_factor = 1.0
        else:
            # add out-of-plane slope component
            if it < 10:
                slope_factor = mag_r * mag_r / (mag_ri * mag_ri)
            else:
                slope_factor = 1.0 + r_dot_n / mag_r  # wrong, but more reliable when far away
            if slope_factor > MAX_SLOPE_FACTOR:
                slope_factor = MAX_SLOPE_FACTOR
            if instrument:
                print("    slope_factor", slope_factor)
        u = mult(rTangent, slope_factor)
        # limit by curvature and distance to other_x
        nm = location[0]
        np = (nm + 1) % nnCount
        nCurvature = getCubicHermiteCurvatureSimple(nx[nm], nd1[nm], nx[np], nd1[np], location[1])[0]
        om = otherLocation[0]
        op = (om + 1) % onCount
        oCurvature = getCubicHermiteCurvatureSimple(ox[om], od1[om], ox[op], od1[op], otherLocation[1])[0]
        curvature = nCurvature + oCurvature
        uNormal = sub(r, u)
        un = magnitude(uNormal)
        curvatureFactor = 1.0 / (un * curvature + 1.0)
        mag_u = magnitude(u) * curvatureFactor
        # never go further than parallel, based on curvature from initial angle
        parallelFactor = 1.0
        if curvature > 0.0:
            max_u = math.atan(mag_ri / mag_ro) / curvature
            if mag_u > max_u:
                parallelFactor = max_u / mag_u
                mag_u = max_u
        if instrument:
            print("--> curvature", curvature, "curvature factor", curvatureFactor, "parallel factor", parallelFactor)
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
            dxi *= osc_factor
            mag_dxi *= osc_factor
            if instrument:
                print("    osc_factor", osc_factor, "dxi", dxi)
        if (it % 20) == 19:
            MAX_MAG_DXI *= 0.5
            if instrument:
                print("    reduce MAX_MAG_DXI to", MAX_MAG_DXI)
        last_dxi = dxi
        bxi, faceNumber = incrementXiOnLine(location[1], dxi)
        location = (location[0], bxi)
        if faceNumber:
            location, onBoundary = updateCurveLocationToFaceNumber(location, faceNumber, neCount, nLoop)
            if onBoundary and lastOnBoundary:
                # print("getNearestLocationBetweenCurves:  Found nearest on boundary in", it + 1, "iterations")
                break
            lastOnBoundary = onBoundary
        else:
            lastOnBoundary = False
        if mag_dxi < XI_TOL:
            if instrument:
                print("getNearestLocationBetweenCurves:  Found nearest in", it + 1, "iterations, dxi", mag_dxi)
            break
    else:
        print('getNearestLocationBetweenCurves did not converge:  Reached max iterations', it + 1,
              'closeness in xi', mag_dxi)
    return location, otherLocation, False

def getCurvaturesAlongCurve(cx, cd, radialVectors, loop=False):
    """
    Calculate curvatures for points lying along a curve.
    :param cx: coordinates on curve
    :param cd: derivative of coordinates on curve
    :param radialVectors: radial direction, assumed normal to curve tangent at coordinate
    :param loop: True if curve is a closed loop
    :return: curvatures along coordinates on curve
    """
    curvatures = []
    cCount = len(cx)
    for c in range(cCount):
        kappa = None
        if (c > 0) or loop:
            cm = c - 1
            kappa = getCubicHermiteCurvature(cx[cm], cd[cm], cx[c], cd[c], radialVectors[c], 1.0)
        if (c < (cCount - 1)) or loop:
            cp = c + 1 - cCount
            kappap = getCubicHermiteCurvature(cx[c], cd[c], cx[cp], cd[cp], radialVectors[c], 0.0)
            if kappa is None:
                kappa = kappap
            else:
                kappa = 0.5 * (kappa + kappap)
        curvatures.append(kappa)
    return curvatures


def get_curve_from_points(px, maximum_element_length=None, number_of_elements=None):
    """
    Get approximate 1-D Hermite curve of equal-sized elements by sampling a series of point coordinates.
    :param px: List of point coordinates from one end of curve to the other. Each entry in list is a list of
    coordinate components.
    :param maximum_element_length: Target maximum length of elements or None to use a fixed number.
    :param number_of_elements: Number of elements to fit, or None to use a target length.
    :return: cx, cd1 (lists of hermite coordinates and derivatives)
    """
    points_count = len(px)
    assert points_count > 1
    assert maximum_element_length or number_of_elements
    assert (((maximum_element_length is None) or (maximum_element_length > 0.0)) or
            ((number_of_elements is None) or (number_of_elements > 0)))
    # get lengths from distances between consecutive points
    total_length = 0.0
    lengths = []
    for i in range(points_count - 1):
        length = magnitude(sub(px[i + 1], px[i]))
        lengths.append(length)
        total_length += length
    assert total_length > 0.0
    elements_count = number_of_elements if number_of_elements else math.ceil(total_length / maximum_element_length)
    if elements_count < 2:
        elements_count = 2  # start with 2 so at least one internal sample to get a better initial shape
    # get half range of lengths to average coordinates over
    delta_length = total_length / (4.0 * elements_count)
    nx = [copy.copy(px[0])]
    components_count = len(px[0])
    zero = [0.0] * components_count
    nd1 = [zero]
    i = 0
    sum_length = 0.0
    for n in range(1, elements_count):
        middle_length = total_length * (n / elements_count)
        start_length = middle_length - delta_length
        end_length = middle_length + delta_length
        while sum_length < start_length:
            sum_length += lengths[i]
            i += 1
        if end_length < sum_length:
            # simple interpolation within one linear segment
            wt0 = (sum_length - middle_length) / lengths[i - 1]
            wt1 = 1.0 - wt0
            x = add(mult(px[i - 1], wt0), mult(px[i], wt1))
        else:
            # weighted sum over several linear segments including part start and part end lengths
            part_length = sum_length - start_length
            wt0 = part_length * 0.5 * part_length / lengths[i - 1]
            wt1 = part_length - wt0
            sum_x = add(mult(px[i - 1], wt0), mult(px[i], wt1))
            while True:
                sum_length += lengths[i]
                i += 1
                segment_length = lengths[i - 1]
                if sum_length < end_length:
                    wtx = mult(add(px[i - 1], px[i]), 0.5 * segment_length)
                    sum_x = add(sum_x, wtx)
                else:
                    part_length = end_length - (sum_length - segment_length)
                    wt1 = part_length * 0.5 * part_length / segment_length
                    wt0 = part_length - wt1
                    wtx = add(mult(px[i - 1], wt0), mult(px[i], wt1))
                    sum_x = add(sum_x, wtx)
                    break
            x = div(sum_x, 2.0 * delta_length)
        nx.append(x)
        nd1.append(zero)
    nx.append(copy.copy(px[-1]))
    nd1.append(zero)
    # smooth with harmonic mean, get the length and resample to the desired number of even-sized elements
    nd1 = smoothCubicHermiteDerivativesLine(nx, nd1, magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
    curve_length = getCubicHermiteCurvesLength(nx, nd1)
    elements_count = number_of_elements if number_of_elements else math.ceil(curve_length / maximum_element_length)
    cx, cd1 = sampleCubicHermiteCurvesSmooth(nx, nd1, elements_count)[0:2]
    return cx, cd1


def track_curve_side_direction(cx, cd1, start_direction, start_location, end_location, forward=True):
    """
    Track side direction from start to end location on curve assuming no twisting.
    :param cx: Curve coordinate parameters.
    :param cd1: Curve derivative parameters.
    :param start_direction: Unit vector orthogonal to curve at start location.
    :param start_location: Curve location (element index, xi) to track from.
    :param end_location: Curve location (element index, xi) to track to.
    :param forward: True to track forward (end >= start), False to track backward (end <= start).
    :return: normalized dir1, dir2, dir3 (updated side direction) at end location.
    """
    assert (forward and (end_location >= start_location)) or ((not forward) and (end_location <= start_location))
    xi_increment = 0.1
    last_element_index = len(cx) - 2
    location = start_location
    direction = start_direction
    tracking = True
    while tracking:
        element_index = location[0]
        xi = location[1]
        if forward:
            xi += xi_increment
            if xi >= 1.0:
                if element_index < last_element_index:
                    element_index += 1
                    xi -= 1.0
                else:
                    xi = 1.0
                    tracking = False
            location = (element_index, xi)
            if location > end_location:
                location = end_location
                tracking = False
        else:  # reverse
            xi -= xi_increment
            if xi < 0.0:
                if element_index > 0:
                    element_index -= 1
                    xi += 1.0
                else:
                    xi = 0.0
                    tracking = False
            location = (element_index, xi)
            if location < end_location:
                location = end_location
                tracking = False
        n1 = location[0]
        n2 = location[0] + 1
        xi = location[1]
        d1 = interpolateCubicHermiteDerivative(cx[n1], cd1[n1], cx[n2], cd1[n2], xi)
        d2 = cross(direction, d1)
        direction = normalize(cross(d1, d2))

    return normalize(d1), normalize(d2), direction


def linearlyInterpolateVectors(u, v, xi, magnitudeScalingMode=DerivativeScalingMode.ARITHMETIC_MEAN):
    """
    Linearly interpolate two 3-component vectors less than 180 degrees apart to get an in-between vector
    rotated from direction of u to direction of v proportionatly to xi, with magnitude linearly interpolated by xi.
    :param u: First vector.
    :param v: Second vector from 0 to less than 180 degrees away from u.
    :param xi: Proportion from u to v.
    :param magnitudeScalingMode: A value from enum DerivativeScalingMode specifying
    expression used to get interpolated vector magnitude from magnitudes of u and v.
    :return: Interpolated vector.
    """
    mag_u = magnitude(u)
    mag_v = magnitude(v)
    if magnitudeScalingMode==DerivativeScalingMode.HARMONIC_MEAN:
        if (mag_u == 0.0) or (mag_v == 0):
            mag = 0
        else:
            mag = 1.0 / (((1.0 - xi) / mag_u) + (xi / mag_v))
    else:  # magnitudeScalingMode==erivativeScalingMode.ARITHMETIC_MEAN:
        mag = (1.0 - xi) * mag_u + xi * mag_v
    dirn_u = div(u, mag_u)
    dirn_v = div(v, mag_v)
    cos_theta = dot(dirn_u, dirn_v)
    if cos_theta > 0.999999:
        dirn = dirn_u
    else:
        theta = math.acos(cos_theta)
        phi = xi * theta
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        axis1 = dirn_u
        cross_u_v = cross(dirn_u, dirn_v)
        if magnitude(cross_u_v) == 0.0:
            return [0.0, 0.0, 0.0]
        axis3 = normalize(cross_u_v)
        axis2 = cross(axis3, dirn_u)
        dirn = add(mult(axis1, cos_phi), mult(axis2, sin_phi))
    return set_magnitude(dirn, mag)


def sampleHermiteCurve(start_x, start_d1, start_d2, end_x, end_d1, end_d2, elements_count,
                       start_weight=None, end_weight=None, overweighting=1.0, end_transition=False):
    """
    Samples a hermite curve from initial points and directions. Also interpolates side derivatives.
    :param start_x: start coordinate.
    :param start_d1: start direction.
    :param start_d2: optional start side direction. Must be supplied if end_d2 is supplied.
    :param end_x: end coordinate.
    :param end_d1: end direction, or None to estimate from start_d1.
    Must be supplied with end_transition.
    :param end_d2: optional end side direction, or None to use normal to end direction closest to start_d2.
    :param elements_count: Number of elements to sample.
    :param start_weight, end_weight: Optional relative weights for start/end d1.
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
            normal = normalize(cross(pd1[-1], start_d2))
            end_d2 = cross(normal, pd1[-1])
        pd2 = [start_d2]
        if end_transition:
            # adjust xi scale to get propor proportion when magnitude of end_d1 differs from regular derivatives
            reg_d1_mag = magnitude(pd1[-2])
            end_d1_mag = magnitude(end_d1)
            xi_scale = reg_d1_mag / ((elements_count - 0.5) * reg_d1_mag + 0.5 * end_d1_mag)
        else:
            xi_scale = 1.0 / elements_count
        for n in range(1, elements_count):
            xi = n * xi_scale
            d2 = linearlyInterpolateVectors(start_d2, end_d2, xi)
            pd2.append(d2)
        pd2.append(end_d2)
        return px, pd1, pd2
    return px, pd1

def get_nway_point(sx, sd, element_counts, sample_curve, move_x_to_surface=None,
                    prescribed_x_nway=None, nway_d_factor=0.6):
    """
    Get 3-way (surface) or 4-way (volume) location and derivatives.
    :param sx: List of N side coordinates.
    :param sd: List of N side inward derivatives.
    :param element_counts: List of N number of elements from side N to n-way point.
    :param sample_curve: Curve sampling function with signature of sampleHermiteCurve.
    :param move_x_to_surface: Optional function take argument x and returning nearest coordinate on surface.
    :param prescribed_x_nway: Optional known 3-way point coordinates to determine derivatives to.
    :param nway_d_factor: Value, normally from 0.5 to 1.0 giving n-way derivative magnitude as a proportion
    of the minimum regular magnitude sampled to the n-way point. This reflects that distances from the mid-side
    of a triangle to the centre are shorter, so the derivative in the middle must be smaller.
    :return: x_nway, d_nway[N]
    """
    n_count = len(sx)
    assert n_count == 3 or n_count == 4
    permutations = [[0, 1], [1, 2], [2, 0]] if n_count == 3 else [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]]
    if prescribed_x_nway:
        x_nway = prescribed_x_nway
    else:
        sum_weight = 0.0
        sum_weighted_x = [0.0, 0.0, 0.0]
        for n1, n2 in permutations:
            px, pd = sample_curve(
                sx[n1], sd[n1], None, sx[n2], [-d for d in sd[n2]], None, element_counts[n1] + element_counts[n2],
                start_weight=element_counts[n1], end_weight=element_counts[n2])
            x = px[element_counts[n1]]
            length1 = getCubicHermiteCurvesLength(px[:element_counts[n1] + 1], pd[:element_counts[n1] + 1])
            length2 = getCubicHermiteCurvesLength(px[element_counts[n1]:], pd[element_counts[n1]:])
            weight = 1.0 / length1 + 1.0 / length2
            sum_weight += weight
            sum_weighted_x = add(sum_weighted_x, mult(x, weight))
        x_nway = div(sum_weighted_x, sum_weight)
        if move_x_to_surface:
            x_nway = move_x_to_surface(x_nway)

    # sample with hermite-lagrange interpolation from sides to 3-way point to get end directions
    directions = []
    for n in range(n_count):
        px, pd = sample_curve(sx[n], sd[n], None, x_nway, None, None, element_counts[n])
        directions.append(normalize(pd[-1]))
    angle_120 = math.pi / 1.5
    if n_count == 3:
        # make directions 120 degrees apart, around the normal
        normal = [0.0, 0.0, 0.0]
        for n1, n2 in permutations:
            np = (n + 1) % n_count
            # this gives the most weight to pairs 90 degrees apart and no weight to collinear pairs
            normal = add(normal, cross(directions[n1], directions[n2]))
        normal = normalize(normal)
        axis2 = normalize(cross(normal, directions[0]))
        axis1 = cross(axis2, normal)
        # get angles around normal, starting at axis1
        ideal_angle = 0.0
        delta_angle = 0.0
        for i in range(1, 3):
            actual_angle = math.atan2(dot(directions[i], axis2), dot(directions[i], axis1))
            if actual_angle < 0.0:
                actual_angle += 2.0 * math.pi
            ideal_angle += angle_120
            delta_angle += (actual_angle - ideal_angle)
        delta_angle /= 3.0
        new_directions = []
        angle = delta_angle
        for i in range(3):
            new_directions.append(add(mult(axis1, math.cos(angle)), mult(axis2, math.sin(angle))))
            angle += angle_120
    else:
        cos_tetra_angle = -1.0 / 3.0
        tetra_angle = math.acos(cos_tetra_angle)
        sin_tetra_angle = math.sin(tetra_angle)
        axis1 = directions[3]
        axis3 = normalize(cross(axis1, directions[0]))
        axis2 = cross(axis3, axis1)
        reverse_1__3 = mult(axis1, cos_tetra_angle)
        ideal_directions = [add(reverse_1__3, mult(axis2, sin_tetra_angle))]
        cos_120 = math.cos(angle_120)
        sin_120 = math.sin(angle_120)
        rotation_matrix = axis_angle_to_rotation_matrix(axis1, -angle_120)
        for i in range(1, 3):
            ideal_directions.append(matrix_vector_mult(rotation_matrix, ideal_directions[-1]))
        ideal_directions.append(axis1)
        def rotation_objective(euler_angles, *args):
            directions, ideal_directions = args
            rotation_matrix = euler_to_rotation_matrix(euler_angles)
            return sum(1.0 - dot(matrix_vector_mult(rotation_matrix, ideal_directions[i]), directions[i])
                       for i in range(3))
        res = minimize(rotation_objective, [0.0] * 3, args=(directions, ideal_directions),
                       # method='Nelder-Mead', tol=1.0E-9)  # options={'xatol': 1.0E-7, 'disp': True})
                       method = 'Powell', tol=1.0E-10)  # options={'xtol': 1.0E-10, 'disp': True})
        if res.success:
            euler_angles = res.x
            rotation_matrix = euler_to_rotation_matrix(euler_angles)
            new_directions = [matrix_vector_mult(rotation_matrix, direction) for direction in ideal_directions]
        else:
            print("get_nway_point.  Could not optimise 4-way directions")
            new_directions = ideal_directions

    # smooth sample from sides to n-way point with new directions and weights to get derivative magnitudes
    d_mags = []
    min_weight = 1.0 # GRC review, remove?
    for n in range(n_count):
        px, pd = sample_curve(
            sx[n], sd[n], None, x_nway, new_directions[n], None, element_counts[n],
            start_weight=element_counts[0] + min_weight, end_weight=1.0 + min_weight)
        d_mags.append(magnitude(pd[-1]))
    d_mag = nway_d_factor * min(d_mags)
    d_nway = [set_magnitude(direction, d_mag) for direction in new_directions]
    return x_nway, d_nway
