'''
Utility functions for geometry.
'''

from __future__ import division

import copy
import math

from cmlibs.maths.vectorops import (
    cross, distance, dot, magnitude, mult, normalize, projection, set_magnitude, rejection)
from scaffoldmaker.utils.interpolation import (
    computeCubicHermiteDerivativeScaling, computeHermiteLagrangeDerivativeScaling, getCubicHermiteArcLength,
    interpolateHermiteLagrangeDerivative, linearlyInterpolateVectors, sampleCubicHermiteCurves,
    sampleCubicHermiteCurvesSmooth)
from scaffoldmaker.utils.tracksurface import calculate_surface_delta_xi


def getApproximateEllipsePerimeter(a, b):
    '''
    Get perimeter of ellipse using Ramanujan II approximation.
    :param a: Major axis length.
    :param b: Minor axis length.
    :return: Perimeter length.
    '''
    h = ((a-b)/(a+b))**2
    return math.pi*(a + b)*(1.0 + 3.0*h/(10.0 + math.sqrt(4.0 - 3.0*h)))

def getEllipseAngleFromVector(a, b, x, y):
    '''
    Given an ellipse with major and minor axes a and b cented at (0.0, 0.0),
    get angle in radians in direction of vector (dx, dy)
    :param a: Major axis length.
    :param b: Minor axis length.
    :param x, y: Non-zero vector direction
    :return: Angle in radians.
    '''
    # Compute theta such that
    # r*x = a*cos_theta
    # r*y = b*sin_theta
    # hence:
    # y/x = b/a * tan(theta)
    # (a*y)/(b*x) = tan(theta)
    return math.atan2(a*y, b*x)

def getEllipseArcLength(a, b, angle1Radians, angle2Radians, method='line segment'):
    '''
    Calculates perimeter distance between two angles, by integration if method is integrate by summing line segments at regular angles.
    :param a: Major axis length (On x, 0 / PI).
    :param b: Minor axis length.(On y, PI/2, 3PI/2).
    :param angle1Radians: First angle anticlockwise from major axis.
    :param angle2Radians: Second angle anticlockwise from major axis.
    :return: Perimeter length, positive if anticlockwise, otherwise negative.
    '''
    angle1 = min(angle1Radians, angle2Radians)
    angle2 = max(angle1Radians, angle2Radians)

    if method == 'integrate':
        from scipy.integrate import quad
        import numpy as np

        def integrand(t, m):
            return np.sqrt(1 - m * np.sin(t) ** 2)

        m = 1 - (a/b) ** 2
        result = quad(integrand, angle1Radians, angle2Radians, args=(m,))
        error = result[1]
        length = b * result[0]
        return length
    elif method == 'line segment':
        # Max 100 segments around ellipse
        segmentCount = int(math.ceil(50*(angle2-angle1)/math.pi))
        length = 0.0
        lastX = None
        for i in range(segmentCount + 1):
            r = i/segmentCount
            angle = r*angle1 + (1.0 - r)*angle2
            x = ( a*math.cos(angle), b*math.sin(angle) )
            if i > 0:
                delta = math.sqrt((x[0] - lastX[0])*(x[0] - lastX[0]) + (x[1] - lastX[1])*(x[1] - lastX[1]))
                length += delta
            lastX = x
        if angle1Radians < angle2Radians:
            return length
        else:
            return -length
    else:
        assert(False), 'Method is not implemented'

def updateEllipseAngleByArcLength(a, b, inAngleRadians, arcLength, tol=1.0E-4, method=None):
    '''
    Update angle around ellipse to subtend arcLength around the perimeter.
    Iterates using Newton's method.
    :param inAngleRadians: Initial angle anticlockwise from major axis.
    :param arcLength: Arc length to traverse. Positive=anticlockwise, negative=clockwise.
    :param a: Major axis length (On x, 0 / PI).
    :param b: Minor axis length.(On y, PI/2, 3PI/2).
    :param tol: Tolerance used for length tolerance.
    :return: New angle, in radians.
    '''
    if method == 'Newton':
        def func(angle):
            return getEllipseArcLength(a, b, inAngleRadians, angle, method='integrate') - arcLength
        from scipy.optimize import newton
        angle = newton(func, inAngleRadians)
    else:
        angle = inAngleRadians
        lengthMoved = 0.0
        lengthTol = (a + b)*tol  # broader tolerance due to reliance on inexact getEllipseArcLength()
        counter=0
        #print('updateEllipseAngleByArcLength', a, b, 'inAngleRadians', inAngleRadians, ', arcLength', arcLength)
        while math.fabs(arcLength - lengthMoved) > lengthTol:
            angleOld = angle
            t = ( -a*math.sin(angle), b*math.cos(angle) )
            dlength_dangle = math.sqrt(t[0]*t[0] + t[1]*t[1])
            angle += (arcLength - lengthMoved)/dlength_dangle
            if counter >= 100:
                angle=(angle+angleOld)/2
            lengthMoved = getEllipseArcLength(a, b, inAngleRadians, angle)
            counter+=1
            #print('lengthMoved', lengthMoved)
        #print('updateEllipseAngleByArcLength a', a, 'b', b, ', angle', inAngleRadians, ', arcLength', arcLength, ' -> ', angle)
    return angle

def getEllipseRadiansToX(ax, bx, dx, initialTheta):
    '''
    Iteratively compute theta in ax*cos(theta) + bx*sin(theta) = dx
    from initialTheta. Uses Newton's method.
    :param ax: x component of major axis.
    :param bx: x component of minor axis.
    :param dx: x distance from centre of ellipse.
    :param initialTheta: Initial value of theta needed since multi-valued.
    :return: theta in radians
    '''
    theta = initialTheta
    iters = 0
    fTol = math.sqrt(ax*ax + bx*bx)*1.0E-10
    while True:
        cosAngle = math.cos(theta)
        sinAngle = math.sin(theta)
        f = ax*cosAngle + bx*sinAngle - dx
        if math.fabs(f) < fTol:
            break;
        df = -ax*sinAngle + bx*cosAngle
        #print(iters, '. theta', theta, 'f', f,'df',df,'-->',theta - f/df)
        theta -= f/df
        iters += 1
        if iters == 100:
            print('getEllipseRadiansToX: did not converge!')
            break
    return theta


def getEllipsePointAtTrueAngle(a, b, angle_radians, origin=[0.0, 0.0]):
    """
    Get coordinates of intersection point of ellipse centred at [0.0, 0.0] with line radiating from origin at an angle.
    :param a: x/major axis length.
    :param b: y/minor axis length.
    :param angle_radians: Angle in radians starting at x axis, increasing towards y axis.
    :param origin: Origin angles radiate from. Must be inside ellipsoid. Default is ellipsoid centre [0.0, 0.0].
    :return: [x, y] on ellipse.
    """
    # ellipse equation: x ** 2 / a ** 2 + y ** 2 / b ** 2 - 1 = 0
    aa = a * a
    bb = b * b
    assert (origin[0] * origin[0] / aa + (origin[1] * origin[1]) / bb) < 1.0
    # normal to line direction:
    # line equation: ni * x + nj * y - k = 0
    ni = math.sin(angle_radians)
    nj = -math.cos(angle_radians)
    k = dot([ni, nj], origin)
    ii = ni * ni
    jj = nj * nj
    kk = k * k
    if math.fabs(nj) > math.fabs(ni):
        # substitute y and solve for x with quadratic equation
        jj_bb = jj * bb
        qa = 1.0 / aa + ii / jj_bb
        qb = -2.0 * k * ni / jj_bb
        qc = kk / jj_bb - 1.0
        det = qb * qb - 4.0 * qa * qc
        sqrt_det = math.sqrt(det)
        x1 = (-qb + sqrt_det) / (2.0 * qa)
        x2 = (-qb - sqrt_det) / (2.0 * qa)
        if x1 * nj < 0.0:
            x = x1
        else:
            x = x2
        y = (k - ni * x) / nj
    else:
        # substitute x and solve for y with quadratic equation
        ii_aa = ii * aa
        qa = 1.0 / bb + jj / ii_aa
        qb = -2.0 * k * nj / ii_aa
        qc = kk / ii_aa - 1.0
        det = qb * qb - 4.0 * qa * qc
        sqrt_det = math.sqrt(det)
        y1 = (-qb + sqrt_det) / (2.0 * qa)
        y2 = (-qb - sqrt_det) / (2.0 * qa)
        if y1 * ni > 0.0:
            y = y1
        else:
            y = y2
        x = (k - nj * y) / ni
    return [x, y]


def getEllipseTangentAtPoint(a, b, x):
    """
    Get unit tangent direction on ellipse centred at origin at giving point on it.
    :param a: x/major axis length.
    :param b: y/minor axis length.
    :param x: Coordinates on ellipse, list of 2 real values.
    :return: [dx, dy] (in anticlockwise direction moving x to y axis and around) unit scale.
    """
    return normalize([-x[1] / (b * b), x[0] / (a * a)])


def sampleEllipsePoints(centre: list, majorAxis: list, minorAxis: list, angle1Radians: float, angle2Radians: float,
                        elementCount: int):
    """
    Sample evenly spaced points around ellipse from start to end angle.
    :param centre: Centre point of ellipse.
    :param majorAxis: Major axis vector, at angle 0PI.
    :param minorAxis: Minor axis vector, at angle PI/2.
    :param angle1Radians: First angle in radians from major axis towards minor axis.
    :param angle2Radians: Second angle in radians from major axis towards minor axis.
    :param elementCount: Number of elements to sample.
    :return: px[], dp1[]
    """
    a = magnitude(majorAxis)
    b = magnitude(minorAxis)
    TOL = max(a, b) * 1.0E-6
    totalArclength = getEllipseArcLength(a, b, angle1Radians, angle2Radians, method='integrate')
    elementArcLength = totalArclength / elementCount
    px = []
    pd1 = []
    radians = angle1Radians
    for n in range(elementCount + 1):
        cosRadians = math.cos(radians)
        sinRadians = math.sin(radians)
        x = [(centre[c] + cosRadians * majorAxis[c] + sinRadians * minorAxis[c]) for c in range(3)]
        px.append(x)
        d1 = [(-sinRadians * majorAxis[c] + cosRadians * minorAxis[c]) for c in range(3)]
        scale = elementArcLength/magnitude(d1)
        d1 = mult(d1, scale)
        pd1.append(d1)
        radians = updateEllipseAngleByArcLength(a, b, radians, elementArcLength, TOL, method="Newton")
    return px, pd1


def createCirclePoints(cx, axis1, axis2, elementsCountAround, startRadians = 0.0):
    '''
    Create circular ring of points centred at cx, from axis1 around through axis2.
    Assumes axis1 and axis2 are orthogonal and equal magnitude.
    Dimension 3 only.
    :param cx: centre
    :param axis1:  Vector from cx to inside at zero angle
    :param axis2:  Vector from cx to inside at 90 degree angle.
    :param elementsCountAround: Number of elements around.
    :return: lists px, pd1
    '''
    px = []
    pd1 = []
    radiansPerElementAround = 2.0*math.pi/elementsCountAround
    radiansAround = startRadians
    for n in range(elementsCountAround):
        cosRadiansAround = math.cos(radiansAround)
        sinRadiansAround = math.sin(radiansAround)
        px.append([ (cx[c] + cosRadiansAround*axis1[c] + sinRadiansAround*axis2[c]) for c in range(3) ])
        pd1.append([ radiansPerElementAround*(-sinRadiansAround*axis1[c] + cosRadiansAround*axis2[c]) for c in range(3) ])
        radiansAround += radiansPerElementAround
    return px, pd1


def createEllipsoidPoints(centre, poleAxis, sideAxis, elementsCountAround, elementsCountUp, height):
    '''
    Generate a set of points and derivatives for circle of revolution of an ellipse
    starting at pole poleAxis from centre.
    :param centre: Centre of full ellipsoid.
    :param poleAxis: Vector in direction of starting pole, magnitude is ellipse axis length.
    :param sideAxis: Vector normal to poleAxis, magnitude is ellipse side axis length.
    :param height: Height of arc of ellipsoid from starting pole along poleAxis.
    :return: Lists nx, nd1, nd2. Ordered fastest around, starting at pole. Suitable for passing to TrackSurface.
    '''
    nx  = []
    nd1 = []
    nd2 = []
    magPoleAxis = magnitude(poleAxis)
    magSideAxis = magnitude(sideAxis)
    unitPoleAxis = normalize(poleAxis)
    unitSideAxis1 = normalize(sideAxis)
    unitSideAxis2 = normalize(cross(sideAxis, poleAxis))
    useHeight = min(max(0.0, height), 2.0*magPoleAxis)
    totalRadiansUp = getEllipseRadiansToX(magPoleAxis, 0.0, magPoleAxis - useHeight, initialTheta = 0.5*math.pi*useHeight/magPoleAxis)
    radiansUp = 0.0
    lengthUp = getEllipseArcLength(magPoleAxis, magSideAxis, radiansUp, totalRadiansUp)
    elementLengthUp = lengthUp/elementsCountUp
    radiansPerElementAround = 2.0*math.pi/elementsCountAround
    for n2 in range(elementsCountUp + 1):
        cosRadiansUp = math.cos(radiansUp)
        sinRadiansUp = math.sin(radiansUp)
        radius = sinRadiansUp*magSideAxis
        d2r, d2z = set_magnitude([ cosRadiansUp*magSideAxis, sinRadiansUp*magPoleAxis ], elementLengthUp)
        cx = [ (centre[c] + cosRadiansUp*poleAxis[c]) for c in range(3) ]
        elementLengthAround = radius*radiansPerElementAround
        radiansAround = 0.0
        for n in range(elementsCountAround):
            cosRadiansAround = math.cos(radiansAround)
            sinRadiansAround = math.sin(radiansAround)
            nx .append([ (cx[c] + radius*(cosRadiansAround*unitSideAxis1[c] + sinRadiansAround*unitSideAxis2[c])) for c in range(3) ])
            nd1.append([ (elementLengthAround*(-sinRadiansAround*unitSideAxis1[c] + cosRadiansAround*unitSideAxis2[c])) for c in range(3) ])
            nd2.append([ (d2r*(cosRadiansAround*unitSideAxis1[c] + sinRadiansAround*unitSideAxis2[c]) - d2z*unitPoleAxis[c]) for c in range(3) ])
            radiansAround += radiansPerElementAround
        radiansUp = updateEllipseAngleByArcLength(magPoleAxis, magSideAxis, radiansUp, elementLengthUp)
    return nx, nd1, nd2


def getEllipsoidPolarCoordinatesTangents(a: float, b: float, c: float, u: float, v: float):
    """
    Get rate of change of x, y, z with u and v at current u, v.
    Given parametric equation for ellipsoid:
    x = a*cos(u)*sin(v)
    y = b*sin(u)*sin(v)
    z = c*-cos(v)
    Note cross(dx/du, dx/dv) gives an outward normal.
    Fails at apexes.
    :param a: Axis length in x direction.
    :param b: Axis length in y direction.
    :param c: Axis length in z direction.
    :param u: Polar coordinate (radians from positive x towards y) from -pi to + pi
    :param v: Polar coordinate (radians from negative z upwards) from 0 to pi
    :return: 3 lists (x, y, z), d(x, y, z)/du d(x, y, z)/dv.
    """
    cos_u = math.cos(u)
    sin_u = math.sin(u)
    cos_v = math.cos(v)
    sin_v = math.sin(v)
    x = [
        a * cos_u * sin_v,
        b * sin_u * sin_v,
        c * -cos_v
    ]
    dx_du = [
        a * -sin_u * sin_v,
        b * cos_u * sin_v,
        0.0
    ]
    dx_dv = [
        a * cos_u * cos_v,
        b * sin_u * cos_v,
        c * sin_v
    ]
    return x, dx_du, dx_dv


def getEllipsoidPolarCoordinatesFromPosition(a: float, b: float, c: float, pos: list):
    """
    Convert position in x, y, z to polar coordinates u, v at nearest location on ellipsoid centred at origin.
    Given parametric equation for ellipsoid:
    x = a*cos(u)*sin(v)
    y = b*sin(u)*sin(v)
    z = c*cos(v)
    Fails at apex.
    :param a: Axis length in x direction.
    :param b: Axis length in y direction.
    :param c: Axis length in z direction.
    :param pos: Position of points, list of 3 coordinates in x, y, z.
    :return: Polar coordinates u, v in radians.
    """
    # initial guess
    rx = pos[0] / a
    ry = pos[1] / b
    rz = pos[2] / c
    u = math.atan2(ry, rx)
    v = math.atan2(math.sqrt(rx*rx + ry*ry), rz)
    # move along tangents
    TOL = 1.0E-6
    iters = 0
    while True:
        iters += 1
        x, dx_du, dx_dv = getEllipsoidPolarCoordinatesTangents(a, b, c, u, v)
        deltax = [pos[c] - x[c] for c in range(3)]
        du, dv = calculate_surface_delta_xi(dx_du, dx_dv, deltax)
        u += du
        v += dv
        if (math.fabs(du) < TOL) and (math.fabs(dv) < TOL):
            break
        if iters == 100:
            print('getEllipsoidPolarCoordinatesFromPosition: did not converge!')
            break
    return u, v


def getEllipsoidPlaneA(a: float, b: float, c: float, midx, majorx):
    """
    Get ellipse function for intersection of ellipsoid and plane where the
    plane normal has no component in the x/a direction.
    Returned minorAxis[3] will be in the +x direction from centre.
    :param a: Axis length in x direction.
    :param b: Axis length in y direction.
    :param c: Axis length in z direction.
    :param midx: Point inside ellipsoid on plane at z = 0.
    :param majorx: Point on surface of ellipsoid at z = 0.
    :return: centre[3], majorAxis[3], minorAxis[3]
    """
    assert midx[0] == 0.0
    assert majorx[0] == 0.0
    # get equation of line through midx and majorx fy + gz + h = 0
    dx = [majorx[c] - midx[c] for c in range(3)]
    abs_dy = math.fabs(dx[1])
    abs_dz = math.fabs(dx[2])
    yzero = majorx[1] - dx[1] * (majorx[2] / dx[2]) if (abs_dz > 0.0) else None
    zzero = majorx[2] - dx[2] * (majorx[1] / dx[1]) if (abs_dy > 0.0) else None
    aa = a * a
    bb = b * b
    cc = c * c
    TOL = 1.0E-7
    if abs_dy > abs_dz:
        dz_dy = dx[2] / dx[1]
        f = -dz_dy
        g = 1.0
        h = -zzero
        # form quadratic equation qa.y2 + qb.y + qc = 0
        cc_gg = cc * g * g
        qa = 1.0 / bb + (f * f) / cc_gg
        qb = 2.0 * h * f / cc_gg
        qc = (h * h) / cc_gg - 1.0
        det = qb * qb - 4.0 * qa * qc
        assert det >= 0.0, "getEllipsoidPlaneA:  No intersection"
        sqrt_det = math.sqrt(det)
        y1 = (-qb + sqrt_det) / (2.0 * qa)
        y2 = (-qb - sqrt_det) / (2.0 * qa)
        y1_match = math.fabs((y1 - majorx[1]) / b) < TOL
        y2_match = math.fabs((y2 - majorx[1]) / b) < TOL
        assert y1_match != y2_match, "getEllipsoidPlaneA:  Exactly one root must match majorx"
        opp_y = y2 if y1_match else y1
        if yzero is not None:
            opp_z = dz_dy * (opp_y - yzero)
        else:
            opp_z = majorx[2]
        opp_majorx = [0.0, opp_y, opp_z]
    else:
        dy_dz = dx[1] / dx[2]
        f = 1.0
        g = -dy_dz
        h = -yzero
        # form quadratic equation qa.z2 + qb.z + qc = 0
        bb_ff = bb * f * f
        qa = 1.0 / cc + (g * g) / bb_ff
        qb = 2.0 * h * g / bb_ff
        qc = (h * h) / bb_ff - 1.0
        det = qb * qb - 4.0 * qa * qc
        assert det >= 0.0, "getEllipsoidPlaneA:  No intersection"
        sqrt_det = math.sqrt(det)
        z1 = (-qb + sqrt_det) / (2.0 * qa)
        z2 = (-qb - sqrt_det) / (2.0 * qa)
        z1_match = math.fabs((z1 - majorx[2]) / c) < TOL
        z2_match = math.fabs((z2 - majorx[2]) / c) < TOL
        assert z1_match != z2_match, "getEllipsoidPlaneA:  Exactly one root must match majorx"
        opp_z = z2 if z1_match else z1
        if zzero is not None:
            opp_y = dy_dz * (opp_z - zzero)
        else:
            opp_y = majorx[1]
        opp_majorx = [0.0, opp_y, opp_z]
    centre = [0.5 * (majorx[c] + opp_majorx[c]) for c in range(3)]
    majorAxis = [0.5 * (majorx[c] - opp_majorx[c]) for c in range(3)]
    # get axis2 length in +x direction at centre
    # form quadratic equation: qa.x2 + qb.x + qc = 0
    qa = 1.0 / aa
    # qb = 0.0
    qc = centre[1] * centre[1] / bb + centre[2] * centre[2] / cc - 1.0
    det = -4.0 * qa * qc
    assert det >= 0.0, "getEllipsoidPlaneA:  Invalid quadratic"
    sqrt_det = math.sqrt(det)
    minorAxis = [sqrt_det / (2.0 * qa), 0.0, 0.0]
    return centre, majorAxis, minorAxis


def moveCoordinatesToEllipsoidSurface(a, b, c, start_x):
    """
    Get the nearest point on ellipsoid surface to start_x.
    :param a: x-axis length.
    :param b: y-axis length.
    :param c: z-axis length.
    :param start_x: Coordinates of point near surface of ellipsoid.
    :return: Coordinates on surface of ellipsoid.
    """
    aa = a * a
    bb = b * b
    cc = c * c
    x = copy.copy(start_x)
    for iter in range(100):
        f = (x[0] * x[0]) / aa + (x[1] * x[1]) / bb + (x[2] * x[2]) / cc - 1.0
        if math.fabs(f) < 1.0E-8:
            # print("moveCoordinatesToEllipsoidSurface converged in", iter, "iterations. f =", f)
            break
        df = [2.0 * x[0] / aa, 2.0 * x[1] / bb, 2.0 * x[2] / cc]
        mag_df = magnitude(df)
        eta = f / (mag_df * mag_df)
        x = [(x[c] - df[c] * eta) for c in range(3)]
    else:
        print("moveCoordinatesToEllipsoidSurface failed to converged in", iter, "iterations. f =", f)
    return x


def moveDerivativeToEllipsoidSurface(a, b, c, x, start_d):
    """
    Convert derivative at point on surface of ellipsoid to be tangential to it.
    :param a: x-axis length.
    :param b: y-axis length.
    :param c: z-axis length.
    :param x: Coordinates on surface of ellipsoid.
    :param start_d: Derivative near tangential to ellipsoid surface at x.
    :return: Derivative made tangential to surface with same magnitude.
    """
    n = [2.0 * x[0] / (a * a), 2.0 * x[1] / (b * b), 2.0 * x[2] / (c * c)]
    return set_magnitude(rejection(start_d, n), magnitude(start_d))


def moveDerivativeToEllipsoidSurfaceInPlane(a, b, c, x, pn, start_d):
    """
    Convert derivative at point on surface of ellipsoid to be tangential to it and normal to pn.
    :param a: x-axis length.
    :param b: y-axis length.
    :param c: z-axis length.
    :param x: Coordinates on surface of ellipsoid.
    :param pn: Direction normal to plane to force result to be in plane.
    :param start_d: Derivative near tangential to ellipsoid surface at x.
    :return: Derivative made tangential to surface with same magnitude.
    """
    en = [2.0 * x[0] / (a * a), 2.0 * x[1] / (b * b), 2.0 * x[2] / (c * c)]
    cp = cross(pn, en)
    return set_magnitude(projection(start_d, cp), magnitude(start_d))


def sampleCurveOnEllipsoid(a, b, c, start_x, start_d1, start_d2, end_x, end_d1, end_d2, elements_count,
                           start_weight=None, end_weight=None, overweighting=1.0, end_transition=False):
    """
    Samples an even-spaced Hermite curve from start point and direction to end point and direction.
    Also interpolates side derivatives.
    :param a: x-axis length.
    :param b: y-axis length.
    :param c: z-axis length.
    :param start_x: start coordinate on surface of ellipsoid.
    :param start_d1: start direction on surface of ellipsoid.
    :param start_d2: optional start side direction on surface of ellipsoid. Must be supplied if end_d2 is
    supplied.
    :param end_x: end coordinate on surface of ellipsoid.
    :param end_d1: end direction on surface of ellipsoid, or None to estimate from start_d1.
    Must be supplied with end_transition.
    :param end_d2: optional end side direction on surface of ellipsoid, or None to use surface tangent normal to end
    direction on same side as start_d2.
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
        end_d = moveDerivativeToEllipsoidSurface(a, b, c, end_x, end_d)
    use_start_weight = (start_weight if start_weight else 1.0) * magnitude(end_x)
    use_end_weight = (end_weight if end_weight else 1.0) * magnitude(start_x)
    start_d = set_magnitude(start_d, 2.0 * length * use_start_weight / (use_start_weight + use_end_weight))
    end_d = set_magnitude(end_d, 2.0 * length * use_end_weight / (use_start_weight + use_end_weight))
    scaling = computeCubicHermiteDerivativeScaling(start_x, start_d, end_x, end_d) * overweighting
    start_d = mult(start_d, scaling)
    end_d = mult(end_d, scaling)
    px, pd1 = sampleCubicHermiteCurvesSmooth([start_x, end_x], [start_d, end_d], elements_count)[0:2]
    iter_count = 2
    for iter in range(iter_count):
        for n in range(1, elements_count):
            px[n] = moveCoordinatesToEllipsoidSurface(a, b, c, px[n])
            pd1[n] = moveDerivativeToEllipsoidSurface(a, b, c, px[n], pd1[n])
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
            end_d2 = moveDerivativeToEllipsoidSurface(a, b, c, px[-1], start_d2)
            normal = normalize(cross(pd1[-1], end_d2))
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
            pd2.append(moveDerivativeToEllipsoidSurface(a, b, c, px[n], d2))
        pd2.append(end_d2)
        return px, pd1, pd2
    return px, pd1


def getCircleProjectionAxes(ax, ad1, ad2, ad3, length, angle1radians, angle2radians, angle3radians = None):
    '''
    Project coordinates and orthogonal unit axes ax, ad1, ad2, ad3 by length in
    direction of ad3, rotated towards ad1 by angle1radians and ad2 by
    angle2radians such that it conforms to a circular arc of the given length,
    tangential to ad3 at the start and the final bd3 at the end.
    All vectors must be length 3.
    Assumes angles 1 and 2 are not large e.g. less than 90 degrees.
    Note: not robust for all inputs.
    :param angle3radians: Optional final rotation of projection axes about bd3.
    :return: Final coordinates and orthogonal unit axes: bx, bd1, bd2, bd3
    '''
    if (math.fabs(angle1radians) < 0.000001) and (math.fabs(angle2radians) < 0.000001):
        bx = [ (ax[c] + length*ad3[c]) for c in range(3) ]
        bd1 = copy.deepcopy(ad1)
        bd2 = copy.deepcopy(ad2)
        bd3 = copy.deepcopy(ad3)
    else:
        cosAngle1 = math.cos(angle1radians)
        sinAngle1 = math.sin(angle1radians)
        cosAngle2 = math.cos(angle2radians)
        sinAngle2 = math.sin(angle2radians)
        f1 = sinAngle1*cosAngle2
        f2 = cosAngle1*sinAngle2
        f3 = cosAngle1*cosAngle2
        angleAroundRadians = math.atan2(f2, f1)
        fh = math.sqrt(f1*f1 + f2*f2)
        arcAngleRadians = 0.5*math.pi - math.atan2(f3, fh)
        arcRadius = length/arcAngleRadians
        br = arcRadius*(1.0 - math.cos(arcAngleRadians))
        w1 = br*math.cos(angleAroundRadians)  # f1/fh
        w2 = br*math.sin(angleAroundRadians)  # f2/fh
        w3 = arcRadius*math.sin(arcAngleRadians)
        bx = [ (ax[c] + w1*ad1[c] + w2*ad2[c] + w3*ad3[c]) for c in range(3) ]
        bd3 = normalize([ (f1*ad1[c] + f2*ad2[c] + f3*ad3[c]) for c in range(3) ])
        bd1 = normalize(cross(ad2, bd3))
        bd2 = cross(bd3, bd1)
    if angle3radians:
        cosAngle3 = math.cos(angle3radians)
        sinAngle3 = math.sin(angle3radians)
        bd1, bd2 = [ (cosAngle3*bd1[c] + sinAngle3*bd2[c]) for c in range(3) ], [ (cosAngle3*bd2[c] - sinAngle3*bd1[c]) for c in range(3) ]
    return bx, bd1, bd2, bd3

def getSurfaceProjectionAxes(ax, ad1, ad2, ad3, angle1radians, angle2radians, length):
    '''
    Project coordinates and orthogonal unit axes ax, ad1, ad2, ad3 in direction
    of ad3, rotated towards ad1 by angle1radians and ad2 by angle2radians by
    simple rotation.
    All vectors are/must be length 3.
    Assumes angles are not large e.g. less than 90 degrees.
    Note: not robust for all inputs.
    :return: Final coordinates and orthogonal unit axes: bx, bd1, bd2, bd3
    '''
    cosAngle1 = math.cos(angle1radians)
    sinAngle1 = math.sin(angle1radians)
    cosAngle2 = math.cos(angle2radians)
    sinAngle2 = math.sin(angle2radians)
    f1 = sinAngle1*cosAngle2
    f2 = cosAngle1*sinAngle2
    f3 = cosAngle1*cosAngle2
    bd3 = [ (f1*ad1[c] + f2*ad2[c] + f3*ad3[c]) for c in range (3) ]
    bx = [ (ax[c] + length*bd3[c]) for c in range(3) ]
    bd1 = cross(ad2, bd3)
    bd2 = cross(bd3, bd1)
    return bx, bd1, bd2, bd3

def createEllipsePoints(cx, radian, axis1, axis2, elementsCountAround, startRadians = 0.0):
    '''
    Create ellipse points centred at cx, from axis1 around through axis2.
    Assumes axis1 and axis2 are orthogonal.
    :param cx: centre
    :param radian: Part of ellipse to be created based on radian (will be 2*math.pi for a complete ellipse).
    :param axis1: Vector from cx to inside at zero angle
    :param axis2: Vector from cx to inside at 90 degree angle.
    :param elementsCountAround: Number of elements around.
    :param startRadians: Angle from axis1 to start creating the ellipse.
    :return: lists px, pd1
    '''
    px = []
    pd1 = []
    magAxis1 = magnitude(axis1)
    magAxis2 = magnitude(axis2)
    totalEllipsePerimeter = getApproximateEllipsePerimeter(magAxis1, magAxis2)
    partOfEllipsePerimeter = radian * totalEllipsePerimeter / (2 * math.pi)
    elementLength = partOfEllipsePerimeter / elementsCountAround
    if radian != 2 * math.pi:
        elementsCountAround = elementsCountAround + 1

    unitSideAxis1 = normalize(axis1)
    unitSideAxis2 = normalize(axis2)
    for n in range(elementsCountAround):
        angle = startRadians
        arcLength = n * elementLength
        newAngle = updateEllipseAngleByArcLength(magAxis1, magAxis2, angle, arcLength)
        cosRadiansAround = math.cos(newAngle)
        sinRadiansAround = math.sin(newAngle)
        x = [
            cx[0] + cosRadiansAround * axis1[0] - sinRadiansAround * axis2[0],
            cx[1] + cosRadiansAround * axis1[1] + sinRadiansAround * axis2[1],
            cx[2] + cosRadiansAround * axis1[2] + sinRadiansAround * axis2[2]
        ]
        px.append(x)
        rd1 = [magAxis1 * (-sinRadiansAround * unitSideAxis1[c]) + magAxis2 * (cosRadiansAround * unitSideAxis2[c]) for c in range(3)]
        rd1Norm = normalize(rd1)
        pd1.append([elementLength * (rd1Norm[c])for c in range(3)])

    return px, pd1

