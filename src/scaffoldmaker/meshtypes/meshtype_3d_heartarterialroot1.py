"""
Generates a 3-D heart arterial root scaffold with semilunar valve,
for attaching to a 6-element-around bicubic-linear orifice.
"""

from __future__ import division

import math

from cmlibs.maths.vectorops import set_magnitude, magnitude, cross
from cmlibs.utils.zinc.field import findOrCreateFieldCoordinates
from cmlibs.utils.zinc.finiteelement import getMaximumElementIdentifier, getMaximumNodeIdentifier
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup
from scaffoldmaker.annotation.heart_terms import get_heart_term
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.utils import interpolation as interp
from scaffoldmaker.utils.eft_utils import remapEftLocalNodes, remapEftNodeValueLabel, setEftScaleFactorIds
from scaffoldmaker.utils.eftfactory_bicubichermitelinear import eftfactory_bicubichermitelinear
from scaffoldmaker.utils.geometry import getApproximateEllipsePerimeter, createCirclePoints
from scaffoldmaker.utils.meshrefinement import MeshRefinement


class MeshType_3d_heartarterialroot1(Scaffold_base):
    '''
    Generates a 3-D heart arterial root scaffold with semilunar valve,
    for attaching to a 6-element-around bicubic-linear orifice.
    '''

    @staticmethod
    def getName():
        return '3D Heart Arterial Root 1'

    @staticmethod
    def getDefaultOptions(parameterSetName='Default'):
        return {
            'Unit scale' : 1.0,
            'Outer height' : 0.5,
            'Inner depth' : 0.15,
            'Cusp height' : 0.6,
            'Inner diameter': 1.0,
            'Sinus radial displacement': 0.1,
            'Wall thickness': 0.06,
            'Cusp thickness' : 0.02,
            'Aortic not pulmonary' : True,
            'Refine' : False,
            'Refine number of elements surface' : 4,
            'Refine number of elements through wall' : 1,
            'Use cross derivatives' : False
        }

    @staticmethod
    def getOrderedOptionNames():
        return [
            'Unit scale',
            'Outer height',
            'Inner depth',
            'Cusp height',
            'Inner diameter',
            'Sinus radial displacement',
            'Wall thickness',
            'Cusp thickness',
            'Aortic not pulmonary',
            'Refine',
            'Refine number of elements surface',
            'Refine number of elements through wall'
        ]

    @staticmethod
    def checkOptions(options):
        '''
        :return:  True if dependent options changed, otherwise False.
        '''
        dependentChanges = False
        for key in [
            'Unit scale',
            'Outer height',
            'Inner depth',
            'Cusp height',
            'Inner diameter',
            'Sinus radial displacement',
            'Wall thickness',
            'Cusp thickness']:
            if options[key] < 0.0:
                options[key] = 0.0
        for key in [
            'Refine number of elements surface',
            'Refine number of elements through wall']:
            if options[key] < 1:
                options[key] = 1
        return dependentChanges

    @classmethod
    def generateBaseMesh(cls, region, options, baseCentre=[ 0.0, 0.0, 0.0 ], axisSide1=[ 0.0, -1.0, 0.0 ], axisUp=[ 0.0, 0.0, 1.0]):
        """
        Generate the base bicubic-linear Hermite mesh.
        Optional extra parameters allow centre and axes to be set.
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :param baseCentre: Centre of valve on ventriculo-arterial junction.
        :param axisSide: Unit vector in first side direction where angle around starts.
        :param axisUp: Unit vector in outflow direction of valve.
        :return: list of AnnotationGroup, None
        """
        unitScale = options['Unit scale']
        outerHeight = unitScale*options['Outer height']
        innerDepth = unitScale*options['Inner depth']
        cuspHeight = unitScale*options['Cusp height']
        innerRadius = unitScale*0.5*options['Inner diameter']
        sinusRadialDisplacement = unitScale*options['Sinus radial displacement']
        wallThickness = unitScale*options['Wall thickness']
        cuspThickness = unitScale*options['Cusp thickness']
        aorticNotPulmonary = options['Aortic not pulmonary']
        useCrossDerivatives = False

        fm = region.getFieldmodule()
        fm.beginChange()
        coordinates = findOrCreateFieldCoordinates(fm)
        cache = fm.createFieldcache()

        mesh = fm.findMeshByDimension(3)

        if aorticNotPulmonary:
            arterialRootGroup = AnnotationGroup(region, get_heart_term("root of aorta"))
            # following is disputed in ontologies
            # cuspGroups = [
            #     AnnotationGroup(region, get_heart_term("posterior cusp of aortic valve")),
            #     AnnotationGroup(region, get_heart_term("right cusp of aortic valve")),
            #     AnnotationGroup(region, get_heart_term("left cusp of aortic valve")) ]
            cuspGroup = AnnotationGroup(region, get_heart_term("aortic valve leaflet"))
            cuspGroups = [cuspGroup] * 3
        else:
            arterialRootGroup = AnnotationGroup(region, get_heart_term("root of pulmonary trunk"))
            # following is disputed in ontologies
            # cuspGroups = [
            #     AnnotationGroup(region, get_heart_term("right cusp of pulmonary valve")),
            #     AnnotationGroup(region, get_heart_term("anterior cusp of pulmonary valve")),
            #     AnnotationGroup(region, get_heart_term("left cusp of pulmonary valve")) ]
            cuspGroup = AnnotationGroup(region, get_heart_term("pulmonary valve leaflet"))
            cuspGroups = [cuspGroup] * 3

        allGroups = [ arterialRootGroup ]  # groups that all elements in scaffold will go in
        annotationGroups = allGroups + cuspGroups

        nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        #################
        # Create nodes
        #################

        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)
        # most nodes in this scaffold do not have a DS3 derivative
        nodetemplateLinearS3 = nodes.createNodetemplate()
        nodetemplateLinearS3.defineField(coordinates)
        nodetemplateLinearS3.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        nodetemplateLinearS3.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        nodetemplateLinearS3.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        # several only have a DS1 derivative
        nodetemplateLinearS2S3 = nodes.createNodetemplate()
        nodetemplateLinearS2S3.defineField(coordinates)
        nodetemplateLinearS2S3.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        nodetemplateLinearS2S3.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)

        nodeIdentifier = max(1, getMaximumNodeIdentifier(nodes) + 1)

        elementsCountAround = 6
        radiansPerElementAround = 2.0*math.pi/elementsCountAround
        axisSide2 = cross(axisUp, axisSide1)
        outerRadius = innerRadius + wallThickness
        cuspOuterLength2 = 0.5*getApproximateEllipsePerimeter(innerRadius, cuspHeight)
        cuspOuterWallArcLength = cuspOuterLength2*innerRadius/(innerRadius + cuspHeight)
        noduleOuterAxialArcLength = cuspOuterLength2 - cuspOuterWallArcLength
        noduleOuterRadialArcLength = innerRadius
        cuspOuterWalld1 = interp.interpolateLagrangeHermiteDerivative([ innerRadius, outerHeight + innerDepth - cuspHeight ], [ 0.0, 0.0 ], [ -innerRadius, 0.0 ], 0.0)

        sin60 = math.sin(math.pi/3.0)
        cuspThicknessLowerFactor = 4.5  # GRC fudge factor
        cuspInnerLength2 = 0.5*getApproximateEllipsePerimeter(innerRadius - cuspThickness/sin60, cuspHeight - cuspThicknessLowerFactor*cuspThickness)

        noduleInnerAxialArcLength = cuspInnerLength2*(cuspHeight - cuspThicknessLowerFactor*cuspThickness)/(innerRadius - cuspThickness/sin60 + cuspHeight - cuspThicknessLowerFactor*cuspThickness)
        noduleInnerRadialArcLength = innerRadius - cuspThickness/math.tan(math.pi/3.0)
        nMidCusp = 0 if aorticNotPulmonary else 1

        # lower points
        ix, id1 = createCirclePoints([ (baseCentre[c] - axisUp[c]*innerDepth) for c in range(3) ],
            [ axisSide1[c]*innerRadius for c in range(3) ], [ axisSide2[c]*innerRadius for c in range(3) ],
            elementsCountAround)
        ox, od1 = getSemilunarValveSinusPoints(baseCentre, axisSide1, axisSide2, outerRadius, sinusRadialDisplacement,
            startMidCusp=aorticNotPulmonary)
        lowerx, lowerd1 = [ ix, ox ], [ id1, od1 ]

        # upper points
        topCentre = [ (baseCentre[c] + axisUp[c]*outerHeight) for c in range(3) ]
        # twice as many on inner:
        ix, id1 = createCirclePoints(topCentre,
            [ axisSide1[c]*innerRadius for c in range(3) ], [ axisSide2[c]*innerRadius for c in range(3) ],
            elementsCountAround*2)
        # tweak inner points so elements attached to cusps are narrower
        cuspRadiansFactor = 0.25  # GRC fudge factor
        midDerivativeFactor = 1.0 + 0.5*(1.0 - cuspRadiansFactor)  # GRC test compromise
        cuspAttachmentRadians = cuspRadiansFactor*radiansPerElementAround
        cuspAttachmentRadialDisplacement = wallThickness*0.333  # GRC fudge factor
        cuspAttachmentRadius = innerRadius - cuspAttachmentRadialDisplacement
        for cusp in range(3):
            n1 = cusp*2 - 1 + nMidCusp
            n2 = n1*2
            id1[n2 + 2] = [ 2.0*d for d in id1[n2 + 2] ]
            # side 1
            radiansAround = n1*radiansPerElementAround + cuspAttachmentRadians
            rcosRadiansAround = cuspAttachmentRadius*math.cos(radiansAround)
            rsinRadiansAround = cuspAttachmentRadius*math.sin(radiansAround)
            ix[n2 + 1] = [ (topCentre[c] + rcosRadiansAround*axisSide1[c] + rsinRadiansAround*axisSide2[c]) for c in range(3) ]
            id1[n2 + 1] = interp.interpolateLagrangeHermiteDerivative(ix[n2 + 1], ix[n2 + 2], id1[n2 + 2], 0.0)
            # side 2
            n1 = ((cusp + 1)*2 - 1 + nMidCusp)%elementsCountAround
            n2 = n1*2
            radiansAround = n1*radiansPerElementAround - cuspAttachmentRadians
            rcosRadiansAround = cuspAttachmentRadius*math.cos(radiansAround)
            rsinRadiansAround = cuspAttachmentRadius*math.sin(radiansAround)
            ix[n2 - 1] = [ (topCentre[c] + rcosRadiansAround*axisSide1[c] + rsinRadiansAround*axisSide2[c]) for c in range(3) ]
            id1[n2 - 1] = interp.interpolateHermiteLagrangeDerivative(ix[n2 - 2], id1[n2 - 2], ix[n2 - 1], 1.0)
        ox, od1 = createCirclePoints(topCentre,
            [ axisSide1[c]*outerRadius for c in range(3) ], [ axisSide2[c]*outerRadius for c in range(3) ],
            elementsCountAround)
        upperx, upperd1 = [ ix, ox ], [ id1, od1 ]

        # get lower and upper derivative 2
        zero = [ 0.0, 0.0, 0.0 ]
        upperd2factor = outerHeight
        upd2 = [ d*upperd2factor for d in axisUp ]
        lowerOuterd2 = interp.smoothCubicHermiteDerivativesLine([ lowerx[1][nMidCusp], upperx[1][nMidCusp] ], [ upd2, upd2 ],
            fixStartDirection=True, fixEndDerivative=True)[0]
        lowerd2factor = 2.0*(outerHeight + innerDepth) - upperd2factor
        lowerInnerd2 = [ d*lowerd2factor for d in axisUp ]
        lowerd2 = [ [ lowerInnerd2 ]*elementsCountAround, [ lowerOuterd2 ]*elementsCountAround ]  # some lowerd2[0] to be fitted below
        upperd2 = [ [ upd2 ]*(elementsCountAround*2), [ upd2 ]*elementsCountAround ]

        # get lower and upper derivative 1 or 2 pointing to/from cusps
        for n1 in range(elementsCountAround):
            radiansAround = n1*radiansPerElementAround
            cosRadiansAround = math.cos(radiansAround)
            sinRadiansAround = math.sin(radiansAround)
            if (n1 % 2) == nMidCusp:
                lowerd2[0][n1] = [ -cuspOuterWallArcLength*(cosRadiansAround*axisSide1[c] + sinRadiansAround*axisSide2[c]) for c in range(3) ]
            else:
                upperd1[0][n1*2] = [ (cuspOuterWalld1[0]*(cosRadiansAround*axisSide1[c] + sinRadiansAround*axisSide2[c]) + cuspOuterWalld1[1]*axisUp[c]) for c in range(3) ]

        # inner wall and mid sinus points; only every second one is used
        sinusDepth = innerDepth - cuspThicknessLowerFactor*cuspThickness  # GRC test
        sinusCentre = [ (baseCentre[c] - sinusDepth*axisUp[c]) for c in range(3) ]
        sinusx, sinusd1 = createCirclePoints(sinusCentre,
            [ axisSide1[c]*innerRadius for c in range(3) ], [ axisSide2[c]*innerRadius for c in range(3) ],
            elementsCountAround)
        # get sinusd2, parallel to lower inclined lines
        sd2 = interp.smoothCubicHermiteDerivativesLine([ [ innerRadius, -sinusDepth ], [ innerRadius, outerHeight ] ],
            [ [ wallThickness + sinusRadialDisplacement, innerDepth ], [ 0.0, upperd2factor ] ],
            fixStartDirection=True, fixEndDerivative=True)[0]
        sinusd2 = [ None ]*elementsCountAround
        for cusp in range(3):
            n1 = cusp*2 + nMidCusp
            radiansAround = n1*radiansPerElementAround
            cosRadiansAround = math.cos(radiansAround)
            sinRadiansAround = math.sin(radiansAround)
            sinusd2[n1] = [ (sd2[0]*(cosRadiansAround*axisSide1[c] + sinRadiansAround*axisSide2[c]) + sd2[1]*axisUp[c]) for c in range(3) ]

        # get points on arc between mid sinus and upper cusp points
        arcx = []
        arcd1 = []
        scaled1 = 2.5  # GRC fudge factor
        for cusp in range(3):
            n1 = cusp*2 + nMidCusp
            n1m = n1 - 1
            n1p = (n1 + 1)%elementsCountAround
            n2m = n1m*2 + 1
            n2p = n1p*2 - 1
            ax, ad1 = interp.sampleCubicHermiteCurves([ upperx[0][n2m], sinusx[n1] ], [ [ -scaled1*d for d in upperd2[0][n2m] ], [ scaled1*d for d in sinusd1[n1] ] ],
                elementsCountOut=2, addLengthStart=0.5*magnitude(upperd2[0][n2m]), lengthFractionStart=0.5,
                addLengthEnd=0.5*magnitude(sinusd1[n1]), lengthFractionEnd=0.5, arcLengthDerivatives=False)[0:2]
            arcx .append(ax [1])
            arcd1.append(ad1[1])
            ax, ad1 = interp.sampleCubicHermiteCurves([ sinusx[n1], upperx[0][n2p], ], [ [ scaled1*d for d in sinusd1[n1] ], [ scaled1*d for d in upperd2[0][n2p] ] ],
                elementsCountOut=2, addLengthStart=0.5*magnitude(sinusd1[n1]), lengthFractionStart=0.5,
                addLengthEnd=0.5*magnitude(upperd2[0][n2p]), lengthFractionEnd=0.5, arcLengthDerivatives=False)[0:2]
            arcx .append(ax [1])
            arcd1.append(ad1[1])
        if nMidCusp == 0:
            arcx .append(arcx .pop(0))
            arcd1.append(arcd1.pop(0))

        # cusp nodule points
        noduleCentre = [ (baseCentre[c] + axisUp[c]*(cuspHeight - innerDepth)) for c in range(3) ]
        nodulex  = [ [], [] ]
        noduled1 = [ [], [] ]
        noduled2 = [ [], [] ]
        noduled3 = [ [], [] ]
        cuspRadialThickness = cuspThickness/sin60
        for i in range(3):
            nodulex[0].append(noduleCentre)
            n1 = i*2 + nMidCusp
            radiansAround = n1*radiansPerElementAround
            cosRadiansAround = math.cos(radiansAround)
            sinRadiansAround = math.sin(radiansAround)
            nodulex[1].append([ (noduleCentre[c] + cuspRadialThickness*(cosRadiansAround*axisSide1[c] + sinRadiansAround*axisSide2[c])) for c in range(3) ])
            n1 = i*2 - 1 + nMidCusp
            radiansAround = n1*radiansPerElementAround
            cosRadiansAround = math.cos(radiansAround)
            sinRadiansAround = math.sin(radiansAround)
            noduled1[0].append([ noduleOuterRadialArcLength*(cosRadiansAround*axisSide1[c] + sinRadiansAround*axisSide2[c]) for c in range(3) ])
            noduled1[1].append(set_magnitude(noduled1[0][i], noduleInnerRadialArcLength))
            n1 = i*2 + 1 + nMidCusp
            radiansAround = n1*radiansPerElementAround
            cosRadiansAround = math.cos(radiansAround)
            sinRadiansAround = math.sin(radiansAround)
            noduled2[0].append([ noduleOuterRadialArcLength*(cosRadiansAround*axisSide1[c] + sinRadiansAround*axisSide2[c]) for c in range(3) ])
            noduled2[1].append(set_magnitude(noduled2[0][i], noduleInnerRadialArcLength))
            noduled3[0].append([ noduleOuterAxialArcLength*axisUp[c] for c in range(3) ])
            noduled3[1].append([ noduleInnerAxialArcLength*axisUp[c] for c in range(3) ])

        # Create nodes

        lowerNodeId = [ [], [] ]
        for n3 in range(2):
            for n1 in range(elementsCountAround):
                node = nodes.createNode(nodeIdentifier, nodetemplateLinearS3)
                cache.setNode(node)
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, lowerx [n3][n1])
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, lowerd1[n3][n1])
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, lowerd2[n3][n1])
                lowerNodeId[n3].append(nodeIdentifier)
                nodeIdentifier += 1

        sinusNodeId = []
        for n1 in range(elementsCountAround):
            if (n1%2) != nMidCusp:
                sinusNodeId.append(None)
                continue
            node = nodes.createNode(nodeIdentifier, nodetemplateLinearS3)
            cache.setNode(node)
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, sinusx [n1])
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, sinusd1[n1])
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, sinusd2[n1])
            sinusNodeId.append(nodeIdentifier)
            nodeIdentifier += 1

        arcNodeId = []
        for n1 in range(elementsCountAround):
            node = nodes.createNode(nodeIdentifier, nodetemplateLinearS2S3)
            cache.setNode(node)
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, arcx [n1])
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, arcd1[n1])
            arcNodeId.append(nodeIdentifier)
            nodeIdentifier += 1

        noduleNodeId = [ [], [] ]
        for n3 in range(2):
            for n1 in range(3):
                node = nodes.createNode(nodeIdentifier, nodetemplate)
                cache.setNode(node)
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, nodulex [n3][n1])
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, noduled1[n3][n1])
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, noduled2[n3][n1])
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, noduled3[n3][n1])
                noduleNodeId[n3].append(nodeIdentifier)
                nodeIdentifier += 1

        upperNodeId = [ [], [] ]
        for n3 in range(2):
            for n1 in range(len(upperx[n3])):
                node = nodes.createNode(nodeIdentifier, nodetemplateLinearS3)
                cache.setNode(node)
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, upperx [n3][n1])
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, upperd1[n3][n1])
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, upperd2[n3][n1])
                upperNodeId[n3].append(nodeIdentifier)
                nodeIdentifier += 1


        #################
        # Create elements
        #################

        allMeshGroups = [ allGroup.getMeshGroup(mesh) for allGroup in allGroups ]
        cuspMeshGroups = [ cuspGroup.getMeshGroup(mesh) for cuspGroup in cuspGroups ]

        linearHermiteLinearBasis = fm.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        linearHermiteLinearBasis.setFunctionType(2, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)

        hermiteLinearLinearBasis = fm.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        hermiteLinearLinearBasis.setFunctionType(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)

        bicubichermitelinear = eftfactory_bicubichermitelinear(mesh, useCrossDerivatives)
        eftDefault = bicubichermitelinear.createEftNoCrossDerivatives()

        elementIdentifier = max(1, getMaximumElementIdentifier(mesh) + 1)

        elementtemplate1 = mesh.createElementtemplate()
        elementtemplate1.setElementShapeType(Element.SHAPE_TYPE_CUBE)

        # wall elements
        for cusp in range(3):
            n1 = cusp*2 - 1 + nMidCusp
            n2 = n1*2
            for e in range(6):
                eft1 = None
                scalefactors = None

                if (e == 0) or (e == 5):
                    # 6 node linear-hermite-linear collapsed wedge element expanding from zero width on outer wall of root, attaching to vertical part of cusp
                    eft1 = mesh.createElementfieldtemplate(linearHermiteLinearBasis)
                    # switch mappings to use DS2 instead of default DS1
                    remapEftNodeValueLabel(eft1, [ 1, 2, 3, 4, 5, 6, 7, 8 ], Node.VALUE_LABEL_D_DS1, [ ( Node.VALUE_LABEL_D_DS2, [] ) ])
                    if e == 0:
                        nids = [ lowerNodeId[0][n1], arcNodeId[n1], upperNodeId[0][n2], upperNodeId[0][n2 + 1], lowerNodeId[1][n1], upperNodeId[1][n1] ]
                        setEftScaleFactorIds(eft1, [1], [])
                        scalefactors = [ -1.0 ]
                        remapEftNodeValueLabel(eft1, [ 2 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS1, [1] ) ])
                    else:
                        nids = [ arcNodeId[n1 + 1], lowerNodeId[0][n1 - 4], upperNodeId[0][n2 + 3], upperNodeId[0][n2 - 8], lowerNodeId[1][n1 - 4], upperNodeId[1][n1 - 4] ]
                        remapEftNodeValueLabel(eft1, [ 1 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS1, [] ) ])
                    ln_map = [ 1, 2, 3, 4, 5, 5, 6, 6 ]
                    remapEftLocalNodes(eft1, 6, ln_map)
                elif (e == 1) or (e == 4):
                    # 6 node hermite-linear-linear collapsed wedge element on lower wall
                    eft1 = mesh.createElementfieldtemplate(hermiteLinearLinearBasis)
                    if e == 1:
                        nids = [ lowerNodeId[0][n1], lowerNodeId[0][n1 + 1], arcNodeId[n1], sinusNodeId[n1 + 1], lowerNodeId[1][n1], lowerNodeId[1][n1 + 1] ]
                    else:
                        nids = [ lowerNodeId[0][n1 + 1], lowerNodeId[0][n1 - 4], sinusNodeId[n1 + 1], arcNodeId[n1 + 1], lowerNodeId[1][n1 + 1], lowerNodeId[1][n1 - 4] ]
                    ln_map = [ 1, 2, 3, 4, 5, 6, 5, 6 ]
                    remapEftLocalNodes(eft1, 6, ln_map)
                else:
                    # 8 node elements with wedges on two sides
                    if e == 2:
                        eft1 = bicubichermitelinear.createEftNoCrossDerivatives()
                        setEftScaleFactorIds(eft1, [1], [])
                        scalefactors = [ -1.0 ]
                        nids = [ arcNodeId[n1], sinusNodeId[n1 + 1], upperNodeId[0][n2 + 1], upperNodeId[0][n2 + 2],
                                 lowerNodeId[1][n1], lowerNodeId[1][n1 + 1], upperNodeId[1][n1], upperNodeId[1][n1 + 1] ]
                        remapEftNodeValueLabel(eft1, [ 1 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS1, [1] ) ])
                    else:
                        eft1 = eftDefault
                        nids = [ sinusNodeId[n1 + 1], arcNodeId[n1 + 1], upperNodeId[0][n2 + 2], upperNodeId[0][n2 + 3],
                                 lowerNodeId[1][n1 + 1], lowerNodeId[1][n1 - 4], upperNodeId[1][n1 + 1], upperNodeId[1][n1 - 4] ]
                        remapEftNodeValueLabel(eft1, [ 2 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS1, [] ) ])

                result = elementtemplate1.defineField(coordinates, -1, eft1)
                element = mesh.createElement(elementIdentifier, elementtemplate1)
                result2 = element.setNodesByIdentifier(eft1, nids)
                result3 = element.setScaleFactors(eft1, scalefactors) if scalefactors else None
                #print('create arterial root wall', cusp, e, 'element',elementIdentifier, result, result2, result3, nids)
                elementIdentifier += 1

                for meshGroup in allMeshGroups:
                    meshGroup.addElement(element)

        # cusps (leaflets)
        for cusp in range(3):
            n1 = cusp*2 - 1 + nMidCusp
            n2 = n1*2
            meshGroups = allMeshGroups + [ cuspMeshGroups[cusp] ]
            for e in range(2):
                eft1 = bicubichermitelinear.createEftNoCrossDerivatives()
                setEftScaleFactorIds(eft1, [1], [])
                scalefactors = [ -1.0 ]

                if e == 0:
                    nids = [ lowerNodeId[0][n1], lowerNodeId[0][n1 + 1], upperNodeId[0][n2], noduleNodeId[0][cusp],
                             arcNodeId[n1], sinusNodeId[n1 + 1], upperNodeId[0][n2 + 1], noduleNodeId[1][cusp] ]
                    remapEftNodeValueLabel(eft1, [ 4, 8 ], Node.VALUE_LABEL_D_DS1, [ ( Node.VALUE_LABEL_D_DS1, [1] ) ])
                    remapEftNodeValueLabel(eft1, [ 4, 8 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS3, [] ) ])
                    remapEftNodeValueLabel(eft1, [ 5 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS1, [1] ) ])
                    remapEftNodeValueLabel(eft1, [ 6 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS2, [1] ) ])
                    remapEftNodeValueLabel(eft1, [ 7 ], Node.VALUE_LABEL_D_DS1, [ ( Node.VALUE_LABEL_D_DS1, [1] ) ])
                else:
                    nids = [ lowerNodeId[0][n1 + 1], lowerNodeId[0][n1 - 4], noduleNodeId[0][cusp], upperNodeId[0][n2 - 8], 
                             sinusNodeId[n1 + 1], arcNodeId[n1 + 1], noduleNodeId[1][cusp], upperNodeId[0][n2 + 3] ]
                    remapEftNodeValueLabel(eft1, [ 3, 7 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS3, [] ) ])
                    remapEftNodeValueLabel(eft1, [ 3, 7 ], Node.VALUE_LABEL_D_DS1, [ ( Node.VALUE_LABEL_D_DS2, [] ) ])
                    remapEftNodeValueLabel(eft1, [ 4, 8 ], Node.VALUE_LABEL_D_DS1, [ ( Node.VALUE_LABEL_D_DS1, [1] ) ])
                    remapEftNodeValueLabel(eft1, [ 5 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS2, [1] ) ])
                    remapEftNodeValueLabel(eft1, [ 6 ], Node.VALUE_LABEL_D_DS2, [ ( Node.VALUE_LABEL_D_DS1, [] ) ])

                result = elementtemplate1.defineField(coordinates, -1, eft1)
                element = mesh.createElement(elementIdentifier, elementtemplate1)
                result2 = element.setNodesByIdentifier(eft1, nids)
                result3 = element.setScaleFactors(eft1, scalefactors) if scalefactors else None
                #print('create semilunar cusp', cusp, e, 'element',elementIdentifier, result, result2, result3, nids)
                elementIdentifier += 1

                for meshGroup in meshGroups:
                    meshGroup.addElement(element)

        fm.endChange()
        return annotationGroups, None

    @classmethod
    def refineMesh(cls, meshrefinement, options):
        """
        Refine source mesh into separate region, with change of basis.
        :param meshrefinement: MeshRefinement, which knows source and target region.
        :param options: Dict containing options. See getDefaultOptions().
        """
        assert isinstance(meshrefinement, MeshRefinement)
        refineElementsCountSurface = options['Refine number of elements surface']
        refineElementsCountThroughWall = options['Refine number of elements through wall']
        # arterial wall
        for cusp in range(3):
            for e in range(6):
                element = meshrefinement._sourceElementiterator.next()
                numberInXi1 = refineElementsCountSurface
                numberInXi2 = refineElementsCountSurface
                numberInXi3 = refineElementsCountThroughWall
                if (e == 0) or (e == 5):
                    numberInXi1 = refineElementsCountThroughWall
                elif (e == 1) or (e == 4):
                    numberInXi2 = refineElementsCountThroughWall
                meshrefinement.refineElementCubeStandard3d(element, numberInXi1, numberInXi2, numberInXi3)
        # semilunar cusps
        # Avoid merging nodes on separate cusps where they are coincident along commissures
        numberInXi1 = refineElementsCountSurface
        numberInXi2 = refineElementsCountSurface
        numberInXi3 = refineElementsCountThroughWall
        for cusp in range(3):
            for e in range(2):
                element = meshrefinement._sourceElementiterator.next()
                meshrefinement.refineElementCubeStandard3d(element, numberInXi1, numberInXi2, numberInXi3)


def getSemilunarValveSinusPoints(centre, axisSide1, axisSide2, radius, sinusRadialDisplacement, startMidCusp, elementsCountAround = 6):
    '''
    Get points around a circle of radius with every second point displaced radially.
    :return: x[], d1[] for 6 points around
    '''
    assert elementsCountAround == 6, 'getArterialRootLowerPoints.  Only supports 6 elements around'
    px = []
    pd1 = []
    # every second outer points is displaced by sinusRadialDisplacement to make sinus dilatation
    nMidCusp = 0 if startMidCusp else 1
    radiusPlus = radius + sinusRadialDisplacement
    radiansPerElementAround = 2.0*math.pi/elementsCountAround
    for n in range(elementsCountAround):
        radiansAround = n*radiansPerElementAround
        midCusp = (n % 2) == nMidCusp
        r = radiusPlus if midCusp else radius
        rcosRadiansAround = r*math.cos(radiansAround)
        rsinRadiansAround = r*math.sin(radiansAround)
        px.append([ (centre[c] + rcosRadiansAround*axisSide1[c] + rsinRadiansAround*axisSide2[c]) for c in range(3) ])
        pd1.append([ radiansPerElementAround*(-rsinRadiansAround*axisSide1[c] + rcosRadiansAround*axisSide2[c]) for c in range(3) ])
    # smooth to get derivative in sinus
    sd1 = interp.smoothCubicHermiteDerivativesLine(px[1 - nMidCusp:3 - nMidCusp], pd1[1 - nMidCusp:3 - nMidCusp], fixStartDerivative=True, fixEndDirection=True)
    magSinus = magnitude(sd1[1])
    for n in range(nMidCusp, elementsCountAround, 2):
        pd1[n] = set_magnitude(pd1[n], magSinus)

    return px, pd1
