'''
Utility functions for element field templates shared by mesh generators.
'''
from cmlibs.maths.vectorops import add, cross, dot, magnitude, matrix_inv, mult, normalize, sub, transpose
from cmlibs.zinc.element import Elementbasis, Elementfieldtemplate
from cmlibs.zinc.node import Node
from cmlibs.zinc.result import RESULT_OK
from scaffoldmaker.utils.interpolation import (
    computeCubicHermiteEndDerivative, interpolateHermiteLagrangeDerivative, interpolateLagrangeHermiteDerivative)
import copy
import math


def getEftTermScaling(eft, functionIndex, termIndex):
    '''
    Convenience function to get the scale factor indexes scaling a term as a list.
    :eft:  Element field template to query.
    :functionIndex:  Function index from 1 to number of basis functions.
    :termIndex:  Term index from 1 to function number of terms.
    :return:  List containing any scale factor indexes scaling term. Empty if unscaled.
    '''
    scaleFactorCount, scaleFactorIndexes = eft.getTermScaling(functionIndex, termIndex, 1)
    if scaleFactorCount < 0:
        if eft.getNumberOfLocalScaleFactors() > 0:
            print('getEftTermScaling function', functionIndex, ' term', termIndex, ' scaleFactorCount ', scaleFactorCount)
        return []
    if scaleFactorCount == 0:
        return []
    if scaleFactorCount == 1:
        return [ scaleFactorIndexes ]
    scaleFactorCount, scaleFactorIndexes = eft.getTermScaling(functionIndex, termIndex, scaleFactorCount)
    return scaleFactorIndexes


def mapEftFunction1Node1Term(eft, function, localNode, valueLabel, version, scaleFactors):
    '''
    Set function of eft to map valueLabel, version from localNode with scaleFactors
    '''
    eft.setFunctionNumberOfTerms(function, 1)
    eft.setTermNodeParameter(function, 1, localNode, valueLabel, version)
    eft.setTermScaling(function, 1, scaleFactors)


def mapEftFunction1Node2Terms(eft, function, localNode, valueLabel1, version1, scaleFactors1, valueLabel2, version2, scaleFactors2):
    '''
    Set function of eft to sum 2 terms the respective valueLabels, versions from localNode with scaleFactors
    '''
    eft.setFunctionNumberOfTerms(function, 2)
    eft.setTermNodeParameter(function, 1, localNode, valueLabel1, version1)
    eft.setTermScaling(function, 1, scaleFactors1)
    eft.setTermNodeParameter(function, 2, localNode, valueLabel2, version2)
    eft.setTermScaling(function, 2, scaleFactors2)


def remapEftLocalNodes(eft, newNodeCount, localNodeIndexes):
    '''
    Remaps current local nodes to the new local ids, changing number of local nodes.
    Assumes node parameters are in use.
    :param localNodeIds: new local node identifiers starting at 1 for each current local node id - 1 referenced.
    '''
    functionCount = eft.getNumberOfFunctions()
    for f in range(1, functionCount + 1):
        termCount = eft.getFunctionNumberOfTerms(f)
        for t in range(1, termCount + 1):
            eft.setTermNodeParameter(f, t, localNodeIndexes[eft.getTermLocalNodeIndex(f, t) - 1], eft.getTermNodeValueLabel(f, t), eft.getTermNodeVersion(f, t))
    eft.setNumberOfLocalNodes(newNodeCount)


def remapEftNodeValueLabel(eft, localNodeIndexes, fromValueLabel, expressionTerms):
    '''
    Remap all uses of the given valueLabels to the expressionTerms.
    Note: Assumes valueLabel is currently single term and unscaled!
    :param localNodeIndexes:  List of local node indexes >= 1 to remap at.
    :param fromValueLabel:  Node value label to be remapped.
    :param expressionTerms: List of (valueLabel, scaleFactorIndexesList ) to remap to.
        e.g. [ (Node.VALUE_LABEL_D_DS2, []), (Node.VALUE_LABEL_D_DS3, [5, 6]) ]
    '''
    functionCount = eft.getNumberOfFunctions()
    for f in range(1, functionCount + 1):
        if eft.getFunctionNumberOfTerms(f) == 1:
            localNodeIndex = eft.getTermLocalNodeIndex(f, 1)
            if ((localNodeIndex in localNodeIndexes) and (eft.getTermNodeValueLabel(f, 1) == fromValueLabel) and
                    (not getEftTermScaling(eft, f, 1))):
                termCount = len(expressionTerms)
                eft.setFunctionNumberOfTerms(f, termCount)
                version = eft.getTermNodeVersion(f, 1)
                for t in range(1, termCount + 1):
                    expressionTerm = expressionTerms[t - 1]
                    eft.setTermNodeParameter(f, t, localNodeIndex, expressionTerm[0], version)
                    if expressionTerm[1]:
                        eft.setTermScaling(f, t, expressionTerm[1])


def remapEftNodeValueLabelVersion(eft, localNodeIndexes, fromValueLabel, expressionTerms):
    '''
    Remap all uses of the given valueLabels to the expressionTerms.
    Note: Assumes valueLabel is currently single term and unscaled!
    :param localNodeIndexes:  List of local node indexes >= 1 to remap at.
    :param fromValueLabel:  Node value label to be remapped.
    :param expressionTerms: List of (valueLabel, version, scaleFactorIndexesList) to remap to.
        e.g. [ (Node.VALUE_LABEL_D_DS2, 1, []), (Node.VALUE_LABEL_D_DS3, 2, [5, 6]) ]
    '''
    functionCount = eft.getNumberOfFunctions()
    for f in range(1, functionCount + 1):
        if eft.getFunctionNumberOfTerms(f) == 1:
            localNodeIndex = eft.getTermLocalNodeIndex(f, 1)
            if ((localNodeIndex in localNodeIndexes) and (eft.getTermNodeValueLabel(f, 1) == fromValueLabel) and
                    (not getEftTermScaling(eft, f, 1))):
                termCount = len(expressionTerms)
                eft.setFunctionNumberOfTerms(f, termCount)
                for t in range(1, termCount + 1):
                    expressionTerm = expressionTerms[t - 1]
                    eft.setTermNodeParameter(f, t, localNodeIndex, expressionTerm[0], expressionTerm[1])
                    if expressionTerm[2]:
                        eft.setTermScaling(f, t, expressionTerm[2])


def remapEftNodeValueLabelsVersion(eft, localNodeIndexes, valueLabels, version):
    '''
    Remap all uses of the given valueLabels to use the version.
    :param localNodeIndexes:  List of local node indexes >= 1 to remap at.
    :param valueLabels:  List of node value labels to be remapped.
    :param version: Version >= 1
    '''
    functionCount = eft.getNumberOfFunctions()
    for f in range(1, functionCount + 1):
        termCount = eft.getFunctionNumberOfTerms(f)
        for t in range(1, termCount + 1):
            localNodeIndex = eft.getTermLocalNodeIndex(f, t)
            valueLabel = eft.getTermNodeValueLabel(f, t)
            if (localNodeIndex in localNodeIndexes) and (valueLabel in valueLabels):
                result = eft.setTermNodeParameter(f, t, localNodeIndex, valueLabel, version)
                #print('remap result', result)


def remapEftNodeValueLabelWithNodes(eft, localNodeIndex, fromValueLabel, expressionTerms):
    '''
    Remap all uses of the given valueLabel to the expressionTerms.
    Note: Assumes valueLabel is currently single term and unscaled!
    :param localNodeIndex:  Local node index >= 1 to remap at.
    :param fromValueLabel:  Node value label to be remapped.
    :param expressionTerms: List of (localNodeIndex, valueLabel, scaleFactorIndexesList ) to remap to.
        e.g. [ (6, Node.VALUE_LABEL_VALUE, [1]), (7, Node.VALUE_LABEL_VALUE, []) ]
    '''
    functionCount = eft.getNumberOfFunctions()
    for f in range(1, functionCount + 1):
        if eft.getFunctionNumberOfTerms(f) == 1:
            if (eft.getTermLocalNodeIndex(f, 1) == localNodeIndex) and (eft.getTermNodeValueLabel(f, 1) == fromValueLabel) and (not getEftTermScaling(eft, f, 1)):
                termCount = len(expressionTerms)
                eft.setFunctionNumberOfTerms(f, termCount)
                version = eft.getTermNodeVersion(f, 1)
                for t in range(1, termCount + 1):
                    expressionTerm = expressionTerms[t - 1]
                    eft.setTermNodeParameter(f, t, expressionTerm[0], expressionTerm[1], version)
                    if expressionTerm[2]:
                        eft.setTermScaling(f, t, expressionTerm[2])


def scaleEftNodeValueLabels(eft, localNodeIndexes, valueLabels, addScaleFactorIndexes):
    '''
    Multiply all uses of the given ValueLabels by scale factor at scaleFactorIndex.
    Handles general maps.
    :param localNodeIndexes:  List of local node indexes >= 1 to scale value label.
    :param valueLabels:  List of node value label to be scaled.
    :param addScaleFactorIndexes: List of valid scale factor indexes >= 1 to append to existing scale factors.
    '''
    functionCount = eft.getNumberOfFunctions()
    for f in range(1, functionCount + 1):
        termCount = eft.getFunctionNumberOfTerms(f)
        for t in range(1, termCount + 1):
            if eft.getTermLocalNodeIndex(f, t) in localNodeIndexes:
                if eft.getTermNodeValueLabel(f, t) in valueLabels:
                    scaleFactorIndexes = getEftTermScaling(eft, f, t)
                    scaleFactorIndexes += addScaleFactorIndexes
                    eft.setTermScaling(f, t, scaleFactorIndexes)


def setEftScaleFactorIds(eft, globalScaleFactorIds, nodeScaleFactorIds, elementScaleFactorCount = 0):
    """
    Set general followed by node scale factor and element scale factor identifiers.
    :param eft: Elementfieldtemplate to modify.
    :param globalScaleFactorIds: List of global scale factor identifiers.
    :param nodeScaleFactorIds: List of node scale factor identifiers.
    :param elementScaleFactorCount: number of element scale factor identifiers.
    """
    eft.setNumberOfLocalScaleFactors(len(globalScaleFactorIds) + len(nodeScaleFactorIds) + elementScaleFactorCount)
    s = 1
    for id in globalScaleFactorIds:
        eft.setScaleFactorType(s, Elementfieldtemplate.SCALE_FACTOR_TYPE_GLOBAL_GENERAL)
        eft.setScaleFactorIdentifier(s, id)
        s += 1
    for id in nodeScaleFactorIds:
        eft.setScaleFactorType(s, Elementfieldtemplate.SCALE_FACTOR_TYPE_NODE_GENERAL)
        eft.setScaleFactorIdentifier(s, id)
        s += 1
    for e in range(elementScaleFactorCount):
        eft.setScaleFactorType(s, Elementfieldtemplate.SCALE_FACTOR_TYPE_ELEMENT_GENERAL)
        s += 1


def addEftNodeScaleFactorIds(eft, nodeScaleFactorIds):
    """
    Add more node-based scale factors to EFT.
    :param eft: Elementfieldtemplate to modify.
    :param nodeScaleFactorIds: List of node scale factor identifiers.
    :return: Local indexes of added scale factors.
    """
    oldCount = eft.getNumberOfLocalScaleFactors()
    addCount = len(nodeScaleFactorIds)
    eft.setNumberOfLocalScaleFactors(oldCount + addCount)
    s = oldCount + 1
    for id in nodeScaleFactorIds:
        eft.setScaleFactorType(s, Elementfieldtemplate.SCALE_FACTOR_TYPE_NODE_GENERAL)
        eft.setScaleFactorIdentifier(s, id)
    return [oldCount + n for n in range(1, addCount + 1)]


def addScaleEftNodesValueLabel(eft, localNodeIndexes, valueLabel, nodeScaleFactorId):
    """
    Create new node scale factors for each local node index with the nodeScaleFactorId and
    scale each instance of just value label at the local node indexes by the respective scale factor.
    :param eft:  Elementfieldtemplate to modify.
    :param localNodeIndexes: List of local node indexes to scale at.
    :param valueLabel: Node value label to scale.
    :param nodeScaleFactorId: Node-local scale factor ID to use.
    """
    sfIndexes = addEftNodeScaleFactorIds(eft, [nodeScaleFactorId] * len(localNodeIndexes))
    for n in range(len(localNodeIndexes)):
        scaleEftNodeValueLabels(eft, [localNodeIndexes[n]], [Node.VALUE_LABEL_D_DS3], [sfIndexes[n]])


def createEftElementSurfaceLayer(elementIn, eftIn, eftfactory, eftStd, removeNodeValueLabel=None):
    """
    Create eft for layer above xi3 = 1, from either tricubic hermite or bicubic hermite linear.
    Extracts mapping information from top xi3=1 surface of elementIn with eftIn.
    Returns eft for new element and scalefactors for it.
    :param elementIn: The element to build on.
    :param eftIn: The eft for the base element field. Assumed validated.
    :param eftfactory: The eft factory to create new eft from, if needed.
    Either eftfactory_tricubichermite or eftfactory_bicubichermitelinear.
    :param eftStd: The standard eft for the new element as created by eftfactory.createEftNoCrossDerivatives.
    Assumed unscaled, without cross derivatives, max 1 version and not collapsed. Never modified.
    :param removeNodeValueLabel: If not None, remove mappings from the value label in newly created layer.
    :return: eftStd or new eft if remapped, scalefactors
    """
    eft = eftStd
    scalefactors = None
    scaleFactorCountIn = eftIn.getNumberOfLocalScaleFactors()
    scaleFactorsUsed = [False]*scaleFactorCountIn  # flag scale factors used on top surface of element
    functionCountIn = eftIn.getNumberOfFunctions()
    halfFunctionCountIn = functionCountIn // 2
    elementBasis = eftfactory.getElementbasis()
    functionCountOut = elementBasis.getNumberOfFunctions()
    halfFunctionCountOut = functionCountOut // 2
    assert functionCountOut == eftStd.getNumberOfFunctions()
    assert (functionCountIn in (32, 64)) and (functionCountOut in (32, 64)) and (functionCountIn > functionCountOut)
    assert eftIn.getNumberOfLocalNodes() == 8
    assert eftStd.getNumberOfLocalNodes() == 8
    functionsPerNodeIn = functionCountIn // 8
    functionsPerNodeOut = functionCountOut // 8
    for i in range(halfFunctionCountOut):
        f = 1 + i
        fIn = halfFunctionCountIn + 1 + ((i // functionsPerNodeOut) * functionsPerNodeIn) + (i % functionsPerNodeOut)
        termCountIn = eftIn.getFunctionNumberOfTerms(fIn)
        noRemap = True
        newt = 0
        for j in range(termCountIn):
            t = 1 + j
            localNodeIndex = eftIn.getTermLocalNodeIndex(fIn, t) - 4
            nodeValueLabel = eftIn.getTermNodeValueLabel(fIn, t)
            nodeVersion = eftIn.getTermNodeVersion(fIn, t)
            scalingCount, scalingIndexes = eftIn.getTermScaling(fIn, t, 0)
            if scalingCount > 0:
                scalingIndexes = eftIn.getTermScaling(fIn, t, scalingCount)[1]
                #if (scalingCount > 1) or (scalingIndexes != 1):
                #    print("Scaling", scalingIndexes)
                if scalingCount == 1:
                    scalingIndexes = [scalingIndexes]
                for scalingIndex in scalingIndexes:
                    scaleFactorsUsed[scalingIndex - 1] = True
            if ((t > 1) or (scalingCount > 0)
                    or (eft.getTermLocalNodeIndex(f, 1) != localNodeIndex)
                    or (eft.getTermNodeValueLabel(f, 1) != nodeValueLabel)
                    or (eft.getTermNodeVersion(f, 1) != nodeVersion)):
                noRemap = False
                if eft is eftStd:
                    eft = eftfactory.createEftNoCrossDerivatives()
                    # initially use same number of scale factors as eftIn, with default type and identifiers
                    # later set type and identifiers, duplicating node scale factor indexes for top row
                    eft.setNumberOfLocalScaleFactors(scaleFactorCountIn)
                if t == 2:
                    eft.setFunctionNumberOfTerms(f, termCountIn)
                eft.setTermNodeParameter(f, t, localNodeIndex, nodeValueLabel, nodeVersion)
                if scalingCount > 0:
                    eft.setTermScaling(f, t, scalingIndexes)
                if nodeValueLabel != removeNodeValueLabel:
                    newt += 1
                    if newt > 1:
                        eft.setFunctionNumberOfTerms(f + halfFunctionCountOut, newt)
                    # top surface is the same, with node offset
                    assert localNodeIndex <= 4
                    assert RESULT_OK == eft.setTermNodeParameter(f + halfFunctionCountOut, newt, localNodeIndex + 4,
                                                                 nodeValueLabel, nodeVersion), localNodeIndex + 4
                    if scalingCount > 0:
                        eft.setTermScaling(f + halfFunctionCountOut, newt, scalingIndexes)
            else:
                newt = t
        assert noRemap or (newt > 0), "Element " + str(elementIn.getIdentifier()) + " f " + str(f) + \
            " terms " + str(termCountIn)
    if True in scaleFactorsUsed:
        # get used scale factors, reorder indexes
        result, scalefactorsIn = elementIn.getScaleFactors(eftIn, scaleFactorCountIn)
        if isinstance(scalefactorsIn, float):
            scalefactorsIn = [scalefactorsIn]
        scalefactors = []
        oldToNewScaleFactorIndex = {}
        globalScaleFactorIds = []
        for i in range(scaleFactorCountIn):
            if scaleFactorsUsed[i]:
                sf = i + 1
                scaleFactorType = eftIn.getScaleFactorType(sf)
                if scaleFactorType == Elementfieldtemplate.SCALE_FACTOR_TYPE_GLOBAL_GENERAL:
                    globalScaleFactorIds.append(eftIn.getScaleFactorIdentifier(sf))
                    scalefactors.append(scalefactorsIn[i])
                    oldToNewScaleFactorIndex[sf] = len(scalefactors)
                else:
                    # only other type current supported here:
                    assert scaleFactorType == Elementfieldtemplate.SCALE_FACTOR_TYPE_NODE_GENERAL
        nodeScaleFactorIds = []
        for i in range(scaleFactorCountIn):
            if scaleFactorsUsed[i]:
                sf = i + 1
                scaleFactorType = eftIn.getScaleFactorType(sf)
                if scaleFactorType == Elementfieldtemplate.SCALE_FACTOR_TYPE_NODE_GENERAL:
                    nodeScaleFactorIds.append(eftIn.getScaleFactorIdentifier(sf))
                    scalefactors.append(scalefactorsIn[i])
                    oldToNewScaleFactorIndex[sf] = len(scalefactors)
        # must duplicate node scale factors as separate for top layer
        setEftScaleFactorIds(eft, globalScaleFactorIds, nodeScaleFactorIds + nodeScaleFactorIds)
        globalScaleFactorCount = len(globalScaleFactorIds)
        nodeScaleFactorCount = len(nodeScaleFactorIds)
        if nodeScaleFactorCount:
            scalefactors += scalefactors[-nodeScaleFactorCount:]
        # remap scale factor indexes
        for f in range(1, functionCountOut + 1):
            termCount = eft.getFunctionNumberOfTerms(f)
            for t in range(1, termCount + 1):
                scalingCount = eft.getTermScaling(f, t, 0)[0]
                if scalingCount > 0:
                    oldScalingIndexes = eft.getTermScaling(f, t, scalingCount)[1]
                    if scalingCount == 1:
                        oldScalingIndexes = [oldScalingIndexes]
                    scalingIndexes = [oldToNewScaleFactorIndex[index] for index in oldScalingIndexes]
                    if nodeScaleFactorCount and (f > halfFunctionCountOut):
                        # remap top row node scale factor indexes to extra set
                        for i in range(scalingCount):
                            if scalingIndexes[i] > globalScaleFactorCount:
                                scalingIndexes[i] += nodeScaleFactorCount
                    if scalingIndexes != oldScalingIndexes:
                        eft.setTermScaling(f, t, scalingIndexes)
    elif eft is not eftStd:
        eft.setNumberOfLocalScaleFactors(0)
    if eft is not eftStd:
        assert eft.validate(), "Element " + str(elementIn.getIdentifier()) + " eft not validated"

    return eft, scalefactors


def determineTricubicHermiteEft(mesh, nodeParameters, nodeDerivativeFixedWeights=None, serendipity=False,
                                mapCrossDerivatives=False):
    """
    Determine the tricubic Hermite normal/serendipity element field template for
    interpolating node parameters at corners of a cube, by matching deltas
    between corners with node derivatives.
    Node lists use zinc ordering which varies nodes fastest over lower element coordinate.
    :param mesh: A Zinc mesh of dimension 3.
    :param nodeParameters: List over 8 local nodes in Zinc ordering of 4 parameter
    vectors x, d1, d2, d3 each with 3 components.
    :param nodeDerivativeFixedWeights: Optional list over 8 local nodes in Zinc ordering of
    list of up to 3 derivatives each containing weights for global d1, d2, d3 to fix that
    derivative, or None to use default search.
    Example: [[], [None, [0.0, -1.0, 1.0]], [], [], [], [], [], []]
    Local nodes 1, 3-8 use the standard search for all derivatives.
    Local node 2 forces d/dxi2 = -d2 + d3, d/dxi1 and d/dxi3 use the standard search.
    :param serendipity: Set to True for serendipity basis.
    :param mapCrossDerivatives: For non-serendipity Hermite basis, map cross derivatives
    as appropriate for first derivatives. For example, if d/dxi1 = d2 and d/dxi2 = d3,
    d2/dxi1dxi2 should map to -d23. If any element derivative is a sum of node derivatives,
    eliminate cross derivatives which use that direction.
    :return: eft, scale factors list [-1.0] or None. Returned eft can be further modified.
    """
    if nodeDerivativeFixedWeights is None:
        nodeDerivativeFixedWeights = [[]] * 8
    assert mesh.getDimension() == 3
    assert len(nodeParameters) == 8
    assert len(nodeParameters[0]) == 4
    assert len(nodeDerivativeFixedWeights) == 8
    assert not (serendipity and mapCrossDerivatives)
    delta12 = sub(nodeParameters[1][0], nodeParameters[0][0])
    delta34 = sub(nodeParameters[3][0], nodeParameters[2][0])
    delta56 = sub(nodeParameters[5][0], nodeParameters[4][0])
    delta78 = sub(nodeParameters[7][0], nodeParameters[6][0])
    delta13 = sub(nodeParameters[2][0], nodeParameters[0][0])
    delta24 = sub(nodeParameters[3][0], nodeParameters[1][0])
    delta57 = sub(nodeParameters[6][0], nodeParameters[4][0])
    delta68 = sub(nodeParameters[7][0], nodeParameters[5][0])
    delta15 = sub(nodeParameters[4][0], nodeParameters[0][0])
    delta26 = sub(nodeParameters[5][0], nodeParameters[1][0])
    delta37 = sub(nodeParameters[6][0], nodeParameters[2][0])
    delta48 = sub(nodeParameters[7][0], nodeParameters[3][0])
    deltas = [
        [delta12, delta13, delta15],
        [delta12, delta24, delta26],
        [delta34, delta13, delta37],
        [delta34, delta24, delta48],
        [delta56, delta57, delta15],
        [delta56, delta68, delta26],
        [delta78, delta57, delta37],
        [delta78, delta68, delta48]
    ]
    fieldmodule = mesh.getFieldmodule()
    tricubicHermiteBasis = fieldmodule.createElementbasis(
        3, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY if serendipity
        else Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
    eft = mesh.createElementfieldtemplate(tricubicHermiteBasis)
    scalefactors = None
    derivativeLabels = [Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3]
    crossDerivativeLabels = [Node.VALUE_LABEL_D2_DS1DS2, Node.VALUE_LABEL_D2_DS1DS3,
                             Node.VALUE_LABEL_D2_DS2DS3, Node.VALUE_LABEL_D3_DS1DS2DS3]
    for n in range(8):
        ln = n + 1
        derivativeExpressionTerms = []
        for ed in range(3):
            if mapCrossDerivatives:
                functionNumber = n * 8 + derivativeLabels[ed]
            else:
                functionNumber = n * 4 + ed + 2
            expressionTerms = []
            if nodeDerivativeFixedWeights[n] and (ed < len(nodeDerivativeFixedWeights[n])):
                weights = nodeDerivativeFixedWeights[n][ed]
                if weights is not None:
                    for nd in range(len(weights)):
                        weight = weights[nd]
                        if weight:
                            expressionTerms.append((derivativeLabels[nd], [1] if (weight < 0.0) else []))
            if not expressionTerms:
                delta = deltas[n][ed]  # delta side coordinates for nominal derivative direction
                magDelta = magnitude(delta)
                greatestSimilarity = 0.0  # can be negative in which can
                derivativeLabel = None
                for nd in range(3):
                    derivative = nodeParameters[n][nd + 1]
                    magDerivative = magnitude(derivative)
                    cosineSimilarity = dot(derivative, delta) / (magDerivative * magDelta)
                    magnitudeSimilarity = math.exp(-math.fabs((magDerivative - magDelta) / magDelta))
                    similarity = cosineSimilarity * magnitudeSimilarity
                    if math.fabs(similarity) > math.fabs(greatestSimilarity):
                        greatestSimilarity = similarity
                        derivativeLabel = derivativeLabels[nd]
                expressionTerms.append((derivativeLabel, [1] if (greatestSimilarity < 0.0) else []))
            termCount = len(expressionTerms)
            if termCount != 1:
                eft.setFunctionNumberOfTerms(functionNumber, termCount)
            for t in range(termCount):
                term = t + 1
                eft.setTermNodeParameter(functionNumber, term, ln, expressionTerms[t][0], 1)
                scaling = expressionTerms[t][1]
                if scaling:
                    if not scalefactors:
                        # assumes only
                        setEftScaleFactorIds(eft, [1], [])
                        scalefactors = [-1.0]
                    eft.setTermScaling(functionNumber, term, scaling)
            if mapCrossDerivatives:
                derivativeExpressionTerms.append(expressionTerms)
        if mapCrossDerivatives:
            for cd in range(4):
                functionNumber = n * 8 + crossDerivativeLabels[cd]
                derivativeIndexes = \
                    [0, 1] if (cd == 0) else \
                    [0, 2] if (cd == 1) else \
                    [1, 2] if (cd == 2) else \
                    [0, 1, 2]
                sign = 1.0
                crossDerivativeLabel = Node.VALUE_LABEL_VALUE
                for ed in derivativeIndexes:
                    if len(derivativeExpressionTerms[ed]) != 1:
                        crossDerivativeLabel = Node.VALUE_LABEL_VALUE
                        break
                    crossDerivativeLabel += derivativeExpressionTerms[ed][0][0] - Node.VALUE_LABEL_VALUE
                    if derivativeExpressionTerms[ed][0][1]:
                        sign *= -1.0
                if crossDerivativeLabel == Node.VALUE_LABEL_VALUE:
                    eft.setFunctionNumberOfTerms(functionNumber, 0)
                else:
                    eft.setTermNodeParameter(functionNumber, 1, ln, crossDerivativeLabel, 1)
                    if sign < 0.0:
                        eft.setTermScaling(functionNumber, 1, [1])
    return eft, scalefactors


class HermiteNodeLayout:
    """
    From the allowable derivative directions expressed as weighted sums, determines all
    possible permutations of them giving a right-handed corner without other directions inside them.
    """

    def __init__(self, directions, baseNodeLayout=None, limitDirections=None):
        """
        Construct a new node layout from either directions, or baseNodeLayout with limitDirections.
        :param directions: List of allowed directions as weighted sums of d1, d2, or d1, d2, d3.
        Lists with 2 weights are for bicubic; with each weight paired with the following
        in the list to make valid permutations. Do not supply if baseNodeLayout is supplied.
        :param baseNodeLayout: Optional base node layout to copy direections and permutations from
        when supplying limitDirections.
        :param limitDirections: Optional list over element directions of lists of allowable weights for that
        direction, or None to not filter. Default None for whole list does not filter any directions.
        For example, with no d3, [None, [[0.0, 1.0], [0.0, -1.0]]] places no limit on the first
        derivative, but derivative 2 must be [0, +/-1].
        """
        assert (directions and not baseNodeLayout) or (baseNodeLayout and limitDirections and not directions)
        self._directions = directions if directions else baseNodeLayout.getDirections()
        self._limitDirections = limitDirections if limitDirections else []
        self._cubicDimensions = len(self._directions[0])
        assert self._cubicDimensions in (2, 3)
        self._permutations = baseNodeLayout.getPermutations() if baseNodeLayout else self._determinePermutations()

    def _determinePermutations(self):
        """
        Determine permutations of weights from directions.
        :return: Permutations list.
        """
        permutations = []
        directionsCount = len(self._directions)
        if self._cubicDimensions == 2:
            for i in range(directionsCount):
                j = (i + 1) % directionsCount
                permutations.append((self._directions[i], self._directions[j]))
                # disallowing reverse as messes up bifurcations with 6, 10, 14 etc. around
                # would be needed for moebius strip
                # permutations.append((directions[j], directions[i]))
        else:
            normDir = [normalize(dir) for dir in self._directions]
            for i in range(directionsCount - 2):
                for j in range(i + 1, directionsCount - 1):
                    if dot(normDir[i], normDir[j]) < -0.9:
                        continue
                    ijCross = cross(normDir[i], normDir[j])
                    for k in range(j + 1, directionsCount):
                        if (dot(normDir[i], normDir[k]) < -0.9) or (dot(normDir[j], normDir[k]) < -0.9):
                            continue
                        dotCross = dot(ijCross, normDir[k])
                        if dotCross == 0.0:
                            continue
                        swizzle = dotCross < 0.0
                        # don't allow if another direction is within i j k octant by projecting onto basis
                        a = transpose([normDir[i], normDir[k], normDir[j]] if swizzle else
                                      [normDir[i], normDir[j], normDir[k]])
                        a_inv = matrix_inv(a)
                        for m in range(directionsCount):
                            if m in [i, j, k]:
                                continue
                            for n in range(3):
                                if dot(a_inv[n], normDir[m]) < 0.0:
                                    break  # outside octant
                            else:
                                break  # another direction is within i j k octant
                        else:
                            ii, jj, kk = (i, k, j) if swizzle else (i, j, k)
                            permutations.append((self._directions[ii], self._directions[jj], self._directions[kk]))
                            permutations.append((self._directions[jj], self._directions[kk], self._directions[ii]))
                            permutations.append((self._directions[kk], self._directions[ii], self._directions[jj]))
        return permutations

    def getComplexity(self):
        """
        :return: Integer complexity equal to number of axis permutations in node layout.
        """
        return len(self._permutations)

    def getDirections(self):
        return self._directions

    def getPermutations(self):
        return self._permutations

    def getDerivativeWeightsList(self, nodeDeltas, nodeDerivatives, localNodeIndex):
        """
        Get derivative weights for permutation making nodeDerivatives closest to nodeDeltas.
        :param nodeDeltas: List of 3 delta side coordinates to match.
        :param nodeDerivatives: List of [d1, d2, d3] parameters from node. d3 is None if linear through wall or 2-D.
        :param localNodeIndex: Local node index from 0 to 7 in Zinc order, which flips allowable directions.
        :return: List of weights for d1, d2, d3 to give d/dxi1, d/dxi2, d/dxi3.
        """
        derivativesPerNode = self._cubicDimensions
        if self._cubicDimensions == 2:
            flips = [localNodeIndex in [1, 3, 5, 7], localNodeIndex in [2, 3, 6, 7]]
            # need to swizzle indexes if odd number of flips, to keep right-handed layout
            flipCount = sum(1 for flip in flips if flip)
            swizzle = (flipCount % 2) == 1
            swizzleIndexes = [1, 0] if swizzle else [0, 1]
        else:
            flips = [localNodeIndex in [1, 3, 5, 7], localNodeIndex in [2, 3, 6, 7], localNodeIndex in [4, 5, 6, 7]]
            # need to swizzle indexes if odd number of flips, to keep right-handed layout
            flipCount = sum(1 for flip in flips if flip)
            swizzle = (flipCount % 2) == 1
            swizzleIndexes = [0, 2, 1] if swizzle else [0, 1, 2]

        # modify deltas to point inward towards opposite node
        inwardNodeDeltas = [[-d for d in nodeDeltas[i]] if flips[i] else nodeDeltas[i] for i in swizzleIndexes]
        derivativeWeightsList = None
        greatestSimilarity = -1.0
        for permutation in self._permutations:
            # skip permutations using directions not in any supplied limitDirections
            skipPermutation = False
            limitIndex = 0
            for limitDirections in self._limitDirections:
                if limitDirections:
                    weights = permutation[swizzleIndexes[limitIndex]]
                    if flips[limitIndex]:
                        weights = [-wt for wt in weights]
                    for limitDirection in limitDirections:
                        if magnitude(sub(weights, limitDirection)) < 1.0E-6:
                            break
                    else:
                        skipPermutation = True
                        break
                limitIndex += 1
            if skipPermutation:
                continue
            similarity = 0.0
            for d in range(derivativesPerNode):
                weights = permutation[d]
                derivative = [0.0, 0.0, 0.0]
                for i in range(derivativesPerNode):
                    if weights[i]:
                        weight = weights[i]
                        nodeDerivative = nodeDerivatives[i]
                        for c in range(3):
                            derivative[c] += weight * nodeDerivative[c]
                delta = inwardNodeDeltas[d]
                magDelta = magnitude(delta)
                magDerivative = magnitude(derivative)
                if magDerivative > 0.0:
                    cosineSimilarity = dot(derivative, delta) / (magDerivative * magDelta)
                    # magnitudeSimilarity = math.exp(-math.fabs((magDerivative - magDelta) / magDelta))
                    similarity += cosineSimilarity  # * magnitudeSimilarity
            if similarity > greatestSimilarity:
                greatestSimilarity = similarity
                derivativeWeightsList = permutation
        finalWeightsList = [
            [-w for w in derivativeWeightsList[swizzleIndexes[i]]] if flips[i]
            else derivativeWeightsList[swizzleIndexes[i]] for i in range(derivativesPerNode)
        ]
        return finalWeightsList


class HermiteNodeLayoutManager:
    """
    Managers permutations of hermite node layouts for easy recall.
    """

    def __init__(self):
        self._nodeLayoutRegularPermuted = HermiteNodeLayout(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        self._nodeLayoutRegularPermuted_d3Defined = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        self._nodeLayout5Way12 = HermiteNodeLayout(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        self._nodeLayout5Way12_d3Defined = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
             [0.0, 0.0, 1.0]])
        self._nodeLayout6Way12 = HermiteNodeLayout(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0]])
        self._nodeLayout6Way12_d3Defined = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [0.0, -1.0, 0.0],
             [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        self._nodeLayoutBifurcationCoreTransitionTopGeneral = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
             [0.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]])
        self._nodeLayoutBifurcationCoreTransitionBottomGeneral = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
             [0.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]])
        self._nodeLayout8Way12 = HermiteNodeLayout(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]])
        self._nodeLayout8Way12_d3Defined = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [0.0, -1.0, 0.0], [1.0, -1.0, 0.0],
             [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        self._nodeLayoutTriplePoint2DQ1 = HermiteNodeLayout(
            [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
        self._nodeLayoutTriplePoint2DQ2 = HermiteNodeLayout(
            [[0.0, -1.0], [1.0, 0.0], [-1.0, 1.0]])
        self._nodeLayoutTriplePoint2DQ3 = HermiteNodeLayout(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        self._nodeLayoutTriplePoint2DQ4 = HermiteNodeLayout(
            [[0.0, 1.0], [-1.0, 0.0], [1.0, -1.0]])
        self._nodeLayout3WayPoints12 = [
            HermiteNodeLayout([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
            HermiteNodeLayout([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
            HermiteNodeLayout([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
            HermiteNodeLayout([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])]
        self._nodeLayout3WayPoints13 = [
            HermiteNodeLayout([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
            HermiteNodeLayout([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
            HermiteNodeLayout([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
            HermiteNodeLayout([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])]
        self._nodeLayout3WayPoints23 = [
            HermiteNodeLayout([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, -1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            HermiteNodeLayout([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            HermiteNodeLayout([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            HermiteNodeLayout([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
        self._nodeLayout4WayPoints = [
            HermiteNodeLayout([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]),
            HermiteNodeLayout([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, -1.0]]),
            HermiteNodeLayout([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 1.0, -1.0]]),
            HermiteNodeLayout([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, -1.0]]),
            HermiteNodeLayout([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, -1.0, 1.0]]),
            HermiteNodeLayout([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, -1.0, 1.0]]),
            HermiteNodeLayout([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 1.0, 1.0]]),
            HermiteNodeLayout([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]])]
        self._nodeLayoutTriplePointTopLeft = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
        self._nodeLayoutTriplePointTopRight = HermiteNodeLayout(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 1.0]])
        self._nodeLayoutTriplePointBottomLeft = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, -1.0]])
        self._nodeLayoutTriplePointBottomRight = HermiteNodeLayout(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
        self._nodeLayoutTriplePoint23Front = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 1.0]])
        self._nodeLayoutTriplePoint23Back = HermiteNodeLayout(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        self.nodeLayoutsBifurcation6WayTriplePoint = {}

    def getNodeLayoutRegularPermuted(self, d3Defined, limitDirections=None):
        """
        Get node layout for permutations of +/- d1, d2, d3, optionally limiting some directions.
        Note that variants using limitDirections are not currently stored so client should hold them if reusing.
        :param d3Defined: Set to True to use tricubic variant with d3 defined, otherwise bicubic is used.
        :param limitDirections: Optional list over element directions of lists of allowable weights for that
        direction, or None to not filter. Default None for whole list does not filter any directions.
        For example, with d3, [None, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], [[0.0, 0.0, 1.0]]] places no
        limits on the first derivative, but derivative 2 must be [0, +/-1, 0] and d3 must be [0, 0, 1].
        :return: HermiteNodeLayout.
        """
        nodeLayout = self._nodeLayoutRegularPermuted_d3Defined if d3Defined else self._nodeLayoutRegularPermuted
        if limitDirections:
            nodeLayout = HermiteNodeLayout(None, nodeLayout, limitDirections)
        return nodeLayout

    def getNodeLayout5Way12(self, d3Defined, limitDirections=None):
        """
        Get node layout for 5-way junction in 1-2 plane, including d1, -d1, d2, -d2, d1 + d2.
        :param d3Defined: Set to True to use tricubic variant with d3 defined, otherwise bicubic is used.
        :param limitDirections: Optional list over element directions of lists of allowable weights for that
        direction, or None to not filter. Default None for whole list does not filter any directions.
        For example, with d3, [None, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], [[0.0, 0.0, 1.0]]] places no
        limits on the first derivative, but derivative 2 must be [0, +/-1, 0] and d3 must be [0, 0, 1].
        :return: HermiteNodeLayout.
        """
        nodeLayout = self._nodeLayout5Way12_d3Defined if d3Defined else self._nodeLayout5Way12
        if limitDirections:
            nodeLayout = HermiteNodeLayout(None, nodeLayout, limitDirections)
        return nodeLayout

    def getNodeLayout6Way12(self, d3Defined, limitDirections=None):
        """
        Get node layout for 6-way junction in 1-2 plane, including d1 + d2, -d1 - d2.
        :param d3Defined: Set to True to use tricubic variant with d3 defined, otherwise bicubic is used.
        :param limitDirections: Optional list over element directions of lists of allowable weights for that
        direction, or None to not filter. Default None for whole list does not filter any directions.
        For example, with d3, [None, [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], [[0.0, 0.0, 1.0]]] places no
        limits on the first derivative, but derivative 2 must be [0, +/-1, 0] and d3 must be [0, 0, 1].
        :return: HermiteNodeLayout.
        """
        nodeLayout = self._nodeLayout6Way12_d3Defined if d3Defined else self._nodeLayout6Way12
        if limitDirections:
            nodeLayout = HermiteNodeLayout(None, nodeLayout, limitDirections)
        return nodeLayout

    def getNodeLayout8Way12(self, d3Defined):
        """
        Get node layout for 8-way junction in 1-2 plane, including d1 + d2, -d1 + d2, -d1 - d2, d1 - d2.
        :param d3Defined: Set to True to use tricubic variant with d3 defined, otherwise bicubic is used.
        :return: HermiteNodeLayout.
        """
        return self._nodeLayout8Way12_d3Defined if d3Defined else self._nodeLayout8Way12

    def getNodeLayoutTriplePoint23Front(self):
        """
        Get node layout for triple-point junction in 2-3 plane, including d1, -d1, -d3, -d2, d2 + d3.
        :return: HermiteNodeLayout.
        """
        nodeLayout = self._nodeLayoutTriplePoint23Front
        return nodeLayout

    def getNodeLayoutTriplePoint23Back(self):
        """
        Get node layout for triple point junction in 2-3 plane, including d1, -d1, -d3, -d2, d2 - d3.
        :return: HermiteNodeLayout.
        """
        nodeLayout = self._nodeLayoutTriplePoint23Back

        return nodeLayout

    def getNodeLayoutTriplePoint2D(self):
        """
        Get node layout for triple-point corners of 2D quadrants.
        :return: List of 4 HermiteNodeLayout.
        """
        nodeLayouts = [self._nodeLayoutTriplePoint2DQ1, self._nodeLayoutTriplePoint2DQ2,
                       self._nodeLayoutTriplePoint2DQ3, self._nodeLayoutTriplePoint2DQ4]
        return nodeLayouts

    def getNodeLayout3WayPoints12(self):
        """
        Get 3-way node layouts for quadrants 12 = NN, NP, PN, PP.
        :return: List of 4 HermiteNodeLayout.
        """
        return self._nodeLayout3WayPoints12

    def getNodeLayout3WayPoints13(self):
        """
        Get 3-way node layouts for quadrants 13 = NN, NP, PN, PP.
        :return: List of 4 HermiteNodeLayout.
        """
        return self._nodeLayout3WayPoints13

    def getNodeLayout3WayPoints23(self):
        """
        Get 3-way node layouts for quadrants 23 = NN, NP, PN, PP.
        :return: List of 4 HermiteNodeLayout.
        """
        return self._nodeLayout3WayPoints23

    def getNodeLayout4WayPoints(self):
        """
        Get node layouts from a regular core for octants 123: NNN, NNP, NPN, NPP, PNN, PNP, PPN, PPP.
        :return: List of 8 HermiteNodeLayout.
        """
        return self._nodeLayout4WayPoints

    def getNodeLayoutTriplePoint(self):
        """
        Get node layout for triple-point corners of core box elements. There are four corners (Top Left, Top Right,
        Bottom Left, and Bottom Right) each with its specific node layout.
        :return: List of 4 HermiteNodeLayout.
        """
        nodeLayouts = [self._nodeLayoutTriplePointTopLeft, self._nodeLayoutTriplePointTopRight,
                       self._nodeLayoutTriplePointBottomLeft, self._nodeLayoutTriplePointBottomRight]
        return nodeLayouts

    def getNodeLayoutBifurcation6WayTriplePoint(self, segmentsIn, sequence, maxMajorSegment, top):
        """
        Get node layout for special case of 3-way junction on core transition corner of box.
        :param segmentsIn: Directions of 3 segments relative to junction.
        :param sequence: Sequence of bifurcation: [0, 1, 2] (normal) or [0, 2, 1] (reverse)
        :param maxMajorSegment: Which segment has the most major elements, i.e. has elements going to both others.
        :param top: True for top, False for bottom.
        :return: HermiteNodeLayout
        """
        reverse = sequence == [0, 2, 1]
        sir = (segmentsIn[0], segmentsIn[1], segmentsIn[2], reverse)
        layoutIndex = maxMajorSegment
        if not top:
            layoutIndex += 3
        nodeLayouts = self.nodeLayoutsBifurcation6WayTriplePoint.get(sir)
        if not nodeLayouts:
            if sir in [(False, False, False, False), (False, False, False, True),
                       (True, False, False, False), (True, False, False, True),
                       (True, True, True, False), (True, True, True, True)]:
                nodeLayoutBifurcationCoreTransitionMOP = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0], [-1.0, 0.0, 1.0]])
                nodeLayoutBifurcationCoreTransitionOMP = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0], [0.0, -1.0, 1.0]])
                nodeLayoutBifurcationCoreTransitionPPP = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]])
                nodeLayoutBifurcationCoreTransitionMOM = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0], [-1.0, 0.0, -1.0]])
                nodeLayoutBifurcationCoreTransitionOMM = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0], [0.0, -1.0, -1.0]])
                nodeLayoutBifurcationCoreTransitionPPM = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0], [1.0, 1.0, -1.0]])
                nodeLayouts = (
                    nodeLayoutBifurcationCoreTransitionPPP,
                    nodeLayoutBifurcationCoreTransitionMOP,
                    nodeLayoutBifurcationCoreTransitionOMP,
                    nodeLayoutBifurcationCoreTransitionPPM,
                    nodeLayoutBifurcationCoreTransitionMOM,
                    nodeLayoutBifurcationCoreTransitionOMM)
                self.nodeLayoutsBifurcation6WayTriplePoint[(False, False, False, False)] = nodeLayouts
                self.nodeLayoutsBifurcation6WayTriplePoint[(True, False, False, False)] = nodeLayouts
                self.nodeLayoutsBifurcation6WayTriplePoint[(True, True, True, False)] = nodeLayouts
                nodeLayoutsReverse = (
                    nodeLayoutBifurcationCoreTransitionPPP,
                    nodeLayoutBifurcationCoreTransitionOMP,
                    nodeLayoutBifurcationCoreTransitionMOP,
                    nodeLayoutBifurcationCoreTransitionPPM,
                    nodeLayoutBifurcationCoreTransitionOMM,
                    nodeLayoutBifurcationCoreTransitionMOM)
                self.nodeLayoutsBifurcation6WayTriplePoint[(False, False, False, True)] = nodeLayoutsReverse
                self.nodeLayoutsBifurcation6WayTriplePoint[(True, False, False, True)] = nodeLayoutsReverse
                self.nodeLayoutsBifurcation6WayTriplePoint[(True, True, True, True)] = nodeLayoutsReverse
                nodeLayouts = self.nodeLayoutsBifurcation6WayTriplePoint.get(sir)
            elif sir in [(True, True, False, False), (True, True, False, True)]:
                nodeLayoutBifurcationCoreTransitionPOP = HermiteNodeLayout(
                    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0], [1.0, 0.0, 1.0]])
                nodeLayoutBifurcationCoreTransitionOPP = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
                nodeLayoutBifurcationCoreTransitionMMP = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0],
                     [0.0, 0.0, -1.0], [-1.0, -1.0, 1.0]])
                nodeLayoutBifurcationCoreTransitionPOM = HermiteNodeLayout(
                    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
                nodeLayoutBifurcationCoreTransitionOPM = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0], [0.0, 1.0, -1.0]])
                nodeLayoutBifurcationCoreTransitionMMM = HermiteNodeLayout(
                    [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 1.0, 0.0],
                     [0.0, 0.0, 11.0], [-1.0, -1.0, -1.0]])
                nodeLayouts = (
                    nodeLayoutBifurcationCoreTransitionPOP,
                    nodeLayoutBifurcationCoreTransitionOPP,
                    nodeLayoutBifurcationCoreTransitionMMP,
                    nodeLayoutBifurcationCoreTransitionPOM,
                    nodeLayoutBifurcationCoreTransitionOPM,
                    nodeLayoutBifurcationCoreTransitionMMM)
                self.nodeLayoutsBifurcation6WayTriplePoint[(True, True, False, False)] = nodeLayouts
                nodeLayoutsReverse = (
                    nodeLayoutBifurcationCoreTransitionOPP,
                    nodeLayoutBifurcationCoreTransitionPOP,
                    nodeLayoutBifurcationCoreTransitionMMP,
                    nodeLayoutBifurcationCoreTransitionOPM,
                    nodeLayoutBifurcationCoreTransitionPOM,
                    nodeLayoutBifurcationCoreTransitionMMM)
                self.nodeLayoutsBifurcation6WayTriplePoint[(True, True, False, True)] = nodeLayoutsReverse
                nodeLayouts = self.nodeLayoutsBifurcation6WayTriplePoint.get(sir)
            if not nodeLayouts:
                print("getNodeLayoutBifurcation6WayTriplePoint not implemented for case:", sir)
                if top:
                    return self._nodeLayoutBifurcationCoreTransitionTopGeneral
                else:
                    return self._nodeLayoutBifurcationCoreTransitionBottomGeneral
        return nodeLayouts[layoutIndex]

def determineCubicHermiteSerendipityEft(mesh, nodeParameters, nodeLayouts):
    """
    Determine the bicubic or tricubic Hermite serendipity element field template for
    interpolating node parameters at corners of a square or cube, by matching deltas
    between corners with node derivatives.
    Node lists use zinc ordering which varies nodes fastest over lower element coordinate.
    :param mesh: A Zinc mesh of dimension 2 or 3.
    :param nodeParameters: List over 4 (2-D) or 8 (3-D) local nodes in Zinc ordering of
    4 parameter vectors x, d1, d2, d3 each with 3 components. d3 is not used in 2-D, and in
    3-d if d3 is omitted the basis is linear in that direction.
    :param nodeLayouts: List over 4 or 8 local nodes of HermiteNodeLayout objects describing the
    list of allowable derivative combinations for each corner node. None value for a node
    keeps the standard, regular layout.
    :return: eft, scale factors list [-1.0] or None. Returned eft can be further modified.
    """
    meshDimension = mesh.getDimension()
    nodesCount = len(nodeParameters)
    assert ((meshDimension == 2) and (nodesCount == 4)) or ((meshDimension == 3) and (nodesCount == 8))
    assert len(nodeParameters[0]) == 4
    d3Defined = (meshDimension == 3) and (nodeParameters[0][3] is not None)
    assert len(nodeLayouts) == nodesCount
    delta12 = sub(nodeParameters[1][0], nodeParameters[0][0])
    delta34 = sub(nodeParameters[3][0], nodeParameters[2][0])
    delta13 = sub(nodeParameters[2][0], nodeParameters[0][0])
    delta24 = sub(nodeParameters[3][0], nodeParameters[1][0])
    if meshDimension == 3:
        delta15 = sub(nodeParameters[4][0], nodeParameters[0][0])
        delta26 = sub(nodeParameters[5][0], nodeParameters[1][0])
        delta37 = sub(nodeParameters[6][0], nodeParameters[2][0])
        delta48 = sub(nodeParameters[7][0], nodeParameters[3][0])
    else:
        delta15 = delta26 = delta37 = delta48 = [0.0, 0.0, 0.0]
    deltas = [
        [delta12, delta13, delta15],
        [delta12, delta24, delta26],
        [delta34, delta13, delta37],
        [delta34, delta24, delta48]
    ]
    if meshDimension == 3:
        delta56 = sub(nodeParameters[5][0], nodeParameters[4][0])
        delta78 = sub(nodeParameters[7][0], nodeParameters[6][0])
        delta57 = sub(nodeParameters[6][0], nodeParameters[4][0])
        delta68 = sub(nodeParameters[7][0], nodeParameters[5][0])
        deltas += [
            [delta56, delta57, delta15],
            [delta56, delta68, delta26],
            [delta78, delta57, delta37],
            [delta78, delta68, delta48]
        ]
    fieldmodule = mesh.getFieldmodule()
    elementbasis = fieldmodule.createElementbasis(meshDimension, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY)
    if (meshDimension == 3) and not d3Defined:
        elementbasis.setFunctionType(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh.createElementfieldtemplate(elementbasis)
    scalefactors = None
    derivativeLabels = [Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3]
    derivativesPerNode = 3 if d3Defined else 2
    functionsPerNode = 1 + derivativesPerNode
    # order local nodes from default then simplest to most complex node layout
    nodeOrder = []
    for n in range(nodesCount):
        if not nodeLayouts[n]:
            nodeOrder.append(n)
    while len(nodeOrder) < nodesCount:
        lowestComplexity = 0
        next_n = None
        for n in range(nodesCount):
            if n in nodeOrder:
                continue
            complexity = nodeLayouts[n].getComplexity()
            if (next_n is None) or (complexity < lowestComplexity):
                lowestComplexity = complexity
                next_n = n
        nodeOrder.append(next_n)
    for n in nodeOrder:
        ln = n + 1
        nodeLayout = nodeLayouts[n]
        nodeDerivatives = [
            nodeParameters[n][1],
            nodeParameters[n][2],
            nodeParameters[n][3] if d3Defined else None]
        derivativeWeightsList =\
            nodeLayout.getDerivativeWeightsList(deltas[n], nodeDerivatives, n) if nodeLayout else None
        for ed in range(derivativesPerNode):
            if nodeLayout:
                derivativeWeights = derivativeWeightsList[ed]
                functionNumber = n * functionsPerNode + ed + 2
                termsCount = sum(1 for wt in derivativeWeights if wt != 0.0)
                eft.setFunctionNumberOfTerms(functionNumber, termsCount)
                term = 0
                elementDerivative = [0.0, 0.0, 0.0]
                for i in range(derivativesPerNode):
                    weight = derivativeWeights[i]
                    if weight:
                        term += 1
                        eft.setTermNodeParameter(functionNumber, term, ln, derivativeLabels[i], 1)
                        if weight < 0.0:
                            if not scalefactors:
                                setEftScaleFactorIds(eft, [1], [])
                                scalefactors = [-1.0]
                            eft.setTermScaling(functionNumber, term, [1])
                        for c in range(3):
                            elementDerivative[c] += weight * nodeDerivatives[i][c]
            else:
                elementDerivative = nodeDerivatives[ed]
            # update delta to equal the exact derivative
            oldDeltaMagnitude = magnitude(deltas[n][ed])
            deltas[n][ed] = elementDerivative
            # get other node index interpolated with this element derivative
            on = n ^ (1 << ed)
            if nodeOrder.index(on) > nodeOrder.index(n):
                derivativeMagnitude = magnitude(elementDerivative)
                newMagnitude = 2.0 / ((1.0 / oldDeltaMagnitude) + (1.0 / derivativeMagnitude))
                elementDerivative = mult(elementDerivative, newMagnitude / derivativeMagnitude)
                # update other node delta to work in with this derivative
                otherElementDerivative = (
                    interpolateHermiteLagrangeDerivative(nodeParameters[n][0], elementDerivative, nodeParameters[on][0], 1.0)
                    if (on > n) else
                    interpolateLagrangeHermiteDerivative(nodeParameters[on][0], nodeParameters[n][0], elementDerivative, 0.0))
                deltas[on][ed] = otherElementDerivative

    return eft, scalefactors


CubicHermiteSerendipityValueLabels = [
    Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D_DS3]


def getTricubicHermiteSerendipityElementNodeParameter(eft, scalefactors, nodeParameters, localNodeIndex, valueLabel):
    """
    Get effective element parameter for basis' original valueLabel at local node index.
    Currently assumes extra versions are not used, but this is not checked.
    :param eft: Element field template.
    :param scalefactors: List of scale factors.
    :param nodeParameters: List over 8 (3-D) local nodes in Zinc ordering of 4 parameter vectors
    x, d1, d2, d3 each with 3 components.
    :param localNodeIndex: 1-based local node index on original basis.
    :param valueLabel: A value label mapped by CubicHermiteSerendipityFunctionOffset.
    :return: Parameter.
    """
    parameter = [0.0, 0.0, 0.0]
    bn = localNodeIndex - 1
    valueIndex = CubicHermiteSerendipityValueLabels.index(valueLabel)
    func = 4 * bn + valueIndex + 1
    termCount = eft.getFunctionNumberOfTerms(func)
    for term in range(1, termCount + 1):
        termValueLabel = eft.getTermNodeValueLabel(func, term)
        termValueIndex = CubicHermiteSerendipityValueLabels.index(termValueLabel)
        component = nodeParameters[eft.getTermLocalNodeIndex(func, term) - 1][termValueIndex]
        scalefactorCount, scalefactorIndexes = eft.getTermScaling(func, term, 0)
        if scalefactorCount > 0:
            scalefactorCount, scalefactorIndexes = eft.getTermScaling(func, term, scalefactorCount)
            if scalefactorCount == 1:
                scalefactorIndexes = [scalefactorIndexes]
            scalefactor = 1.0
            for sfi in scalefactorIndexes:
                scalefactor *= scalefactors[sfi - 1]
            component = mult(component, scalefactor)
        parameter = add(parameter, component)
    return parameter


def addTricubicHermiteSerendipityEftParameterScaling(eft, scalefactors, nodeParameters, localNodeIndexes, valueLabel,
                                                     version=1):
    """
    Scale tricubic hermite serendipity element field template value label at local nodes to fit actual parameters.
    Calculation assumes value label parameter is an unscaled single term expression at the local nodes.
    :param eft: Element field template e.g. returned from determineCubicHermiteSerendipityEft.
    Note: be careful to pass a separate eft to this function from any cached which do not needing scaling!
    :param scalefactors: Existing scale factors for element
    :param nodeParameters: As passed to determineCubicHermiteSerendipityEft: list over 8 (3-D)
    local nodes in Zinc ordering of 4 parameter vectors x, d1, d2, d3 each with 3 components.
    :param localNodeIndexes: Local node indexes to scale value label at. Currently must be in [5, 6, 7, 8].
    :param valueLabel: Single value label to scale. Currently only implemened for D_DS3.
    :param version: Set to a number > 1 to use that derivative version instead of scale factors, and client is
    responsible for assigning that derivative version in proportion to the returned scale factors.
    :return: Modified eft, final scalefactors, add scalefactors (multiplying the affected derivatives; these are
    included in final scalefactors if version == 1, otherwise client must use these to scale versioned derivatives).
    """
    assert len(nodeParameters) == 8
    assert valueLabel == Node.VALUE_LABEL_D_DS3
    addScalefactors = []
    for ln in localNodeIndexes:
        assert ln in [5, 6, 7, 8]
        na = ln - 5
        nb = na + 4
        xa = nodeParameters[na][0]
        da = getTricubicHermiteSerendipityElementNodeParameter(eft, scalefactors, nodeParameters, na + 1, valueLabel)
        xb = nodeParameters[nb][0]
        db = nodeParameters[nb][3]
        dbScaled = computeCubicHermiteEndDerivative(xa, da, xb, db)
        scalefactor = magnitude(dbScaled) / magnitude(db)
        addScalefactors.append(scalefactor)
    if version == 1:
        newScalefactors = scalefactors + addScalefactors if scalefactors else addScalefactors
        addScaleEftNodesValueLabel(eft, localNodeIndexes, Node.VALUE_LABEL_D_DS3, 3)
        return eft, newScalefactors, addScalefactors
    remapEftNodeValueLabelsVersion(eft, localNodeIndexes, [valueLabel], version)
    return eft, scalefactors, addScalefactors
