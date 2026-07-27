"""
Generates a 3D spinal nerve scaffold using tube network mesh.
"""
from cmlibs.maths.vectorops import cross, mult, set_magnitude, magnitude, sub, euler_to_rotation_matrix, \
    matrix_vector_mult, add
from cmlibs.utils.zinc.field import Field, find_or_create_field_coordinates
from cmlibs.zinc.node import Node
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, findOrCreateAnnotationGroupForTerm
from scaffoldmaker.annotation.spinal_nerve_terms import get_spinal_nerve_term
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils.interpolation import sampleCubicHermiteCurves, smoothCubicHermiteDerivativesLine
from scaffoldmaker.utils.networkmesh import NetworkMesh
from scaffoldmaker.utils.tubenetworkmesh import TubeNetworkMeshBuilder, TubeNetworkMeshGenerateData
import math


class MeshType_1d_human_spinal_nerve_network_layout1(MeshType_1d_network_layout1):
    """
    Defines spinal nerve network layout.
    """

    @classmethod
    def getName(cls):
        return "1D Human Spinal Nerve Network Layout 1"

    @classmethod
    def getParameterSetNames(cls):
        return ["Default",
                "Human 1",
                "Human pair 1",
                "Human whole spine 1"]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):

        options = {}
        options["Base parameter set"] = parameterSetName
        if parameterSetName == "Human 1":
            options["Structure"] = (
                "1-2.1,"
                "2.2-3-4-5,"
                "2.3-6-7, 7-8-9-10-11, 11-12-13")
        else:
            options["Structure"] = (
                "1-2.1,"
                "2.2-3-4-5,"
                "2.3-6-7, 7-8-9-10-11, 11-12-13,"
                "14-15.1,"
                "15.2-16-17-18,"
                "15.3-19-20, 20-21-22-23-24, 24-25-26")

        options["Define inner coordinates"] = True
        options["Spinal nerve root length"] = 0.5
        options["Spinal nerve root diameter"] = 0.1
        options["Dorsal root diameter"] = 0.1
        options["Dorsal root ganglion thickest diameter"] = 0.18
        options["Ventral root diameter"] = 0.1
        options["Spinal cord major diameter"] = 1.5
        options["Spinal cord minor diameter"] = 0.5
        options["Distance between bifurcation to spinal cord"] = 0.1
        options['Pitch angle degrees'] = 10.0
        options["Inner proportion default"] = 0.75

        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Spinal nerve root length",
            "Spinal nerve root diameter",
            "Dorsal root diameter",
            "Dorsal root ganglion thickest diameter",
            "Ventral root diameter",
            "Spinal cord major diameter",
            "Spinal cord minor diameter",
            "Distance between bifurcation to spinal cord",
            "Pitch angle degrees",
            "Inner proportion default"
        ]

    @classmethod
    def checkOptions(cls, options):
        dependentChanges = False
        for key in [
            "Spinal nerve root length",
            "Spinal nerve root diameter",
            "Dorsal root diameter",
            "Dorsal root ganglion thickest diameter",
            "Ventral root diameter",
            "Spinal cord major diameter",
            "Spinal cord minor diameter",
            "Distance between bifurcation to spinal cord"
        ]:
            if options[key] < 0.1:
                options[key] = 0.1
        for key in [
            "Inner proportion default",
        ]:
            if options[key] < 0.1:
                options[key] = 0.1
            elif options[key] > 0.9:
                options[key] = 0.9
        for key, angleRange in {
                "Pitch angle degrees": (-60.0, 60.0)
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
        nerveRootRadius = 0.5 * options["Spinal nerve root diameter"]
        nerveRootLength = options["Spinal nerve root length"]
        dorsalRootRadius = 0.5 * options["Dorsal root diameter"]
        ventralRootRadius = 0.5 * options["Ventral root diameter"]
        dRGThickestRadius = 0.5 * options["Dorsal root ganglion thickest diameter"]
        nervePitch = options["Pitch angle degrees"]
        spinalCordMajorRadius = 0.5 * options["Spinal cord major diameter"]
        spinalCordMinorRadius = 0.5 * options["Spinal cord minor diameter"]
        distToSpinalCord = options["Distance between bifurcation to spinal cord"]
        innerProportionDefault = options["Inner proportion default"]

        numberOfNetworkNodes = 13
        numberOfLevels = 1
        scales = [1.0]
        rotMatRight = euler_to_rotation_matrix([0.0, math.radians(-nervePitch), 0.0])
        rotMatLeft = euler_to_rotation_matrix([0.0, math.radians(nervePitch), 0.0])
        scCentres = [[0.0, 0.0, 0.0]]

        levelGroups = []
        segmentGroups = []
        annotationGroups = []

        if parameterSetName == "Human whole spine 1":
            numberOfLevels = 31
            cervicalGroup = AnnotationGroup(region, get_spinal_nerve_term("cervical spinal nerve"))
            c1Group = AnnotationGroup(region, get_spinal_nerve_term("C1 spinal nerve"))
            c2Group = AnnotationGroup(region, get_spinal_nerve_term("C2 spinal nerve"))
            c3Group = AnnotationGroup(region, get_spinal_nerve_term("C3 spinal nerve"))
            c4Group = AnnotationGroup(region, get_spinal_nerve_term("C4 spinal nerve"))
            c5Group = AnnotationGroup(region, get_spinal_nerve_term("C5 spinal nerve"))
            c6Group = AnnotationGroup(region, get_spinal_nerve_term("C6 spinal nerve"))
            c7Group = AnnotationGroup(region, get_spinal_nerve_term("C7 spinal nerve"))
            c8Group = AnnotationGroup(region, get_spinal_nerve_term("C8 spinal nerve"))

            thoracicGroup = AnnotationGroup(region, get_spinal_nerve_term("thoracic nerve"))
            t1Group = AnnotationGroup(region, get_spinal_nerve_term("T1 spinal nerve"))
            t2Group = AnnotationGroup(region, get_spinal_nerve_term("T2 spinal nerve"))
            t3Group = AnnotationGroup(region, get_spinal_nerve_term("T3 spinal nerve"))
            t4Group = AnnotationGroup(region, get_spinal_nerve_term("T4 spinal nerve"))
            t5Group = AnnotationGroup(region, get_spinal_nerve_term("T5 spinal nerve"))
            t6Group = AnnotationGroup(region, get_spinal_nerve_term("T6 spinal nerve"))
            t7Group = AnnotationGroup(region, get_spinal_nerve_term("T7 spinal nerve"))
            t8Group = AnnotationGroup(region, get_spinal_nerve_term("T8 spinal nerve"))
            t9Group = AnnotationGroup(region, get_spinal_nerve_term("T9 spinal nerve"))
            t10Group = AnnotationGroup(region, get_spinal_nerve_term("T10 spinal nerve"))
            t11Group = AnnotationGroup(region, get_spinal_nerve_term("T11 spinal nerve"))
            t12Group = AnnotationGroup(region, get_spinal_nerve_term("T12 spinal nerve"))

            lumbarGroup = AnnotationGroup(region, get_spinal_nerve_term("lumbar nerve"))
            l1Group = AnnotationGroup(region, get_spinal_nerve_term("L1 spinal nerve"))
            l2Group = AnnotationGroup(region, get_spinal_nerve_term("L2 spinal nerve"))
            l3Group = AnnotationGroup(region, get_spinal_nerve_term("L3 spinal nerve"))
            l4Group = AnnotationGroup(region, get_spinal_nerve_term("L4 spinal nerve"))
            l5Group = AnnotationGroup(region, get_spinal_nerve_term("L5 spinal nerve"))

            sacralGroup = AnnotationGroup(region, get_spinal_nerve_term("sacral nerve"))
            s1Group = AnnotationGroup(region, get_spinal_nerve_term("S1 spinal nerve"))
            s2Group = AnnotationGroup(region, get_spinal_nerve_term("S2 spinal nerve"))
            s3Group = AnnotationGroup(region, get_spinal_nerve_term("S3 spinal nerve"))
            s4Group = AnnotationGroup(region, get_spinal_nerve_term("S4 spinal nerve"))
            s5Group = AnnotationGroup(region, get_spinal_nerve_term("S5 spinal nerve"))

            coccyxGroup = AnnotationGroup(region, get_spinal_nerve_term("coccygeal nerve"))

            # Scaled to transverse diameter of spinal column
            # https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2016.00238/full
            scales = [11.3, 11.5,12, 12.8, 13.3, 13.1, 12.5, 11.3, 10.7, 10, 9.6, 9.5, 9.2, 8.7, 8.4, 8.3, 8.6, 8.6,
                      8.3, 8.2, 8.6, 9.1, 9.4, 9.3, 8.8, 8.4, 7.1, 6.3, 5.5, 4.7, 4.5]
            scales = [3.0 * c for c in scales]

            scCentres = [[-1.111,-62.119,1480.149],
                         [-1.098,-62.413,1461.157],
                         [-1.093,-63.518,1442.197],
                         [-1.106,-65.530,1427.154],
                         [-1.118,-68.288,1414.134],
                         [-1.501,-69.819,1400.915],
                         [-1.152,-68.236,1385.901],
                         [-1.374,-64.510,1371.839],
                         [-1.156,-59.522,1358.831],
                         [-1.118,-52.803,1341.064],
                         [-1.489,-44.788,1319.923],
                         [-1.037,-36.641,1296.426],
                         [-1.030,-30.415,1268.655],
                         [-1.084,-27.936,1240.284],
                         [-1.140,-27.975,1211.798],
                         [-1.135,-30.239,1183.408],
                         [-1.086,-34.353,1159.670],
                         [-1.022,-41.577,1129.848],
                         [-0.939,-50.471,1100.482],
                         [-0.882,-60.099,1067.288],
                         [-0.911,-68.357,1038.908],
                         [-0.919,-72.331,1009.820],
                         [-0.846,-70.503,982.717],
                         [-0.761,-63.680,952.211],
                         [-0.757,-53.459,925.676],
                         [-0.901, -31.278, 894.878],
                         [-0.974, -21.369, 878.707],
                         [-1.013, -15.403, 860.708],
                         [-1.033, -12.027, 842.027],
                         [-1.039, -10.864, 823.078],
                         [-1.031, -11.819, 804.117]]

            structure = ""
            for level in range(numberOfLevels):
                for side in range(2):
                    i = level * (numberOfNetworkNodes * 2) + side * numberOfNetworkNodes + 1
                    baseStructure = \
                        (str(i) + '-' + str(i + 1) + '.' + str(1) + ',' +
                         str(i + 1) + '.' + str(2) + '-' + str(i + 2) + '-' + str(i + 3) + '-' + str(i + 4) + ',' +
                         str(i + 1) + '.' + str(3) + '-' + str(i + 5) + '-' + str(i + 6) + ',' +
                         str(i + 6) + '-' + str(i + 7) + '-' + str(i + 8) + '-' + str(i + 9) + '-' + str(i + 10) + ',' +
                         str(i + 10) + '-' + str(i + 11) + '-' + str(i + 12)) + ','
                    structure += baseStructure
            structure = structure[:-1]

            levelGroups = [c1Group, c2Group, c3Group, c4Group, c5Group, c6Group, c7Group, c8Group,
                           t1Group, t2Group, t3Group, t4Group, t5Group, t6Group, t7Group, t8Group,
                           t9Group, t10Group, t11Group, t12Group,
                           l1Group, l2Group, l3Group, l4Group, l5Group,
                           s1Group, s2Group, s3Group, s4Group, s5Group, coccyxGroup]

            segmentGroups = [cervicalGroup, thoracicGroup, lumbarGroup, sacralGroup, coccyxGroup]
            numberPerSegment = [0, 8, 20, 25, 30, 31]

        networkMesh = NetworkMesh(structure)
        networkMesh.create1DLayoutMesh(region)

        fieldmodule = region.getFieldmodule()
        mesh = fieldmodule.findMeshByDimension(1)

        halfDistBetweenLeftRight = 0.2

        # set up element annotations
        sideGroups = []
        if parameterSetName in ("Human whole spine 1", "Human pair 1"):
            leftGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region,("left", ""))
            rightGroup = findOrCreateAnnotationGroupForTerm(annotationGroups, region,("right", ""))
            sideGroups = [rightGroup, leftGroup]

        nerveGroup = AnnotationGroup(region, get_spinal_nerve_term("spinal nerve"))
        dorsalRootGanglionGroup = AnnotationGroup(region, get_spinal_nerve_term("dorsal root ganglion"))
        dorsalRootGroup = AnnotationGroup(region, get_spinal_nerve_term("dorsal root of spinal cord"))
        ventralRootGroup = AnnotationGroup(region, get_spinal_nerve_term("ventral root of spinal cord"))

        annotationGroups = [nerveGroup, dorsalRootGroup, dorsalRootGanglionGroup, ventralRootGroup] + \
                           levelGroups + segmentGroups + sideGroups

        nerveMeshGroup = nerveGroup.getMeshGroup(mesh)
        elementIdentifier = 1

        for level in range(numberOfLevels):
            levelMeshGroup = levelGroups[level].getMeshGroup(mesh) if levelGroups else []

            if segmentGroups:
                for segment in range(len(segmentGroups)):
                    if numberPerSegment[segment] <= level < numberPerSegment[segment + 1]:
                        segmentMeshGroup = segmentGroups[segment].getMeshGroup(mesh)

            for i in range(2):
                sideMeshGroup = sideGroups[i].getMeshGroup(mesh) if sideGroups else []
                nerveRootElementsCount = 1
                meshGroups = [nerveMeshGroup, levelMeshGroup, segmentMeshGroup] if levelGroups \
                    else [nerveMeshGroup]
                if sideGroups:
                    meshGroups.append(sideMeshGroup)
                for e in range(nerveRootElementsCount):
                    element = mesh.findElementByIdentifier(elementIdentifier)
                    for meshGroup in meshGroups:
                        meshGroup.addElement(element)
                    elementIdentifier += 1

                ventralRootElementsCount = 3
                meshGroups = [ventralRootGroup.getMeshGroup(mesh), levelMeshGroup, segmentMeshGroup] if levelGroups \
                    else [ventralRootGroup.getMeshGroup(mesh)]
                if sideGroups:
                    meshGroups.append(sideMeshGroup)
                for e in range(ventralRootElementsCount):
                    element = mesh.findElementByIdentifier(elementIdentifier)
                    for meshGroup in meshGroups:
                        meshGroup.addElement(element)
                    elementIdentifier += 1

                dorsalRootElementsCount = 8
                meshGroups = [dorsalRootGroup.getMeshGroup(mesh), levelMeshGroup, segmentMeshGroup] if levelGroups \
                    else [dorsalRootGroup.getMeshGroup(mesh)]
                meshGroupsWithDRG = [dorsalRootGroup.getMeshGroup(mesh), dorsalRootGanglionGroup.getMeshGroup(mesh),
                                     levelMeshGroup, segmentMeshGroup] if levelGroups \
                    else [dorsalRootGroup.getMeshGroup(mesh), dorsalRootGanglionGroup.getMeshGroup(mesh)]
                if sideGroups:
                    meshGroups.append(sideMeshGroup)
                    meshGroupsWithDRG.append(sideMeshGroup)
                for e in range(dorsalRootElementsCount):
                    element = mesh.findElementByIdentifier(elementIdentifier)
                    for meshGroup in (meshGroupsWithDRG if (1 < e < 5) else meshGroups):
                        meshGroup.addElement(element)
                    elementIdentifier += 1

        # set coordinates (outer)
        fieldcache = fieldmodule.createFieldcache()
        coordinates = find_or_create_field_coordinates(fieldmodule)
        # need to ensure inner coordinates are at least defined:
        cls.defineInnerCoordinates(region, coordinates, options, networkMesh, innerProportion=0.75)
        innerCoordinates = find_or_create_field_coordinates(fieldmodule, "inner coordinates")
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        nodeIdentifier = 1
        for level in range(numberOfLevels):
            scale = scales[level]
            translateToSC = scCentres[level]

            for side in ('right' if parameterSetName == "Human 1" else ('right', 'left')):
                nerveRootScale = nerveRootLength / nerveRootElementsCount
                rotMat = rotMatRight if side == "right" else rotMatLeft

                d1 = matrix_vector_mult(rotMat, [nerveRootScale * (1 if side == 'right' else -1), 0.0, 0.0])
                d2 = matrix_vector_mult(rotMat, [0.0, nerveRootRadius *(1 if side == 'right' else -1), 0.0])
                d3 = matrix_vector_mult(rotMat, [0.0, 0.0, nerveRootRadius])

                # Scale
                d1 = mult(d1, scale)
                d2 = mult(d2, scale)
                d3 = mult(d3, scale)

                id2 = mult(d2, innerProportionDefault)
                id3 = mult(d3, innerProportionDefault)

                for i in range(nerveRootElementsCount + 1):
                    node = nodes.findNodeByIdentifier(nodeIdentifier)
                    fieldcache.setNode(node)
                    if side == 'right':
                        x = [-(spinalCordMajorRadius + distToSpinalCord + halfDistBetweenLeftRight + nerveRootLength) +
                             nerveRootScale * i, 0.0, 0.0]
                    else:
                        x = [spinalCordMajorRadius + distToSpinalCord + halfDistBetweenLeftRight + nerveRootLength -
                             nerveRootScale * i, 0.0, 0.0]

                    xJunction = x
                    x = matrix_vector_mult(rotMat, x)
                    x = mult(x, scale)
                    x = add(x, translateToSC)

                    setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
                    setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3)
                    nodeIdentifier += 1

                nerveJunctionNodeIdentifier = nodeIdentifier - 1

                # Dorsal and ventral roots
                for nerve in ("ventral", "dorsal"):
                    d2List = []
                    sideNerveElementsCount = ventralRootElementsCount if nerve == "ventral" else dorsalRootElementsCount
                    sideNerveRadius = ventralRootRadius if nerve == "ventral" else dorsalRootRadius
                    xStart = xJunction
                    dStart = [0.0, (-1 if nerve == "ventral" else 1) * spinalCordMinorRadius, 0.0]
                    xEnd = [halfDistBetweenLeftRight * (-1 if side == 'right' else 1),
                            (-1 if nerve == "ventral" else 1) * spinalCordMinorRadius,
                            0.0]
                    dEnd = [spinalCordMajorRadius * (1 if side == 'right' else -1), 0.0, 0.0]
                    xList, d1List = sampleCubicHermiteCurves([xStart, xEnd], [dStart, dEnd],
                                                             sideNerveElementsCount)[0:2]
                    d1List = smoothCubicHermiteDerivativesLine(xList, d1List, fixEndDirection=True)

                    for i in range(len(xList)):
                        xList[i] = matrix_vector_mult(rotMat, xList[i])
                        xList[i] = mult(xList[i], scale)
                        d1List[i] = matrix_vector_mult(rotMat, d1List[i])
                        d1List[i] = mult(d1List[i], scale)

                    d3 = matrix_vector_mult(rotMat, [0.0, 0.0, sideNerveRadius])
                    d3 = mult(d3, scale)

                    d2 = set_magnitude(cross(d3, d1List[0]), magnitude(d3))
                    id3 = mult(d3, innerProportionDefault)
                    id2 = mult(d2, innerProportionDefault)

                    d2Next = set_magnitude(cross(d3, d1List[1]), magnitude(d3))
                    d12 = sub(d2Next, d2)
                    id12 = mult(d12, innerProportionDefault)
                    d13 = matrix_vector_mult(rotMat, [0.0, 0.0, 0.0])
                    id13 = matrix_vector_mult(rotMat, [0.0, 0.0, 0.0])

                    if nerve == "dorsal":
                        d3List = [d3 for i in range(dorsalRootElementsCount + 1)]
                        d3Mid = 0.5 * (dRGThickestRadius + sideNerveRadius)
                        d3List[int(dorsalRootElementsCount * 0.5) - 1] = \
                            mult(matrix_vector_mult(rotMat, [0.0, 0.0, d3Mid]), scale)
                        d3List[int(dorsalRootElementsCount * 0.5)] = \
                            mult(matrix_vector_mult(rotMat, [0.0, 0.0, dRGThickestRadius]), scale)
                        d3List[int(dorsalRootElementsCount * 0.5) + 1] = \
                            mult(matrix_vector_mult(rotMat,[0.0, 0.0, d3Mid]), scale)

                        id3List = []
                        d2List = []
                        id2List = []
                        d12List = []
                        d13List = []
                        id12List = []
                        id13List = []

                        for i in range(len(d3List)):
                            id3List.append(mult(d3List[i], innerProportionDefault))

                            d2List.append(set_magnitude(cross(d3List[i], d1List[i]), magnitude(d3List[i])))
                            id2List.append(mult(d2List[-1], innerProportionDefault))

                        for i in range(len(d3List) - 1):
                            d13List.append(sub(d3List[i + 1], d3List[i]))
                            d12List.append(sub(d2List[i + 1], d2List[i]))
                            id12List.append(mult(d12List[-1], innerProportionDefault))
                            id13List.append(mult(d13List[-1], innerProportionDefault))
                        d13List.append(d13List[-1])
                        d12List.append(d12List[-1])
                        id13List.append(id13List[-1])
                        id12List.append(id12List[-1])

                    node = nodes.findNodeByIdentifier(nerveJunctionNodeIdentifier)
                    fieldcache.setNode(node)
                    version = 2 if (nerve == "ventral") else 3

                    if nerve == 'dorsal':
                        d2, d3, id2, id3, d12, d13, id12, id13 = \
                            d2List[0], d3List[0], id2List[0], id3List[0], d12List[0], d13List[0], id12List[0], \
                            id13List[0]

                    setNodeFieldVersionDerivatives(coordinates, fieldcache, version, d1List[0], d2, d3, d12, d13)
                    setNodeFieldVersionDerivatives(innerCoordinates, fieldcache, version, d1List[0], id2, id3, id12,
                                                   id13)

                    # remaining part of nerve
                    for i in range(1, sideNerveElementsCount + 1):
                        node = nodes.findNodeByIdentifier(nodeIdentifier)
                        fieldcache.setNode(node)
                        x = xList[i]
                        x = add(x, translateToSC)
                        d1 = d1List[i]

                        if nerve == 'dorsal':
                            d2, d3, id2, id3, d12, d13, id12, id13 = \
                                d2List[i], d3List[i], id2List[i], id3List[i], d12List[i], d13List[i], id12List[i],\
                                id13List[i]
                        else:
                            d2 = set_magnitude(cross(d3, d1List[i]), magnitude(d3))
                            id2 = mult(d2, innerProportionDefault)
                            if i < sideNerveElementsCount:
                                d2Next = set_magnitude(cross(d3, d1List[i + 1]), magnitude(d3))
                                d12 = sub(d2Next, d2)
                            else:
                                d12 = [0.0, 0.0, 0.0]
                            id12 = mult(d12, innerProportionDefault)

                        setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12, d13)
                        setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12, id13)
                        nodeIdentifier += 1

        return annotationGroups, networkMesh

    @classmethod
    def getInteractiveFunctions(cls):
        """
        Edit base class list to include only valid functions.
        """
        interactiveFunctions = super(MeshType_1d_human_spinal_nerve_network_layout1, cls).getInteractiveFunctions()
        for interactiveFunction in interactiveFunctions:
            if interactiveFunction[0] == "Edit structure...":
                interactiveFunctions.remove(interactiveFunction)
                break
        return interactiveFunctions


class MeshType_3d_spinalnerve1(Scaffold_base):
    """
    Generates a 3-D hermite bifurcating tube network with core representing the human spinal nerve.
    """

    @classmethod
    def getName(cls):
        return "3D Spinal Nerve 1"

    @classmethod
    def getParameterSetNames(cls):
        return [
            "Default",
            "Human 1",
            "Human pair 1",
            "Human whole spine 1"
        ]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {}
        useParameterSetName = "Human 1" if (parameterSetName == "Default") else parameterSetName
        options["Base parameter set"] = useParameterSetName
        options["Spinal nerve network layout"] = ScaffoldPackage(MeshType_1d_human_spinal_nerve_network_layout1,
                                                                 defaultParameterSetName=useParameterSetName)
        options["Number of elements along nerve root"] = 3
        options["Number of elements along dorsal root beside DRG"] = 1
        options["Number of elements along dorsal root ganglion"] = 3
        options["Number of elements along ventral root"] = 5
        options["Number of elements around spinal nerve"] = 8
        options["Number of elements through shell"] = 0
        options["Show trim surfaces"] = False
        options["Use Core"] = True
        options["Number of elements across core box minor"] = 2
        options["Number of elements across core transition"] = 1

        return options

    @classmethod
    def getOrderedOptionNames(cls):
        optionNames = [
            "Spinal nerve network layout",
            "Number of elements along nerve root",
            "Number of elements along dorsal root beside DRG",
            "Number of elements along dorsal root ganglion",
            "Number of elements along ventral root",
            "Number of elements around spinal nerve",
            "Number of elements through shell",
            "Show trim surfaces",
            "Use Core",
            "Number of elements across core box minor",
            "Number of elements across core transition"]
        return optionNames

    @classmethod
    def getOptionValidScaffoldTypes(cls, optionName):
        if optionName == "Spinal nerve network layout":
            return [MeshType_1d_human_spinal_nerve_network_layout1]
        return []

    @classmethod
    def getOptionScaffoldPackage(cls, optionName, scaffoldType, parameterSetName=None):
        """
        :param parameterSetName:  Name of valid parameter set for option Scaffold, or None for default.
        :return: ScaffoldPackage.
        """
        if parameterSetName:
            assert parameterSetName in cls.getOptionScaffoldTypeParameterSetNames(optionName, scaffoldType), \
                "Invalid parameter set " + str(parameterSetName) + " for scaffold " + str(scaffoldType.getName()) + \
                " in option " + str(optionName) + " of scaffold " + cls.getName()
        if optionName == "Spinal nerve network layout":
            if not parameterSetName:
                parameterSetName = "Default"
            return ScaffoldPackage(MeshType_1d_human_spinal_nerve_network_layout1,
                                   defaultParameterSetName=parameterSetName)
        assert False, cls.__name__ + ".getOptionScaffoldPackage:  Option " + optionName + " is not a scaffold"

    @classmethod
    def checkOptions(cls, options):
        dependentChanges = False
        if (options["Spinal nerve network layout"].getScaffoldType() not in
                cls.getOptionValidScaffoldTypes("Spinal nerve network layout")):
            options["Spinal nerve network layout"] = ScaffoldPackage(MeshType_1d_human_spinal_nerve_network_layout1)
        for key in [
            "Number of elements along nerve root",
            "Number of elements along dorsal root beside DRG",
            "Number of elements along ventral root"
        ]:
            if options[key] < 1:
                options[key] = 1
        minElementsCountAround = None
        for key in [
            "Number of elements around spinal nerve"
        ]:
            if options[key] < 8:
                options[key] = 8
            elif options[key] % 4:
                options[key] += 4 - (options[key] % 4)
            if (minElementsCountAround is None) or (options[key] < minElementsCountAround):
                minElementsCountAround = options[key]

        if options["Number of elements through shell"] < 0:
            options["Number of elements through shell"] = 0

        if options["Number of elements across core transition"] < 1:
            options["Number of elements across core transition"] = 1

        maxElementsCountCoreBoxMinor = minElementsCountAround // 2 - 2
        for key in [
            "Number of elements across core box minor"
        ]:
            if options[key] < 2:
                options[key] = 2
            elif options[key] > maxElementsCountCoreBoxMinor:
                options[key] = maxElementsCountCoreBoxMinor
                dependentChanges = True
            elif options[key] % 2:
                options[key] += options[key] % 2

        return dependentChanges

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the base hermite-bilinear mesh. See also generateMesh().
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: list of AnnotationGroup, None
        """
        networkLayout = options["Spinal nerve network layout"]
        elementsCountAlongNerveRoot = options["Number of elements along nerve root"]
        elementsCountAlongDorsalRoot = options["Number of elements along dorsal root beside DRG"]
        elementsCountAlongDorsalRootGanglion = options["Number of elements along dorsal root ganglion"]
        elementsCountAlongVentralRoot = options["Number of elements along ventral root"]
        elementsCountAroundSpinalNerve = options["Number of elements around spinal nerve"]
        shell_count = options["Number of elements through shell"]
        core = options["Use Core"]

        layoutRegion = region.createRegion()
        networkLayout.generate(layoutRegion)  # ask scaffold to generate to get user-edited parameters
        layoutAnnotationGroups = networkLayout.getAnnotationGroups()
        networkMesh = networkLayout.getConstructionObject()

        annotationAlongCounts = []
        annotationAroundCounts = []
        for layoutAnnotationGroup in layoutAnnotationGroups:
            alongCount = 0
            aroundCount = elementsCountAroundSpinalNerve
            name = layoutAnnotationGroup.getName()
            if name == "spinal nerve":
                alongCount = elementsCountAlongNerveRoot
            elif name == "dorsal root of spinal cord":
                alongCount = elementsCountAlongDorsalRoot
            elif name == "dorsal root ganglion":
                alongCount = elementsCountAlongDorsalRootGanglion
            elif name == "ventral root of spinal cord":
                alongCount = elementsCountAlongVentralRoot

            annotationAlongCounts.append(alongCount)
            annotationAroundCounts.append(aroundCount)

        tubeNetworkMeshBuilder = TubeNetworkMeshBuilder(
            networkMesh,
            targetElementDensityAlongLongestSegment=2.0,
            layoutAnnotationGroups=layoutAnnotationGroups,
            annotationElementsCountsAlong=annotationAlongCounts,
            defaultElementsCountAround=options["Number of elements around spinal nerve"],
            annotationElementsCountsAround=annotationAroundCounts,
            shell_count=shell_count,
            core=core,
            transition_count=options['Number of elements across core transition'],
            defaultElementsCountCoreBoxMinor=options["Number of elements across core box minor"],
            annotationElementsCountsCoreBoxMinor=[],
            useOuterTrimSurfaces=True)

        meshDimension = 3 if (core or shell_count) else 2
        tubeNetworkMeshBuilder.build()
        generateData = TubeNetworkMeshGenerateData(
            region, meshDimension,
            isLinearThroughShell=False,
            isShowTrimSurfaces=options["Show trim surfaces"])
        tubeNetworkMeshBuilder.generateMesh(generateData)
        annotationGroups = generateData.getAnnotationGroups()

        return annotationGroups, None

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
        is_core = options["Use Core"]
        shell_count = options["Number of elements through shell"]
        TubeNetworkMeshBuilder.defineFaceAnnotations(region, annotationGroups, is_core, shell_count)

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
