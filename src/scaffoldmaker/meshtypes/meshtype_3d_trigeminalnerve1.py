"""
Generates a 3D trigeminal nerve scaffold using tube network mesh.
"""
from cmlibs.maths.vectorops import cross, mult, set_magnitude, sub
from cmlibs.utils.zinc.field import Field, find_or_create_field_coordinates
from cmlibs.zinc.node import Node
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup
from scaffoldmaker.annotation.trigeminal_nerve_terms import get_trigeminal_nerve_term
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils.interpolation import (getCubicHermiteArcLength, interpolateLagrangeHermiteDerivative,
                                               sampleCubicHermiteCurvesSmooth)
from scaffoldmaker.utils.networkmesh import NetworkMesh
from scaffoldmaker.utils.tubenetworkmesh import TubeNetworkMeshBuilder, TubeNetworkMeshGenerateData
import math


class MeshType_1d_human_trigeminal_nerve_network_layout1(MeshType_1d_network_layout1):
    """
    Defines trigeminal nerve network layout.
    """

    @classmethod
    def getName(cls):
        return "1D Human Trigeminal Nerve Network Layout 1"

    @classmethod
    def getParameterSetNames(cls):
        return ["Default"]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {}
        options["Base parameter set"] = parameterSetName
        options["Structure"] = (
            "1-2-3,"
            "3-4-5.1,"
            "5.2-6-7-8-9,"
            "5.3-10-11-12-13,"
            "5.4-14-15-16-17")
        options["Define inner coordinates"] = True
        options["Trigeminal nerve root length"] = 3.0
        options["Trigeminal nerve root width"] = 1.2
        options["Trigeminal ganglion length"] = 2.0
        options["Trigeminal ganglion width"] = 3.5
        options["Mandibular nerve length"] = 1.5
        options["Mandibular nerve start width"] = 1.6
        options["Mandibular nerve end width"] = 1.0
        options["Maxillary nerve length"] = 3.0
        options["Maxillary nerve start width"] = 2.0
        options["Maxillary nerve end width"] = 0.25
        options["Ophthalmic nerve length"] = 3.0
        options["Ophthalmic nerve start width"] = 1.0
        options["Ophthalmic nerve end width"] = 0.6
        options["Trigeminal nerve thickness"] = 0.5
        options["Inner proportion default"] = 0.75

        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Trigeminal nerve root length",
            "Trigeminal nerve root width",
            "Trigeminal ganglion length",
            "Trigeminal ganglion width",
            "Mandibular nerve length",
            "Mandibular nerve start width",
            "Mandibular nerve end width",
            "Maxillary nerve length",
            "Maxillary nerve start width",
            "Maxillary nerve end width",
            "Ophthalmic nerve length",
            "Ophthalmic nerve start width",
            "Ophthalmic nerve end width",
            "Trigeminal nerve thickness",
            "Inner proportion default"
        ]

    @classmethod
    def checkOptions(cls, options):
        dependentChanges = False
        for key in [
            "Trigeminal nerve root length",
            "Trigeminal nerve root width",
            "Trigeminal ganglion length",
            "Trigeminal ganglion width",
            "Mandibular nerve length",
            "Mandibular nerve start width",
            "Mandibular nerve end width",
            "Maxillary nerve length",
            "Maxillary nerve start width",
            "Maxillary nerve end width",
            "Ophthalmic nerve length",
            "Ophthalmic nerve start width",
            "Ophthalmic nerve end width",
            "Trigeminal nerve thickness"
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
        return dependentChanges

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the unrefined mesh.
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: [] empty list of AnnotationGroup, NetworkMesh
        """
        structure = options["Structure"]
        halfNerveThickness = 0.5 * options["Trigeminal nerve thickness"]

        nerveRootLength = options["Trigeminal nerve root length"]
        halfNerveRootWidth = 0.5 * options["Trigeminal nerve root width"]

        trigeminalGanglionLength = options["Trigeminal ganglion length"]
        halfTrigeminalGanglionWidth = 0.5 * options["Trigeminal ganglion width"]

        mandibularLength = options["Mandibular nerve length"]
        halfMandibularStartWidth = 0.5 * options["Mandibular nerve start width"]
        halfMandibularEndWidth = 0.5 * options["Mandibular nerve end width"]
        mandibularAngleRadians = math.radians(-45.0)

        maxillaryLength = options["Maxillary nerve length"]
        halfMaxillaryStartWidth = 0.5 * options["Maxillary nerve start width"]
        halfMaxillaryEndWidth = 0.5 * options["Maxillary nerve end width"]
        maxillaryAngleRadians = math.radians(0.0)

        ophthalmicLength = options["Ophthalmic nerve length"]
        halfOphthalmicStartWidth = 0.5 * options["Ophthalmic nerve start width"]
        halfOphthalmicEndWidth = 0.5 * options["Ophthalmic nerve end width"]
        ophthalmicAngleRadians = math.radians(45.0)

        innerProportionDefault = options["Inner proportion default"]

        networkMesh = NetworkMesh(structure)
        networkMesh.create1DLayoutMesh(region)

        fieldmodule = region.getFieldmodule()
        mesh = fieldmodule.findMeshByDimension(1)

        # set up element annotations
        trigeminalNerveGroup = AnnotationGroup(region, get_trigeminal_nerve_term("trigeminal nerve"))
        nerveRootGroup = AnnotationGroup(region, get_trigeminal_nerve_term("trigeminal nerve root"))
        trigeminalGanglionGroup = AnnotationGroup(region, get_trigeminal_nerve_term("trigeminal ganglion"))
        mandibularNerveGroup = AnnotationGroup(region, get_trigeminal_nerve_term("mandibular nerve"))
        maxillaryNerveGroup = AnnotationGroup(region, get_trigeminal_nerve_term("maxillary nerve"))
        ophthalmicNerveGroup = AnnotationGroup(region, get_trigeminal_nerve_term("ophthalmic nerve"))
        annotationGroups = [trigeminalNerveGroup, nerveRootGroup, trigeminalGanglionGroup,
                            mandibularNerveGroup, maxillaryNerveGroup, ophthalmicNerveGroup]
        trigeminalNerveMeshGroup = trigeminalNerveGroup.getMeshGroup(mesh)
        elementIdentifier = 1
        nerveRootElementsCount = 2
        meshGroups = [trigeminalNerveMeshGroup, nerveRootGroup.getMeshGroup(mesh)]
        for e in range(nerveRootElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1
        trigeminalGanglionElementsCount = 2
        meshGroups = [trigeminalNerveMeshGroup, trigeminalGanglionGroup.getMeshGroup(mesh)]
        for e in range(trigeminalGanglionElementsCount):
            element = mesh.findElementByIdentifier(elementIdentifier)
            for meshGroup in meshGroups:
                meshGroup.addElement(element)
            elementIdentifier += 1

        nerveBranchElementsCount = 4
        for nerveBranch in ('mandibular','ophthalmic','maxillary'):
            if nerveBranch == 'mandibular':
                nerveGroup = mandibularNerveGroup
            elif nerveBranch == 'maxillary':
                nerveGroup = maxillaryNerveGroup
            elif nerveBranch == 'ophthalmic':
                nerveGroup = ophthalmicNerveGroup
            meshGroups = [trigeminalNerveMeshGroup, nerveGroup.getMeshGroup(mesh)]
            for e in range(nerveBranchElementsCount):
                element = mesh.findElementByIdentifier(elementIdentifier)
                for meshGroup in meshGroups:
                    meshGroup.addElement(element)
                elementIdentifier += 1

        # set coordinates (outer)
        fieldcache = fieldmodule.createFieldcache()
        coordinates = find_or_create_field_coordinates(fieldmodule)
        # need to ensure inner coordinates are at least defined:
        cls.defineInnerCoordinates(region, coordinates, options, networkMesh, innerProportion=0.75)
        innerCoordinates = find_or_create_field_coordinates(fieldmodule, "inner coordinates")
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        nerveRootScale = nerveRootLength / nerveRootElementsCount
        nodeIdentifier = 1
        d1 = [nerveRootScale, 0.0, 0.0]
        d2 = [0.0, halfNerveRootWidth, 0.0]
        d3 = [0.0, 0.0, halfNerveThickness]
        id2 = mult(d2, innerProportionDefault)
        id3 = mult(d3, innerProportionDefault)
        for i in range(nerveRootElementsCount):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = [nerveRootScale * i, 0.0, 0.0]
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3)
            nodeIdentifier += 1

        trigeminalGanglionScale = trigeminalGanglionLength / trigeminalGanglionElementsCount
        d12_mag = (halfTrigeminalGanglionWidth - halfNerveRootWidth) / trigeminalGanglionElementsCount
        d3 = [0.0, 0.0, halfNerveThickness]
        id3 = mult(d3, innerProportionDefault)
        for i in range(trigeminalGanglionElementsCount + 1):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = [nerveRootLength + trigeminalGanglionScale * i, 0.0, 0.0]
            d1 = [0.5 * (nerveRootScale + trigeminalGanglionScale) if (i == 0) else trigeminalGanglionScale, 0.0, 0.0]
            sHalfWidth = halfNerveRootWidth + i * trigeminalGanglionScale * d12_mag
            d2 = [0.0, sHalfWidth, 0.0]
            d12 = [0.0, d12_mag, 0.0]
            id2 = mult(d2, innerProportionDefault)
            id12 = mult(d12, innerProportionDefault)
            setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
            setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12)
            nodeIdentifier += 1

        nerveJunctionNodeIdentifier = nodeIdentifier - 1
        xJunction = x

        # Mandibular and ophthlamic nerves
        for nerve in ("mandibular", "ophthalmic"):
            sideNerveShoulderDrop = 0.8
            sideNerveAngleRadians = mandibularAngleRadians if (nerve == "mandibular") else ophthalmicAngleRadians
            sideNerveShoulderRotationFactor = 1.0 - math.cos(0.5 * sideNerveAngleRadians)
            sideNerveShoulderLimitAngleRadians = math.asin(2.0 * sideNerveShoulderDrop / halfTrigeminalGanglionWidth)
            sideNerveShoulderAngleRadians = sideNerveShoulderRotationFactor * sideNerveShoulderLimitAngleRadians

            sideNerveStartX = nerveRootLength + trigeminalGanglionLength + sideNerveShoulderDrop - \
                               halfTrigeminalGanglionWidth * math.sin(sideNerveShoulderAngleRadians)
            sideNerveStartY = halfTrigeminalGanglionWidth * (-1 if (nerve == "mandibular") else 1) * \
                               math.cos(sideNerveShoulderAngleRadians)
            x = [sideNerveStartX, sideNerveStartY, 0.0]

            sideNerveScale = (mandibularLength if (nerve == "mandibular") else ophthalmicLength) / \
                             (nerveBranchElementsCount - 2)
            sd1 = interpolateLagrangeHermiteDerivative(xJunction, x, d1, 0.0)
            nx, nd1 = sampleCubicHermiteCurvesSmooth([xJunction, x], [sd1, d1], 2,
                                                     derivativeMagnitudeEnd=sideNerveScale)[0:2]
            arcLengths = [getCubicHermiteArcLength(nx[i], nd1[i], nx[i + 1], nd1[i + 1]) for i in range(2)]

            cosSideNerveAngle = math.cos(sideNerveAngleRadians)
            sinSideNerveAngle = math.sin(sideNerveAngleRadians)
            d1 = [sideNerveScale * cosSideNerveAngle, sideNerveScale * sinSideNerveAngle, 0.0]

            sideNerveHalfStartWidth = halfMandibularStartWidth if (nerve == "mandibular") else halfOphthalmicStartWidth
            sideNerveHalfEndWidth = halfMandibularEndWidth if (nerve == "mandibular") else halfOphthalmicEndWidth
            d12_mag = (sideNerveHalfEndWidth - sideNerveHalfStartWidth) / (nerveBranchElementsCount - 2)
            d3 = [0.0, 0.0, halfNerveThickness]
            id3 = mult(d3, innerProportionDefault)

            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            sd2_list = []

            sNodeIdentifiers = []
            for i in range(2):
                sNodeIdentifiers.append(nodeIdentifier if (i > 0) else nerveJunctionNodeIdentifier)
                node = nodes.findNodeByIdentifier(sNodeIdentifiers[-1])
                fieldcache.setNode(node)
                version = 1 if (i > 0) else 2 if (nerve == "mandibular") else 3
                sd1 = nd1[i]
                sDistance = sum(arcLengths[i:])
                sHalfWidth = sideNerveHalfStartWidth + sDistance * -d12_mag
                sd2 = set_magnitude(cross(d3, sd1), sHalfWidth)
                sid2 = mult(sd2, innerProportionDefault)
                sd2_list.append(sd2)
                if i > 0:
                    for field in (coordinates, innerCoordinates):
                        field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, nx[i])
                    nodeIdentifier += 1
                setNodeFieldVersionDerivatives(coordinates, fieldcache, version, sd1, sd2, d3)
                setNodeFieldVersionDerivatives(innerCoordinates, fieldcache, version, sd1, sid2, id3)
            sd2_list.append([-sideNerveHalfStartWidth * sinSideNerveAngle,
                             sideNerveHalfStartWidth * cosSideNerveAngle, 0.0])

            for i in range(2):
                node = nodes.findNodeByIdentifier(sNodeIdentifiers[i])
                fieldcache.setNode(node)
                version = 1 if (i > 0) else 2 if (nerve == "mandibular") else 4
                sd12 = sub(sd2_list[i + 1], sd2_list[i])
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, version, sd12)
                sid12 = mult(sd12, innerProportionDefault)
                innerCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, version, sid12)

            # remaining part of nerve
            for i in range(nerveBranchElementsCount - 1):
                node = nodes.findNodeByIdentifier(nodeIdentifier)
                fieldcache.setNode(node)
                x = [sideNerveStartX + d1[0] * i, sideNerveStartY + d1[1] * i, d1[2] * i]
                sHalfWidth = sideNerveHalfStartWidth + i * sideNerveScale * d12_mag
                d2 = set_magnitude(cross(d3, d1), sHalfWidth)
                d12 = [-d12_mag * sinSideNerveAngle, d12_mag * cosSideNerveAngle, 0.0]
                id2 = mult(d2, innerProportionDefault)
                id12 = mult(d12, innerProportionDefault)
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12)
                nodeIdentifier += 1

        # maxillary nerve
        maxillaryScale = maxillaryLength / nerveBranchElementsCount
        maxillaryStartX = nerveRootLength + trigeminalGanglionLength
        sx = [maxillaryStartX, 0.0, 0.0]
        cosMaxillaryAngle = math.cos(maxillaryAngleRadians)
        sinMaxillaryAngle = math.sin(maxillaryAngleRadians)
        d12_mag = (halfMaxillaryEndWidth - halfMaxillaryStartWidth) / nerveBranchElementsCount
        for i in range(nerveBranchElementsCount + 1):
            node = nodes.findNodeByIdentifier(nodeIdentifier)
            fieldcache.setNode(node)
            x = [sx[0] + maxillaryScale * i * cosMaxillaryAngle,
                 sx[1] + maxillaryScale * i * sinMaxillaryAngle,
                 0.0]
            d1 = [maxillaryScale * cosMaxillaryAngle, maxillaryScale * sinMaxillaryAngle, 0.0]
            d3 = [0.0, 0.0, halfNerveThickness]
            id3 = mult(d3, innerProportionDefault)
            sHalfWidth = halfMaxillaryStartWidth + i * maxillaryScale * d12_mag
            d2 = set_magnitude(cross(d3, d1), sHalfWidth)
            d12 = [-d12_mag * sinMaxillaryAngle, d12_mag * cosMaxillaryAngle, 0.0]
            id2 = mult(d2, innerProportionDefault)
            id12 = mult(d12, innerProportionDefault)
            if i == 0:
                node = nodes.findNodeByIdentifier(nerveJunctionNodeIdentifier)
                fieldcache.setNode(node)
                version = 4
                setNodeFieldVersionDerivatives(coordinates, fieldcache, version, d1, d2, d3)
                setNodeFieldVersionDerivatives(innerCoordinates, fieldcache, version, d1, id2, id3)
            else:
                setNodeFieldParameters(coordinates, fieldcache, x, d1, d2, d3, d12)
                setNodeFieldParameters(innerCoordinates, fieldcache, x, d1, id2, id3, id12)
                nodeIdentifier += 1

        return annotationGroups, networkMesh

    @classmethod
    def getInteractiveFunctions(cls):
        """
        Edit base class list to include only valid functions.
        """
        interactiveFunctions = super(MeshType_1d_human_trigeminal_nerve_network_layout1, cls).getInteractiveFunctions()
        for interactiveFunction in interactiveFunctions:
            if interactiveFunction[0] == "Edit structure...":
                interactiveFunctions.remove(interactiveFunction)
                break
        return interactiveFunctions


class MeshType_3d_trigeminalnerve1(Scaffold_base):
    """
    Generates a 3-D hermite bifurcating tube network with core representing the human trigeminal nerve.
    """

    @classmethod
    def getName(cls):
        return "3D Trigeminal Nerve 1"

    @classmethod
    def getParameterSetNames(cls):
        return [
            "Default",
            "Human 1"
        ]

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {}
        useParameterSetName = "Human 1" if (parameterSetName == "Default") else parameterSetName
        options["Base parameter set"] = useParameterSetName
        options["Trigeminal nerve network layout"] = ScaffoldPackage(MeshType_1d_human_trigeminal_nerve_network_layout1)
        options["Number of elements along nerve root"] = 3
        options["Number of elements along trigeminal ganglion"] = 2
        options["Number of elements along mandibular nerve"] = 3
        options["Number of elements along maxillary nerve"] = 3
        options["Number of elements along ophthalmic nerve"] = 3
        options["Number of elements around trigeminal nerve"] = 12
        options["Number of elements through shell"] = 0
        options["Show trim surfaces"] = False
        options["Use Core"] = True
        options["Number of elements across core box minor"] = 2
        options["Number of elements across core transition"] = 1

        return options

    @classmethod
    def getOrderedOptionNames(cls):
        optionNames = [
            "Trigeminal nerve network layout",
            "Number of elements along nerve root",
            "Number of elements along trigeminal ganglion",
            "Number of elements along maxillary nerve",
            "Number of elements along mandibular nerve",
            "Number of elements along ophthalmic nerve",
            "Number of elements around trigeminal nerve",
            "Number of elements through shell",
            "Show trim surfaces",
            "Use Core",
            "Number of elements across core box minor",
            "Number of elements across core transition"]
        return optionNames

    @classmethod
    def getOptionValidScaffoldTypes(cls, optionName):
        if optionName == "Trigeminal nerve network layout":
            return [MeshType_1d_human_trigeminal_nerve_network_layout1]
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
        if optionName == "Trigeminal nerve network layout":
            if not parameterSetName:
                parameterSetName = "Default"
            return ScaffoldPackage(MeshType_1d_human_trigeminal_nerve_network_layout1, defaultParameterSetName=parameterSetName)
        assert False, cls.__name__ + ".getOptionScaffoldPackage:  Option " + optionName + " is not a scaffold"

    @classmethod
    def checkOptions(cls, options):
        dependentChanges = False
        if (options["Trigeminal nerve network layout"].getScaffoldType() not in
                cls.getOptionValidScaffoldTypes("Trigeminal nerve network layout")):
            options["Trigeminal nerve network layout"] = ScaffoldPackage(MeshType_1d_human_trigeminal_nerve_network_layout1)
        for key in [
            "Number of elements along nerve root",
            "Number of elements along trigeminal ganglion",
            "Number of elements along maxillary nerve",
            "Number of elements along mandibular nerve",
            "Number of elements along ophthalmic nerve"
        ]:
            if options[key] < 1:
                options[key] = 1
        minElementsCountAround = None
        for key in [
            "Number of elements around trigeminal nerve"
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
        networkLayout = options["Trigeminal nerve network layout"]
        elementsCountAlongNerveRoot = options["Number of elements along nerve root"]
        elementsCountAlongTrigeminalGanglion = options["Number of elements along trigeminal ganglion"]
        elementsCountAlongMaxillaryNerve = options["Number of elements along maxillary nerve"]
        elementsCountAlongMandibularNerve = options["Number of elements along mandibular nerve"]
        elementsCountAlongOphthalmicNerve = options["Number of elements along ophthalmic nerve"]
        elementsCountAroundTrigeminalNerve = options["Number of elements around trigeminal nerve"]
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
            aroundCount = elementsCountAroundTrigeminalNerve
            name = layoutAnnotationGroup.getName()
            if "nerve root" in name:
                alongCount = elementsCountAlongNerveRoot
            elif "ganglion" in name:
                alongCount = elementsCountAlongTrigeminalGanglion
            elif "maxillary" in name:
                alongCount = elementsCountAlongMaxillaryNerve
            elif "mandibular" in name:
                alongCount = elementsCountAlongMandibularNerve
            elif "ophthalmic" in name:
                alongCount = elementsCountAlongOphthalmicNerve
            annotationAlongCounts.append(alongCount)
            annotationAroundCounts.append(aroundCount)

        tubeNetworkMeshBuilder = TubeNetworkMeshBuilder(
            networkMesh,
            targetElementDensityAlongLongestSegment=2.0,
            layoutAnnotationGroups=layoutAnnotationGroups,
            annotationElementsCountsAlong=annotationAlongCounts,
            defaultElementsCountAround=options["Number of elements around trigeminal nerve"],
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
