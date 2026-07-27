"""
Generates a 3-D Hermite bifurcating tube network (with optional solid core).
"""
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils.tubenetworkmesh import TubeNetworkMeshBuilder, TubeNetworkMeshGenerateData


class MeshType_3d_tubenetwork1(Scaffold_base):
    """
    Generates a 3-D hermite tube network from a network layout description, with linear or cubic basis through the
    shell wall, optionally with a solid core. The number of elements generated are primarily controlled by the
    "Number of elements around" and "Number of elements across core box minor" settings.
    The Number of elements around parameter determines the number of elements around the elliptical shell.
    The core is built out of a regular box with specified number of elements across core box minor.
    The minor direction is in the direction of d3 in the network layout it is defined from.
    The number of elements across the core box major axis is calculated internally from the above.
    The reason for preferring the minor number is that this must currently match when using the core,
    plus it is planned to eventually use an odd number to specify a half phase difference in starting node
    even without a core, as that is required to work with odd minor numbers in the core, once supported.
    There are any number of transition elements between the box and the shell forming further concentric rim elements.
    """

    @classmethod
    def getName(cls):
        return "3D Tube Network 1"

    @classmethod
    def getParameterSetNames(cls):
        return MeshType_1d_network_layout1.getParameterSetNames()

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        options = {
            "Network layout": ScaffoldPackage(MeshType_1d_network_layout1,
                                              {"scaffoldSettings": {"Define inner coordinates": True}},
                                              defaultParameterSetName=parameterSetName),
            "Number of elements around": 8,
            "Annotation numbers of elements around": [0],
            "Target element density along longest segment": 4.0,
            "Annotation numbers of elements along": [0],
            "Number of elements through shell": 1,
            "Use linear through shell": False,
            "Use outer trim surfaces": True,
            "Show trim surfaces": False,
            "Core": False,
            "Number of elements across core box minor": 2,
            "Number of elements across core transition": 1,
            "Annotation numbers of elements across core box minor": [0],
            "Refine": False,
            "Refine number of elements": 4,
            "Refine number of elements through shell": 2
        }
        if parameterSetName in ["Loop", "Snake", "Vase"]:
            options["Target element density along longest segment"] = 12.0
        elif "Bone" == parameterSetName:
            options["Target element density along longest segment"] = 8.0
        elif "Sphere" == parameterSetName:
            options["Number of elements around"] = 16
            options["Number of elements across core box minor"] = 4
            options["Target element density along longest segment"] = 5.0
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Network layout",
            "Number of elements around",
            "Annotation numbers of elements around",
            "Target element density along longest segment",
            "Annotation numbers of elements along",
            "Number of elements through shell",
            "Use linear through shell",
            "Use outer trim surfaces",
            "Show trim surfaces",
            "Core",
            "Number of elements across core box minor",
            "Number of elements across core transition",
            "Annotation numbers of elements across core box minor",
            "Refine",
            "Refine number of elements",
            "Refine number of elements through shell"
        ]

    @classmethod
    def getOptionValidScaffoldTypes(cls, optionName):
        if optionName == "Network layout":
            return [MeshType_1d_network_layout1]
        return []

    @classmethod
    def getOptionScaffoldTypeParameterSetNames(cls, optionName, scaffoldType):
        assert scaffoldType in cls.getOptionValidScaffoldTypes(optionName), \
            cls.__name__ + ".getOptionScaffoldTypeParameterSetNames.  " + \
            "Invalid option \"" + optionName + "\" scaffold type " + scaffoldType.getName()
        return scaffoldType.getParameterSetNames()  # use the defaults from the network layout scaffold

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
        if optionName == "Network layout":
            if not parameterSetName:
                parameterSetName = "Default"
            return ScaffoldPackage(scaffoldType,
                                   {"scaffoldSettings": {"Define inner coordinates": True}},
                                   defaultParameterSetName=parameterSetName)
        assert False, cls.__name__ + ".getOptionScaffoldPackage:  Option " + optionName + " is not a scaffold"

    @classmethod
    def checkOptions(cls, options):
        if not options["Network layout"].getScaffoldType() in cls.getOptionValidScaffoldTypes("Network layout"):
            options["Network layout"] = cls.getOptionScaffoldPackage("Network layout", MeshType_1d_network_layout1)
        dependentChanges = False

        # element counts across core box major/minor are used for domes even without a core so must be checked here
        networkLayoutOptions = options["Network layout"].getScaffoldSettings()
        structure_string = networkLayoutOptions["Structure"]
        has_domes = any(end_style in structure_string for end_style in ('(', ')'))
        core = options["Core"]

        if core or has_domes:
            if options["Number of elements around"] < 8:
                options["Number of elements around"] = 8
                dependentChanges = True
            elif options["Number of elements around"] % 4:
                options["Number of elements around"] += 4 - (options["Number of elements around"] % 4)
                dependentChanges = True

            annotationAroundCounts = options["Annotation numbers of elements around"]
            minAroundCount = options["Number of elements around"]
            if len(annotationAroundCounts) == 0:
                annotationAroundCounts = options["Annotation numbers of elements around"] = [0]
            else:
                for i in range(len(annotationAroundCounts)):
                    if annotationAroundCounts[i] < 0:
                        annotationAroundCounts[i] = 0
                        dependentChanges = True
                    else:
                        if annotationAroundCounts[i] < 8:
                            annotationAroundCounts[i] = 8
                            dependentChanges = True
                        elif annotationAroundCounts[i] % 4:
                            annotationAroundCounts[i] += 4 - (annotationAroundCounts[i] % 4)
                            dependentChanges = True
                        if annotationAroundCounts[i] < minAroundCount:
                            minAroundCount = annotationAroundCounts[i]

            maxCoreBoxMinorCount = options["Number of elements around"] // 2 - 2
            if options["Number of elements across core box minor"] < 2:
                options["Number of elements across core box minor"] = 2
                dependentChanges = True
            elif options["Number of elements across core box minor"] > maxCoreBoxMinorCount:
                options["Number of elements across core box minor"] = maxCoreBoxMinorCount
                dependentChanges = True
            elif options["Number of elements across core box minor"] % 2:
                options["Number of elements across core box minor"] += 1
                dependentChanges = True

            annotationCoreBoxMinorCounts = options["Annotation numbers of elements across core box minor"]
            if len(annotationCoreBoxMinorCounts) == 0:
                annotationCoreBoxMinorCounts = options["Annotation numbers of elements across core box minor"] = [0]
                dependentChanges = True
            if len(annotationCoreBoxMinorCounts) > len(annotationAroundCounts):
                annotationCoreBoxMinorCounts = options["Annotation numbers of elements across core box minor"] = \
                    annotationCoreBoxMinorCounts[:len(annotationAroundCounts)]
                dependentChanges = True
            for i in range(len(annotationCoreBoxMinorCounts)):
                aroundCount = annotationAroundCounts[i] if annotationAroundCounts[i] \
                    else options["Number of elements around"]
                maxCoreBoxMinorCount = aroundCount // 2 - 2
                if annotationCoreBoxMinorCounts[i] <= 0:
                    annotationCoreBoxMinorCounts[i] = 0
                    # this may reduce the default
                    if maxCoreBoxMinorCount < options["Number of elements across core box minor"]:
                        options["Number of elements across core box minor"] = maxCoreBoxMinorCount
                        dependentChanges = True
                elif annotationCoreBoxMinorCounts[i] < 2:
                    annotationCoreBoxMinorCounts[i] = 2
                    dependentChanges = True
                elif annotationCoreBoxMinorCounts[i] > maxCoreBoxMinorCount:
                    annotationCoreBoxMinorCounts[i] = maxCoreBoxMinorCount
                    dependentChanges = True
                elif annotationCoreBoxMinorCounts[i] % 2:
                    annotationCoreBoxMinorCounts[i] += 1
                    dependentChanges = True
        else:
            if options["Number of elements around"] < 4:
                options["Number of elements around"] = 4
            annotationAroundCounts = options["Annotation numbers of elements around"]
            if len(annotationAroundCounts) == 0:
                options["Annotation numbers of elements around"] = [0]
            else:
                for i in range(len(annotationAroundCounts)):
                    if annotationAroundCounts[i] <= 0:
                        annotationAroundCounts[i] = 0
                    elif annotationAroundCounts[i] < 4:
                        annotationAroundCounts[i] = 4

        if core and options["Use linear through shell"]:
            options["Use linear through shell"] = False
            dependentChanges = True

        if options["Number of elements through shell"] < 0:
            options["Number of elements through shell"] = 0

        if options["Number of elements across core transition"] < 1:
            options["Number of elements across core transition"] = 1

        if options["Target element density along longest segment"] < 1.0:
            options["Target element density along longest segment"] = 1.0
        annotationAlongCounts = options["Annotation numbers of elements along"]
        if len(annotationAlongCounts) == 0:
            options["Annotation numbers of elements along"] = [0]
        else:
            for i in range(len(annotationAlongCounts)):
                if annotationAlongCounts[i] <= 0:
                    annotationAlongCounts[i] = 0
                elif annotationAlongCounts[i] < 1:
                    annotationAlongCounts[i] = 1

        for key in [
            "Refine number of elements",
            "Refine number of elements through shell"
        ]:
            if options[key] < 1:
                options[key] = 1

        return dependentChanges

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the base tricubic hermite or bicubic hermite-linear mesh. See also generateMesh().
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: list of AnnotationGroup, None
        """
        layoutRegion = region.createRegion()
        networkLayout = options["Network layout"]
        networkLayout.generate(layoutRegion)  # ask scaffold to generate to get user-edited parameters
        networkMesh = networkLayout.getConstructionObject()

        shell_count = options["Number of elements through shell"]
        core = options["Core"]
        mesh_dimension = 3 if (shell_count or core) else 2
        isLinearThroughShell = options["Use linear through shell"] or (mesh_dimension == 2)

        tubeNetworkMeshBuilder = TubeNetworkMeshBuilder(
            networkMesh,
            targetElementDensityAlongLongestSegment=options["Target element density along longest segment"],
            layoutAnnotationGroups=networkLayout.getAnnotationGroups(),
            annotationElementsCountsAlong=options["Annotation numbers of elements along"],
            defaultElementsCountAround=options["Number of elements around"],
            annotationElementsCountsAround=options["Annotation numbers of elements around"],
            shell_count=shell_count,
            core=core,
            transition_count=options["Number of elements across core transition"],
            defaultElementsCountCoreBoxMinor=options["Number of elements across core box minor"],
            annotationElementsCountsCoreBoxMinor=options["Annotation numbers of elements across core box minor"],
            useOuterTrimSurfaces=options["Use outer trim surfaces"])
        tubeNetworkMeshBuilder.build()
        generateData = TubeNetworkMeshGenerateData(
            region, mesh_dimension,
            isLinearThroughShell=isLinearThroughShell,
            isShowTrimSurfaces=options["Show trim surfaces"])
        tubeNetworkMeshBuilder.generateMesh(generateData)
        annotationGroups = generateData.getAnnotationGroups()

        return annotationGroups, tubeNetworkMeshBuilder

    @classmethod
    def refineMesh(cls, meshRefinement, options):
        """
        Refine source mesh into separate region, with change of basis.
        :param meshRefinement: MeshRefinement, which knows source and target region.
        :param options: Dict containing options. See getDefaultOptions().
        """
        refine_count = options["Refine number of elements"]
        refine_shell_count = options["Refine number of elements through shell"]
        shell_count = options["Number of elements through shell"]
        core = options["Core"]
        mesh_dimension = 3 if (shell_count or core) else 2
        if mesh_dimension == 3:
            refine_count3 = refine_count if core else refine_shell_count
            annotation_refinements = {
                "shell": (refine_count, refine_count, refine_shell_count)} if (shell_count and core) else None
            meshRefinement.refineAllElementsCubeStandard3d(
                refine_count, refine_count, refine_count3, annotation_refinements)
        else:
            meshRefinement.refineAllElementsSquareStandard2d(refine_count, refine_count)

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
        TubeNetworkMeshBuilder.defineFaceAnnotations(
            region, annotationGroups, options["Core"], options["Number of elements through shell"])
