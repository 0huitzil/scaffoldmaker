"""
Generates a solid 3d ellipsoid of hexahedral elements or a shell of hex (3d) or quad (2d) elements.
"""
import math
from cmlibs.utils.zinc.field import find_or_create_field_coordinates
from cmlibs.zinc.element import Element
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, findOrCreateAnnotationGroupForTerm
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.utils.ellipsoidmesh import EllipsoidMesh, EllipsoidSurfaceD3Mode
from scaffoldmaker.utils.meshgeneratedata import MeshGenerateData
from scaffoldmaker.utils.meshrefinement import MeshRefinement


class MeshType_3d_ellipsoid1(Scaffold_base):
    """
    Generates a solid 3d ellipsoid of hexahedral elements or a shell of hex (3d) or quad (2d) elements.
    """

    @classmethod
    def getName(cls):
        return "3D Ellipsoid 1"

    @classmethod
    def getDefaultOptions(cls, parameterSetName='Default'):
        options = {
            "Numbers of elements across axes": [4, 6, 8],
            "Number of elements through shell": 0,
            "Number of elements across core transition": 1,
            "Axes lengths": [1.0, 1.5, 2.0],
            "Axes shell thicknesses": [0.2, 0.2, 0.2],
            "Axis 2 x-rotation degrees": 0.0,
            "Axis 3 x-rotation degrees": 90.0,
            "Use linear through shell": False,
            "Core": True,
            "Core shell scaling mode": 1,
            "Advanced n-way derivative factor": 0.6,
            "Advanced surface D3 mode": EllipsoidSurfaceD3Mode.SURFACE_NORMAL.value,
            "Refine": False,
            "Refine number of elements": 4,
            "Refine number of elements through shell": 2
        }
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Numbers of elements across axes",
            "Number of elements through shell",
            "Number of elements across core transition",
            "Axes lengths",
            "Axes shell thicknesses",
            "Axis 2 x-rotation degrees",
            "Axis 3 x-rotation degrees",
            "Use linear through shell",
            "Core",
            "Core shell scaling mode",
            "Advanced n-way derivative factor",
            "Advanced surface D3 mode",
            "Refine",
            "Refine number of elements",
            "Refine number of elements through shell"
        ]

    @classmethod
    def checkOptions(cls, options):
        dependent_changes = False

        max_rim_count = None
        axes_numbers = options["Numbers of elements across axes"]
        count = len(axes_numbers)
        if count < 3:
            for i in range(3 - count):
                axes_numbers.append(axes_numbers[-1])
        elif count > 3:
            del axes_numbers[3:]
        for i, number in enumerate(axes_numbers):
            if number < 4:
                axes_numbers[i] = 4
            elif number % 2:
                axes_numbers[i] += 1
            transition_count = (axes_numbers[i] // 2) - 1
            if (max_rim_count is None) or (transition_count < max_rim_count):
                max_rim_count = transition_count

        # prioritise shell count if possible
        shell_count = options["Number of elements through shell"]
        if shell_count < 0:
            shell_count = options["Number of elements through shell"] = 0
        elif shell_count > max_rim_count - 1:
            shell_count = options["Number of elements through shell"] = max_rim_count - 1
            dependent_changes = True
        transition_count = options["Number of elements across core transition"]
        if transition_count < 1:
            transition_count = options["Number of elements across core transition"] = 1
        elif (shell_count + transition_count) > max_rim_count:
            transition_count = options["Number of elements across core transition"] = max_rim_count - shell_count
            dependent_changes = True

        axes_lengths = options["Axes lengths"]
        count = len(axes_lengths)
        if count < 3:
            for i in range(3 - count):
                axes_lengths.append(axes_lengths[-1])
        elif count > 3:
            del axes_lengths[3:]
        for i, length in enumerate(axes_lengths):
            if length <= 0.0:
                axes_lengths[i] = 1.0

        axes_thicknesses = options["Axes shell thicknesses"]
        count = len(axes_thicknesses)
        if count < 3:
            for i in range(3 - count):
                axes_thicknesses.append(axes_thicknesses[-1])
        elif count > 3:
            del axes_thicknesses[3:]
        for i, thickness in enumerate(axes_thicknesses):
            if thickness <= 0.0:
                axes_thicknesses[i] = axes_lengths[i] * 0.2

        core = options["Core"]
        if core and options["Use linear through shell"]:
            options["Use linear through shell"] = False
            dependentChanges = True

        if options["Core shell scaling mode"] not in (1, 2):
            options["Core shell scaling mode"] = 1
        if options["Advanced n-way derivative factor"] < 0.1:
            options["Advanced n-way derivative factor"] = 0.1
        elif options["Advanced n-way derivative factor"] > 1.0:
            options["Advanced n-way derivative factor"] = 1.0

        try:
            mode = EllipsoidSurfaceD3Mode(options["Advanced surface D3 mode"])
        except ValueError:
            options["Advanced surface D3 mode"] = EllipsoidSurfaceD3Mode.SURFACE_NORMAL.value

        for key in [
            "Refine number of elements",
            "Refine number of elements through shell"
        ]:
            if options[key] < 1:
                options[key] = 1

        return dependent_changes

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the base tricubic Hermite mesh. See also generateMesh().
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: empty list of AnnotationGroup, None
        """
        element_counts = options["Numbers of elements across axes"]
        shell_count = options["Number of elements through shell"]
        transition_count = options["Number of elements across core transition"]
        axes_lengths = options["Axes lengths"]
        axes_shell_thicknesses = options["Axes shell thicknesses"]

        axis2_x_rotation_radians = math.radians(options["Axis 2 x-rotation degrees"])
        axis3_x_rotation_radians = math.radians(options["Axis 3 x-rotation degrees"])
        core = options["Core"]
        core_shell_scaling_mode = options["Core shell scaling mode"]
        nway_d_factor = options["Advanced n-way derivative factor"]
        surface_d3_mode = EllipsoidSurfaceD3Mode(options["Advanced surface D3 mode"])

        fieldmodule = region.getFieldmodule()
        coordinates = find_or_create_field_coordinates(fieldmodule)

        ellipsoid = EllipsoidMesh(element_counts, shell_count, transition_count, core,
                                  core_shell_scaling_mode=core_shell_scaling_mode)

        left_group = AnnotationGroup(region, ("left", ""))
        right_group = AnnotationGroup(region, ("right", ""))
        back_group = AnnotationGroup(region, ("back", ""))
        front_group = AnnotationGroup(region, ("front", ""))
        bottom_group = AnnotationGroup(region, ("bottom", ""))
        top_group = AnnotationGroup(region, ("top", ""))
        annotation_groups = [left_group, right_group, back_group, front_group, bottom_group, top_group]
        octant_group_lists = []
        for octant in range(8):
            octant_group_list = []
            octant_group_list.append((right_group if (octant & 1) else left_group).getGroup())
            octant_group_list.append((front_group if (octant & 2) else back_group).getGroup())
            octant_group_list.append((top_group if (octant & 4) else bottom_group).getGroup())
            octant_group_lists.append(octant_group_list)
        ellipsoid.set_octant_group_lists(octant_group_lists)

        if core:
            box_group = AnnotationGroup(region, ("box", ""))
            transition_group = AnnotationGroup(region, ("transition", ""))
            core_group = AnnotationGroup(region, ("core", ""))
            annotation_groups += [box_group, core_group, transition_group]
            shell_group = AnnotationGroup(region, ("shell", "")) if shell_count else None
            if shell_group:
                annotation_groups.append(shell_group)
            ellipsoid.set_box_transition_groups(box_group.getGroup(), transition_group.getGroup())
            ellipsoid.set_core_shell_groups(core_group.getGroup(), shell_group.getGroup() if shell_group else None)

        ellipsoid.build(axes_lengths, axis2_x_rotation_radians, axis3_x_rotation_radians, axes_shell_thicknesses,
                        nway_d_factor=nway_d_factor, surface_d3_mode=surface_d3_mode)
        generate_data = MeshGenerateData(region, meshDimension=(2 if ((shell_count == 0) and not core) else 3))
        generate_data.setLinearThroughShell(options["Use linear through shell"])
        ellipsoid.generate_mesh(generate_data)

        return annotation_groups, None

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
        core = options["Core"]
        shell_count = shell_count = options["Number of elements through shell"]
        if core and (shell_count == 0):
            fieldmodule = region.getFieldmodule()
            mesh2d = fieldmodule.findMeshByDimension(2)
            shell = findOrCreateAnnotationGroupForTerm(annotationGroups, region, ("shell", ""))
            is_exterior = fieldmodule.createFieldIsExterior()
            is_exterior_face_xi3_1 = fieldmodule.createFieldAnd(
                fieldmodule.createFieldIsExterior(), fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
            shell.getMeshGroup(mesh2d).addElementsConditional(is_exterior_face_xi3_1)
