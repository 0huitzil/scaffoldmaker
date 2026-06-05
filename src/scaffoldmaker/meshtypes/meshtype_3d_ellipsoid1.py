"""
Generates a solid ellipsoid of hexahedral elements.
"""
import math
from cmlibs.utils.zinc.field import find_or_create_field_coordinates
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.utils.ellipsoidmesh import EllipsoidMesh, EllipsoidSurfaceD3Mode
from scaffoldmaker.utils.meshrefinement import MeshRefinement


class MeshType_3d_ellipsoid1(Scaffold_base):
    """
    Generates a solid ellipsoid of hexahedral elements.
    """

    @classmethod
    def getName(cls):
        return "3D Ellipsoid 1"

    @classmethod
    def getDefaultOptions(cls, parameterSetName='Default'):
        options = {
            "Number of elements across axis 1": 4,
            "Number of elements across axis 2": 6,
            "Number of elements across axis 3": 8,
            "2D surface only": False,
            "Number of transition elements": 1,
            "Axis length x": 1.0,
            "Axis length y": 1.5,
            "Axis length z": 2.0,
            "Axis 2 x-rotation degrees": 0.0,
            "Axis 3 x-rotation degrees": 90.0,
            "Advanced n-way derivative factor": 0.6,
            "Advanced surface D3 mode": EllipsoidSurfaceD3Mode.SURFACE_NORMAL.value,
            "Refine": False,
            "Refine number of elements": 4,
        }
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            "Number of elements across axis 1",
            "Number of elements across axis 2",
            "Number of elements across axis 3",
            "Number of transition elements",
            "2D surface only",
            "Axis length x",
            "Axis length y",
            "Axis length z",
            "Axis 2 x-rotation degrees",
            "Axis 3 x-rotation degrees",
            "Advanced n-way derivative factor",
            "Advanced surface D3 mode",
            "Refine",
            "Refine number of elements"
        ]

    @classmethod
    def checkOptions(cls, options):
        dependent_changes = False
        max_transition_count = None
        for key in [
            "Number of elements across axis 1",
            "Number of elements across axis 2",
            "Number of elements across axis 3"
        ]:
            if options[key] < 4:
                options[key] = 4
            elif options[key] % 2:
                options[key] += 1
            transition_count = (options[key] // 2) - 1
            if (max_transition_count is None) or (transition_count < max_transition_count):
                max_transition_count = transition_count

        if options["Number of transition elements"] < 1:
            options["Number of transition elements"] = 1
        elif options["Number of transition elements"] > max_transition_count:
            options["Number of transition elements"] = max_transition_count
            dependent_changes = True

        for key in [
            "Axis length x",
            "Axis length y",
            "Axis length z"
        ]:
            if options[key] <= 0.0:
                options[key] = 1.0

        if options["Advanced n-way derivative factor"] < 0.1:
            options["Advanced n-way derivative factor"] = 0.1
        elif options["Advanced n-way derivative factor"] > 1.0:
            options["Advanced n-way derivative factor"] = 1.0

        try:
            mode = EllipsoidSurfaceD3Mode(options["Advanced surface D3 mode"])
        except ValueError:
            options["Advanced surface D3 mode"] = EllipsoidSurfaceD3Mode.SURFACE_NORMAL.value

        for key in [
            "Refine number of elements"
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
        element_counts = [options[key] for key in [
            "Number of elements across axis 1", "Number of elements across axis 2", "Number of elements across axis 3"]]
        transition_element_count = options["Number of transition elements"]
        a = options["Axis length x"]
        b = options["Axis length y"]
        c = options["Axis length z"]
        axis2_x_rotation_radians = math.radians(options["Axis 2 x-rotation degrees"])
        axis3_x_rotation_radians = math.radians(options["Axis 3 x-rotation degrees"])
        surface_only = options["2D surface only"]
        nway_derivative_factor = options["Advanced n-way derivative factor"]
        surface_d3_mode = EllipsoidSurfaceD3Mode(options["Advanced surface D3 mode"])

        fieldmodule = region.getFieldmodule()
        coordinates = find_or_create_field_coordinates(fieldmodule)

        ellipsoid = EllipsoidMesh(a, b, c, element_counts, transition_element_count, surface_only)

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

        if not surface_only:
            box_group = AnnotationGroup(region, ("box", ""))
            transition_group = AnnotationGroup(region, ("transition", ""))
            annotation_groups += [box_group, transition_group]
            ellipsoid.set_box_transition_groups(box_group.getGroup(), transition_group.getGroup())

        ellipsoid.set_nway_derivative_factor(nway_derivative_factor)
        ellipsoid.set_surface_d3_mode(surface_d3_mode)

        ellipsoid.build(axis2_x_rotation_radians, axis3_x_rotation_radians)
        ellipsoid.generate_mesh(fieldmodule, coordinates)

        return annotation_groups, None

    @classmethod
    def refineMesh(cls, meshRefinement, options):
        """
        Refine source mesh into separate region, with change of basis.
        :param meshRefinement: MeshRefinement, which knows source and target region.
        :param options: Dict containing options. See getDefaultOptions().
        """
        assert isinstance(meshRefinement, MeshRefinement)
        refineElementsCount = options["Refine number of elements"]
        meshRefinement.refineAllElementsCubeStandard3d(refineElementsCount, refineElementsCount, refineElementsCount)
