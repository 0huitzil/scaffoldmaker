from cmlibs.utils.zinc.finiteelement import evaluateFieldNodesetRange
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Element
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from cmlibs.zinc.result import RESULT_ERROR_NOT_FOUND, RESULT_OK
from scaffoldmaker.annotation.annotationgroup import findAnnotationGroupByName
from scaffoldmaker.meshtypes.meshtype_3d_ellipsoid1 import MeshType_3d_ellipsoid1
from testutils import assertAlmostEqualList
import unittest


class EllipsoidScaffoldTestCase(unittest.TestCase):

    def test_ellipsoid_2d(self):
        """
        Test creation of 2-D ellipsoid surface.
        """
        scaffold_class = MeshType_3d_ellipsoid1
        parameter_set_names = scaffold_class.getParameterSetNames()
        self.assertEqual(parameter_set_names, ["Default"])
        options = scaffold_class.getDefaultOptions("Default")
        self.assertEqual(15, len(options))
        self.assertEqual([4, 6, 8], options["Numbers of elements across axes"])
        self.assertEqual(0, options["Number of elements through shell"])
        self.assertEqual(1, options["Number of elements across core transition"])
        self.assertEqual([1.0, 1.5, 2.0], options["Axes lengths"])
        self.assertEqual([0.2, 0.2, 0.2], options["Axes shell thicknesses"])
        self.assertTrue(options["Core"])
        self.assertEqual(1, options["Core shell scaling mode"])
        self.assertEqual(0.0, options["Axis 2 x-rotation degrees"])
        self.assertEqual(90.0, options["Axis 3 x-rotation degrees"])
        self.assertEqual(0.6, options["Advanced n-way derivative factor"])
        self.assertEqual(1, options["Advanced surface D3 mode"])
        self.assertFalse(options["Use linear through shell"])
        self.assertFalse(options["Refine"])
        self.assertEqual(4, options["Refine number of elements"])
        # set test options
        options["Core"] = False

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotation_groups = scaffold_class.generateMesh(region, options)[0]
        self.assertEqual(6, len(annotation_groups))

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(0, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(88, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(176, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(90, nodes.getSize())

        # check coordinates range, sphere volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        TOL = 1.0E-6
        assertAlmostEqualList(self, minimums, [-1.0, -1.5, -2.0], TOL)
        assertAlmostEqualList(self, maximums, [1.0, 1.5, 2.0], TOL)
        # test symmetry of 3-way points
        fieldcache = fieldmodule.createFieldcache()
        node_3way1 = nodes.findNodeByIdentifier(90)
        fieldcache.setNode(node_3way1)
        result, x_3way1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
        result, d1_3way1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
        result, d2_3way1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
        node_3way2 = nodes.findNodeByIdentifier(78)
        fieldcache.setNode(node_3way2)
        result, x_3way2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
        result, d1_3way2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
        result, d2_3way2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
        assertAlmostEqualList(self, x_3way2, [x_3way1[0], -x_3way1[1], x_3way1[2]], TOL)
        assertAlmostEqualList(self, d1_3way2, [d1_3way1[0], -d1_3way1[1], d1_3way1[2]], TOL)
        assertAlmostEqualList(self, d2_3way2, [-d2_3way1[0], d2_3way1[1], -d2_3way1[2]], TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            surface_area_field = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surface_area_field.setNumbersOfPoints(4)
        result, surface_area = surface_area_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surface_area, 27.86848567909992, delta=TOL)

        for annotation_group in annotation_groups:
            self.assertEqual(44, annotation_group.getMeshGroup(mesh2d).getSize())

    def test_ellipsoid_3d_core_0shell(self):
        """
        Test creation of 3-D ellipsoid volume.
        """
        scaffold_class = MeshType_3d_ellipsoid1
        parameter_set_names = scaffold_class.getParameterSetNames()
        self.assertEqual(parameter_set_names, ["Default"])
        options = scaffold_class.getDefaultOptions("Default")
        self.assertEqual(15, len(options))
        self.assertEqual([4, 6, 8], options["Numbers of elements across axes"])
        self.assertEqual(0, options["Number of elements through shell"])
        self.assertEqual(1, options["Number of elements across core transition"])
        self.assertEqual([1.0, 1.5, 2.0], options["Axes lengths"])
        self.assertEqual([0.2, 0.2, 0.2], options["Axes shell thicknesses"])
        self.assertTrue(options["Core"])
        self.assertEqual(1, options["Core shell scaling mode"])
        self.assertEqual(0.0, options["Axis 2 x-rotation degrees"])
        self.assertEqual(90.0, options["Axis 3 x-rotation degrees"])
        self.assertEqual(0.6, options["Advanced n-way derivative factor"])
        self.assertEqual(1, options["Advanced surface D3 mode"])
        self.assertFalse(options["Use linear through shell"])
        self.assertFalse(options["Refine"])
        self.assertEqual(4, options["Refine number of elements"])
        self.assertEqual(2, options["Refine number of elements through shell"])

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotation_groups = scaffold_class.generateMesh(region, options)[0]
        self.assertEqual(10, len(annotation_groups))

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(136, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(452, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(510, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(195, nodes.getSize())

        # check coordinates range, sphere volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        TOL = 1.0E-6
        assertAlmostEqualList(self, minimums, [-1.0, -1.5, -2.0], TOL)
        assertAlmostEqualList(self, maximums, [1.0, 1.5, 2.0], TOL)
        # test symmetry of 4-way points
        fieldcache = fieldmodule.createFieldcache()
        node_3way1 = nodes.findNodeByIdentifier(180)
        fieldcache.setNode(node_3way1)
        result, x_3way1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
        result, d1_3way1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
        result, d2_3way1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
        result, d3_3way1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)
        node_3way2 = nodes.findNodeByIdentifier(168)
        fieldcache.setNode(node_3way2)
        result, x_3way2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
        result, d1_3way2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
        result, d2_3way2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
        result, d3_3way2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)
        assertAlmostEqualList(self, x_3way2, [x_3way1[0], -x_3way1[1], x_3way1[2]], TOL)
        assertAlmostEqualList(self, d1_3way2, [d1_3way1[0], -d1_3way1[1], d1_3way1[2]], TOL)
        assertAlmostEqualList(self, d2_3way2, [-d2_3way1[0], d2_3way1[1], -d2_3way1[2]], TOL)
        assertAlmostEqualList(self, d3_3way2, [d3_3way1[0], -d3_3way1[1], d3_3way1[2]], TOL)

        with (ChangeManager(fieldmodule)):
            is_exterior = fieldmodule.createFieldIsExterior()
            surface_group = fieldmodule.createFieldGroup()
            surface_mesh_group = surface_group.createMeshGroup(mesh2d)
            surface_mesh_group.addElementsConditional(is_exterior)
            self.assertEqual(88, surface_mesh_group.getSize())
            one = fieldmodule.createFieldConstant(1.0)
            surface_area_field = fieldmodule.createFieldMeshIntegral(one, coordinates, surface_mesh_group)
            surface_area_field.setNumbersOfPoints(4)
            volume_field = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volume_field.setNumbersOfPoints(4)
        result, total_surface_area = surface_area_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        result, total_volume = volume_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        expected_surface_area = 27.86848567909992
        expected_box_volume = 3.873227965966105
        expected_transition_volume = 8.684168536163744
        # note exact ellipsoid volume is 4.0 / 3.0 * math.pi * a * b * c = 12.566370614359173
        expected_total_volume = expected_box_volume + expected_transition_volume  # 12.557396502129784

        expected_sizes_3d = {
            "box": (48, expected_box_volume),
            "transition": (88, expected_transition_volume),
            "core": (136, expected_total_volume),
            "left": (68, 0.5 * expected_total_volume),
            "right": (68, 0.5 * expected_total_volume),
            "back": (68, 0.5 * expected_total_volume),
            "front": (68, 0.5 * expected_total_volume),
            "bottom": (68, 0.5 * expected_total_volume),
            "top": (68, 0.5 * expected_total_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expected_sizes_3d:
            annotation_group = findAnnotationGroupByName(annotation_groups, name)
            size = annotation_group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expected_sizes_3d[name][0], size, name)
            volume_mesh_group = annotation_group.getMeshGroup(mesh3d)
            volume_field = fieldmodule.createFieldMeshIntegral(one, coordinates, volume_mesh_group)
            volume_field.setNumbersOfPoints(4)
            result, volume = volume_field.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expected_sizes_3d[name][1], msg=name, delta=TOL)

        self.assertAlmostEqual(total_surface_area, expected_surface_area, delta=TOL)
        self.assertAlmostEqual(total_volume, expected_total_volume, delta=TOL)

        expected_sizes_2d = {
            "shell": (88, expected_surface_area)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expected_sizes_2d:
            annotation_group = findAnnotationGroupByName(annotation_groups, name)
            size = annotation_group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expected_sizes_2d[name][0], size, name)
            surface_area_mesh_group = annotation_group.getMeshGroup(mesh2d)
            surface_area_field = fieldmodule.createFieldMeshIntegral(one, coordinates, surface_area_mesh_group)
            surface_area_field.setNumbersOfPoints(4)
            result, surface_area = surface_area_field.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surface_area, expected_sizes_2d[name][1], msg=name, delta=TOL)

    def test_ellipsoid_3d_core_symmetry(self):
        """
        Test creation of 3-D ellipsoid volume.
        """
        scaffold_class = MeshType_3d_ellipsoid1
        options = scaffold_class.getDefaultOptions()
        options["Numbers of elements across axes"] = [6, 6, 6]
        options["Number of elements through shell"] = 0
        options["Axes lengths"] = [1.0, 1.0, 1.0]
        options["Axes shell thicknesses"] = [0.2, 0.2, 0.2]
        options["Core"] = True

        context = Context("Test")
        region = context.getDefaultRegion()
        scaffold_class.generateMesh(region, options)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(160, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(528, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(590, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(223, nodes.getSize())

        # check coordinates range, sphere volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        TOL = 2.0E-8
        assertAlmostEqualList(self, minimums, [-1.0, -1.0, -1.0], TOL)
        assertAlmostEqualList(self, maximums, [1.0, 1.0, 1.0], TOL)

        fieldcache = fieldmodule.createFieldcache()
        with (ChangeManager(fieldmodule)):
            is_exterior = fieldmodule.createFieldIsExterior()
            surface_group = fieldmodule.createFieldGroup()
            surface_mesh_group = surface_group.createMeshGroup(mesh2d)
            surface_mesh_group.addElementsConditional(is_exterior)
            self.assertEqual(96, surface_mesh_group.getSize())
            one = fieldmodule.createFieldConstant(1.0)
            surface_area_field = fieldmodule.createFieldMeshIntegral(one, coordinates, surface_mesh_group)
            surface_area_field.setNumbersOfPoints(4)
            volume_field = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volume_field.setNumbersOfPoints(3)
        result, surface_area = surface_area_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        result, volume = volume_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surface_area, 12.561730858137276, delta=TOL)
        # note exact ellipsoid volume is 4.0 / 3.0 * math.pi * a * b * c = 12.566370614359173
        self.assertAlmostEqual(volume, 4.186442287004547, delta=TOL)

        # these nodes should have +/- symmetry of coordinates
        node_identifiers = [63, 65, 77, 79, 145, 147, 159, 161]
        expected_value = 0.29334831
        for node_identifier in node_identifiers:
            node = nodes.findNodeByIdentifier(node_identifier)
            fieldcache.setNode(node)
            result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
            self.assertEqual(result, RESULT_OK)
            for value in x:
                self.assertAlmostEqual(abs(value), expected_value, delta=TOL)

    def test_ellipsoid_3d_core_1shell(self):
        """
        Test creation of 3-D ellipsoid surface with 1 shell layer.
        """
        scaffold_class = MeshType_3d_ellipsoid1
        parameter_set_names = scaffold_class.getParameterSetNames()
        self.assertEqual(parameter_set_names, ["Default"])
        options = scaffold_class.getDefaultOptions("Default")
        self.assertEqual(15, len(options))
        self.assertEqual([4, 6, 8], options["Numbers of elements across axes"])
        self.assertEqual(0, options["Number of elements through shell"])
        self.assertEqual(1, options["Number of elements across core transition"])
        self.assertEqual([1.0, 1.5, 2.0], options["Axes lengths"])
        self.assertEqual([0.2, 0.2, 0.2], options["Axes shell thicknesses"])
        self.assertTrue(options["Core"])
        self.assertEqual(1, options["Core shell scaling mode"])
        self.assertEqual(0.0, options["Axis 2 x-rotation degrees"])
        self.assertEqual(90.0, options["Axis 3 x-rotation degrees"])
        self.assertEqual(0.6, options["Advanced n-way derivative factor"])
        self.assertEqual(1, options["Advanced surface D3 mode"])
        self.assertFalse(options["Use linear through shell"])
        self.assertFalse(options["Refine"])
        self.assertEqual(4, options["Refine number of elements"])
        self.assertEqual(2, options["Refine number of elements through shell"])
        options["Numbers of elements across axes"] = [6, 6, 8]
        options["Number of elements through shell"] = 1

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotation_groups = scaffold_class.generateMesh(region, options)[0]
        self.assertEqual(10, len(annotation_groups))

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(96, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(308, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(340, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(129, nodes.getSize())

        # check coordinates range, sphere volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        TOL = 1.0E-6
        assertAlmostEqualList(self, minimums, [-1.0, -1.5, -2.0], TOL)
        assertAlmostEqualList(self, maximums, [1.0, 1.5, 2.0], TOL)

        fieldcache = fieldmodule.createFieldcache()

        with (ChangeManager(fieldmodule)):
            is_exterior = fieldmodule.createFieldIsExterior()
            surface_group = fieldmodule.createFieldGroup()
            surface_mesh_group = surface_group.createMeshGroup(mesh2d)
            surface_mesh_group.addElementsConditional(is_exterior)
            self.assertEqual(40, surface_mesh_group.getSize())
            one = fieldmodule.createFieldConstant(1.0)
            surface_area_field = fieldmodule.createFieldMeshIntegral(one, coordinates, surface_mesh_group)
            surface_area_field.setNumbersOfPoints(4)
            volume_field = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volume_field.setNumbersOfPoints(4)
        result, total_surface_area = surface_area_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        result, total_volume = volume_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        expected_total_surface_area = 27.834374833736717
        expected_box_volume = 1.6952914143656526
        expected_transition_volume = 6.135660301203088
        expected_core_volume = expected_box_volume + expected_transition_volume
        expected_shell_volume = 4.703184358570596
        # note exact ellipsoid volume is 4.0 / 3.0 * math.pi * a * b * c = 12.566370614359173
        expected_total_volume = expected_core_volume + expected_shell_volume  # 12.534136074139385

        expected_sizes_3d = {
            "box": (16, expected_box_volume),
            "transition": (40, expected_transition_volume),
            "core": (56, expected_core_volume),
            "shell": (40, expected_shell_volume),
            "left": (48, 0.5 * expected_total_volume),
            "right": (48, 0.5 * expected_total_volume),
            "back": (48, 0.5 * expected_total_volume),
            "front": (48, 0.5 * expected_total_volume),
            "bottom": (48, 0.5 * expected_total_volume),
            "top": (48, 0.5 * expected_total_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expected_sizes_3d:
            annotation_group = findAnnotationGroupByName(annotation_groups, name)
            size = annotation_group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expected_sizes_3d[name][0], size, name)
            volume_mesh_group = annotation_group.getMeshGroup(mesh3d)
            volume_field = fieldmodule.createFieldMeshIntegral(one, coordinates, volume_mesh_group)
            volume_field.setNumbersOfPoints(4)
            result, volume = volume_field.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expected_sizes_3d[name][1], msg=name, delta=TOL)

        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=TOL)
        self.assertAlmostEqual(total_volume, expected_total_volume, delta=TOL)

    def test_ellipsoid_3d_2shell_linear(self):
        """
        Test creation of 3-D ellipsoid surface with only a 2-element thick linear shell layer.
        """
        scaffold_class = MeshType_3d_ellipsoid1
        parameter_set_names = scaffold_class.getParameterSetNames()
        self.assertEqual(parameter_set_names, ["Default"])
        options = scaffold_class.getDefaultOptions("Default")
        self.assertEqual(15, len(options))
        self.assertEqual([4, 6, 8], options["Numbers of elements across axes"])
        self.assertEqual(0, options["Number of elements through shell"])
        self.assertEqual(1, options["Number of elements across core transition"])
        self.assertEqual([1.0, 1.5, 2.0], options["Axes lengths"])
        self.assertEqual([0.2, 0.2, 0.2], options["Axes shell thicknesses"])
        self.assertTrue(options["Core"])
        self.assertEqual(1, options["Core shell scaling mode"])
        self.assertEqual(0.0, options["Axis 2 x-rotation degrees"])
        self.assertEqual(90.0, options["Axis 3 x-rotation degrees"])
        self.assertEqual(0.6, options["Advanced n-way derivative factor"])
        self.assertEqual(1, options["Advanced surface D3 mode"])
        self.assertFalse(options["Use linear through shell"])
        self.assertFalse(options["Refine"])
        self.assertEqual(4, options["Refine number of elements"])
        self.assertEqual(2, options["Refine number of elements through shell"])
        options["Numbers of elements across axes"] = [8, 8, 8]
        options["Number of elements through shell"] = 2
        options["Core"] = False
        options["Use linear through shell"] = True

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotation_groups = scaffold_class.generateMesh(region, options)[0]
        self.assertEqual(6, len(annotation_groups))

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(48, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(168, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(196, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(78, nodes.getSize())

        # check coordinates range, sphere volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        TOL = 1.0E-6
        assertAlmostEqualList(self, minimums, [-1.0, -1.5, -2.0], TOL)
        assertAlmostEqualList(self, maximums, [1.0, 1.5, 2.0], TOL)

        fieldcache = fieldmodule.createFieldcache()

        with (ChangeManager(fieldmodule)):
            node1 = nodes.findNodeByIdentifier(1)
            fieldcache.setNode(node1)
            result = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)[0]
            self.assertEqual(result, RESULT_ERROR_NOT_FOUND)

            is_exterior = fieldmodule.createFieldIsExterior()
            is_xi3_0 = fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0)
            is_xi3_1 = fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_1)
            inner_surface_group = fieldmodule.createFieldGroup()
            inner_surface_mesh_group = inner_surface_group.createMeshGroup(mesh2d)
            inner_surface_mesh_group.addElementsConditional(fieldmodule.createFieldAnd(is_exterior, is_xi3_0))
            self.assertEqual(24, inner_surface_mesh_group.getSize())
            outer_surface_group = fieldmodule.createFieldGroup()
            outer_surface_mesh_group = outer_surface_group.createMeshGroup(mesh2d)
            outer_surface_mesh_group.addElementsConditional(fieldmodule.createFieldAnd(is_exterior, is_xi3_1))
            self.assertEqual(24, outer_surface_mesh_group.getSize())
            one = fieldmodule.createFieldConstant(1.0)
            inner_surface_area_field = fieldmodule.createFieldMeshIntegral(one, coordinates, inner_surface_mesh_group)
            inner_surface_area_field.setNumbersOfPoints(4)
            outer_surface_area_field = fieldmodule.createFieldMeshIntegral(one, coordinates, outer_surface_mesh_group)
            outer_surface_area_field.setNumbersOfPoints(4)
            volume_field = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volume_field.setNumbersOfPoints(4)
        result, inner_surface_area = inner_surface_area_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        result, outer_surface_area = outer_surface_area_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        result, total_volume = volume_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(inner_surface_area, 20.87041984582838, delta=TOL)
        self.assertAlmostEqual(outer_surface_area, 27.824012333529215, delta=TOL)
        expected_total_volume = 4.680468618329442

        for annotation_group in annotation_groups:
            name = annotation_group.getName()
            self.assertTrue(name in ["left", "right", "back", "front", "bottom", "top"])
            self.assertEqual(24, annotation_group.getMeshGroup(mesh3d).getSize(), name)
            volume_mesh_group = annotation_group.getMeshGroup(mesh3d)
            volume_field = fieldmodule.createFieldMeshIntegral(one, coordinates, volume_mesh_group)
            volume_field.setNumbersOfPoints(4)
            result, volume = volume_field.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, 0.5 * expected_total_volume, msg=name, delta=TOL)

        self.assertAlmostEqual(total_volume, expected_total_volume, delta=TOL)


if __name__ == "__main__":
    unittest.main()
