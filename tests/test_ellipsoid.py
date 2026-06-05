from cmlibs.utils.zinc.finiteelement import evaluateFieldNodesetRange
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.context import Context
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from cmlibs.zinc.result import RESULT_OK
from scaffoldmaker.meshtypes.meshtype_3d_ellipsoid1 import MeshType_3d_ellipsoid1
from testutils import assertAlmostEqualList
import unittest


class EllipsoidScaffoldTestCase(unittest.TestCase):

    def test_ellipsoid_2D(self):
        """
        Test creation of 2-D ellipsoid surface.
        """
        scaffold_class = MeshType_3d_ellipsoid1
        parameter_set_names = scaffold_class.getParameterSetNames()
        self.assertEqual(parameter_set_names, ["Default"])
        options = scaffold_class.getDefaultOptions("Default")
        self.assertEqual(14, len(options))
        self.assertEqual(4, options["Number of elements across axis 1"])
        self.assertEqual(6, options["Number of elements across axis 2"])
        self.assertEqual(8, options["Number of elements across axis 3"])
        self.assertFalse(options["2D surface only"])
        self.assertEqual(1, options["Number of transition elements"])
        self.assertEqual(1.0, options["Axis length x"])
        self.assertEqual(1.5, options["Axis length y"])
        self.assertEqual(2.0, options["Axis length z"])
        self.assertEqual(0.0, options["Axis 2 x-rotation degrees"])
        self.assertEqual(90.0, options["Axis 3 x-rotation degrees"])
        self.assertEqual(0.6, options["Advanced n-way derivative factor"])
        self.assertEqual(1, options["Advanced surface D3 mode"])
        self.assertFalse(options["Refine"])
        self.assertEqual(4, options["Refine number of elements"])
        # set test options
        options["2D surface only"] = True

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

    def test_ellipsoid_3D(self):
        """
        Test creation of 3-D ellipsoid surface.
        """
        scaffold_class = MeshType_3d_ellipsoid1
        parameter_set_names = scaffold_class.getParameterSetNames()
        self.assertEqual(parameter_set_names, ["Default"])
        options = scaffold_class.getDefaultOptions("Default")
        self.assertEqual(14, len(options))
        self.assertEqual(4, options["Number of elements across axis 1"])
        self.assertEqual(6, options["Number of elements across axis 2"])
        self.assertEqual(8, options["Number of elements across axis 3"])
        self.assertFalse(options["2D surface only"])
        self.assertEqual(1, options["Number of transition elements"])
        self.assertEqual(1.0, options["Axis length x"])
        self.assertEqual(1.5, options["Axis length y"])
        self.assertEqual(2.0, options["Axis length z"])
        self.assertEqual(0.0, options["Axis 2 x-rotation degrees"])
        self.assertEqual(90.0, options["Axis 3 x-rotation degrees"])
        self.assertEqual(0.6, options["Advanced n-way derivative factor"])
        self.assertEqual(1, options["Advanced surface D3 mode"])
        self.assertFalse(options["Refine"])
        self.assertEqual(4, options["Refine number of elements"])

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotation_groups = scaffold_class.generateMesh(region, options)[0]
        self.assertEqual(8, len(annotation_groups))

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
            volume_field.setNumbersOfPoints(3)
        result, surface_area = surface_area_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        result, volume = volume_field.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surface_area, 27.86848567909992, delta=TOL)
        # note exact ellipsoid volume is 4.0 / 3.0 * math.pi * a * b * c = 12.566370614359173
        self.assertAlmostEqual(volume, 12.557389634764352, delta=TOL)

        for annotation_group in annotation_groups:
            name = annotation_group.getName()
            if name == "box":
                self.assertEqual(48, annotation_group.getMeshGroup(mesh3d).getSize())
            elif name == "transition":
                self.assertEqual(88, annotation_group.getMeshGroup(mesh3d).getSize())
            else:
                self.assertTrue(name in ["left", "right", "back", "front", "bottom", "top"])
                self.assertEqual(68, annotation_group.getMeshGroup(mesh3d).getSize())


if __name__ == "__main__":
    unittest.main()
