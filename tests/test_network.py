import math
import unittest

from cmlibs.maths.vectorops import magnitude
from cmlibs.utils.zinc.field import find_or_create_field_group
from cmlibs.utils.zinc.finiteelement import evaluateFieldNodesetRange
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.utils.zinc.group import identifier_ranges_to_string, mesh_group_add_identifier_ranges, \
    mesh_group_to_identifier_ranges
from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Element
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from cmlibs.zinc.result import RESULT_ERROR_NOT_FOUND, RESULT_OK
from scaffoldmaker.annotation.annotationgroup import findAnnotationGroupByName
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.meshtype_3d_boxnetwork1 import MeshType_3d_boxnetwork1
from scaffoldmaker.meshtypes.meshtype_3d_tubenetwork1 import MeshType_3d_tubenetwork1
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.utils.meshrefinement import MeshRefinement
from scaffoldmaker.utils.zinc_utils import get_nodeset_path_ordered_field_parameters


from testutils import assertAlmostEqualList


class NetworkScaffoldTestCase(unittest.TestCase):

    def test_network_layout(self):
        """
        Test creation of network layout scaffold.
        """
        scaffold = MeshType_1d_network_layout1
        options = scaffold.getDefaultOptions()
        self.assertEqual(3, len(options))
        self.assertEqual("1-2", options.get("Structure"))
        options["Structure"] = "1-2-3,3-4,3.2-5"

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotationGroups, networkMesh = scaffold.generateBaseMesh(region, options)
        self.assertEqual(0, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(4, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(5, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        interactiveFunctions = scaffold.getInteractiveFunctions()
        functionOptions = None
        for interactiveFunction in interactiveFunctions:
            if interactiveFunction[0] == "Smooth derivatives...":
                functionOptions = interactiveFunction[1]
                break
        functionOptions["Update directions"] = True
        scaffold.smoothDerivatives(region, options, None, functionOptions, "meshEdits")

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [0.0, -0.5, 0.0], 1.0E-6)
        assertAlmostEqualList(self, maximums, [3.0, 0.5, 0.0], 1.0E-6)

        networkSegments = networkMesh.getNetworkSegments()
        self.assertEqual(3, len(networkSegments))
        self.assertEqual([1, 2, 3], networkSegments[0].getNodeIdentifiers())
        self.assertEqual([1, 1, 1], networkSegments[0].getNodeVersions())
        self.assertEqual([3, 4], networkSegments[1].getNodeIdentifiers())
        self.assertEqual([1, 1], networkSegments[1].getNodeVersions())
        self.assertEqual([3, 5], networkSegments[2].getNodeIdentifiers())
        self.assertEqual([2, 1], networkSegments[2].getNodeVersions())

        # get path parameters with versions
        nx, nd1 = get_nodeset_path_ordered_field_parameters(
            nodes, coordinates, [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1],
            networkSegments[2].getNodeIdentifiers(), networkSegments[2].getNodeVersions())
        self.assertEqual(2, len(nx))
        assertAlmostEqualList(self, nx[0], [2.0, 0.0, 0.0], 1.0E-6)
        assertAlmostEqualList(self, nx[1], [3.0, 0.5, 0.0], 1.0E-6)
        expected_nd = [nx[1][c] - nx[0][c] for c in range(3)]
        assertAlmostEqualList(self, nd1[0], expected_nd, 1.0E-6)
        assertAlmostEqualList(self, nd1[1], expected_nd, 1.0E-6)

    def test_2d_tube_network_bifurcation(self):
        """
        Test 2D tube bifurcation is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Bifurcation")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        settings["Target element density along longest segment"] = 3.3
        settings["Number of elements through shell"] = 0
        settings["Core"] = False
        MeshType_3d_tubenetwork1.checkOptions(settings)

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(88, mesh2d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(99, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [0.0, -0.5894427190999916, -0.10000000000000002], X_TOL)
        assertAlmostEqualList(self, maximums, [2.044721359549996, 0.5894427190999916, 0.10000000000000002], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, 1.930234746193136, delta=X_TOL)  # same as converging bifurcation

    def test_2d_tube_network_converging_bifurcation(self):
        """
        Test 2D tube converging bifurcation is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Converging bifurcation")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        settings["Target element density along longest segment"] = 3.3
        settings["Number of elements through shell"] = 0
        settings["Core"] = False
        MeshType_3d_tubenetwork1.checkOptions(settings)

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(88, mesh2d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(99, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.044721359549995794, -0.5894427190999916, -0.10000000000000002], X_TOL)
        assertAlmostEqualList(self, maximums, [2.0, 0.5894427190999916, 0.10000000000000002], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, 1.930234746193136, delta=X_TOL)  # same as bifurcation

    def test_2d_tube_network_snake(self):
        """
        Test 2D tube snake has radial elements.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Snake")
        settings = scaffoldPackage.getScaffoldSettings()
        self.assertEqual(12.0, settings["Target element density along longest segment"])
        settings["Number of elements through shell"] = 0
        settings["Core"] = False
        MeshType_3d_tubenetwork1.checkOptions(settings)

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(96, mesh2d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(104, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.1, -0.5195896012378123, -0.1], X_TOL)
        assertAlmostEqualList(self, maximums, [4.1, 0.5195896012378123, 0.1], X_TOL)

        with ChangeManager(fieldmodule):
            # check range of d2 shows element sizes vary from inside to outside of curves
            d2 = fieldmodule.createFieldNodeValue(coordinates, Node.VALUE_LABEL_D_DS2, 1)
            mag_d2 = fieldmodule.createFieldMagnitude(d2)
            min_mag_d2, max_mag_d2 = evaluateFieldNodesetRange(mag_d2, nodes)

            one = fieldmodule.createFieldConstant(1.0)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(min_mag_d2, 0.41887416310350095, delta=X_TOL)
            self.assertAlmostEqual(max_mag_d2, 0.6283112446552643, delta=X_TOL)
            self.assertAlmostEqual(surfaceArea, 3.884872678133455, delta=X_TOL)

    def test_2d_tube_network_sphere_cube(self):
        """
        Test 2D sphere cube is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Sphere cube")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        settings["Number of elements through shell"] = 0
        settings["Core"] = False
        MeshType_3d_tubenetwork1.checkOptions(settings)

        context = Context("Test")
        region = context.getDefaultRegion()

        # add a user-defined annotation group to network layout. Must generate first
        tmpRegion = region.createRegion()
        tmpFieldmodule = tmpRegion.getFieldmodule()
        networkLayoutScaffoldPackage.generate(tmpRegion)

        annotationGroup1 = networkLayoutScaffoldPackage.createUserAnnotationGroup(("bob", "BOB:1"))
        group = annotationGroup1.getGroup()
        mesh1d = tmpFieldmodule.findMeshByDimension(1)
        meshGroup = group.createMeshGroup(mesh1d)
        mesh_group_add_identifier_ranges(meshGroup, [[1, 1], [5, 5]])
        self.assertEqual(2, meshGroup.getSize())
        self.assertEqual(1, annotationGroup1.getDimension())
        identifier_ranges_string = identifier_ranges_to_string(mesh_group_to_identifier_ranges(meshGroup))
        self.assertEqual("1,5", identifier_ranges_string)
        networkLayoutScaffoldPackage.updateUserAnnotationGroups()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(1, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(32 * 12, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(8 * 7 * 12 + 4 * 3 * 8, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(8 * 3 * 12 + (2 + 3 * 3) * 8, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        # check annotation group transferred to 2D tube
        annotationGroup = annotationGroups[0]
        self.assertEqual("bob", annotationGroup.getName())
        self.assertEqual("BOB:1", annotationGroup.getId())
        self.assertEqual(64, annotationGroup.getMeshGroup(fieldmodule.findMeshByDimension(2)).getSize())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.5664486520285047, -0.5965021612011158, -0.5986899625064467], X_TOL)
        assertAlmostEqualList(self, maximums, [0.5664486520285047, 0.5965021612011158, 0.5986899625064468], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, 4.04123479243703, delta=X_TOL)

    def test_2d_tube_network_trifurcation(self):
        """
        Test 2D tube trifurcation is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Trifurcation")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        settings["Number of elements through shell"] = 0
        settings["Core"] = False
        MeshType_3d_tubenetwork1.checkOptions(settings)

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(112, mesh2d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(126, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [0.0, -1.0707106781186548, -0.10000000000000002], X_TOL)
        assertAlmostEqualList(self, maximums, [2.0707106781186546, 1.0707106781186548, 0.1], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, 2.79226277780631, delta=X_TOL)

    def test_2d_tube_network_vase(self):
        """
        Test 2D tube vase has near constant length elements despite radius changes.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Vase")
        settings = scaffoldPackage.getScaffoldSettings()
        self.assertEqual(12.0, settings["Target element density along longest segment"])
        settings["Number of elements through shell"] = 0
        settings["Core"] = False
        MeshType_3d_tubenetwork1.checkOptions(settings)

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(96, mesh2d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(104, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-1.5, -1.5, 0.0], X_TOL)
        assertAlmostEqualList(self, maximums, [1.5, 1.5, 4.0], X_TOL)

        with ChangeManager(fieldmodule):
            # check range of d2 shows near constant element sizes
            d2 = fieldmodule.createFieldNodeValue(coordinates, Node.VALUE_LABEL_D_DS2, 1)
            mag_d2 = fieldmodule.createFieldMagnitude(d2)
            min_mag_d2, max_mag_d2 = evaluateFieldNodesetRange(mag_d2, nodes)

            one = fieldmodule.createFieldConstant(1.0)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(min_mag_d2, 0.38251009640445927, delta=X_TOL)
            self.assertAlmostEqual(max_mag_d2, 0.3828163259239028, delta=X_TOL)
            self.assertAlmostEqual(surfaceArea, 28.820047346225028, delta=X_TOL)

    def test_3d_tube_network_bifurcation(self):
        """
        Test bifurcation 3-D tube network is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Bifurcation")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(1, settings["Number of elements through shell"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        self.assertFalse(settings["Use linear through shell"])
        self.assertTrue(settings["Use outer trim surfaces"])
        self.assertFalse(settings["Show trim surfaces"])
        self.assertFalse(settings["Core"])
        self.assertEqual(2, settings["Number of elements across core box minor"])
        self.assertEqual(1, settings["Number of elements across core transition"])
        self.assertEqual([0], settings["Annotation numbers of elements across core box minor"])
        self.assertFalse(settings["Refine"])
        self.assertEqual(4, settings["Refine number of elements"])
        self.assertEqual(2, settings["Refine number of elements through shell"])

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()

        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(8 * 4 * 3, mesh3d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual((8 * 4 * 3 + 3 * 3 + 2) * 2, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [0.0, -0.5894427190999916, -0.10000000000000002], X_TOL)
        assertAlmostEqualList(self, maximums, [2.044721359549996, 0.5894427190999916, 0.10000000000000002], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            isExteriorXi3_0 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0))
            isExteriorXi3_1 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            outerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_1, coordinates, mesh2d)
            outerSurfaceAreaField.setNumbersOfPoints(4)
            result, outerSurfaceArea = outerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            innerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_0, coordinates, mesh2d)
            innerSurfaceAreaField.setNumbersOfPoints(4)
            result, innerSurfaceArea = innerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.03494530171207773, delta=X_TOL)
            self.assertAlmostEqual(outerSurfaceArea, 1.9287678191612518, delta=X_TOL)
            self.assertAlmostEqual(innerSurfaceArea, 1.559753598132764, delta=X_TOL)

    def test_3d_tube_network_bifurcation_core(self):
        """
        Test bifurcation 3-D tube network with solid core is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Bifurcation")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(1, settings["Number of elements through shell"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        self.assertFalse(settings["Use linear through shell"])
        self.assertFalse(settings["Show trim surfaces"])
        self.assertFalse(settings["Core"])
        self.assertEqual(2, settings["Number of elements across core box minor"])
        self.assertEqual(1, settings["Number of elements across core transition"])
        self.assertEqual([0], settings["Annotation numbers of elements across core box minor"])
        settings["Core"] = True

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)

        self.assertEqual((8 * 4 * 3) * 2 + (4 * 4 * 3), mesh3d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual((8 * 4 * 3 + 3 * 3 + 2) * 2 + (9 * 4 * 3 + 3 * 4), nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [0.0, -0.5894427190999916, -0.10000000000000002], X_TOL)
        assertAlmostEqualList(self, maximums, [2.044721359549996, 0.5894427190999916, 0.10000000000000002], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.09887558242149766, delta=X_TOL)
            self.assertAlmostEqual(surfaceArea, 2.0229136766511058, delta=X_TOL)

    def test_3d_tube_network_converging_bifurcation_core(self):
        """
        Test bifurcation 3-D tube network with solid core and 12, 12, 8 elements around.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Bifurcation")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(1, settings["Number of elements through shell"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        self.assertFalse(settings["Use linear through shell"])
        self.assertFalse(settings["Show trim surfaces"])
        self.assertFalse(settings["Core"])
        self.assertEqual(2, settings["Number of elements across core box minor"])
        self.assertEqual(1, settings["Number of elements across core transition"])
        self.assertEqual([0], settings["Annotation numbers of elements across core box minor"])
        settings["Core"] = True
        settings["Number of elements around"] = 12
        settings["Annotation numbers of elements around"] = [8]

        context = Context("Test")
        region = context.getDefaultRegion()

        # add a user-defined annotation group to network layout to vary elements count around. Must generate first
        tmpRegion = region.createRegion()
        tmpFieldmodule = tmpRegion.getFieldmodule()
        networkLayoutScaffoldPackage.generate(tmpRegion)

        annotationGroup1 = networkLayoutScaffoldPackage.createUserAnnotationGroup(("segment 3", "SEGMENT:3"))
        group = annotationGroup1.getGroup()
        mesh1d = tmpFieldmodule.findMeshByDimension(1)
        meshGroup = group.createMeshGroup(mesh1d)
        mesh_group_add_identifier_ranges(meshGroup, [[3, 3]])
        self.assertEqual(1, meshGroup.getSize())
        self.assertEqual(1, annotationGroup1.getDimension())
        identifier_ranges_string = identifier_ranges_to_string(mesh_group_to_identifier_ranges(meshGroup))
        self.assertEqual("3", identifier_ranges_string)
        networkLayoutScaffoldPackage.updateUserAnnotationGroups()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(3, len(annotationGroups))
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "core") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "shell") is not None)

        fieldmodule = region.getFieldmodule()

        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(336, mesh3d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(460, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        # check annotation group transferred to 3D tube
        annotationGroup = findAnnotationGroupByName(annotationGroups, "segment 3")
        self.assertTrue(annotationGroup is not None)
        self.assertEqual("SEGMENT:3", annotationGroup.getId())
        self.assertEqual(80, annotationGroup.getMeshGroup(fieldmodule.findMeshByDimension(3)).getSize())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [0.0, -0.5894427190999916, -0.10000000000000002], X_TOL)
        assertAlmostEqualList(self, maximums, [2.044721359549996, 0.5894427190999916, 0.10000000000000002], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.09854389580373706, delta=X_TOL)
            self.assertAlmostEqual(surfaceArea, 2.0220606390309976, delta=X_TOL)

    def test_3d_tube_network_line_core_transition2(self):
        """
        Test line 3-D tube network with solid core and 2 transition elements.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Default")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(1, settings["Number of elements through shell"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        self.assertFalse(settings["Use linear through shell"])
        self.assertFalse(settings["Show trim surfaces"])
        self.assertFalse(settings["Core"])
        self.assertEqual(2, settings["Number of elements across core box minor"])
        self.assertEqual(1, settings["Number of elements across core transition"])
        self.assertEqual([0], settings["Annotation numbers of elements across core box minor"])
        settings["Core"] = True
        settings["Number of elements across core transition"] = 2

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)

        self.assertEqual(112, mesh3d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(165, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [0.0, -0.1, -0.1], X_TOL)
        assertAlmostEqualList(self, maximums, [1.0, 0.1, 0.1], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.03138195249662126, delta=X_TOL)
            self.assertAlmostEqual(surfaceArea, 0.6907451706120391, delta=X_TOL)

    def test_3d_tube_network_bone_core_12around_4along(self):
        """
        Test bone 3-D tube network with solid core is generated correctly with 4 along.
        This tests the case where the 4-way cap layer of the domes join the junction.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Bone")
        settings = scaffoldPackage.getScaffoldSettings()
        settings["Number of elements around"] = 12
        settings["Target element density along longest segment"] = 4.0
        settings["Core"] = True

        context = Context("Test")
        region = context.getDefaultRegion()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(2, len(annotationGroups))
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "core") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "shell") is not None)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(320, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(1024, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(1110, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(407, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.6593730298094371, -0.8660254037844386, -0.5], X_TOL)
        assertAlmostEqualList(self, maximums, [4.659373029509067, 0.8660254037844387, 0.5], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, total_volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            # make groups for each half
            cmiss_number = fieldmodule.findFieldByName("cmiss_number")
            self.assertTrue(cmiss_number.isValid())
            middle_element_number = fieldmodule.createFieldConstant(160.5)
            half1 = scaffoldPackage.createUserAnnotationGroup(("half1", ""))
            half1_mesh3d = half1.getMeshGroup(mesh3d)
            half1_mesh3d.addElementsConditional(fieldmodule.createFieldLessThan(cmiss_number, middle_element_number))
            half2 = scaffoldPackage.createUserAnnotationGroup(("half2", ""))
            half2_mesh3d = half2.getMeshGroup(mesh3d)
            half2_mesh3d.addElementsConditional(fieldmodule.createFieldGreaterThan(cmiss_number, middle_element_number))

        expected_core_volume = 2.9639678526918485
        expected_shell_volume = 1.733506855128808
        expected_total_volume = expected_core_volume + expected_shell_volume  # 4.667658800439896
        expected_total_surface_area = 19.86510683832899

        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(4, len(annotationGroups))

        expectedSizes3d = {
            "core": (192, expected_core_volume),
            "shell": (128, expected_shell_volume),
            "half1": (160, 0.5 * expected_total_volume),
            "half2": (160, 0.5 * expected_total_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes3d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=X_TOL)

        self.assertAlmostEqual(total_volume, expected_total_volume, delta=X_TOL)
        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

        # check derivatives around centre of ends are of the expected size
        expected_d1 = [0.0, 0.0, 0.28745150226471494]
        for node_identifier in [73, 352]:
            node = nodes.findNodeByIdentifier(node_identifier)
            fieldcache.setNode(node)
            result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
            self.assertEqual(result, RESULT_OK)
            assertAlmostEqualList(self, d1, expected_d1, delta=X_TOL)

    def test_3d_tube_network_bone_core_8around_8along_1shell(self):
        """
        Test bone 3-D tube network with solid core is generated correctly with 8 along, 1 shell element.
        This tests the case where regular layers of the domes join the junction.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Bone")
        settings = scaffoldPackage.getScaffoldSettings()
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(8.0, settings["Target element density along longest segment"])
        self.assertEqual(1, settings["Number of elements through shell"])
        settings["Core"] = True

        context = Context("Test")
        region = context.getDefaultRegion()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(2, len(annotationGroups))
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "core") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "shell") is not None)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(352, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(1128, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(1226, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(451, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.6612836030569434, -0.8660254037844386, -0.5], X_TOL)
        assertAlmostEqualList(self, maximums, [4.661283603014371, 0.8660254037844387, 0.5], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, total_volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            # make groups for each half
            cmiss_number = fieldmodule.findFieldByName("cmiss_number")
            self.assertTrue(cmiss_number.isValid())
            middle_element_number = fieldmodule.createFieldConstant(176.5)
            half1 = scaffoldPackage.createUserAnnotationGroup(("half1", ""))
            half1_mesh3d = half1.getMeshGroup(mesh3d)
            half1_mesh3d.addElementsConditional(fieldmodule.createFieldLessThan(cmiss_number, middle_element_number))
            half2 = scaffoldPackage.createUserAnnotationGroup(("half2", ""))
            half2_mesh3d = half2.getMeshGroup(mesh3d)
            half2_mesh3d.addElementsConditional(fieldmodule.createFieldGreaterThan(cmiss_number, middle_element_number))

            core = findAnnotationGroupByName(annotationGroups, "core")
            half1_core = scaffoldPackage.createUserAnnotationGroup(("half1_core", ""))
            half1_core_mesh3d = half1_core.getMeshGroup(mesh3d)
            half1_core_mesh3d.addElementsConditional(fieldmodule.createFieldAnd(half1.getGroup(), core.getGroup()))
            half2_core = scaffoldPackage.createUserAnnotationGroup(("half2_core", ""))
            half2_core_mesh3d = half2_core.getMeshGroup(mesh3d)
            half2_core_mesh3d.addElementsConditional(fieldmodule.createFieldAnd(half2.getGroup(), core.getGroup()))

            shell = findAnnotationGroupByName(annotationGroups, "shell")
            half1_shell = scaffoldPackage.createUserAnnotationGroup(("half1_shell", ""))
            half1_shell_mesh3d = half1_shell.getMeshGroup(mesh3d)
            half1_shell_mesh3d.addElementsConditional(fieldmodule.createFieldAnd(half1.getGroup(), shell.getGroup()))
            half2_shell = scaffoldPackage.createUserAnnotationGroup(("half2_shell", ""))
            half2_shell_mesh3d = half2_shell.getMeshGroup(mesh3d)
            half2_shell_mesh3d.addElementsConditional(fieldmodule.createFieldAnd(half2.getGroup(), shell.getGroup()))

        expected_core_volume = 2.93335897647307
        expected_shell_volume = 1.7346580626984778
        expected_total_volume = expected_core_volume + expected_shell_volume  # 4.667658696690623
        expected_total_surface_area = 19.860657310573725

        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(8, len(annotationGroups))

        expectedSizes3d = {
            "core": (208, expected_core_volume),
            "shell": (144, expected_shell_volume),
            "half1": (176, 0.5 * expected_total_volume),
            "half2": (176, 0.5 * expected_total_volume),
            "half1_core": (104, 0.5 * expected_core_volume),
            "half2_core": (104, 0.5 * expected_core_volume),
            "half1_shell": (72, 0.5 * expected_shell_volume),
            "half2_shell": (72, 0.5 * expected_shell_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes3d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=X_TOL)

        self.assertAlmostEqual(total_volume, expected_total_volume, delta=X_TOL)
        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

        # check symmetry of 6-way points between halves
        expected_6way_x = [0.0019108636038874666, 1.4486745136821355e-12, 0.49742413411115977]
        expected_6way_d1 = [0.21323449908997336, -0.36687564649608395, 0.006704466598067782]
        expected_6way_d2 = [0.2132344990888123, 0.36687564649664645, 0.006704466604210613]
        expected_6way_d3 = [-0.015227869859240711, 1.3415194880887309e-15, 0.09462613958074884]
        for node_identifier in (72, 333):
            node = nodes.findNodeByIdentifier(node_identifier)
            fieldcache.setNode(node)
            if node_identifier == 333:
                expected_6way_x[0] = 4.0 - expected_6way_x[0]
                expected_6way_d1[2] = -expected_6way_d1[2]
                expected_6way_d2[2] = -expected_6way_d2[2]
                expected_6way_d3[0] = -expected_6way_d3[0]
            result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
            self.assertEqual(result, RESULT_OK)
            assertAlmostEqualList(self, x, expected_6way_x, delta=X_TOL)
            result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
            self.assertEqual(result, RESULT_OK)
            assertAlmostEqualList(self, d1, expected_6way_d1, delta=X_TOL)
            result, d2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
            self.assertEqual(result, RESULT_OK)
            assertAlmostEqualList(self, d2, expected_6way_d2, delta=X_TOL)
            result, d3 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)
            self.assertEqual(result, RESULT_OK)
            assertAlmostEqualList(self, d3, expected_6way_d3, delta=X_TOL)

    def test_3d_tube_network_bone_core_8around_8along_0shell(self):
        """
        Test bone 3-D tube network with solid core is generated correctly with 8 along, 0 shell elements.
        This tests the case where regular layers of the domes join the junction.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Bone")
        settings = scaffoldPackage.getScaffoldSettings()
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(8.0, settings["Target element density along longest segment"])
        settings["Core"] = True
        settings["Number of elements through shell"] = 0

        context = Context("Test")
        region = context.getDefaultRegion()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(2, len(annotationGroups))
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "core") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "shell") is not None)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(208, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(696, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(792, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(305, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.6612836030569434, -0.8660254037844386, -0.5], X_TOL)
        assertAlmostEqualList(self, maximums, [4.661283603014371, 0.8660254037844387, 0.5], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, total_volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            # make groups for each half
            cmiss_number = fieldmodule.findFieldByName("cmiss_number")
            self.assertTrue(cmiss_number.isValid())
            middle_element_number = fieldmodule.createFieldConstant(104.5)
            half1 = scaffoldPackage.createUserAnnotationGroup(("half1", ""))
            half1_mesh3d = half1.getMeshGroup(mesh3d)
            half1_mesh3d.addElementsConditional(fieldmodule.createFieldLessThan(cmiss_number, middle_element_number))
            half2 = scaffoldPackage.createUserAnnotationGroup(("half2", ""))
            half2_mesh3d = half2.getMeshGroup(mesh3d)
            half2_mesh3d.addElementsConditional(fieldmodule.createFieldGreaterThan(cmiss_number, middle_element_number))
            shell = findAnnotationGroupByName(annotationGroups, "shell")

        expected_total_volume = 4.668017185448212
        expected_total_surface_area = 19.860657310573725

        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(4, len(annotationGroups))

        expectedSizes3d = {
            "core": (208, expected_total_volume),
            "half1": (104, 0.5 * expected_total_volume),
            "half2": (104, 0.5 * expected_total_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes3d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=X_TOL)

        expectedSizes2d = {
            "shell": (144, expected_total_surface_area)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes2d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name][0], size, name)
            surfaceAreaMeshGroup = annotationGroup.getMeshGroup(mesh2d)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, surfaceAreaMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, expectedSizes2d[name][1], delta=X_TOL)

        self.assertAlmostEqual(total_volume, expected_total_volume, delta=X_TOL)
        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

    def test_2d_tube_network_bone_8around_8along_0shell(self):
        """
        Test bone 3-D tube network with solid core is generated correctly with 8 along, 0 shell elements.
        This tests the case where regular layers of the domes join the junction.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Bone")
        settings = scaffoldPackage.getScaffoldSettings()
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(8.0, settings["Target element density along longest segment"])
        settings["Core"] = False
        settings["Number of elements through shell"] = 0

        context = Context("Test")
        region = context.getDefaultRegion()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(0, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(144, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(288, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(146, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.6612836030569434, -0.8660254037844386, -0.5], X_TOL)
        assertAlmostEqualList(self, maximums, [4.661283603014371, 0.8660254037844387, 0.5], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            # make groups for each half
            cmiss_number = fieldmodule.findFieldByName("cmiss_number")
            self.assertTrue(cmiss_number.isValid())
            middle_element_number = fieldmodule.createFieldConstant(72.5)
            half1 = scaffoldPackage.createUserAnnotationGroup(("half1", ""))
            half1_mesh2d = half1.getMeshGroup(mesh2d)
            half1_mesh2d.addElementsConditional(fieldmodule.createFieldLessThan(cmiss_number, middle_element_number))
            half2 = scaffoldPackage.createUserAnnotationGroup(("half2", ""))
            half2_mesh2d = half2.getMeshGroup(mesh2d)
            half2_mesh2d.addElementsConditional(fieldmodule.createFieldGreaterThan(cmiss_number, middle_element_number))

        expected_total_surface_area = 19.860657310573725

        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(2, len(annotationGroups))

        expectedSizes2d = {
            "half1": (72, 0.5 * expected_total_surface_area),
            "half2": (72, 0.5 * expected_total_surface_area)
            }
        for name in expectedSizes2d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name][0], size, name)
            surfaceAreaMeshGroup = annotationGroup.getMeshGroup(mesh2d)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, surfaceAreaMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, expectedSizes2d[name][1], delta=X_TOL)

        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

        # refine 2x2 and check
        # first remove faces/lines and any surface annotation groups as they are re-added by defineFaceAnnotations
        removeAnnotationGroups = []
        for annotationGroup in annotationGroups:
            if annotationGroup.getDimension() == 1:
                removeAnnotationGroups.append(annotationGroup)
        for annotationGroup in removeAnnotationGroups:
            annotationGroups.remove(annotationGroup)
        self.assertEqual(2, len(annotationGroups))
        # must keep all faces and lines as used for refinement

        refineRegion = region.createRegion()
        refineFieldmodule = refineRegion.getFieldmodule()
        settings['Refine number of elements'] = 2
        settings['Refine number of elements through shell'] = 1
        meshrefinement = MeshRefinement(region, refineRegion, annotationGroups)
        MeshType_3d_tubenetwork1.refineMesh(meshrefinement, settings)
        annotationGroups = meshrefinement.getAnnotationGroups()
        coordinates = refineFieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        refineFieldmodule.defineAllFaces()
        self.assertEqual(2, len(annotationGroups))

        mesh3d = refineFieldmodule.findMeshByDimension(3)
        self.assertEqual(0, mesh3d.getSize())
        mesh2d = refineFieldmodule.findMeshByDimension(2)
        self.assertEqual(576, mesh2d.getSize())
        mesh1d = refineFieldmodule.findMeshByDimension(1)
        self.assertEqual(1152, mesh1d.getSize())
        nodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(578, nodes.getSize())

        with ChangeManager(refineFieldmodule):
            one = refineFieldmodule.createFieldConstant(1.0)
            mesh2d = refineFieldmodule.findMeshByDimension(2)
            fieldcache = refineFieldmodule.createFieldcache()

            surfaceAreaField = refineFieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            # check no exterior lines
            exterior_group = find_or_create_field_group(refineFieldmodule, "exterior")
            exterior_lines = exterior_group.createMeshGroup(mesh1d)
            exterior_lines.addElementsConditional(refineFieldmodule.createFieldIsExterior())
            self.assertEqual(0, exterior_lines.getSize())
            del exterior_lines
            del exterior_group

        expected_total_surface_area = 19.63988770848942
        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

        refineExpectedSizes2d = {
            "half1": (288, 0.5 * expected_total_surface_area),
            "half2": (288, 0.5 * expected_total_surface_area)
            }
        for name in expectedSizes2d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh2d).getSize()
            self.assertEqual(refineExpectedSizes2d[name][0], size, name)
            surfaceAreaMeshGroup = annotationGroup.getMeshGroup(mesh2d)
            surfaceAreaField = refineFieldmodule.createFieldMeshIntegral(one, coordinates, surfaceAreaMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, refineExpectedSizes2d[name][1], delta=X_TOL)

    def test_3d_tube_network_bone_8around_8along_1shell_linear(self):
        """
        Test bone 3-D tube network with no core and 1 linear shell element is generated correctly with 8 along.
        This tests the case where regular layers of the domes join the junction.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Bone")
        settings = scaffoldPackage.getScaffoldSettings()
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(8.0, settings["Target element density along longest segment"])
        self.assertEqual(1, settings["Number of elements through shell"])
        settings["Core"] = False
        settings["Use linear through shell"] = True

        context = Context("Test")
        region = context.getDefaultRegion()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(144, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(576, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(722, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(292, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.6612836030569434, -0.8660254037844386, -0.5], X_TOL)
        assertAlmostEqualList(self, maximums, [4.661283603014371, 0.8660254037844387, 0.5], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, total_volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            # make groups for each half
            cmiss_number = fieldmodule.findFieldByName("cmiss_number")
            self.assertTrue(cmiss_number.isValid())
            middle_element_number = fieldmodule.createFieldConstant(72.5)
            half1 = scaffoldPackage.createUserAnnotationGroup(("half1", ""))
            half1_mesh3d = half1.getMeshGroup(mesh3d)
            half1_mesh3d.addElementsConditional(fieldmodule.createFieldLessThan(cmiss_number, middle_element_number))
            half2 = scaffoldPackage.createUserAnnotationGroup(("half2", ""))
            half2_mesh3d = half2.getMeshGroup(mesh3d)
            half2_mesh3d.addElementsConditional(fieldmodule.createFieldGreaterThan(cmiss_number, middle_element_number))

        expected_total_volume = 1.7346580626984778
        expected_total_surface_area = 35.31423992969953

        self.assertAlmostEqual(total_volume, expected_total_volume, delta=X_TOL)
        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(2, len(annotationGroups))

        expectedSizes3d = {
            "half1": (72, 0.5 * expected_total_volume),
            "half2": (72, 0.5 * expected_total_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes3d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=X_TOL)

        # check symmetry of 6-way points between halves
        expected_6way_x = [0.0019108636038874666, 1.4486745136821355e-12, 0.49742413411115977]
        expected_6way_d1 = [0.21323449908997336, -0.36687564649608395, 0.006704466598067782]
        expected_6way_d2 = [0.2132344990888123, 0.36687564649664645, 0.006704466604210613]
        for node_identifier in (45, 213):
            node = nodes.findNodeByIdentifier(node_identifier)
            fieldcache.setNode(node)
            if node_identifier == 213:
                expected_6way_x[0] = 4.0 - expected_6way_x[0]
                expected_6way_d1[2] = -expected_6way_d1[2]
                expected_6way_d2[2] = -expected_6way_d2[2]
            result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
            self.assertEqual(result, RESULT_OK)
            assertAlmostEqualList(self, x, expected_6way_x, delta=X_TOL)
            result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
            self.assertEqual(result, RESULT_OK)
            assertAlmostEqualList(self, d1, expected_6way_d1, delta=X_TOL)
            result, d2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
            self.assertEqual(result, RESULT_OK)
            assertAlmostEqualList(self, d2, expected_6way_d2, delta=X_TOL)
            result = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)[0]
            self.assertEqual(result, RESULT_ERROR_NOT_FOUND)

    def test_3d_tube_network_sphere_core_1shell(self):
        """
        Test sphere 3-D tube network with solid core is generated correctly, 1 shell element.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Sphere")
        settings = scaffoldPackage.getScaffoldSettings()
        self.assertEqual(1, settings["Number of elements through shell"])
        settings["Core"] = True

        context = Context("Test")
        region = context.getDefaultRegion()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(2, len(annotationGroups))
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "core") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "shell") is not None)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(256, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(816, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(880, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(321, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-1.0, -1.0, -1.0], X_TOL)
        assertAlmostEqualList(self, maximums, [1.0, 1.0, 1.0], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, total_volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

        # perfect volume would be 4.0 / 3.0 * math.pi = 4.1887902047863909846168578443727
        expected_core_volume = 2.1422349725140952
        expected_shell_volume = 2.041817728215633
        expected_total_volume = expected_core_volume + expected_shell_volume  # 4.1840527007297282
        # perfect surface area would be 4.0 * math.pi = 12.566370614359172953850573533118
        expected_total_surface_area = 12.5569662702283

        expectedSizes3d = {
            "core": (160, expected_core_volume),
            "shell": (96, expected_shell_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes3d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=X_TOL)

        self.assertAlmostEqual(total_volume, expected_total_volume, delta=X_TOL)
        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

        # check nodes have symmetric d1, d3 magnitudes along axis3
        for node_identifier in [63, 104, 161, 218, 259]:
            node = nodes.findNodeByIdentifier(node_identifier)
            fieldcache.setNode(node)
            result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
            self.assertEqual(result, RESULT_OK)
            result, d3 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(magnitude(d1), magnitude(d3), delta=X_TOL)

    def test_3d_tube_network_sphere_core_0shell(self):
        """
        Test sphere 3-D tube network with solid core and no shell is generated correctly with 0 shell elements.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Sphere")
        settings = scaffoldPackage.getScaffoldSettings()
        settings["Core"] = True
        settings["Number of elements through shell"] = 0

        context = Context("Test")
        region = context.getDefaultRegion()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(2, len(annotationGroups))
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "core") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "shell") is not None)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(160, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(528, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(590, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(223, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-1.0, -1.0, -1.0], X_TOL)
        assertAlmostEqualList(self, maximums, [1.0, 1.0, 1.0], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, total_volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

        # perfect volume would be 4.0 / 3.0 * math.pi = 4.1887902047863909846168578443727
        expected_total_volume = 4.1840527007297282
        # perfect surface area would be 4.0 * math.pi = 12.566370614359172953850573533118
        expected_total_surface_area = 12.5569662702283

        expectedSizes3d = {
            "core": (160, expected_total_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes3d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=X_TOL)

        expectedSizes2d = {
            "shell": (96, expected_total_surface_area)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes2d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name][0], size, name)
            surfaceAreaMeshGroup = annotationGroup.getMeshGroup(mesh2d)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, surfaceAreaMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, expectedSizes2d[name][1], delta=X_TOL)

        self.assertAlmostEqual(total_volume, expected_total_volume, delta=X_TOL)
        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

    def test_3d_tube_network_sphere_8around_2along_core(self):
        """
        Test sphere 3-D tube network with solid core is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Sphere")
        settings = scaffoldPackage.getScaffoldSettings()
        settings["Number of elements around"] = 8
        settings["Target element density along longest segment"] = 2.0
        settings["Number of elements across core box minor"] = 2
        settings["Core"] = True

        context = Context("Test")
        region = context.getDefaultRegion()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(2, len(annotationGroups))
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "core") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "shell") is not None)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(56, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(180, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(202, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(79, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-1.0, -1.0, -1.0], X_TOL)
        assertAlmostEqualList(self, maximums, [1.0, 1.0, 1.0], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, total_volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

        # perfect volume would be 4.0 / 3.0 * math.pi = 4.1887902047863909846168578443727
        expected_core_volume = 2.1224259944920227
        expected_shell_volume = 2.0229377082224276
        expected_total_volume = expected_core_volume + expected_shell_volume  # 4.1453637027144503
        # perfect surface area would be 4.0 * math.pi = 12.566370614359172953850573533118
        expected_total_surface_area = 12.482842286611568

        expectedSizes3d = {
            "core": (32, expected_core_volume),
            "shell": (24, expected_shell_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes3d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=X_TOL)

        self.assertAlmostEqual(total_volume, expected_total_volume, delta=X_TOL)
        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

        # check nodes have symmetric d1, d3 magnitudes along axis3
        for node_identifier in [23, 40, 57]:
            node = nodes.findNodeByIdentifier(node_identifier)
            fieldcache.setNode(node)
            result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
            self.assertEqual(result, RESULT_OK)
            result, d3 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(magnitude(d1), magnitude(d3), delta=X_TOL)

    def test_3d_tube_network_line_twist_core(self):
        """
        Test line twist 3-D tube network with solid core is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Line twist")
        settings = scaffoldPackage.getScaffoldSettings()
        settings["Number of elements around"] = 8
        settings["Target element density along longest segment"] = 4.0
        settings["Core"] = True
        settings["Number of elements across core box minor"] = 2
        settings["Number of elements across core transition"] = 2

        context = Context("Test")
        region = context.getDefaultRegion()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(2, len(annotationGroups))
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "core") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "shell") is not None)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(112, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(380, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(432, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(165, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-8

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [0.0, -0.09982782069268835, -0.09982782069268835], X_TOL)
        assertAlmostEqualList(self, maximums, [1.0, 0.09982782069268835, 0.09982782069268835], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, total_volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, total_surface_area = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

        expected_core_volume = 0.020084449597837638
        expected_shell_volume = 0.011235647827508903
        expected_total_volume = expected_core_volume + expected_shell_volume
        expected_total_surface_area = 0.6901264247417831

        expectedSizes3d = {
            "core": (80, expected_core_volume),
            "shell": (32, expected_shell_volume)
            }
        fieldcache = fieldmodule.createFieldcache()
        for name in expectedSizes3d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=X_TOL)

        self.assertAlmostEqual(total_volume, expected_total_volume, delta=X_TOL)
        self.assertAlmostEqual(total_surface_area, expected_total_surface_area, delta=X_TOL)

        # check twist of centre nodes along model
        expected_x = [
            [0.0, 1.734723475976807e-18, -1.0408340855860843e-17],
            [0.25001128145481993, 1.214306433183765e-17, -5.204170427930421e-18],
            [0.5, 2.0816681711721685e-17, -6.938893903907228e-18],
            [0.74998871854518, 8.673617379884035e-18, -7.806255641895632e-18],
            [1.0, -6.938893903907228e-18, -1.214306433183765e-17]]
        expected_d1 = [
            [0.0, -0.025350651258193597, -0.014614550557958546],
            [0.0, -0.025122103521202532, -0.012181503879516015],
            [0.0, -0.026666666666666675, -1.687085467822282e-17],
            [0.0, -0.025122103521202546, 0.012181503879515982],
            [0.0, -0.0253506512581936, 0.014614550557958524]]
        expected_d2 = [
            [0.25002963869709516, -2.6020852139652106e-14, 5.204170427930421e-14],
            [0.24999646305423617, -8.673617379884035e-15, -3.469446951953614e-14],
            [0.24998450968705122, -1.734723475976807e-14, 6.938893903907228e-14],
            [0.24999646305756684, -1.734723475976807e-14, 8.673617379884035e-15],
            [0.25002963869757977, 1.734723475976807e-14, 4.336808689942018e-14]]
        expected_d3 = [
            [0.0, -0.014614550557958551, 0.025350651258193593],
            [0.0, -0.012181503879516027, 0.025122103521202543],
            [0.0, 2.2251485348737508e-17, 0.026666666666666672],
            [0.0, 0.012181503879515979, 0.025122103521202557],
            [0.0, 0.014614550557958522, 0.02535065125819361]]
        for n in range(5):
            xi = n / 4.0
            node_identifier = 5 + n * 33
            node = nodes.findNodeByIdentifier(node_identifier)
            fieldcache.setNode(node)
            result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
            self.assertEqual(result, RESULT_OK)
            result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
            self.assertEqual(result, RESULT_OK)
            result, d2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
            self.assertEqual(result, RESULT_OK)
            result, d3 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)
            self.assertEqual(result, RESULT_OK)
            assertAlmostEqualList(self, x, expected_x[n], X_TOL)
            assertAlmostEqualList(self, d1, expected_d1[n], X_TOL)
            assertAlmostEqualList(self, d2, expected_d2[n], X_TOL)
            assertAlmostEqualList(self, d3, expected_d3[n], X_TOL)

    def test_3d_tube_network_sphere_cube(self):
        """
        Test sphere cube 3-D tube network is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Sphere cube")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(1, settings["Number of elements through shell"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        self.assertFalse(settings["Use linear through shell"])
        self.assertFalse(settings["Show trim surfaces"])
        settings["Number of elements through shell"] = 2
        settings["Use linear through shell"] = True

        context = Context("Test")
        region = context.getDefaultRegion()

        # set custom inner coordinates
        tmpRegion = region.createRegion()
        networkLayoutScaffoldPackage.generate(tmpRegion)
        networkMesh = networkLayoutScaffoldPackage.getConstructionObject()
        functionOptions = {
            "To field": {"coordinates": False, "inner coordinates": True},
            "From field": {"coordinates": True, "inner coordinates": False},
            "Mode": {"Scale": True, "Offset": False},
            "D2 value": 0.8,
            "D3 value": 0.8}
        editGroupName = "meshEdits"
        MeshType_1d_network_layout1.assignCoordinates(tmpRegion, networkLayoutSettings, networkMesh,
                                                      functionOptions, editGroupName=editGroupName)
        # put edited coordinates into scaffold package
        sir = tmpRegion.createStreaminformationRegion()
        srm = sir.createStreamresourceMemory()
        sir.setResourceGroupName(srm, editGroupName)
        sir.setResourceFieldNames(srm, ["coordinates", "inner coordinates"])
        tmpRegion.write(sir)
        result, meshEditsString = srm.getBuffer()
        self.assertEqual(RESULT_OK, result)
        networkLayoutScaffoldPackage.setMeshEdits(meshEditsString)

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(32 * 12 * 2, mesh3d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(8 * 3 * 12 * 3 + (2 + 3 * 3) * 8 * 3, nodes.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(32 * 12 * 5 + 24 * 12 * 2 + 12 * 8 * 2, mesh2d.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.5664486520285047, -0.5965021612011158, -0.5986899625064468], X_TOL)
        assertAlmostEqualList(self, maximums, [0.5664486520285046, 0.5965021612011158, 0.5986899625064467], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            isExteriorXi3_0 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0))
            isExteriorXi3_1 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            outerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_1, coordinates, mesh2d)
            outerSurfaceAreaField.setNumbersOfPoints(4)
            result, outerSurfaceArea = outerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            innerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_0, coordinates, mesh2d)
            innerSurfaceAreaField.setNumbersOfPoints(4)
            result, innerSurfaceArea = innerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.07337971310984492, delta=X_TOL)
            self.assertAlmostEqual(outerSurfaceArea, 4.041234792437032, delta=X_TOL)
            self.assertAlmostEqual(innerSurfaceArea, 3.3266361037274113, delta=X_TOL)

    def test_3d_tube_network_sphere_cube_core(self):
        """
        Test sphere cube 3-D tube network with solid core is generated correctly.
        Use different number of elements around on some segments to mix it up.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Sphere cube")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(1, settings["Number of elements through shell"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        self.assertFalse(settings["Use linear through shell"])
        self.assertFalse(settings["Show trim surfaces"])
        self.assertFalse(settings["Core"])
        self.assertEqual(2, settings["Number of elements across core box minor"])
        self.assertEqual(1, settings["Number of elements across core transition"])
        self.assertEqual([0], settings["Annotation numbers of elements across core box minor"])
        settings["Number of elements through shell"] = 2
        settings["Annotation numbers of elements around"] = [12]
        settings["Core"] = True

        context = Context("Test")
        region = context.getDefaultRegion()

        # set custom inner coordinates
        tmpRegion = region.createRegion()
        networkLayoutScaffoldPackage.generate(tmpRegion)
        networkMesh = networkLayoutScaffoldPackage.getConstructionObject()
        functionOptions = {
            "To field": {"coordinates": False, "inner coordinates": True},
            "From field": {"coordinates": True, "inner coordinates": False},
            "Mode": {"Scale": True, "Offset": False},
            "D2 value": 0.75,
            "D3 value": 0.75}
        editGroupName = "meshEdits"
        MeshType_1d_network_layout1.assignCoordinates(tmpRegion, networkLayoutSettings, networkMesh,
                                                      functionOptions, editGroupName=editGroupName)
        # put edited coordinates into scaffold package
        sir = tmpRegion.createStreaminformationRegion()
        srm = sir.createStreamresourceMemory()
        sir.setResourceGroupName(srm, editGroupName)
        sir.setResourceFieldNames(srm, ["coordinates", "inner coordinates"])
        tmpRegion.write(sir)
        result, meshEditsString = srm.getBuffer()
        self.assertEqual(RESULT_OK, result)
        networkLayoutScaffoldPackage.setMeshEdits(meshEditsString)

        # add a user-defined annotation group to network layout to vary elements count around. Must generate first
        annotationGroup1 = networkLayoutScaffoldPackage.createUserAnnotationGroup(("group1", ""))
        group = annotationGroup1.getGroup()
        tmpFieldmodule = tmpRegion.getFieldmodule()
        mesh1d = tmpFieldmodule.findMeshByDimension(1)
        meshGroup = group.createMeshGroup(mesh1d)
        mesh_group_add_identifier_ranges(meshGroup, [[2, 2], [5, 5], [8, 8], [10, 10]])
        self.assertEqual(4, meshGroup.getSize())
        self.assertEqual(1, annotationGroup1.getDimension())
        identifier_ranges_string = identifier_ranges_to_string(mesh_group_to_identifier_ranges(meshGroup))
        self.assertEqual("2,5,8,10", identifier_ranges_string)
        networkLayoutScaffoldPackage.updateUserAnnotationGroups()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(3, len(annotationGroups))
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "core") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "shell") is not None)
        self.assertTrue(findAnnotationGroupByName(annotationGroups, "group1") is not None)

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(1600, mesh3d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(1836, nodes.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(5024, mesh2d.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.5762176957083269, -0.5965021612011158, -0.5909252304333917], X_TOL)
        assertAlmostEqualList(self, maximums, [0.5762176957083269, 0.5965021612011158, 0.5909252304333917], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            expected_core_volume = 0.1213019484974165
            expected_shell_volume = 0.08857474198624075
            expected_volume = expected_core_volume + expected_shell_volume
            self.assertAlmostEqual(volume, expected_volume, delta=X_TOL)
            self.assertAlmostEqual(surfaceArea, 4.033283436169648, delta=X_TOL)

        expectedSizes3d = {
            "core": (704, expected_core_volume),
            "shell": (896, expected_shell_volume)
            }
        for name in expectedSizes3d:
            annotationGroup = findAnnotationGroupByName(annotationGroups, name)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=X_TOL)

    def test_3d_tube_network_trifurcation_cross(self):
        """
        Test trifurcation cross 3-D tube network is generated correctly with variable elements count around.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Trifurcation cross")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(1, settings["Number of elements through shell"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        self.assertFalse(settings["Use linear through shell"])
        self.assertFalse(settings["Show trim surfaces"])
        settings["Annotation numbers of elements around"] = [10]  # requires annotation group below

        context = Context("Test")
        region = context.getDefaultRegion()

        # add a user-defined annotation group to network layout to vary elements count around. Must generate first
        tmpRegion = region.createRegion()
        tmpFieldmodule = tmpRegion.getFieldmodule()
        networkLayoutScaffoldPackage.generate(tmpRegion)

        annotationGroup1 = networkLayoutScaffoldPackage.createUserAnnotationGroup(("straight", "STRAIGHT:1"))
        group = annotationGroup1.getGroup()
        mesh1d = tmpFieldmodule.findMeshByDimension(1)
        meshGroup = group.createMeshGroup(mesh1d)
        mesh_group_add_identifier_ranges(meshGroup, [[1, 1], [4, 4]])
        self.assertEqual(2, meshGroup.getSize())
        self.assertEqual(1, annotationGroup1.getDimension())
        identifier_ranges_string = identifier_ranges_to_string(mesh_group_to_identifier_ranges(meshGroup))
        self.assertEqual("1,4", identifier_ranges_string)
        networkLayoutScaffoldPackage.updateUserAnnotationGroups()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(1, len(annotationGroups))

        fieldmodule = region.getFieldmodule()

        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(144, mesh3d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(320, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        # check annotation group transferred to 3D tube
        annotationGroup = annotationGroups[0]
        self.assertEqual("straight", annotationGroup.getName())
        self.assertEqual("STRAIGHT:1", annotationGroup.getId())
        self.assertEqual(80, annotationGroup.getMeshGroup(fieldmodule.findMeshByDimension(3)).getSize())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.0447213595499958, -0.5894427190999916, -0.1], X_TOL)
        assertAlmostEqualList(self, maximums, [2.044721359549996, 0.5894427190999916, 0.10000000000000002], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            isExteriorXi3_0 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0))
            isExteriorXi3_1 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            outerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_1, coordinates, mesh2d)
            outerSurfaceAreaField.setNumbersOfPoints(4)
            result, outerSurfaceArea = outerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            innerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_0, coordinates, mesh2d)
            innerSurfaceAreaField.setNumbersOfPoints(4)
            result, innerSurfaceArea = innerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.04718734252940824, delta=X_TOL)
            self.assertAlmostEqual(outerSurfaceArea, 2.5989406802438695, delta=X_TOL)
            self.assertAlmostEqual(innerSurfaceArea, 2.1140966889264514, delta=X_TOL)

    def test_3d_tube_network_trifurcation_cross_core(self):
        """
        Test trifurcation cross 3-D tube network with solid core is generated correctly with
        variable elements count around.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Trifurcation cross")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        self.assertTrue(networkLayoutSettings["Define inner coordinates"])
        self.assertEqual(16, len(settings))
        self.assertEqual(8, settings["Number of elements around"])
        self.assertEqual(1, settings["Number of elements through shell"])
        self.assertEqual([0], settings["Annotation numbers of elements around"])
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])
        self.assertFalse(settings["Use linear through shell"])
        self.assertFalse(settings["Show trim surfaces"])
        self.assertFalse(settings["Core"])
        self.assertEqual(2, settings["Number of elements across core box minor"])
        self.assertEqual(1, settings["Number of elements across core transition"])
        self.assertEqual([0], settings["Annotation numbers of elements across core box minor"])
        settings["Core"] = True
        settings["Annotation numbers of elements around"] = [12]  # requires annotation group below
        settings["Annotation numbers of elements across core box minor"] = [2]

        context = Context("Test")
        region = context.getDefaultRegion()

        # add a user-defined annotation group to network layout to vary elements count around. Must generate first
        tmpRegion = region.createRegion()
        tmpFieldmodule = tmpRegion.getFieldmodule()
        networkLayoutScaffoldPackage.generate(tmpRegion)

        annotationGroup1 = networkLayoutScaffoldPackage.createUserAnnotationGroup(("straight", "STRAIGHT:1"))
        group = annotationGroup1.getGroup()
        mesh1d = tmpFieldmodule.findMeshByDimension(1)
        meshGroup = group.createMeshGroup(mesh1d)
        mesh_group_add_identifier_ranges(meshGroup, [[1, 1], [4, 4]])
        self.assertEqual(2, meshGroup.getSize())
        self.assertEqual(1, annotationGroup1.getDimension())
        identifier_ranges_string = identifier_ranges_to_string(mesh_group_to_identifier_ranges(meshGroup))
        self.assertEqual("1,4", identifier_ranges_string)
        networkLayoutScaffoldPackage.updateUserAnnotationGroups()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(3, len(annotationGroups))

        fieldmodule = region.getFieldmodule()

        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(416, mesh3d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(569, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        # check annotation group transferred to 3D tube
        annotationGroup = findAnnotationGroupByName(annotationGroups, "straight")
        self.assertTrue(annotationGroup is not None)
        self.assertEqual("STRAIGHT:1", annotationGroup.getId())
        self.assertEqual(256, annotationGroup.getMeshGroup(fieldmodule.findMeshByDimension(3)).getSize())

        X_TOL = 1.0E-6

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.0447213595499958, -0.5894427190999916, -0.1], X_TOL)
        assertAlmostEqualList(self, maximums, [2.044721359549996, 0.5894427190999916, 0.10000000000000002], X_TOL)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.13463808777381098, delta=X_TOL)
            self.assertAlmostEqual(surfaceArea, 2.723442249479539, delta=X_TOL)

    def test_3d_box_network_bifurcation(self):
        """
        Test 3-D box network bifurcation is generated correctly.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_boxnetwork1, defaultParameterSetName="Bifurcation")
        settings = scaffoldPackage.getScaffoldSettings()
        self.assertEqual(3, len(settings))
        self.assertEqual(4.0, settings["Target element density along longest segment"])
        self.assertEqual([0], settings["Annotation numbers of elements along"])

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(12, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(63, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(108, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(13, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        X_TOL = 1.0E-6
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [0.0, -0.5, 0.0], X_TOL)
        assertAlmostEqualList(self, maximums, [2.0, 0.5, 0.0], X_TOL)

        L2 = math.sqrt(1.25)
        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(1)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            expectedVolume = 0.2 * 0.2 * (1.0 + 2 * L2)
            self.assertAlmostEqual(volume, expectedVolume, delta=X_TOL)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(1)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            expectedSurfaceArea = 6 * 0.2 * 0.2 + 4 * 0.2 * (1.0 + 2 * L2)
            self.assertAlmostEqual(surfaceArea, expectedSurfaceArea, delta=X_TOL)

    def test_3d_box_network_smooth(self):
        """
        Test 3-D box network derivative smoothing is working between segments sharing a version.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_boxnetwork1)
        settings = scaffoldPackage.getScaffoldSettings()
        settings["Target element density along longest segment"] = 1.0
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        networkLayoutSettings["Structure"] = "1-2-3,3-4"  # 2 unequal-sized segments

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(3, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        node2 = nodes.findNodeByIdentifier(2)
        self.assertTrue(node2.isValid())

        # test magnitude of d1 between segments is harmonic mean of element sizes
        fieldcache = fieldmodule.createFieldcache()
        fieldcache.setNode(node2)
        result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
        self.assertEqual(result, RESULT_OK)
        d1Mag = magnitude(d1)
        self.assertAlmostEqual(4.0 / 3.0, d1Mag, delta=1.0E-12)

    def test_3d_tube_network_loop(self):
        """
        Test loop 3-D tube network is generated correctly.
        This has one segment which loops back on itself so nodes are common at start and end.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Loop")
        settings = scaffoldPackage.getScaffoldSettings()
        settings["Target element density along longest segment"] = 8.0

        context = Context("Test")
        region = context.getDefaultRegion()
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(8 * 8, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(8 * 8 * 4, mesh2d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(8 * 8 * 2, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.6, -0.6, -0.1], 1.0E-8)
        assertAlmostEqualList(self, maximums, [0.6, 0.6, 0.1], 1.0E-8)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            isExteriorXi3_0 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0))
            isExteriorXi3_1 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            outerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_1, coordinates, mesh2d)
            outerSurfaceAreaField.setNumbersOfPoints(4)
            result, outerSurfaceArea = outerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            innerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_0, coordinates, mesh2d)
            innerSurfaceAreaField.setNumbersOfPoints(4)
            result, innerSurfaceArea = innerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.035365666033757015, delta=1.0E-6)
            self.assertAlmostEqual(outerSurfaceArea, 1.968898024252741, delta=1.0E-6)
            self.assertAlmostEqual(innerSurfaceArea, 1.5751177926763344, delta=1.0E-6)

    def test_3d_tube_network_loop_core(self):
        """
        Test loop 3-D tube network with solid core is generated correctly.
        This has one segment which loops back on itself so nodes are common at start and end.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Loop")
        settings = scaffoldPackage.getScaffoldSettings()
        settings["Target element density along longest segment"] = 8.0
        settings["Core"] = True

        context = Context("Test")
        region = context.getDefaultRegion()
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(160, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(512, mesh2d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(200, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.6, -0.6, -0.1], 1.0E-8)
        assertAlmostEqualList(self, maximums, [0.6, 0.6, 0.1], 1.0E-8)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.09823796120488085, delta=1.0E-6)
            self.assertAlmostEqual(surfaceArea, 1.968898024252741, delta=1.0E-6)

    def test_3d_tube_network_loop_two_segments(self):
        """
        Test loop 3-D tube network is generated with 2 segments with fixed element boundary between them.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_tubenetwork1, defaultParameterSetName="Loop")
        settings = scaffoldPackage.getScaffoldSettings()
        networkLayoutScaffoldPackage = settings["Network layout"]
        networkLayoutSettings = networkLayoutScaffoldPackage.getScaffoldSettings()
        # change structure to make two segments but use regular loop parameters:
        networkLayoutSettings["Structure"] = "1-2-3-4-5-6-7,7-8-1"
        settings["Target element density along longest segment"] = 7.0

        context = Context("Test")
        region = context.getDefaultRegion()

        # add a user-defined annotation group to network layout. Must generate first
        tmpRegion = region.createRegion()
        tmpFieldmodule = tmpRegion.getFieldmodule()
        networkLayoutScaffoldPackage.generate(tmpRegion)

        annotationGroup1 = networkLayoutScaffoldPackage.createUserAnnotationGroup(("bob", "BOB:1"))
        group = annotationGroup1.getGroup()
        mesh1d = tmpFieldmodule.findMeshByDimension(1)
        meshGroup = group.createMeshGroup(mesh1d)
        mesh_group_add_identifier_ranges(meshGroup, [[7, 8]])
        self.assertEqual(2, meshGroup.getSize())
        self.assertEqual(1, annotationGroup1.getDimension())
        identifier_ranges_string = identifier_ranges_to_string(mesh_group_to_identifier_ranges(meshGroup))
        self.assertEqual("7-8", identifier_ranges_string)
        networkLayoutScaffoldPackage.updateUserAnnotationGroups()

        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(1, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(10 * 8, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(10 * 8 * 4, mesh2d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(10 * 8 * 2, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.5845902460315318, -0.6, -0.1], 1.0E-8)
        assertAlmostEqualList(self, maximums, [0.6, 0.5845902460315319, 0.1], 1.0E-8)

        bob = fieldmodule.findFieldByName("bob").castGroup()
        self.assertTrue(bob.isValid())
        bobNodes = bob.getNodesetGroup(nodes)
        self.assertTrue(bobNodes.isValid())
        self.assertEqual(4 * 8 * 2, bobNodes.getSize())
        bobMinimums, bobMaximums = evaluateFieldNodesetRange(coordinates, bobNodes)
        assertAlmostEqualList(self, bobMinimums, [0.0, -0.6, -0.1], 1.0E-8)
        assertAlmostEqualList(self, bobMaximums, [0.6, 0.0, 0.1], 1.0E-8)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            isExteriorXi3_0 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_0))
            isExteriorXi3_1 = fieldmodule.createFieldAnd(
                isExterior, fieldmodule.createFieldIsOnFace(Element.FACE_TYPE_XI3_1))
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(4)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            outerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_1, coordinates, mesh2d)
            outerSurfaceAreaField.setNumbersOfPoints(4)
            result, outerSurfaceArea = outerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            innerSurfaceAreaField = fieldmodule.createFieldMeshIntegral(isExteriorXi3_0, coordinates, mesh2d)
            innerSurfaceAreaField.setNumbersOfPoints(4)
            result, innerSurfaceArea = innerSurfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 0.03536527637408515, delta=1.0E-6)
            self.assertAlmostEqual(outerSurfaceArea, 1.968455538630236, delta=1.0E-6)
            self.assertAlmostEqual(innerSurfaceArea, 1.5747640035466601, delta=1.0E-6)


if __name__ == "__main__":
    unittest.main()
