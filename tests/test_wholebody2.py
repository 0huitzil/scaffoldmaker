import math
import unittest

from cmlibs.utils.zinc.finiteelement import evaluateFieldNodesetRange
from cmlibs.utils.zinc.general import ChangeManager

from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Element
from cmlibs.zinc.field import Field
from cmlibs.zinc.result import RESULT_OK

from scaffoldmaker.annotation.annotationgroup import getAnnotationGroupForTerm, findAnnotationGroupByName
from scaffoldmaker.annotation.body_terms import get_body_term, body_terms
from scaffoldmaker.meshtypes.meshtype_3d_wholebody2 import MeshType_3d_wholebody2

from testutils import assertAlmostEqualList, check_annotation_term_ids


class WholeBody2ScaffoldTestCase(unittest.TestCase):

    def test_body_annotations(self):
        """
        Test nomenclature of the body terms. 
        """
        for term_ids in body_terms:
            self.assertTrue(check_annotation_term_ids(term_ids), "Invalid primary term id or order not UBERON < ILX < FMA for body annotation term ids " + str(term_ids)) 

    def test_wholebody2_core_1shell(self):
        """
        Test creation of whole-body scaffold with solid core and 1 shell count.
        """
        scaffold = MeshType_3d_wholebody2
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, ["Default", "Human 1 Coarse", "Human 1 Medium", "Human 1 Fine"])
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        self.assertEqual(19, len(options))
        self.assertEqual(4, options["Number of elements along head"])
        self.assertEqual(1, options["Number of elements along neck"])
        self.assertEqual(2, options["Number of elements along thorax"])
        self.assertEqual(2, options["Number of elements along abdomen"])
        self.assertEqual(5, options["Number of elements along arm to hand"])
        self.assertEqual(1, options["Number of elements along hand"])
        self.assertEqual(4, options["Number of elements along leg to foot"])
        self.assertEqual(2, options["Number of elements along foot"])
        self.assertEqual(12, options["Number of elements around head"])
        self.assertEqual(12, options["Number of elements around torso"])
        self.assertEqual(8, options["Number of elements around arm"])
        self.assertEqual(8, options["Number of elements around leg"])
        self.assertEqual(1, options["Number of elements through shell"])
        self.assertEqual(False, options["Show trim surfaces"])
        self.assertEqual(True, options["Use Core"])
        self.assertEqual(2, options["Number of elements across core box minor"])
        self.assertEqual(1, options["Number of elements across core transition"])

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotationGroups = scaffold.generateMesh(region, options)[0]
        self.assertEqual(40, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(752, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(2444, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(2668, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(977, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # Check coordinates range, volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        tol = 1.0E-4
        assertAlmostEqualList(self, minimums, [0.0, -3.616103797617173, -1.0], tol)
        assertAlmostEqualList(self, maximums, [19.24758095792375, 3.616103797617173, 2.15], tol)
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

            self.assertAlmostEqual(volume, 84.80289228373682, delta=tol)
            self.assertAlmostEqual(surfaceArea, 209.22400741536197, delta=tol)

        # check some annotation groups:

        expectedSizes3d = {
            'abdominal cavity': (40, 7.907927325697995),
            'core': (456, 42.49713792169914),
            'head': (112, 6.950404130313098),
            'shell': (296, 42.30575436203679),
            'thoracic cavity': (40, 5.896123358745541)
        }
        for name in expectedSizes3d:
            term = get_body_term(name)
            annotationGroup = getAnnotationGroupForTerm(annotationGroups, term)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=tol)

        expectedSizes2d = {
            'abdominal cavity boundary surface': (64, 24.11511184594557),
            'diaphragm': (20, 2.4622475940631054),
            'left lower limb skin epidermis outer surface': (68, 49.68059914132895),
            'left upper limb skin epidermis outer surface': (68, 21.928293524879667),
            'right lower limb skin epidermis outer surface': (68, 49.68059914132903),
            'right upper limb skin epidermis outer surface': (68, 21.928293524852375),
            'skin epidermis outer surface': (376, 209.22400741536197),
            'thoracic cavity boundary surface': (64, 18.9648787549131)
        }
        for name in expectedSizes2d:
            term = get_body_term(name)
            annotationGroup = getAnnotationGroupForTerm(annotationGroups, term)
            size = annotationGroup.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name][0], size, name)
            surfaceMeshGroup = annotationGroup.getMeshGroup(mesh2d)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, surfaceMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, expectedSizes2d[name][1], delta=tol)

        expectedSizes1d = {
            'spinal cord': (6, 7.9078846597122805)
            }
        for name in expectedSizes1d:
            term = get_body_term(name)
            annotationGroup = getAnnotationGroupForTerm(annotationGroups, term)
            size = annotationGroup.getMeshGroup(mesh1d).getSize()
            self.assertEqual(expectedSizes1d[name][0], size, name)
            lineMeshGroup = annotationGroup.getMeshGroup(mesh1d)
            lengthField = fieldmodule.createFieldMeshIntegral(one, coordinates, lineMeshGroup)
            lengthField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, length = lengthField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(length, expectedSizes1d[name][1], delta=tol)

    def test_wholebody2_core_0shell(self):
        """
        Test creation of whole-body scaffold with solid core and 0 shell count.
        """
        scaffold = MeshType_3d_wholebody2
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, ["Default", "Human 1 Coarse", "Human 1 Medium", "Human 1 Fine"])
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        self.assertEqual(19, len(options))
        options["Number of elements through shell"] = 0
        self.assertEqual(True, options["Use Core"])

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotationGroups = scaffold.generateMesh(region, options)[0]
        self.assertEqual(34, len(annotationGroups))  # since cavity groups x 4, diaphragm and spinal cord not defined

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(456, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(1540, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(1750, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(667, nodes.getSize())

        # Check coordinates range, volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        tol = 1.0E-4
        assertAlmostEqualList(self, minimums, [0.0, -3.616103797617173, -1.0], tol)
        assertAlmostEqualList(self, maximums, [19.24758095792375, 3.616103797617173, 2.15], tol)

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

            self.assertAlmostEqual(volume, 84.80287779858831, delta=tol)
            self.assertAlmostEqual(surfaceArea, 209.22400680754546, delta=tol)

        # check some annotation groups:

        expectedSizes3d = {
            'core': (456, 84.80287779858831), 
            'head': (68, 6.950404175746005)
        }
        for name in expectedSizes3d:
            term = get_body_term(name)
            annotationGroup = getAnnotationGroupForTerm(annotationGroups, term)
            size = annotationGroup.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name][0], size, name)
            volumeMeshGroup = annotationGroup.getMeshGroup(mesh3d)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, volumeMeshGroup)
            volumeField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(volume, expectedSizes3d[name][1], delta=tol)

        expectedSizes2d = {
            'shell': (296, 208.4052209853951),
            'left lower limb skin epidermis outer surface': (60, 49.68059883742069),
            'left upper limb skin epidermis outer surface': (60, 21.92829352487968),
            'right lower limb skin epidermis outer surface': (60, 49.68059883742075),
            'right upper limb skin epidermis outer surface': (60, 21.928293524852407),
            'skin epidermis outer surface': (344, 209.22400680754546)
        }
        for name in expectedSizes2d:
            term = get_body_term(name)
            annotationGroup = getAnnotationGroupForTerm(annotationGroups, term)
            size = annotationGroup.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name][0], size, name)
            surfaceMeshGroup = annotationGroup.getMeshGroup(mesh2d)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, surfaceMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, expectedSizes2d[name][1], delta=tol)

    def test_wholebody2_1shell(self):
        """
        Test creation of whole-body scaffold without the solid core, 1 shell layer.
        """
        scaffold = MeshType_3d_wholebody2
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        options["Number of elements through shell"] = 1
        options["Use Core"] = False

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotationGroups = scaffold.generateMesh(region, options)[0]
        self.assertEqual(32, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(296, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(1200, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(1526, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(620, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # Check coordinates range, volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        tol = 1.0E-4
        assertAlmostEqualList(self, minimums, [0.0, -3.616103797617173, -1.0], tol)
        assertAlmostEqualList(self, maximums, [19.24758095792375, 3.616103797617173, 2.15], tol)
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

            self.assertAlmostEqual(volume, 42.30575436203679, delta=tol)
            self.assertAlmostEqual(outerSurfaceArea, 208.4052209853952, delta=tol)
            self.assertAlmostEqual(innerSurfaceArea, 148.9434649574258, delta=tol)

        # check some annotationGroups:
        expectedSizes2d = {
            'skin epidermis outer surface': (328, 208.82280181827798)
            }
        for name in expectedSizes2d:
            term = get_body_term(name)
            annotationGroup = getAnnotationGroupForTerm(annotationGroups, term)
            size = annotationGroup.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name][0], size, name)

            surfaceMeshGroup = annotationGroup.getMeshGroup(mesh2d)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, surfaceMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)

            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, expectedSizes2d[name][1], delta=tol)

    def test_wholebody2_0shell(self):
        """
        Test creation of 2d whole-body scaffold without solid core but 0 shell count.
        """
        scaffold = MeshType_3d_wholebody2
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, ["Default", "Human 1 Coarse", "Human 1 Medium", "Human 1 Fine"])
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        self.assertEqual(19, len(options))
        options["Number of elements through shell"] = 0
        options["Use Core"] = False

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotationGroups = scaffold.generateMesh(region, options)[0]
        self.assertEqual(32, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(0, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(296, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(608, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(310, nodes.getSize())

        # Check coordinates range, volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        tol = 1.0E-4
        assertAlmostEqualList(self, minimums, [0.0, -3.616103797617173, -1.0], tol)
        assertAlmostEqualList(self, maximums, [19.24758095792375, 3.616103797617173, 2.15], tol)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            fieldcache = fieldmodule.createFieldcache()

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, 208.40522100096814, delta=tol)

        # check some annotation groups:

        expectedSizes2d = {
            'head': (44, 16.006190273878417),
            'left lower limb skin epidermis outer surface': (48, 49.438940146171966),
            'left upper limb skin epidermis outer surface': (48, 21.76055930620015),
            'right lower limb skin epidermis outer surface': (48, 49.438940146172094),
            'right upper limb skin epidermis outer surface': (48, 21.760559306172855),
            'skin epidermis outer surface': (296, 208.40522100096814)
        }
        for name in expectedSizes2d:
            term = get_body_term(name)
            annotationGroup = getAnnotationGroupForTerm(annotationGroups, term)
            size = annotationGroup.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name][0], size, name)
            surfaceMeshGroup = annotationGroup.getMeshGroup(mesh2d)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, surfaceMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            fieldcache = fieldmodule.createFieldcache()
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            self.assertAlmostEqual(surfaceArea, expectedSizes2d[name][1], delta=tol)


if __name__ == "__main__":
    unittest.main()
