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


    def test_wholebody2_core(self):
        """
        Test creation of Whole-body scaffold with solid core.
        """
        scaffold = MeshType_3d_wholebody2
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, ["Default", "Human 1 Coarse", "Human 1 Medium", "Human 1 Fine"])
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        self.assertEqual(23, len(options))
        self.assertEqual(2, options["Number of elements along head"])
        self.assertEqual(1, options["Number of elements along neck"])
        self.assertEqual(2, options["Number of elements along thorax"])
        self.assertEqual(2, options["Number of elements along abdomen"])
        self.assertEqual(3, options["Number of elements along brachium"])
        self.assertEqual(2, options["Number of elements along antebrachium"])
        self.assertEqual(1, options["Number of elements along hand"])
        self.assertEqual(2, options["Number of elements along hip"])
        self.assertEqual(3, options["Number of elements along upper leg"])
        self.assertEqual(3, options["Number of elements along lower leg"])
        self.assertEqual(1, options["Number of elements along foot"])
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
        self.assertEqual(58, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(904, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(2962, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(3235, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(1195, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # Check coordinates range, volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        tol = 1.0E-4
        assertAlmostEqualList(self, minimums, [0.0, -3.936660189011623, -1.25], tol)
        assertAlmostEqualList(self, maximums, [19.925896216328983, 3.936660189011623, 2.354767205067949], tol)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            isExterior = fieldmodule.createFieldIsExterior()
            mesh2d = fieldmodule.findMeshByDimension(2)
            fieldcache = fieldmodule.createFieldcache()

            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(3)
            result, volume = volumeField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(isExterior, coordinates, mesh2d)
            surfaceAreaField.setNumbersOfPoints(4)
            result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)

            self.assertAlmostEqual(volume, 94.6500967774338, delta=tol)
            self.assertAlmostEqual(surfaceArea, 229.64114786971544, delta=tol)

        # check some annotation groups:

        expectedSizes3d = {
            "abdominal cavity": (40,  9.403829461454576),
            "core": (548, 43.36861548855504),
            "head": (64,  6.909618374858556),
            "thoracic cavity": (40, 6.858627377211818),
            "shell": (356, 51.2784493068052)
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
            "abdominal cavity boundary surface": (64, 25.896219570557346),
            "diaphragm": (20, 3.0778936559347208),
            "left upper limb skin epidermis outer surface": (60, 19.127854715737577),
            "left lower limb skin epidermis outer surface": (64, 49.11716217195525),
            "right upper limb skin epidermis outer surface": (60, 19.127854715737577),
            "right lower limb skin epidermis outer surface": (64, 49.11716217195525),
            "skin epidermis outer surface": (500, 229.64114786971544),
            "thoracic cavity boundary surface": (56, 18.548022753125213)
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
            "spinal cord": (8, 10.263008402034743)
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


    def test_wholebody2_tube(self):
        """
        Test creation of Whole-body scaffold without solid core.
        """
        scaffold = MeshType_3d_wholebody2
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        options["Use Core"] = False
        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotationGroups = scaffold.generateMesh(region, options)[0]
        self.assertEqual(50, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(356, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(1446, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(1843, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(763, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # Check coordinates range, volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        tol = 1.0E-4
        assertAlmostEqualList(self, minimums, [0.0, -3.936660189011623, -1.25], tol)
        assertAlmostEqualList(self, maximums, [19.925896216328983, 3.936660189011623, 2.354767205067949], tol)

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
            volumeField.setNumbersOfPoints(3)
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

            self.assertAlmostEqual(volume, 51.279257855672284, delta=tol)
            self.assertAlmostEqual(outerSurfaceArea, 221.88145706662112, delta=tol)
            self.assertAlmostEqual(innerSurfaceArea, 152.66309373435772, delta=tol)

        # check some annotationGroups:
        expectedSizes2d = {
            "skin epidermis outer surface": (400, 225.0532922435965)
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
