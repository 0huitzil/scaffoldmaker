import unittest

from cmlibs.utils.zinc.finiteelement import evaluateFieldNodesetRange, findNodeWithName
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Element
from cmlibs.zinc.field import Field
from cmlibs.zinc.result import RESULT_OK
from scaffoldmaker.annotation.annotationgroup import getAnnotationGroupForTerm
from scaffoldmaker.annotation.uterus_terms import get_uterus_term, uterus_terms
from scaffoldmaker.meshtypes.meshtype_3d_uterus1 import MeshType_3d_uterus1
from scaffoldmaker.utils.meshrefinement import MeshRefinement
from scaffoldmaker.utils.zinc_utils import createFaceMeshGroupExteriorOnFace

from testutils import assertAlmostEqualList, check_annotation_term_ids


class UterusScaffoldTestCase(unittest.TestCase):

    def test_uterus_annotations(self):
        """
        Test nomenclature of the uterus terms. 
        """
        for term_ids in uterus_terms:
            self.assertTrue(check_annotation_term_ids(term_ids), "Invalid primary term id or order not UBERON < ILX < FMA for uterus annotation term ids " + str(term_ids)) 


    def test_uterus1(self):
        """
        Test creation of uterus scaffold.
        """
        scaffold = MeshType_3d_uterus1
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, ['Default', 'Human 1', 'Human Pregnant 1', 'Mouse 1', 'Rat 1'])
        options = scaffold.getDefaultOptions("Human 1")

        networkLayout = options.get("Network layout")
        networkLayoutSettings = networkLayout.getScaffoldSettings()
        self.assertEqual("1-2-3-4-5-6-7-8-23.1,9-10-11-12-13-14-15-16-23.2,#-17-18-19-20-21-22-23.3,"
                         "23.4-24-25-26-27-28-29,29-30-31,31-32-33-34-35-36-37-38",
                         networkLayoutSettings["Structure"])

        self.assertEqual(15, len(options))
        self.assertEqual(20, options.get("Number of elements around"))
        self.assertEqual(8, options.get("Number of elements around oviduct/uterine horn"))
        self.assertEqual(1, options.get("Number of elements through wall"))
        self.assertEqual(True, options.get("Use linear through wall"))

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotationGroups = scaffold.generateBaseMesh(region, options)[0]
        self.assertEqual(17, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        self.assertEqual(RESULT_OK, fieldmodule.defineAllFaces())
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(320, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(1298, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(1653, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(678, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-2.999999999999999, -14.0, -8.268270767743472], 1.0E-6)
        assertAlmostEqualList(self, maximums, [12.947480545068354, 14.0, 2.991966625368734], 1.0E-6)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            faceMeshGroup = createFaceMeshGroupExteriorOnFace(fieldmodule, Element.FACE_TYPE_XI3_1)
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, faceMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(3)
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 326.53374476529103, delta=5.0E-2)
        self.assertAlmostEqual(volume, 250.8430658356537, delta=5.0E-2)

        fieldmodule.defineAllFaces()
        for annotationGroup in annotationGroups:
            annotationGroup.addSubelements()
        scaffold.defineFaceAnnotations(region, options, annotationGroups)
        self.assertEqual(45, len(annotationGroups))

        # check some annotation groups
        expectedSizes3d = {
            "fundus of uterus": 32,
            "body of uterus": 128,
            "left oviduct": 40,
            "vagina": 80,
            'left broad ligament of uterus': 12,
            'right broad ligament of uterus': 12,
            "uterus": 240
            }

        meshes = [mesh1d, mesh2d, mesh3d]
        for name in expectedSizes3d:
            term = get_uterus_term(name)
            annotationGroup = getAnnotationGroupForTerm(annotationGroups, term)
            size = annotationGroup.getMeshGroup(meshes[annotationGroup.getDimension() - 1]).getSize()
            self.assertEqual(expectedSizes3d[name], size, name)

        # refine 2x2x2 and check result
        # first remove faces/lines and any surface annotation groups as they are re-added by defineFaceAnnotations
        removeAnnotationGroups = []
        for annotationGroup in annotationGroups:
            if annotationGroup.getDimension() in [1, 2]:
                removeAnnotationGroups.append(annotationGroup)
        for annotationGroup in removeAnnotationGroups:
            if "cervix" in annotationGroup.getName():
                continue
            annotationGroups.remove(annotationGroup)
        self.assertEqual(22, len(annotationGroups))
        # must keep all faces and lines as used for refinement

        refineRegion = region.createRegion()
        refineFieldmodule = refineRegion.getFieldmodule()
        options['Refine number of elements along'] = 2
        options['Refine number of elements around'] = 2
        options['Refine number of elements through wall'] = 2
        meshrefinement = MeshRefinement(region, refineRegion, annotationGroups)
        scaffold.refineMesh(meshrefinement, options)
        annotationGroups = meshrefinement.getAnnotationGroups()

        refineFieldmodule.defineAllFaces()
        self.assertEqual(22, len(annotationGroups))

        mesh3d = refineFieldmodule.findMeshByDimension(3)
        self.assertEqual(2560, mesh3d.getSize())
        mesh2d = refineFieldmodule.findMeshByDimension(2)
        self.assertEqual(9032, mesh2d.getSize())
        mesh1d = refineFieldmodule.findMeshByDimension(1)
        self.assertEqual(10418, mesh1d.getSize())
        nodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(3949, nodes.getSize())
        datapoints = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check some refined annotationGroups:
        meshes = [mesh1d, mesh2d, mesh3d]
        sizeScales = [2, 4, 8]
        expectedSizes3d = {
            "fundus of uterus": 32,
            "body of uterus": 128,
            "left oviduct": 40,
            "vagina": 80,
            "uterus": 240
        }
        for name in expectedSizes3d:
            term = get_uterus_term(name)
            annotationGroup = getAnnotationGroupForTerm(annotationGroups, term)
            size = annotationGroup.getMeshGroup(meshes[annotationGroup.getDimension() - 1]).getSize()
            self.assertEqual(expectedSizes3d[name] * sizeScales[annotationGroup.getDimension() - 1], size, name)

        # test finding a marker in refined scaffold
        markerGroup = refineFieldmodule.findFieldByName("marker").castGroup()
        refinedNodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        markerNodes = markerGroup.getNodesetGroup(refinedNodes)
        self.assertEqual(4, markerNodes.getSize())
        markerName = refineFieldmodule.findFieldByName("marker_name")
        self.assertTrue(markerName.isValid())
        markerLocation = refineFieldmodule.findFieldByName("marker_location")
        self.assertTrue(markerLocation.isValid())
        cache = refineFieldmodule.createFieldcache()
        node = findNodeWithName(markerNodes, markerName, "junction of left round ligament with uterus")
        self.assertTrue(node.isValid())
        cache.setNode(node)
        element, xi = markerLocation.evaluateMeshLocation(cache, 3)
        self.assertTrue(element.isValid())


if __name__ == "__main__":
    unittest.main()
