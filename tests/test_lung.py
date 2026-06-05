import copy
import math
import unittest

from cmlibs.utils.zinc.finiteelement import evaluateFieldNodesetRange, findNodeWithName
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.utils.zinc.group import mesh_group_to_identifier_ranges, nodeset_group_to_identifier_ranges
from cmlibs.zinc.context import Context
from cmlibs.zinc.field import Field
from cmlibs.zinc.result import RESULT_OK
from scaffoldmaker.annotation.annotationgroup import getAnnotationGroupForTerm
from scaffoldmaker.annotation.lung_terms import get_lung_term, lung_terms
from scaffoldmaker.meshtypes.meshtype_3d_lung1 import MeshType_3d_lung1
from scaffoldmaker.meshtypes.meshtype_3d_lung2 import MeshType_3d_lung2
from scaffoldmaker.meshtypes.meshtype_3d_lung3 import MeshType_3d_lung3
from scaffoldmaker.meshtypes.meshtype_3d_lung4 import MeshType_3d_lung4
from scaffoldmaker.utils.meshrefinement import MeshRefinement

from testutils import assertAlmostEqualList, check_annotation_term_ids


class LungScaffoldTestCase(unittest.TestCase):

    def test_lung_annotations(self):
        """
        Test nomenclature of the lung terms. 
        """
        for term_ids in lung_terms:
            self.assertTrue(check_annotation_term_ids(term_ids), "Invalid primary term id or order not UBERON < ILX < FMA for lung annotation term ids " + str(term_ids)) 
            
    def test_lung1(self):
        """
        Test creation of heart scaffold.
        """
        scaffold = MeshType_3d_lung1
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, [ "Default", "Human 1", "Mouse 1", "Pig 1", "Rat 1" ])
        options = scaffold.getDefaultOptions(["Human 1"])
        self.assertEqual(3, len(options))
        self.assertFalse(scaffold.checkOptions(options))

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotationGroups = scaffold.generateMesh(region, options)[0]
        self.assertEqual(20, len(annotationGroups))
        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(88, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(292, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(334, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(141, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check coordinates range, outside surface area and volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [ 14.153, 62.444, -353.40564 ], 1.0E-6)
        assertAlmostEqualList(self, maximums, [ 297.502, 245.3003, -23.486 ], 1.0E-6)
        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            upperRightFissureGroup = fieldmodule.findFieldByName('upper lobe of right lung').castGroup()
            self.assertTrue(upperRightFissureGroup.isValid())
            upperRightFissureMeshGroup = upperRightFissureGroup.getMeshGroup(mesh2d)
            self.assertTrue(upperRightFissureMeshGroup.isValid())
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, upperRightFissureMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(3)
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 138894.27893716586, delta=1.0E-2)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(volume, 7225489.982550352, delta=1.0E-2)

        # check some annotationGroups:
        expectedSizes3d = {
            "lung": 88,
            "left lung": 44,
            "right lung": 44,
            "upper lobe of left lung": 24,
            "lower lobe of left lung": 20,
            "upper lobe of right lung": 16,
            "middle lobe of right lung": 8,
            "lower lobe of right lung": 20,
            # "right lung accessory lobe": 0
            }
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name], size, name)
        expectedSizes2d = {
            "horizontal fissure of right lung": 4,
            "oblique fissure of left lung": 8,
            "oblique fissure of right lung": 8
            }
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name], size, name)

        # test finding a marker in scaffold
        markerGroup = fieldmodule.findFieldByName("marker").castGroup()
        markerNodes = markerGroup.getNodesetGroup(nodes)
        self.assertEqual(9, markerNodes.getSize())
        markerName = fieldmodule.findFieldByName("marker_name")
        self.assertTrue(markerName.isValid())
        markerLocation = fieldmodule.findFieldByName("marker_location")
        self.assertTrue(markerLocation.isValid())
        # test apex marker point
        cache = fieldmodule.createFieldcache()
        node = findNodeWithName(markerNodes, markerName, "apex of left lung")
        self.assertTrue(node.isValid())
        cache.setNode(node)
        element, xi = markerLocation.evaluateMeshLocation(cache, 3)
        self.assertEqual(37, element.getIdentifier())
        assertAlmostEqualList(self, xi, [ 0.0, 0.0, 1.0 ], 1.0E-10)

        # refine 4 and check result
        # first remove any face (but not point) annotation groups as they are re-added by defineFaceAnnotations
        removeAnnotationGroups = []
        for annotationGroup in annotationGroups:
            if (not annotationGroup.hasMeshGroup(mesh3d)) and (annotationGroup.hasMeshGroup(mesh2d) or annotationGroup.hasMeshGroup(mesh1d)):
                removeAnnotationGroups.append(annotationGroup)
        for annotationGroup in removeAnnotationGroups:
            annotationGroups.remove(annotationGroup)
        self.assertEqual(17, len(annotationGroups))

        refineRegion = region.createRegion()
        refineFieldmodule = refineRegion.getFieldmodule()
        options['Refine'] = True
        options['Refine number of elements'] = 4
        refineNumberOfElements = options['Refine number of elements']
        meshrefinement = MeshRefinement(region, refineRegion, annotationGroups)
        scaffold.refineMesh(meshrefinement, options)
        annotationGroups = meshrefinement.getAnnotationGroups()

        refineFieldmodule.defineAllFaces()
        oldAnnotationGroups = copy.copy(annotationGroups)
        for annotationGroup in annotationGroups:
            annotationGroup.addSubelements()
        scaffold.defineFaceAnnotations(refineRegion, options, annotationGroups)
        for annotation in annotationGroups:
            if annotation not in oldAnnotationGroups:
                annotationGroup.addSubelements()
        self.assertEqual(20, len(annotationGroups))

        mesh3d = refineFieldmodule.findMeshByDimension(3)
        self.assertEqual(5632, mesh3d.getSize())
        mesh2d = refineFieldmodule.findMeshByDimension(2)
        self.assertEqual(17344, mesh2d.getSize())
        mesh1d = refineFieldmodule.findMeshByDimension(1)
        self.assertEqual(17848, mesh1d.getSize())
        nodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(6147, nodes.getSize())
        datapoints = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check some refined annotationGroups:
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name]*(refineNumberOfElements**3), size, name)
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name]*(refineNumberOfElements**2), size, name)

        # test finding a marker in refined scaffold
        markerGroup = refineFieldmodule.findFieldByName("marker").castGroup()
        refinedNodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        markerNodes = markerGroup.getNodesetGroup(refinedNodes)
        self.assertEqual(9, markerNodes.getSize())
        markerName = refineFieldmodule.findFieldByName("marker_name")
        self.assertTrue(markerName.isValid())
        markerLocation = refineFieldmodule.findFieldByName("marker_location")
        self.assertTrue(markerLocation.isValid())
        # test apex marker point
        cache = refineFieldmodule.createFieldcache()
        node = findNodeWithName(markerNodes, markerName, "apex of left lung")
        self.assertTrue(node.isValid())
        cache.setNode(node)
        element, xi = markerLocation.evaluateMeshLocation(cache, 3)
        self.assertEqual(2353, element.getIdentifier())
        assertAlmostEqualList(self, xi, [ 0.0, 0.0, 1.0 ], 1.0E-10)

    def test_lung2_human(self):
        """
        Test creation of heart scaffold.
        """
        scaffold = MeshType_3d_lung2
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, [ "Default", "Human 1", "Mouse 1", "Rat 1", "Pig 1", "Material" ])
        options = scaffold.getDefaultOptions(["Human 1"])
        self.assertEqual(37, len(options))
        self.assertFalse(scaffold.checkOptions(options))
        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())

        # Need to do the following manually to save originalAnnotationGroups which has some temporary groups
        fieldmodule = region.getFieldmodule()
        # annotationGroups = scaffold.generateMesh(region, options)
        with ChangeManager(fieldmodule):
            annotationGroups = scaffold.generateBaseMesh(region, options)[0]
            fieldmodule.defineAllFaces()
            originalAnnotationGroups = copy.copy(annotationGroups)
            for annotationGroup in annotationGroups:
                annotationGroup.addSubelements()
            scaffold.defineFaceAnnotations(region, options, annotationGroups)
            for annotationGroup in annotationGroups:
                if annotationGroup not in originalAnnotationGroups:
                    annotationGroup.addSubelements()

        self.assertEqual(54, len(annotationGroups))
        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(88, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(292, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(334, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(139, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check coordinates range, outside surface area and volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        material_coordinates = fieldmodule.findFieldByName("lung coordinates").castFiniteElement()
        self.assertTrue(material_coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.5604046150578359, -0.46588462707467704, -0.20848015499159522], 1.0E-6)
        assertAlmostEqualList(self, maximums, [0.5604046150578359, 0.3440517958126045, 0.9604261163849366], 1.0E-6)
        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            upperRightFissureGroup = fieldmodule.findFieldByName('upper lobe of right lung').castGroup()
            self.assertTrue(upperRightFissureGroup.isValid())
            upperRightFissureMeshGroup = upperRightFissureGroup.getMeshGroup(mesh2d)
            self.assertTrue(upperRightFissureMeshGroup.isValid())
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, upperRightFissureMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(3)
            left_lung_group = getAnnotationGroupForTerm(annotationGroups, get_lung_term("left lung"))
            left_lung_mesh_group = left_lung_group.getMeshGroup(mesh3d)
            materialVolumeField = fieldmodule.createFieldMeshIntegral(one, material_coordinates, left_lung_mesh_group)
            materialVolumeField.setNumbersOfPoints(3)
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 1.838244693520495, delta=1.0E-6)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(volume,  0.3859394886952645, delta=1.0E-6)
        result, material_volume = materialVolumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        halfEllipsoidVolume = 0.5 * (4.0 / 3.0) * math.pi * 0.25 * 0.5 * 1.0  # 3 axis lengths
        self.assertAlmostEqual(halfEllipsoidVolume, 0.2617993877991494, delta=1.0E-12)
        # not quite the same as the above due to sharpening on the ventral edge
        self.assertAlmostEqual(material_volume, 0.2597062752904966, delta=1.0E-6)

        # check some annotationGroups:
        expectedSizes3d = {
            "lung": 88,
            "left lung": 44,
            "right lung": 44,
            "upper lobe of left lung": 24,
            "lower lobe of left lung": 20,
            "upper lobe of right lung": 16,
            "middle lobe of right lung": 8,
            "lower lobe of right lung": 20,
            }
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name], size, name)
        expectedSizes2d = {
            "base of left lung surface": 12,
            "base of lower lobe of left lung surface": 8,
            "base of upper lobe of left lung surface": 4,
            "base of right lung surface": 12,
            "base of lower lobe of right lung surface": 8,
            "base of middle lobe of right lung surface": 4,
            "horizontal fissure of right lung": 4,
            "horizontal fissure of middle lobe of right lung": 4,
            "horizontal fissure of upper lobe of right lung": 4,
            'lateral surface of lower lobe of left lung': 10,
            'lateral surface of upper lobe of left lung': 12,
            'lateral surface of lower lobe of right lung': 10,
            'lateral surface of middle lobe of right lung': 4,
            'lateral surface of upper lobe of right lung': 8,
            'medial surface of lower lobe of left lung': 10,
            'medial surface of upper lobe of left lung': 12,
            'medial surface of lower lobe of right lung': 10,
            'medial surface of middle lobe of right lung': 4,
            'medial surface of upper lobe of right lung': 8,
            "oblique fissure of left lung": 8,
            "oblique fissure of right lung": 8,
            "oblique fissure of lower lobe of left lung": 8,
            "oblique fissure of upper lobe of left lung": 8,
            "oblique fissure of lower lobe of right lung": 8,
            # "oblique fissure of middle lobe of right lung": 4,
            # "oblique fissure of upper lobe of right lung": 4,
            "left lung surface": 56,
            "lower lobe of left lung surface": 36,
            "upper lobe of left lung surface": 36,
            "right lung surface": 56,
            "lower lobe of right lung surface": 36,
            "middle lobe of right lung surface": 20,
            "upper lobe of right lung surface": 24,
        }
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name], size, name)
        expectedSizes1d = {
            "anterior border of left lung": 5,
            "anterior border of right lung": 5
            }
        for name in expectedSizes1d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh1d).getSize()
            self.assertEqual(expectedSizes1d[name], size, name)

        # test finding a marker in scaffold
        markerGroup = fieldmodule.findFieldByName("marker").castGroup()
        markerNodes = markerGroup.getNodesetGroup(nodes)
        self.assertEqual(7, markerNodes.getSize())
        markerName = fieldmodule.findFieldByName("marker_name")
        self.assertTrue(markerName.isValid())
        markerLocation = fieldmodule.findFieldByName("marker_location")
        self.assertTrue(markerLocation.isValid())
        # test apex marker point
        cache = fieldmodule.createFieldcache()
        node = findNodeWithName(markerNodes, markerName, "apex of left lung")
        self.assertTrue(node.isValid())
        cache.setNode(node)
        element, xi = markerLocation.evaluateMeshLocation(cache, 3)
        self.assertEqual(40, element.getIdentifier())
        assertAlmostEqualList(self, xi, [ 0.0, 1.0, 1.0 ], 1.0E-10)

        # refine 2x2x2 and check result
        # need to use original annotation groups to get temporaries
        annotationGroups = originalAnnotationGroups
        self.assertEqual(19, len(annotationGroups))  # including 4 temporary groups

        refineRegion = region.createRegion()
        refineFieldmodule = refineRegion.getFieldmodule()
        options['Refine'] = True
        options['Refine number of elements'] = 2
        refineNumberOfElements = options['Refine number of elements']
        meshrefinement = MeshRefinement(region, refineRegion, annotationGroups)
        scaffold.refineMesh(meshrefinement, options)
        annotationGroups = meshrefinement.getAnnotationGroups()

        refineFieldmodule.defineAllFaces()
        oldAnnotationGroups = copy.copy(annotationGroups)
        for annotationGroup in annotationGroups:
            annotationGroup.addSubelements()
        scaffold.defineFaceAnnotations(refineRegion, options, annotationGroups)
        for annotation in annotationGroups:
            if annotation not in oldAnnotationGroups:
                annotationGroup.addSubelements()
        self.assertEqual(54, len(annotationGroups))

        mesh3d = refineFieldmodule.findMeshByDimension(3)
        self.assertEqual(704, mesh3d.getSize())
        mesh2d = refineFieldmodule.findMeshByDimension(2)
        self.assertEqual(2224, mesh2d.getSize())
        mesh1d = refineFieldmodule.findMeshByDimension(1)
        self.assertEqual(2364, mesh1d.getSize())
        nodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(853, nodes.getSize())
        datapoints = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check some refined annotationGroups:
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name]*(refineNumberOfElements**3), size, name)
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name]*(refineNumberOfElements**2), size, name)

        # test finding a marker in refined scaffold
        markerGroup = refineFieldmodule.findFieldByName("marker").castGroup()
        refinedNodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        markerNodes = markerGroup.getNodesetGroup(refinedNodes)
        self.assertEqual(7, markerNodes.getSize())
        markerName = refineFieldmodule.findFieldByName("marker_name")
        self.assertTrue(markerName.isValid())
        markerLocation = refineFieldmodule.findFieldByName("marker_location")
        self.assertTrue(markerLocation.isValid())
        # test apex marker point
        cache = refineFieldmodule.createFieldcache()
        node = findNodeWithName(markerNodes, markerName, "apex of left lung")
        self.assertTrue(node.isValid())
        cache.setNode(node)
        element, xi = markerLocation.evaluateMeshLocation(cache, 3)
        self.assertEqual(319, element.getIdentifier())
        assertAlmostEqualList(self, xi, [ 0.0, 1.0, 1.0 ], 1.0E-10)

    def test_lung2_mouse(self):
        """
        Test creation of heart scaffold.
        """
        scaffold = MeshType_3d_lung2
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, [ "Default", "Human 1", "Mouse 1", "Rat 1", "Pig 1", "Material" ])
        options = scaffold.getDefaultOptions(["Mouse 1"])
        self.assertEqual(37, len(options))
        self.assertFalse(scaffold.checkOptions(options))
        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())

        # Need to do the following manually to save originalAnnotationGroups which has some temporary groups
        fieldmodule = region.getFieldmodule()
        # annotationGroups = scaffold.generateMesh(region, options)
        with ChangeManager(fieldmodule):
            annotationGroups = scaffold.generateBaseMesh(region, options)[0]
            fieldmodule.defineAllFaces()
            originalAnnotationGroups = copy.copy(annotationGroups)
            for annotationGroup in annotationGroups:
                annotationGroup.addSubelements()
            scaffold.defineFaceAnnotations(region, options, annotationGroups)
            for annotationGroup in annotationGroups:
                if annotationGroup not in originalAnnotationGroups:
                    annotationGroup.addSubelements()

        self.assertEqual(54, len(annotationGroups))
        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(108, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(366, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(429, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(187, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check coordinates range, outside surface area and volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.5965062723149795, -0.9035466860119679, -0.6098596059713717], 1.0E-6)
        assertAlmostEqualList(self, maximums, [0.6642156880908214, 0.009214327892980281, 0.669464028759718], 1.0E-6)
        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            upperRightFissureGroup = fieldmodule.findFieldByName('upper lobe of right lung').castGroup()
            self.assertTrue(upperRightFissureGroup.isValid())
            upperRightFissureMeshGroup = upperRightFissureGroup.getMeshGroup(mesh2d)
            self.assertTrue(upperRightFissureMeshGroup.isValid())
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, upperRightFissureMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(3)
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 2.077889388115966, delta=1.0E-2)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(volume,  0.3799935491917908, delta=1.0E-2)

        # check some annotationGroups:
        expectedSizes3d = {
            "lung": 108,
            "left lung": 44,
            "right lung": 44,
            "upper lobe of right lung": 16,
            "middle lobe of right lung": 8,
            "lower lobe of right lung": 20,
            "right lung accessory lobe": 20
            }
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name], size, name)
        expectedSizes2d = {
            "base of left lung surface": 12,
            "base of right lung surface": 12,
            "base of lower lobe of right lung surface": 8,
            "base of middle lobe of right lung surface": 4,
            "horizontal fissure of right lung": 4,
            "horizontal fissure of middle lobe of right lung": 4,
            "horizontal fissure of upper lobe of right lung": 4,
            'lateral surface of lower lobe of right lung': 10,
            'lateral surface of middle lobe of right lung': 4,
            'lateral surface of upper lobe of right lung': 8,
            'medial surface of lower lobe of right lung': 10,
            'medial surface of middle lobe of right lung': 4,
            'medial surface of upper lobe of right lung': 8,
            "oblique fissure of right lung": 8,
            "oblique fissure of lower lobe of right lung": 8,
            # "oblique fissure of middle lobe of right lung": 4,
            # "oblique fissure of upper lobe of right lung": 4,
            "left lung surface": 56,
            "right lung surface": 56,
            "lower lobe of right lung surface": 36,
            "middle lobe of right lung surface": 20,
            "upper lobe of right lung surface": 24,
        }
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name], size, name)
        expectedSizes1d = {
            "anterior border of left lung": 5,
            "anterior border of right lung": 5
            }
        for name in expectedSizes1d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh1d).getSize()
            self.assertEqual(expectedSizes1d[name], size, name)

        # test finding a marker in scaffold
        markerGroup = fieldmodule.findFieldByName("marker").castGroup()
        markerNodes = markerGroup.getNodesetGroup(nodes)
        self.assertEqual(13, markerNodes.getSize())
        markerName = fieldmodule.findFieldByName("marker_name")
        self.assertTrue(markerName.isValid())
        markerLocation = fieldmodule.findFieldByName("marker_location")
        self.assertTrue(markerLocation.isValid())
        # test apex marker point
        cache = fieldmodule.createFieldcache()
        node = findNodeWithName(markerNodes, markerName, "apex of left lung")
        self.assertTrue(node.isValid())
        cache.setNode(node)
        element, xi = markerLocation.evaluateMeshLocation(cache, 3)
        self.assertEqual(40, element.getIdentifier())
        assertAlmostEqualList(self, xi, [ 0.0, 1.0, 1.0 ], 1.0E-10)

        # refine 2x2x2 and check result
        # need to use original annotation groups to get temporaries
        annotationGroups = originalAnnotationGroups
        self.assertEqual(24, len(annotationGroups))  # including 4 temporary groups

        refineRegion = region.createRegion()
        refineFieldmodule = refineRegion.getFieldmodule()
        options['Refine'] = True
        options['Refine number of elements'] = 2
        refineNumberOfElements = options['Refine number of elements']
        meshrefinement = MeshRefinement(region, refineRegion, annotationGroups)
        scaffold.refineMesh(meshrefinement, options)
        annotationGroups = meshrefinement.getAnnotationGroups()

        refineFieldmodule.defineAllFaces()
        oldAnnotationGroups = copy.copy(annotationGroups)
        for annotationGroup in annotationGroups:
            annotationGroup.addSubelements()
        scaffold.defineFaceAnnotations(refineRegion, options, annotationGroups)
        for annotation in annotationGroups:
            if annotation not in oldAnnotationGroups:
                annotationGroup.addSubelements()
        self.assertEqual(54, len(annotationGroups))

        mesh3d = refineFieldmodule.findMeshByDimension(3)
        self.assertEqual(864, mesh3d.getSize())
        mesh2d = refineFieldmodule.findMeshByDimension(2)
        self.assertEqual(2760, mesh2d.getSize())
        mesh1d = refineFieldmodule.findMeshByDimension(1)
        self.assertEqual(2970, mesh1d.getSize())
        nodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(1090, nodes.getSize())
        datapoints = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check some refined annotationGroups:
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name]*(refineNumberOfElements**3), size, name)
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name]*(refineNumberOfElements**2), size, name)
        # for name in expectedSizes1d:
        #     group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
        #     size = group.getMeshGroup(mesh1d).getSize()
        #     self.assertEqual(expectedSizes1d[name]*refineNumberOfElements, size, name)


        # test finding a marker in refined scaffold
        markerGroup = refineFieldmodule.findFieldByName("marker").castGroup()
        refinedNodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        markerNodes = markerGroup.getNodesetGroup(refinedNodes)
        self.assertEqual(13, markerNodes.getSize())
        markerName = refineFieldmodule.findFieldByName("marker_name")
        self.assertTrue(markerName.isValid())
        markerLocation = refineFieldmodule.findFieldByName("marker_location")
        self.assertTrue(markerLocation.isValid())
        # test apex marker point
        cache = refineFieldmodule.createFieldcache()
        node = findNodeWithName(markerNodes, markerName, "apex of left lung")
        self.assertTrue(node.isValid())
        cache.setNode(node)
        element, xi = markerLocation.evaluateMeshLocation(cache, 3)
        self.assertEqual(319, element.getIdentifier())
        assertAlmostEqualList(self, xi, [ 0.0, 1.0, 1.0 ], 1.0E-10)

    def test_lung2_human_openFissures(self):
        """
        Test creation of heart scaffold.
        """
        scaffold = MeshType_3d_lung2
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, [ "Default", "Human 1", "Mouse 1", "Rat 1", "Pig 1", "Material" ])
        options = scaffold.getDefaultOptions(["Human 1"])
        options['Open fissures'] = True
        self.assertEqual(37, len(options))
        self.assertFalse(scaffold.checkOptions(options))
        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())

        annotationGroups = scaffold.generateMesh(region, options)[0]
        self.assertEqual(54, len(annotationGroups))
        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(88, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(312, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(384, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(172, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check coordinates range, outside surface area and volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.5604046150578359, -0.46588462707467704, -0.20848015499159522], 1.0E-6)
        assertAlmostEqualList(self, maximums, [0.5604046150578359, 0.3440517958126045, 0.9604261163849366], 1.0E-6)
        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            upperRightFissureGroup = fieldmodule.findFieldByName('upper lobe of right lung').castGroup()
            self.assertTrue(upperRightFissureGroup.isValid())
            upperRightFissureMeshGroup = upperRightFissureGroup.getMeshGroup(mesh2d)
            self.assertTrue(upperRightFissureMeshGroup.isValid())
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, upperRightFissureMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(3)
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 1.8382446935204972, delta=1.0E-6)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(volume,  0.3859394886952645, delta=1.0E-6)

        # check some annotationGroups:
        expectedSizes3d = {
            "lung": 88,
            "left lung": 44,
            "right lung": 44,
            "upper lobe of left lung": 24,
            "lower lobe of left lung": 20,
            "upper lobe of right lung": 16,
            "middle lobe of right lung": 8,
            "lower lobe of right lung": 20,
            # "right lung accessory lobe": 0
        }
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name], size, name)
        expectedSizes2d = {
            "base of left lung surface": 12,
            "base of lower lobe of left lung surface": 8,
            "base of upper lobe of left lung surface": 4,
            "base of right lung surface": 12,
            "base of lower lobe of right lung surface": 8,
            "base of middle lobe of right lung surface": 4,
            "horizontal fissure of right lung": 8,
            "horizontal fissure of middle lobe of right lung": 4,
            "horizontal fissure of upper lobe of right lung": 4,
            'lateral surface of lower lobe of left lung': 10,
            'lateral surface of upper lobe of left lung': 12,
            'lateral surface of lower lobe of right lung': 10,
            'lateral surface of middle lobe of right lung': 4,
            'lateral surface of upper lobe of right lung': 8,
            'medial surface of lower lobe of left lung': 10,
            'medial surface of upper lobe of left lung': 12,
            'medial surface of lower lobe of right lung': 10,
            'medial surface of middle lobe of right lung': 4,
            'medial surface of upper lobe of right lung': 8,
            "oblique fissure of left lung": 16,
            "oblique fissure of right lung": 16,
            "oblique fissure of lower lobe of left lung": 8,
            "oblique fissure of upper lobe of left lung": 8,
            "oblique fissure of lower lobe of right lung": 8,
            # "oblique fissure of middle lobe of right lung": 4,
            # "oblique fissure of upper lobe of right lung": 4,
            "left lung surface": 56,
            "lower lobe of left lung surface": 36,
            "upper lobe of left lung surface": 36,
            "right lung surface": 56,
            "lower lobe of right lung surface": 36,
            "middle lobe of right lung surface": 20,
            "upper lobe of right lung surface": 24,
        }
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name], size, name)
        expectedSizes1d = {
            "anterior border of left lung": 5,
            "anterior border of right lung": 5
        }
        for name in expectedSizes1d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh1d).getSize()
            self.assertEqual(expectedSizes1d[name], size, name)

        # test finding a marker in scaffold
        markerGroup = fieldmodule.findFieldByName("marker").castGroup()
        markerNodes = markerGroup.getNodesetGroup(nodes)
        self.assertEqual(7, markerNodes.getSize())
        markerName = fieldmodule.findFieldByName("marker_name")
        self.assertTrue(markerName.isValid())
        markerLocation = fieldmodule.findFieldByName("marker_location")
        self.assertTrue(markerLocation.isValid())
        # test apex marker point
        cache = fieldmodule.createFieldcache()
        node = findNodeWithName(markerNodes, markerName, "apex of left lung")
        self.assertTrue(node.isValid())
        cache.setNode(node)
        element, xi = markerLocation.evaluateMeshLocation(cache, 3)
        self.assertEqual(40, element.getIdentifier())
        assertAlmostEqualList(self, xi, [ 0.0, 1.0, 1.0 ], 1.0E-10)

    def test_lung2_mouse_openFissures(self):
        """
        Test creation of heart scaffold.
        """
        scaffold = MeshType_3d_lung2
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames, [ "Default", "Human 1", "Mouse 1", "Rat 1", "Pig 1", "Material" ])
        options = scaffold.getDefaultOptions(["Mouse 1"])
        options['Open fissures'] = True
        self.assertEqual(37, len(options))
        self.assertFalse(scaffold.checkOptions(options))
        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        annotationGroups = scaffold.generateMesh(region, options)[0]
        self.assertEqual(54, len(annotationGroups))
        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(108, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(378, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(459, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(207, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check coordinates range, outside surface area and volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.5965062723149795, -0.9035466860119679, -0.6098596059713717], 1.0E-6)
        assertAlmostEqualList(self, maximums, [0.6642156880908214, 0.009214327892980281, 0.669464028759718], 1.0E-6)
        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            upperRightFissureGroup = fieldmodule.findFieldByName('upper lobe of right lung').castGroup()
            self.assertTrue(upperRightFissureGroup.isValid())
            upperRightFissureMeshGroup = upperRightFissureGroup.getMeshGroup(mesh2d)
            self.assertTrue(upperRightFissureMeshGroup.isValid())
            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, upperRightFissureMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(3)
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 2.077889388115967, delta=1.0E-2)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(volume,  0.3799935491917908, delta=1.0E-2)

        # check some annotationGroups:
        expectedSizes3d = {
            "lung": 108,
            "left lung": 44,
            "right lung": 44,
            "upper lobe of right lung": 16,
            "middle lobe of right lung": 8,
            "lower lobe of right lung": 20,
            "right lung accessory lobe": 20
            }
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name], size, name)
        expectedSizes2d = {
            "base of left lung surface": 12,
            "base of right lung surface": 12,
            "base of lower lobe of right lung surface": 8,
            "base of middle lobe of right lung surface": 4,
            "horizontal fissure of right lung": 8,
            "horizontal fissure of middle lobe of right lung": 4,
            "horizontal fissure of upper lobe of right lung": 4,
            'lateral surface of lower lobe of right lung': 10,
            'lateral surface of middle lobe of right lung': 4,
            'lateral surface of upper lobe of right lung': 8,
            'medial surface of lower lobe of right lung': 10,
            'medial surface of middle lobe of right lung': 4,
            'medial surface of upper lobe of right lung': 8,
            "oblique fissure of right lung": 16,
            "oblique fissure of lower lobe of right lung": 8,
            # "oblique fissure of middle lobe of right lung": 4,
            # "oblique fissure of upper lobe of right lung": 4,
            "left lung surface": 56,
            "right lung surface": 56,
            "lower lobe of right lung surface": 36,
            "middle lobe of right lung surface": 20,
            "upper lobe of right lung surface": 24,
        }
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name], size, name)
        expectedSizes1d = {
            "anterior border of left lung": 5,
            "anterior border of right lung": 5
            }
        for name in expectedSizes1d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh1d).getSize()
            self.assertEqual(expectedSizes1d[name], size, name)

        # test finding a marker in scaffold
        markerGroup = fieldmodule.findFieldByName("marker").castGroup()
        markerNodes = markerGroup.getNodesetGroup(nodes)
        self.assertEqual(13, markerNodes.getSize())
        markerName = fieldmodule.findFieldByName("marker_name")
        self.assertTrue(markerName.isValid())
        markerLocation = fieldmodule.findFieldByName("marker_location")
        self.assertTrue(markerLocation.isValid())
        # test apex marker point
        cache = fieldmodule.createFieldcache()
        node = findNodeWithName(markerNodes, markerName, "apex of left lung")
        self.assertTrue(node.isValid())
        cache.setNode(node)
        element, xi = markerLocation.evaluateMeshLocation(cache, 3)
        self.assertEqual(40, element.getIdentifier())
        assertAlmostEqualList(self, xi, [ 0.0, 1.0, 1.0 ], 1.0E-10)

    def test_lung3_human(self):
        """
        Test creation of human lung scaffold using lung3 mesh.
        """
        scaffold = MeshType_3d_lung3
        parameterSetNames = scaffold.getParameterSetNames()
        self.assertEqual(parameterSetNames,
                         ["Default", "Human 1 Coarse", "Human 1 Medium", "Human 1 Fine",
                          "Ellipsoid Coarse", "Ellipsoid Medium", "Ellipsoid Fine"])
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        self.assertEqual(20, len(options))
        self.assertFalse(scaffold.checkOptions(options))
        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())

        # Need to do the following manually to save originalAnnotationGroups which has some temporary groups
        fieldmodule = region.getFieldmodule()
        with ChangeManager(fieldmodule):
            annotationGroups = scaffold.generateBaseMesh(region, options)[0]
            fieldmodule.defineAllFaces()
            originalAnnotationGroups = copy.copy(annotationGroups)
            for annotationGroup in annotationGroups:
                annotationGroup.addSubelements()
            scaffold.defineFaceAnnotations(region, options, annotationGroups)
            for annotationGroup in annotationGroups:
                if annotationGroup not in originalAnnotationGroups:
                    annotationGroup.addSubelements()
        self.assertEqual(63, len(annotationGroups))

        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)
        self.assertEqual(192, mesh3d.getSize())
        mesh2d = fieldmodule.findMeshByDimension(2)
        self.assertEqual(640, mesh2d.getSize())
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(728, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(282, nodes.getSize())
        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        tol = 1.0E-6
        # check coordinates range, outside surface area and volume
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        # material_coordinates = fieldmodule.findFieldByName("lung coordinates").castFiniteElement()
        # self.assertTrue(material_coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(self, minimums, [-0.5150077744090888, -0.39797018819161917, 0.035835394723902014], tol)
        assertAlmostEqualList(self, maximums, [0.5150077744090888, 0.3290391566888997, 0.9617023416663923], tol)

        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            leftLungGroup = getAnnotationGroupForTerm(annotationGroups, get_lung_term("left lung"))
            leftLungSurfaceMeshGroup = leftLungGroup.getMeshGroup(mesh2d)
            self.assertTrue(leftLungSurfaceMeshGroup.isValid())

            surfaceAreaField = fieldmodule.createFieldMeshIntegral(one, coordinates, leftLungSurfaceMeshGroup)
            surfaceAreaField.setNumbersOfPoints(4)
            volumeField = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh3d)
            volumeField.setNumbersOfPoints(3)

        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        result, volume = volumeField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 4.88745261466682, delta=tol)
        self.assertAlmostEqual(volume, 0.301840862866859, delta=tol)

        # check some annotationGroups:
        expectedSizes3d = {
            "lung": 192,
            "left lung": 96,
            "right lung": 96,
            "upper lobe of left lung": 48,
            "lower lobe of left lung": 48,
            "upper lobe of right lung": 24,
            "middle lobe of right lung": 24,
            "lower lobe of right lung": 48,
        }
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name], size, name)

        expectedSizes2d = {
            "base of left lung surface": 16,
            "base of lower lobe of left lung surface": 8,
            "base of upper lobe of left lung surface": 8,
            "base of right lung surface": 16,
            "base of lower lobe of right lung surface": 8,
            "base of middle lobe of right lung surface": 8,
            "horizontal fissure of right lung": 10,
            "horizontal fissure of middle lobe of right lung": 10,
            "horizontal fissure of upper lobe of right lung": 10,
            'lateral surface of lower lobe of left lung': 16,
            'lateral surface of upper lobe of left lung': 16,
            'lateral surface of lower lobe of right lung': 16,
            'lateral surface of middle lobe of right lung': 8,
            'lateral surface of upper lobe of right lung': 8,
            'medial surface of lower lobe of left lung': 16,
            'medial surface of upper lobe of left lung': 16,
            'medial surface of lower lobe of right lung': 16,
            'medial surface of middle lobe of right lung': 8,
            'medial surface of upper lobe of right lung': 8,
            "oblique fissure of left lung": 20,
            "oblique fissure of right lung": 20,
            "oblique fissure of lower lobe of left lung": 20,
            "oblique fissure of upper lobe of left lung": 20,
            "oblique fissure of lower lobe of right lung": 20,
            "oblique fissure of middle lobe of right lung": 10,
            "oblique fissure of upper lobe of right lung": 10,
            "left lung surface": 64,
            "lower lobe of left lung surface": 52,
            "upper lobe of left lung surface": 52,
            "right lung surface": 64,
            "lower lobe of right lung surface": 52,
            "middle lobe of right lung surface": 36,
            "upper lobe of right lung surface": 36,
        }
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name], size, name)

        expectedSizes1d = {
            "antero-posterior edge of upper lobe of left lung": 8,
            "antero-posterior edge of upper lobe of right lung": 4
        }
        for name in expectedSizes1d:
            group = getAnnotationGroupForTerm(annotationGroups, [name, ""])
            size = group.getMeshGroup(mesh1d).getSize()
            self.assertEqual(expectedSizes1d[name], size, name)

        # refine 2x2x2 and check result
        # need to use original annotation groups to get temporaries
        annotationGroups = originalAnnotationGroups
        self.assertEqual(16, len(annotationGroups))  # including temporary groups

        refineRegion = region.createRegion()
        refineFieldmodule = refineRegion.getFieldmodule()
        options['Refine'] = True
        options['Refine number of elements'] = 2
        refineNumberOfElements = options['Refine number of elements']
        meshrefinement = MeshRefinement(region, refineRegion, annotationGroups)
        scaffold.refineMesh(meshrefinement, options)
        annotationGroups = meshrefinement.getAnnotationGroups()

        refineFieldmodule.defineAllFaces()
        oldAnnotationGroups = copy.copy(annotationGroups)
        for annotationGroup in annotationGroups:
            annotationGroup.addSubelements()
        scaffold.defineFaceAnnotations(refineRegion, options, annotationGroups)
        for annotation in annotationGroups:
            if annotation not in oldAnnotationGroups:
                annotationGroup.addSubelements()
        self.assertEqual(63, len(annotationGroups))

        mesh3d = refineFieldmodule.findMeshByDimension(3)
        self.assertEqual(1536, mesh3d.getSize())
        mesh2d = refineFieldmodule.findMeshByDimension(2)
        self.assertEqual(4864, mesh2d.getSize())
        mesh1d = refineFieldmodule.findMeshByDimension(1)
        self.assertEqual(5168, mesh1d.getSize())
        nodes = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(1842, nodes.getSize())
        datapoints = refineFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self.assertEqual(0, datapoints.getSize())

        # check some refined annotationGroups:
        for name in expectedSizes3d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh3d).getSize()
            self.assertEqual(expectedSizes3d[name] * (refineNumberOfElements ** 3), size, name)
        for name in expectedSizes2d:
            group = getAnnotationGroupForTerm(annotationGroups, get_lung_term(name))
            size = group.getMeshGroup(mesh2d).getSize()
            self.assertEqual(expectedSizes2d[name] * (refineNumberOfElements ** 2), size, name)

    def test_lung4_human(self):
        """
        Test creation of human lung scaffold lung4 with open fissures.
        """
        scaffold = MeshType_3d_lung4
        parameter_set_names = scaffold.getParameterSetNames()
        self.assertEqual(parameter_set_names,
                         ["Default", "Human 1 Coarse", "Human 1 Medium", "Human 1 Fine", "Ellipsoid"])
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        self.assertEqual(16, len(options))
        self.assertFalse(scaffold.checkOptions(options))

        context = Context("Test")
        region = context.getDefaultRegion()
        fieldmodule = region.getFieldmodule()
        self.assertTrue(region.isValid())
        annotation_groups, _  = scaffold.generateBaseMesh(region, options)
        base_annotation_groups = copy.copy(annotation_groups)
        self.assertEqual(25, len(base_annotation_groups))
        fieldmodule.defineAllFaces()
        for annotation_group in annotation_groups:
            annotation_group.addSubelements()
        scaffold.defineFaceAnnotations(region, options, annotation_groups)
        for annotation_group in annotation_groups:
            if annotation_group not in base_annotation_groups:
                annotation_group.addSubelements()
        self.assertEqual(76, len(annotation_groups))

        fieldcache = fieldmodule.createFieldcache()
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(416, nodes.getSize())
        intersection_group_names = [
            ("lower lobe of left lung", "upper lobe of left lung"),
            ("lower lobe of right lung", "middle lobe of right lung"),
            ("lower lobe of right lung", "upper lobe of right lung"),
            ("middle lobe of right lung", "upper lobe of right lung")
        ]
        # check number of common nodes at hilum
        with ChangeManager(fieldmodule):
            for group_name1, group_name2 in intersection_group_names:
                group = fieldmodule.createFieldGroup()
                nodeset_group = group.createNodesetGroup(nodes)
                nodeset_group.addNodesConditional(
                    fieldmodule.createFieldAnd(fieldmodule.findFieldByName(group_name1),
                                               fieldmodule.findFieldByName(group_name2)))
                self.assertEqual(3, nodeset_group.getSize())
                del nodeset_group
                del group

        expected_mesh_sizes = {
            'mesh3d': (232, 0.23531518999948317),
            'mesh2d': (830, 9.322931842916137),
            'mesh1d': (1003, 110.4277660475001),
            'left lung.mesh3d': (116, 0.11765752113724519),
            'lower lobe of left lung.mesh3d': (44, 0.055984781964438096),
            'upper lobe of left lung.mesh3d': (72, 0.06167273917280671),
            'right lung.mesh3d': (116, 0.11765766886223888),
            'lower lobe of right lung.mesh3d': (44, 0.05598478196443805),
            'middle lobe of right lung.mesh3d': (24, 0.015908839863426272),
            'upper lobe of right lung.mesh3d': (48, 0.04576404703437447),
            'oblique fissure of lower lobe of left lung.mesh2d': (18, 0.23573085010104664),
            'oblique fissure of upper lobe of left lung.mesh2d': (20, 0.26116146713690147),
            # base of middle lobe includes medial surface
            'base of middle lobe of right lung surface.mesh2d': (10, 0.11847510573774353),
            'horizontal fissure of middle lobe of right lung.mesh2d': (10, 0.08506039275120454),
            'horizontal fissure of upper lobe of right lung.mesh2d': (10, 0.08506039275120465),
            'oblique fissure of lower lobe of right lung.mesh2d': (18, 0.23573085010104675),
            'oblique fissure of middle lobe of right lung.mesh2d': (10, 0.11847510573774353),
            'oblique fissure of upper lobe of right lung.mesh2d': (10, 0.14268636139915797),
            'lateral edge of base of lower lobe of left lung.mesh1d': (3, 0.6163117802059481),
            'medial edge of base of lower lobe of left lung.mesh1d': (3, 0.5174206016915209),
            'posterior edge of lower lobe of left lung.mesh1d': (6, 0.7217563496699031),
            'lateral edge of base of lower lobe of right lung.mesh1d': (3, 0.6163117802059481),
            'medial edge of base of lower lobe of right lung.mesh1d': (3, 0.5174206016915209),
            # base of middle lobe includes anterior edge
            'lateral edge of base of middle lobe of right lung.mesh1d': (7, 1.0751126241895612),
            'medial edge of base of middle lobe of right lung.mesh1d': (7, 0.8432549927571157),
            'posterior edge of lower lobe of right lung.mesh1d': (6, 0.7217563496699028)
        }
        TOL = 1.0E-6
        with ChangeManager(fieldmodule):
            one = fieldmodule.createFieldConstant(1.0)
            for mesh_name, expected_sizes in expected_mesh_sizes.items():
                expected_size, expected_val = expected_sizes
                mesh_group = fieldmodule.findMeshByName(mesh_name)
                size = mesh_group.getSize()
                self.assertEqual(size, expected_size, msg=mesh_name)
                # volume/area/length
                val_integral = fieldmodule.createFieldMeshIntegral(one, coordinates, mesh_group)
                val_integral.setNumbersOfPoints(4)
                result, val = val_integral.evaluateReal(fieldcache, 1)
                self.assertEqual(result, RESULT_OK)
                self.assertAlmostEqual(val, expected_val, msg=mesh_name, delta=TOL)

        # check markers
        marker_group = fieldmodule.findFieldByName("marker").castGroup()
        marker_nodes = marker_group.getNodesetGroup(nodes)
        self.assertEqual(9, marker_nodes.getSize())
        for annotation_group in annotation_groups:
            if annotation_group.isMarker():
                group_field = annotation_group.getGroup()
                self.assertEqual(group_field, marker_group)
                null_group = fieldmodule.findFieldByName(annotation_group.getName()).castGroup()
                self.assertFalse(null_group.isValid())

        # refine 2x2x2 and check results
        refine_region = region.createRegion()
        refine_fieldmodule = refine_region.getFieldmodule()
        options['Refine'] = True
        options['Refine number of elements'] = 2
        meshrefinement = MeshRefinement(region, refine_region, base_annotation_groups)
        scaffold.refineMesh(meshrefinement, options)
        refine_annotation_groups = meshrefinement.getAnnotationGroups()
        self.assertEqual(25, len(refine_annotation_groups))
        refine_fieldmodule.defineAllFaces()
        for annotation_group in refine_annotation_groups:
            annotation_group.addSubelements()
        old_annotation_groups = copy.copy(refine_annotation_groups)
        scaffold.defineFaceAnnotations(refine_region, options, refine_annotation_groups)
        for annotation_group in refine_annotation_groups:
            if annotation_group not in old_annotation_groups:
                annotation_group.addSubelements()
        self.assertEqual(76, len(refine_annotation_groups))

        refine_fieldcache = refine_fieldmodule.createFieldcache()
        refine_coordinates = refine_fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(refine_coordinates.isValid())
        refine_nodes = refine_fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(2481, refine_nodes.getSize())
        # check number of common nodes at hilum
        with ChangeManager(refine_fieldmodule):
            for group_name1, group_name2 in intersection_group_names:
                group = refine_fieldmodule.createFieldGroup()
                nodeset_group = group.createNodesetGroup(refine_nodes)
                nodeset_group.addNodesConditional(
                    refine_fieldmodule.createFieldAnd(refine_fieldmodule.findFieldByName(group_name1),
                                                      refine_fieldmodule.findFieldByName(group_name2)))
                self.assertEqual(5, nodeset_group.getSize())
                del nodeset_group
                del group

        expected_refine_mesh_sizes = {
            'mesh3d': (1856, 0.230336480247037),
            'mesh2d': (6104, 16.332327494330084),
            'mesh1d': (6718, 358.30880646477294),
            'left lung.mesh3d': (928, 0.1151678888674793),
            'lower lobe of left lung.mesh3d': (352, 0.0548394588491132),
            'upper lobe of left lung.mesh3d': (576, 0.060328430018365395),
            'right lung.mesh3d': (928, 0.11516859137956122),
            'lower lobe of right lung.mesh3d': (352, 0.054839458849113155),
            'middle lobe of right lung.mesh3d': (192, 0.015593500770872093),
            'upper lobe of right lung.mesh3d': (384, 0.04473563175957532),
            'oblique fissure of lower lobe of left lung.mesh2d': (72, 0.23303251949030654),
            'oblique fissure of upper lobe of left lung.mesh2d': (80, 0.25693609525575456),
            # base of middle lobe includes medial surface
            'base of middle lobe of right lung surface.mesh2d': (40, 0.11647413723834865),
            'horizontal fissure of middle lobe of right lung.mesh2d': (40, 0.08446588867418067),
            'horizontal fissure of upper lobe of right lung.mesh2d': (40, 0.08446588867418048),
            'oblique fissure of lower lobe of right lung.mesh2d': (72, 0.23303251949030673),
            'oblique fissure of middle lobe of right lung.mesh2d': (40, 0.11647413723834865),
            'oblique fissure of upper lobe of right lung.mesh2d': (40, 0.14046195801740563),
            'lateral edge of base of lower lobe of left lung.mesh1d': (6, 0.6113060310659696),
            'medial edge of base of lower lobe of left lung.mesh1d': (6, 0.5145764175197279),
            'posterior edge of lower lobe of left lung.mesh1d': (12, 0.7213630968439506),
            'lateral edge of base of lower lobe of right lung.mesh1d': (6, 0.6113060310659696),
            'medial edge of base of lower lobe of right lung.mesh1d': (6, 0.5145764175197279),
            # base of middle lobe includes anterior edge
            'lateral edge of base of middle lobe of right lung.mesh1d': (14, 1.0700457132854544),
            'medial edge of base of middle lobe of right lung.mesh1d': (14, 0.8398626858403571),
            'posterior edge of lower lobe of right lung.mesh1d': (12, 0.7213630968439503)
        }
        with ChangeManager(refine_fieldmodule):
            one = refine_fieldmodule.createFieldConstant(1.0)
            for mesh_name, expected_sizes in expected_refine_mesh_sizes.items():
                expected_size, expected_val = expected_sizes
                mesh_group = refine_fieldmodule.findMeshByName(mesh_name)
                size = mesh_group.getSize()
                self.assertEqual(size, expected_size, msg=mesh_name)
                # volume/area/length
                val_integral = refine_fieldmodule.createFieldMeshIntegral(one, refine_coordinates, mesh_group)
                val_integral.setNumbersOfPoints(4)
                result, val = val_integral.evaluateReal(refine_fieldcache, 1)
                self.assertEqual(result, RESULT_OK)
                self.assertAlmostEqual(val, expected_val, msg=mesh_name, delta=TOL)

        # check refine markers
        refine_marker_group = refine_fieldmodule.findFieldByName("marker").castGroup()
        refine_marker_nodes = refine_marker_group.getNodesetGroup(refine_nodes)
        refine_lung_nodes = refine_fieldmodule.findFieldByName("lung").castGroup().getNodesetGroup(refine_nodes)
        self.assertEqual(9, refine_marker_nodes.getSize())
        for annotation_group in refine_annotation_groups:
            if annotation_group.isMarker():
                marker_node = annotation_group.getMarkerNode()
                self.assertTrue(marker_node.isValid())
                group_field = annotation_group.getGroup()
                self.assertEqual(group_field, refine_marker_group)
                self.assertTrue(refine_marker_nodes.containsNode(marker_node))
                self.assertTrue(refine_lung_nodes.containsNode(marker_node))
                null_group = refine_fieldmodule.findFieldByName(annotation_group.getName()).castGroup()
                self.assertFalse(null_group.isValid())

    def test_lung4_human_left(self):
        """
        Test creation of human lung scaffold lung4 left side only, that objects have expected identifiers.
        """
        scaffold = MeshType_3d_lung4
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        options["Number of elements oblique"] = 4  # minimum, for speed
        options["Right lung"] = False

        context = Context("Test")
        region = context.getDefaultRegion()
        fieldmodule = region.getFieldmodule()
        self.assertTrue(region.isValid())
        annotation_groups, _  = scaffold.generateMesh(region, options)
        self.assertEqual(31, len(annotation_groups))
        # check no right annotations
        for annotation_group in annotation_groups:
            name = annotation_group.getName()
            self.assertFalse("right" in name)
        # get ranges of identifiers in meshes and nodes
        expected_domain_count_ranges = [
            ("mesh3d", 44, [[1, 44]]),
            ("mesh2d", 164, [[1, 164]]),
            ("mesh1d", 208, [[1, 208]]),
            ("nodes", 93, [[1, 89], [187, 190]]),
            ("marker.nodes", 4, [[187, 190]])
        ]
        for domain_name, expected_count, expected_ranges in expected_domain_count_ranges:
            if "mesh" in domain_name:
                domain = fieldmodule.findMeshByName(domain_name)
                ranges = mesh_group_to_identifier_ranges(domain)
            else:
                domain = fieldmodule.findNodesetByName(domain_name)
                ranges = nodeset_group_to_identifier_ranges(domain)
            self.assertEqual(domain.getSize(), expected_count)
            self.assertEqual(ranges, expected_ranges)

    def test_lung4_human_right(self):
        """
        Test creation of human lung scaffold lung4 left side only, that objects have expected identifiers.
        """
        scaffold = MeshType_3d_lung4
        options = scaffold.getDefaultOptions("Human 1 Coarse")
        options["Number of elements oblique"] = 4  # minimum, for speed
        options["Left lung"] = False

        context = Context("Test")
        region = context.getDefaultRegion()
        fieldmodule = region.getFieldmodule()
        self.assertTrue(region.isValid())
        annotation_groups, _  = scaffold.generateMesh(region, options)
        self.assertEqual(46, len(annotation_groups))
        # check no right annotations
        for annotation_group in annotation_groups:
            name = annotation_group.getName()
            self.assertFalse("left" in name)
        # get ranges of identifiers in meshes and nodes
        expected_domain_count_ranges = [
            ("mesh3d", 44, [[45, 88]]),
            ("mesh2d", 170, [[165, 334]]),
            ("mesh1d", 222, [[209, 430]]),
            ("nodes", 102, [[90, 186], [191, 195]]),
            ("marker.nodes", 5, [[191, 195]])
        ]
        for domain_name, expected_count, expected_ranges in expected_domain_count_ranges:
            if "mesh" in domain_name:
                domain = fieldmodule.findMeshByName(domain_name)
                ranges = mesh_group_to_identifier_ranges(domain)
            else:
                domain = fieldmodule.findNodesetByName(domain_name)
                ranges = nodeset_group_to_identifier_ranges(domain)
            self.assertEqual(domain.getSize(), expected_count, msg=domain_name)
            self.assertEqual(ranges, expected_ranges, msg=domain_name)


if __name__ == "__main__":
    unittest.main()
