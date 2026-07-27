import math
import unittest

from cmlibs.maths.vectorops import dot, magnitude, mult, normalize, sub
from cmlibs.utils.zinc.field import find_or_create_field_coordinates, find_or_create_field_group
from cmlibs.utils.zinc.finiteelement import evaluateFieldNodesetRange
from cmlibs.utils.zinc.group import identifier_ranges_from_string, identifier_ranges_to_string, \
    mesh_group_add_identifier_ranges, mesh_group_to_identifier_ranges, \
    nodeset_group_add_identifier_ranges, nodeset_group_to_identifier_ranges
from cmlibs.zinc.context import Context
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from cmlibs.zinc.result import RESULT_OK
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, getAnnotationMarkerNameField
from scaffoldmaker.meshtypes.meshtype_1d_network_layout1 import MeshType_1d_network_layout1
from scaffoldmaker.meshtypes.meshtype_3d_box1 import MeshType_3d_box1
from scaffoldmaker.meshtypes.meshtype_3d_brainstem import MeshType_3d_brainstem1
from scaffoldmaker.meshtypes.meshtype_3d_heartatria1 import MeshType_3d_heartatria1
from scaffoldmaker.meshtypes.meshtype_3d_stomach1 import MeshType_3d_stomach1
from scaffoldmaker.scaffoldpackage import ScaffoldPackage
from scaffoldmaker.scaffolds import Scaffolds
from scaffoldmaker.utils.eft_utils import determineTricubicHermiteEft
from scaffoldmaker.utils.geometry import getEllipsoidPlaneA, getEllipsoidPolarCoordinatesFromPosition, \
    getEllipsoidPolarCoordinatesTangents
from scaffoldmaker.utils.interpolation import computeCubicHermiteSideCrossDerivatives, evaluateCoordinatesOnCurve, \
    getCubicHermiteCurvesLength, getNearestLocationBetweenCurves, getNearestLocationOnCurve, interpolateCubicHermite
from scaffoldmaker.utils.tracksurface import TrackSurface, TrackSurfacePosition
from scaffoldmaker.utils.tubenetworkmesh import (
    TubeNetworkMeshSegment, getPathRawTubeCoordinates, resampleTubeCoordinates)
from scaffoldmaker.utils.zinc_utils import generate_curve_mesh, get_nodeset_path_ordered_field_parameters

from testutils import assertAlmostEqualList


class GeneralScaffoldTestCase(unittest.TestCase):

    def test_transformation(self):
        """
        Test transformation of a box scaffold with scaffold package.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_box1)

        tmpScale = scaffoldPackage.getScale()
        TOL = 1.0E-7
        FINE_TOL = 1.0E-12
        assertAlmostEqualList(self, tmpScale, [1.0, 1.0, 1.0], delta=FINE_TOL)
        newScale = [2.0, 1.5, 0.5]
        scaffoldPackage.setScale(newScale)
        tmpScale = scaffoldPackage.getScale()
        assertAlmostEqualList(self, tmpScale, newScale, delta=FINE_TOL)

        tmpRotation = scaffoldPackage.getRotation()
        assertAlmostEqualList(self, tmpRotation, [0.0, 0.0, 0.0], delta=FINE_TOL)
        newRotation = [30.0, -10.0, 90.0]
        scaffoldPackage.setRotation(newRotation)
        tmpRotation = scaffoldPackage.getRotation()
        assertAlmostEqualList(self, tmpRotation, newRotation, delta=FINE_TOL)

        tmpTranslation = scaffoldPackage.getTranslation()
        assertAlmostEqualList(self, tmpTranslation, [0.0, 0.0, 0.0], delta=FINE_TOL)
        newTranslation = [0.5, 1.2, -0.1]
        scaffoldPackage.setTranslation(newTranslation)
        tmpTranslation = scaffoldPackage.getTranslation()
        assertAlmostEqualList(self, tmpTranslation, newTranslation, delta=FINE_TOL)

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())

        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(8, nodes.getSize())

        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())
        minimums, maximums = evaluateFieldNodesetRange(coordinates, nodes)
        assertAlmostEqualList(
            self, minimums, [2.744244002293470e-01, 6.367511648575830e-01, -1.000000000000000e-01], delta=TOL)
        assertAlmostEqualList(
            self, maximums, [2.455737063904887e+00, 2.184807753012208e+00,  1.724507984852172e+00], delta=TOL)

        node = nodes.findNodeByIdentifier(8)
        self.assertTrue(node.isValid())
        fieldcache = fieldmodule.createFieldcache()
        fieldcache.setNode(node)
        result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
        self.assertEqual(RESULT_OK, result)
        assertAlmostEqualList(
            self, x, [2.230161464134234e+00,  1.621558917869791e+00,  1.724507984852172e+00], delta=TOL)
        # derivative magnitudes must also equal scale
        result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
        self.assertEqual(RESULT_OK, result)
        assertAlmostEqualList(
            self, d1, [1.705737064039425e+00,  9.848077530127952e-01,  3.472963553408093e-01], delta=TOL)
        self.assertAlmostEqual(newScale[0], magnitude(d1), delta=TOL)
        result, d2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
        self.assertEqual(RESULT_OK, result)
        assertAlmostEqualList(
            self, d2, [-2.255755995328457e-01, -1.302361332111701e-01,  1.477211629352659e+00], delta=TOL)
        self.assertAlmostEqual(newScale[1], magnitude(d2), delta=TOL)
        result, d3 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)
        self.assertEqual(RESULT_OK, result)
        assertAlmostEqualList(
            self, d3, [2.499999998128999e-01, -4.330127019169794e-01,  0.000000000000000e+00], delta=TOL)
        self.assertAlmostEqual(newScale[2], magnitude(d3), delta=TOL)

    def test_user_annotation_groups(self):
        """
        Test user annotation group on heartatria1 scaffold with scaffold package.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_heartatria1)
        # can't add user annotation groups until generate is called()
        self.assertRaises(AssertionError, lambda: scaffoldPackage.createUserAnnotationGroup())

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        fieldmodule = region.getFieldmodule()

        scaffoldPackage.generate(region)

        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(33, len(annotationGroups))

        endocardium_of_la = scaffoldPackage.findAnnotationGroupByName('left atrium endocardium')
        self.assertTrue(isinstance(endocardium_of_la, AnnotationGroup))
        self.assertFalse(scaffoldPackage.isUserAnnotationGroup(endocardium_of_la))
        self.assertFalse(scaffoldPackage.deleteAnnotationGroup(endocardium_of_la))  # can't delete auto annotation group

        annotationGroup1 = scaffoldPackage.createUserAnnotationGroup()
        self.assertEqual('group1', annotationGroup1.getName())  # default name
        self.assertEqual('None', annotationGroup1.getId())
        self.assertTrue(scaffoldPackage.isUserAnnotationGroup(annotationGroup1))
        self.assertEqual(-1, annotationGroup1.getDimension())  # -1 = empty
        group = annotationGroup1.getGroup()
        self.assertTrue(group.isValid())
        mesh2d = fieldmodule.findMeshByDimension(2)
        meshGroup = group.createMeshGroup(mesh2d)
        mesh_group_add_identifier_ranges(meshGroup, [[1, 2], [4, 4]])
        self.assertEqual(3, meshGroup.getSize())
        self.assertEqual(2, annotationGroup1.getDimension())
        identifier_ranges_string = identifier_ranges_to_string(mesh_group_to_identifier_ranges(meshGroup))
        self.assertEqual('1-2,4', identifier_ranges_string)

        annotationGroup2 = scaffoldPackage.createUserAnnotationGroup(('bob', 'BOB:1'))
        self.assertEqual('bob', annotationGroup2.getName())
        self.assertEqual('BOB:1', annotationGroup2.getId())
        self.assertTrue(scaffoldPackage.isUserAnnotationGroup(annotationGroup2))
        group = annotationGroup2.getGroup()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodesetGroup = group.createNodesetGroup(nodes)
        nodeset_group_add_identifier_ranges(nodesetGroup, identifier_ranges_from_string('1,3-5,7'))
        self.assertEqual(5, nodesetGroup.getSize())
        self.assertEqual(0, annotationGroup2.getDimension())
        identifier_ranges_string = identifier_ranges_to_string(nodeset_group_to_identifier_ranges(nodesetGroup))
        self.assertEqual('1,3-5,7', identifier_ranges_string)

        annotationGroup3 = scaffoldPackage.createUserAnnotationGroup()
        self.assertEqual('group2', annotationGroup3.getName())  # default name
        self.assertTrue(scaffoldPackage.isUserAnnotationGroup(annotationGroup3))
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(36, len(annotationGroups))

        # rename group1 to fred
        self.assertTrue(annotationGroup1.setName('fred'))
        self.assertTrue(annotationGroup1.setId('FRED:1'))
        self.assertEqual('fred', annotationGroup1.getName())
        self.assertEqual('FRED:1', annotationGroup1.getId())

        self.assertTrue(scaffoldPackage.deleteAnnotationGroup(annotationGroup3))
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(35, len(annotationGroups))

        # test serialisation
        dct = scaffoldPackage.toDict()
        self.assertEqual('3D Heart Atria 1', dct['scaffoldTypeName'])
        scaffoldType = Scaffolds().findScaffoldTypeByName(dct['scaffoldTypeName'])

        scaffoldPackage2 = ScaffoldPackage(scaffoldType, dct)
        region2 = context.createRegion()
        fieldmodule2 = region2.getFieldmodule()

        scaffoldPackage2.generate(region2)

        annotationGroups2 = scaffoldPackage2.getAnnotationGroups()
        self.assertEqual(35, len(annotationGroups2))

        annotationGroup1 = scaffoldPackage2.findAnnotationGroupByName('fred')
        self.assertEqual('fred', annotationGroup1.getName())
        self.assertEqual('FRED:1', annotationGroup1.getId())
        self.assertTrue(scaffoldPackage2.isUserAnnotationGroup(annotationGroup1))
        self.assertEqual(2, annotationGroup1.getDimension())
        mesh2d2 = fieldmodule2.findMeshByDimension(2)
        meshGroup2 = annotationGroup1.getMeshGroup(mesh2d2)
        self.assertEqual(3, meshGroup2.getSize())
        identifier_ranges_string = identifier_ranges_to_string(mesh_group_to_identifier_ranges(meshGroup2))
        self.assertEqual('1-2,4', identifier_ranges_string)

        annotationGroup2 = scaffoldPackage2.findAnnotationGroupByName('bob')
        self.assertEqual('bob', annotationGroup2.getName())
        self.assertEqual('BOB:1', annotationGroup2.getId())
        self.assertTrue(scaffoldPackage2.isUserAnnotationGroup(annotationGroup2))
        self.assertEqual(0, annotationGroup2.getDimension())
        nodes2 = fieldmodule2.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodesetGroup2 = annotationGroup2.getNodesetGroup(nodes2)
        self.assertEqual(5, nodesetGroup.getSize())
        identifier_ranges_string = identifier_ranges_to_string(nodeset_group_to_identifier_ranges(nodesetGroup2))
        self.assertEqual('1,3-5,7', identifier_ranges_string)

    def test_user_marker_points(self):
        """
        Test user marker point on brainstem1 scaffold which defined "brainstem coordinates".
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_brainstem1)

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        fieldmodule = region.getFieldmodule()
        mesh = fieldmodule.findMeshByDimension(3)
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        scaffoldPackage.generate(region)
        # 1 higher than last node in scaffold. Make marker nodes from this number
        nextNodeIdentifier = scaffoldPackage.getNextNodeIdentifier()

        brainstemCoordinatesField = fieldmodule.findFieldByName("brainstem coordinates")
        self.assertTrue(brainstemCoordinatesField.isValid())

        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(18, len(annotationGroups))
        TOL = 1.0E-6  # coarse to handle find xi tolerances

        # check a built-in non-marker annotation group
        ponsGroup = scaffoldPackage.findAnnotationGroupByName('pons')
        self.assertFalse(ponsGroup.isMarker())
        self.assertRaises(AssertionError, lambda: ponsGroup.getMarkerMaterialCoordinates())
        self.assertRaises(AssertionError, lambda: ponsGroup.getMarkerLocation())
        self.assertRaises(AssertionError, lambda: ponsGroup.createMarkerNode(nextNodeIdentifier))

        # check a built-in marker annotation group
        brainstemVentralCranialPointGroup = \
            scaffoldPackage.findAnnotationGroupByName('brainstem ventral midline cranial point')
        self.assertIsNotNone(brainstemVentralCranialPointGroup)
        self.assertTrue(brainstemVentralCranialPointGroup.isMarker())
        brainstemCoordinatesFieldOut, brainstemCoordinatesValueOut = \
            brainstemVentralCranialPointGroup.getMarkerMaterialCoordinates()
        self.assertEqual(brainstemCoordinatesFieldOut, brainstemCoordinatesField)
        assertAlmostEqualList(self, [0.0, -1.0, 8.0], brainstemCoordinatesValueOut, delta=TOL)
        elementOut, xiOut = brainstemVentralCranialPointGroup.getMarkerLocation()
        self.assertEqual(235, elementOut.getIdentifier())
        assertAlmostEqualList(self, [1.0, 1.0, 0.0], xiOut, delta=TOL)
        self.assertRaises(
            AssertionError, lambda: brainstemVentralCranialPointGroup.createMarkerNode(nextNodeIdentifier))

        # check a non-existant annotation group
        bobGroup = scaffoldPackage.findAnnotationGroupByName("bob")
        self.assertIsNone(bobGroup)

        # now make a marker annotation named "bob" at the default location
        bobGroup = scaffoldPackage.createUserAnnotationGroup(('bob', 'BOB:1'), isMarker=True)
        self.assertTrue(scaffoldPackage.isUserAnnotationGroup(bobGroup))
        self.assertFalse(bobGroup.isMarker())
        node = bobGroup.createMarkerNode(nextNodeIdentifier)
        bobNodeIdentifier = node.getIdentifier()
        self.assertEqual(nextNodeIdentifier, bobNodeIdentifier)
        brainstemCoordinatesFieldOut, brainstemCoordinatesValueOut = bobGroup.getMarkerMaterialCoordinates()
        self.assertIsNone(brainstemCoordinatesFieldOut)
        self.assertIsNone(brainstemCoordinatesValueOut)
        elementOut, xiOut = bobGroup.getMarkerLocation()
        self.assertEqual(1, elementOut.getIdentifier())
        assertAlmostEqualList(self, [0.0, 0.0, 0.0], xiOut, delta=TOL)
        # now request brainstem coordinates and let the annotation group determine its values from element:xi
        bobGroup.setMarkerMaterialCoordinates(brainstemCoordinatesField)
        brainstemCoordinatesFieldOut, brainstemCoordinatesValueOut = bobGroup.getMarkerMaterialCoordinates()
        self.assertEqual(brainstemCoordinatesFieldOut, brainstemCoordinatesField)
        # these should be precisely cos(45) but are not due to ellipse approximations
        cos45 = math.cos(math.pi / 4.0)
        assertAlmostEqualList(self, [cos45, -cos45, 0], brainstemCoordinatesValueOut, delta=TOL)
        # set element:xi location and check brainstem coordinates change
        bobGroup.setMarkerLocation(mesh.findElementByIdentifier(33), [0.0, 0.5, 0.0])
        brainstemCoordinatesFieldOut, brainstemCoordinatesValueOut = bobGroup.getMarkerMaterialCoordinates()
        self.assertEqual(brainstemCoordinatesFieldOut, brainstemCoordinatesField)
        assertAlmostEqualList(self, [cos45, -cos45, 1.5], brainstemCoordinatesValueOut, delta=TOL)
        # assign brainstem coordinates and check element:xi has moved
        bobGroup.setMarkerMaterialCoordinates(brainstemCoordinatesField, [-0.1, -0.5, 2.2])
        brainstemCoordinatesFieldOut, brainstemCoordinatesValueOut = bobGroup.getMarkerMaterialCoordinates()
        self.assertEqual(brainstemCoordinatesFieldOut, brainstemCoordinatesField)
        assertAlmostEqualList(self, [-0.1, -0.5, 2.2], brainstemCoordinatesValueOut, delta=TOL)
        elementOut, xiOut = bobGroup.getMarkerLocation()
        self.assertEqual(82, elementOut.getIdentifier())
        assertAlmostEqualList(self, [0.30536897479419056, 0.2, 0.486013009421796], xiOut, delta=TOL)

        # now make a marker annotation named "fred" with brainstem coordinates from the start
        fredGroup = scaffoldPackage.createUserAnnotationGroup(('fred', 'FRED:1'))
        # AnnotationGroup.createMarkerNode increments nextNodeIdentifier to one not used by existing node
        node = fredGroup.createMarkerNode(nextNodeIdentifier, brainstemCoordinatesField, [0.5, 0.5, 4])
        fredNodeIdentifier = node.getIdentifier()
        self.assertEqual(nextNodeIdentifier + 1, fredNodeIdentifier)
        del node
        brainstemCoordinatesFieldOut, brainstemCoordinatesValueOut = fredGroup.getMarkerMaterialCoordinates()
        self.assertEqual(brainstemCoordinatesFieldOut, brainstemCoordinatesField)
        assertAlmostEqualList(self, [0.5, 0.5, 4], brainstemCoordinatesValueOut, delta=TOL)
        elementOut, xiOut = fredGroup.getMarkerLocation()
        self.assertEqual(105, elementOut.getIdentifier())
        assertAlmostEqualList(self, [0.3452673123795837, 1.0, 0.6634646029995092], xiOut, delta=TOL)

        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(20, len(annotationGroups))

        # test renaming group assigns the marker_name field:
        fredGroup.setName("freddy")
        self.assertEqual("freddy", fredGroup.getName())
        node = fredGroup.getMarkerNode()
        markerName = getAnnotationMarkerNameField(fieldmodule)
        fieldcache = fieldmodule.createFieldcache()
        fieldcache.setNode(node)
        markerNodeName = markerName.evaluateString(fieldcache)
        self.assertEqual("freddy", markerNodeName)
        del node

        # test deleting a marker annotation group
        scaffoldPackage.deleteAnnotationGroup(fredGroup)

        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(19, len(annotationGroups))

        # check node fred has gone
        node = nodes.findNodeByIdentifier(fredNodeIdentifier)
        self.assertFalse(node.isValid())

        # re-recreate fred with just element:xi location
        fredGroup = scaffoldPackage.createUserAnnotationGroup(('fred', 'FRED:1'), isMarker=True)
        element = mesh.findElementByIdentifier(105)
        node = fredGroup.createMarkerNode(nextNodeIdentifier, element=element,
                                          xi=[0.3452673123795837, 1.0, 0.6634646029995092])
        self.assertEqual(fredNodeIdentifier, node.getIdentifier())
        del node
        elementOut, xiOut = fredGroup.getMarkerLocation()
        self.assertEqual(105, elementOut.getIdentifier())
        assertAlmostEqualList(self, [0.3452673123795837, 1.0, 0.6634646029995092], xiOut, delta=TOL)

        # check the total number of groups
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(20, len(annotationGroups))

        # test serialisation
        dct = scaffoldPackage.toDict()
        scaffoldType = Scaffolds().findScaffoldTypeByName(dct['scaffoldTypeName'])

        scaffoldPackage2 = ScaffoldPackage(scaffoldType, dct)
        region2 = context.createRegion()
        fieldmodule2 = region2.getFieldmodule()

        scaffoldPackage2.generate(region2)

        brainstemCoordinatesField2 = fieldmodule2.findFieldByName("brainstem coordinates")
        self.assertTrue(brainstemCoordinatesField2.isValid())

        annotationGroups2 = scaffoldPackage2.getAnnotationGroups()
        self.assertEqual(20, len(annotationGroups2))

        # check user markers have been defined correctly for scaffoldPackage2

        bobGroup2 = scaffoldPackage2.findAnnotationGroupByName('bob')
        brainstemCoordinatesFieldOut, brainstemCoordinatesValueOut = bobGroup2.getMarkerMaterialCoordinates()
        self.assertEqual(brainstemCoordinatesFieldOut, brainstemCoordinatesField2)
        assertAlmostEqualList(self, [-0.1, -0.5, 2.2], brainstemCoordinatesValueOut, delta=TOL)
        elementOut, xiOut = bobGroup2.getMarkerLocation()
        self.assertEqual(82, elementOut.getIdentifier())
        assertAlmostEqualList(self, [0.30536897479419256, 0.20000000000000018, 0.48601300942181097], xiOut, delta=TOL)

        fredGroup2 = scaffoldPackage2.findAnnotationGroupByName('fred')
        brainstemCoordinatesFieldOut, brainstemCoordinatesValueOut = fredGroup2.getMarkerMaterialCoordinates()
        self.assertIsNone(brainstemCoordinatesFieldOut)
        self.assertIsNone(brainstemCoordinatesValueOut)
        elementOut, xiOut = fredGroup2.getMarkerLocation()
        self.assertEqual(105, elementOut.getIdentifier())
        assertAlmostEqualList(self, [0.3452673123795837, 1.0, 0.6634646029995092], xiOut, delta=TOL)

    def test_deletion(self):
        """
        Test deletion of element ranges on a stomach scaffold with scaffold package.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_3d_stomach1)
        TOL = 1.0E-7
        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        fieldmodule = region.getFieldmodule()
        fieldcache = fieldmodule.createFieldcache()
        mesh3d = fieldmodule.findMeshByDimension(3)
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

        scaffoldPackage.generate(region)
        coordinates = fieldmodule.findFieldByName("coordinates")
        stomachCoordinatesField = fieldmodule.findFieldByName("stomach coordinates")

        # Check existing markers
        bodyAntrumGCLuminal = \
            scaffoldPackage.findAnnotationGroupByName('body-antrum junction along the greater curvature on luminal '
                                                      'surface')
        self.assertTrue(isinstance(bodyAntrumGCLuminal, AnnotationGroup))
        self.assertTrue(bodyAntrumGCLuminal.isMarker())
        bodyAntrumGCLuminalNode = bodyAntrumGCLuminal.getMarkerNode()
        bodyAntrumGCLuminalNodeIdentifier = bodyAntrumGCLuminalNode.getIdentifier()
        bodyAntrumGCLuminalMaterialCoordinates = bodyAntrumGCLuminal.getMarkerMaterialCoordinates()[1]
        element, xi = bodyAntrumGCLuminal.getMarkerLocation()
        fieldcache.setMeshLocation(element, xi)
        bodyAntrumGCLuminalCoordinates = coordinates.evaluateReal(fieldcache, coordinates.getNumberOfComponents())[1]

        # check a non-existent annotation group
        bobGroup = scaffoldPackage.findAnnotationGroupByName("bob")
        self.assertIsNone(bobGroup)

        # now make a marker annotation named "bob" at a location to be deleted
        bobGroup = scaffoldPackage.createUserAnnotationGroup(('bob', 'BOB:1'), isMarker=True)
        self.assertTrue(scaffoldPackage.isUserAnnotationGroup(bobGroup))
        self.assertFalse(bobGroup.isMarker())
        nextNodeIdentifier = scaffoldPackage.getNextNodeIdentifier()
        node = bobGroup.createMarkerNode(nextNodeIdentifier, stomachCoordinatesField, [-0.165034, 0.12313, 0.484541])
        bobNodeIdentifier = node.getIdentifier()
        self.assertEqual(nextNodeIdentifier, bobNodeIdentifier)
        stomachCoordinatesFieldOut, stomachCoordinatesValueOut = bobGroup.getMarkerMaterialCoordinates()
        self.assertEqual([-0.165034, 0.12313, 0.484541], stomachCoordinatesValueOut)

        # check bob is made before deletion
        bob = scaffoldPackage.findAnnotationGroupByName('bob')
        self.assertTrue(isinstance(bob, AnnotationGroup))
        self.assertTrue(bob.isMarker())

        # make a user marker named "fred" at a location not being deleted, but without material coordinates
        # a bug was causing such markers to be deleted everywhere if any elements were deleted
        fredGroup = scaffoldPackage.createUserAnnotationGroup(('fred', 'FRED:1'))  # not setting isMarker here
        self.assertTrue(scaffoldPackage.isUserAnnotationGroup(fredGroup))
        self.assertFalse(fredGroup.isMarker())
        element300 = mesh3d.findElementByIdentifier(300)
        node = fredGroup.createMarkerNode(nextNodeIdentifier, element=element300, xi=[0.5, 0.5, 1.0])
        fredNodeIdentifier = node.getIdentifier()
        self.assertEqual(nextNodeIdentifier + 1, fredNodeIdentifier)

        # check fred is made before deletion
        fred = scaffoldPackage.findAnnotationGroupByName('fred')
        self.assertTrue(isinstance(fred, AnnotationGroup))
        self.assertTrue(fred.isMarker())

        # delete element ranges for body
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(74, len(annotationGroups))
        scaffoldPackage.deleteElementsInRanges(region, [[313, 496]])
        self.assertEqual(824, mesh3d.getSize())
        element = mesh3d.findElementByIdentifier(400)
        self.assertFalse(element.isValid())

        # check that body-antrum junction marker is still present after deletion
        self.assertIn(bodyAntrumGCLuminal, annotationGroups)
        self.assertTrue(bodyAntrumGCLuminal.isMarker())
        node = nodes.findNodeByIdentifier(bodyAntrumGCLuminalNodeIdentifier)
        self.assertTrue(node.isValid())
        fieldcache.setNode(bodyAntrumGCLuminal.getMarkerNode())
        markerMaterialCoordinates = bodyAntrumGCLuminal.getMarkerMaterialCoordinates()[1]
        element, xi = bodyAntrumGCLuminal.getMarkerLocation()
        fieldcache.setMeshLocation(element, xi)
        result, markerCoordinates = coordinates.evaluateReal(fieldcache, coordinates.getNumberOfComponents())
        assertAlmostEqualList(self, markerCoordinates, bodyAntrumGCLuminalCoordinates, delta=TOL)
        assertAlmostEqualList(self, markerMaterialCoordinates, bodyAntrumGCLuminalMaterialCoordinates, delta=TOL)

        # check that bob is deleted
        annotationGroups = scaffoldPackage.getAnnotationGroups()
        self.assertEqual(71, len(annotationGroups))
        bob = scaffoldPackage.findAnnotationGroupByName('bob')
        self.assertNotIn(bob, annotationGroups)
        node = nodes.findNodeByIdentifier(bobNodeIdentifier)
        self.assertFalse(node.isValid())

        # check that fred is still around
        self.assertIn(fred, annotationGroups)
        node = nodes.findNodeByIdentifier(fredNodeIdentifier)
        self.assertTrue(node.isValid())

    def test_utils_ellipsoid(self):
        """
        Test ellipsoid functions converting between coordinates.
        """
        a = 1.0
        b = 0.75
        c = 2.0
        aa = a * a
        bb = b * b
        cc = c * c
        TOL = 1.0E-6
        u, v = getEllipsoidPolarCoordinatesFromPosition(a, b, c, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(u, 0.0, delta=TOL)
        self.assertAlmostEqual(v, 0.5 * math.pi, delta=TOL)
        x, dx_du, dx_dv = getEllipsoidPolarCoordinatesTangents(a, b, c, u, v)
        assertAlmostEqualList(self, x, [1.0, 0.0, 0.0], delta=TOL)
        assertAlmostEqualList(self, dx_du, [0.0, 0.75, 0.0], delta=TOL)
        assertAlmostEqualList(self, dx_dv, [0.0, 0.0, 2.0], delta=TOL)

        # test a point not on the surface
        u, v = getEllipsoidPolarCoordinatesFromPosition(a, b, c, [1.0, 0.75, 1.0])
        self.assertAlmostEqual(u, 0.6795893730435009, delta=TOL)
        self.assertAlmostEqual(v, 2.038067043010934, delta=TOL)
        x = getEllipsoidPolarCoordinatesTangents(a, b, c, u, v)[0]
        assertAlmostEqualList(self, x, [0.6944481794937621, 0.42082645661076656, 0.9009025175142785], delta=TOL)
        mag = (x[0] * x[0]) / aa + (x[1] * x[1]) / bb + (x[2] * x[2]) / cc
        self.assertAlmostEqual(mag, 1.0, delta=TOL)
        # test the nearest point found on the surface has same polar coordinates
        u, v = getEllipsoidPolarCoordinatesFromPosition(a, b, c, x)
        self.assertAlmostEqual(u, 0.6795893730435009, delta=TOL)
        self.assertAlmostEqual(v, 2.038067043010935, delta=TOL)

        # test a point not on the surface
        u, v = getEllipsoidPolarCoordinatesFromPosition(a, b, c, [-0.9, -0.25, -1.2])
        self.assertAlmostEqual(u, -2.8190484300147065, delta=TOL)
        self.assertAlmostEqual(v, 0.9561038921906282, delta=TOL)
        x = getEllipsoidPolarCoordinatesTangents(a, b, c, u, v)[0]
        assertAlmostEqualList(self, x, [-0.7748223543206793, -0.19421818481623945, -1.153414567503636], delta=TOL)
        mag = (x[0] * x[0]) / aa + (x[1] * x[1]) / bb + (x[2] * x[2]) / cc
        self.assertAlmostEqual(mag, 1.0, delta=TOL)
        # test the nearest point found on the surface has same polar coordinates
        u, v = getEllipsoidPolarCoordinatesFromPosition(a, b, c, x)
        self.assertAlmostEqual(u, -2.8190484300147065, delta=TOL)
        self.assertAlmostEqual(v, 0.9561038921906281, delta=TOL)

        u_in = math.pi / 3.0
        v_in = math.pi / 2.0
        x, dx_du, dx_dv = getEllipsoidPolarCoordinatesTangents(a, b, c, u_in, v_in)
        assertAlmostEqualList(self, x, [0.5, 0.649519052838329, 0.0], delta=TOL)
        assertAlmostEqualList(self, dx_du, [-0.8660254037844386, 0.375, 0.0], delta=TOL)
        assertAlmostEqualList(self, dx_dv, [0.0, 0.0, 2.0], delta=TOL)
        u, v = getEllipsoidPolarCoordinatesFromPosition(a, b, c, x)
        self.assertAlmostEqual(u, u_in, delta=TOL)
        self.assertAlmostEqual(v, v_in, delta=TOL)

        u_in = -0.7 * math.pi
        v_in = 0.3 * math.pi
        x, dx_du, dx_dv = getEllipsoidPolarCoordinatesTangents(a, b, c, u_in, v_in)
        assertAlmostEqualList(self, x, [-0.4755282581475767, -0.4908813728906053, -1.1755705045849463], delta=TOL)
        assertAlmostEqualList(self, dx_du, [0.6545084971874737, -0.35664619361068256, 0.0], delta=TOL)
        assertAlmostEqualList(self, dx_dv, [-0.3454915028125262, -0.3566461936106826, 1.618033988749895], delta=TOL)
        u, v = getEllipsoidPolarCoordinatesFromPosition(a, b, c, x)
        self.assertAlmostEqual(u, u_in, delta=TOL)
        self.assertAlmostEqual(v, v_in, delta=TOL)

        u_in = 0.35 * math.pi
        v_in = 0.65 * math.pi
        x, dx_du, dx_dv = getEllipsoidPolarCoordinatesTangents(a, b, c, u_in, v_in)
        assertAlmostEqualList(self, x, [0.4045084971874737, 0.5954194696096774, 0.9079809994790935], delta=TOL)
        assertAlmostEqualList(self, dx_du, [-0.7938926261462366, 0.3033813728906053, 0.0], delta=TOL)
        assertAlmostEqualList(self, dx_dv, [-0.20610737385376343, -0.30338137289060524, 1.7820130483767358], delta=TOL)
        u, v = getEllipsoidPolarCoordinatesFromPosition(a, b, c, x)
        self.assertAlmostEqual(u, u_in, delta=TOL)
        self.assertAlmostEqual(v, v_in, delta=TOL)

        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, 0.0, 0.0], [0.0, -b, 0.0])
        assertAlmostEqualList(self, centre, [0.0, 0.0, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, -b, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis2, [a, 0.0, 0.0], delta=TOL)

        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, b / 3.0, 0.0], [0.0, -b, 0.0])
        assertAlmostEqualList(self, centre, [0.0, 0.0, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, -b, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis2, [a, 0.0, 0.0], delta=TOL)

        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, 0.0, 0.0], [0.0, 0.0, c])
        assertAlmostEqualList(self, centre, [0.0, 0.0, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, 0.0, c], delta=TOL)
        assertAlmostEqualList(self, axis2, [a, 0.0, 0.0], delta=TOL)

        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, 0.0, c / 4.0], [0.0, 0.0, c])
        assertAlmostEqualList(self, centre, [0.0, 0.0, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, 0.0, c], delta=TOL)
        assertAlmostEqualList(self, axis2, [a, 0.0, 0.0], delta=TOL)

        z = 0.25 * c
        y = -math.sqrt((b * b) * (1.0 - (z * z) / (c * c)))
        x = math.sqrt((a * a) * (1.0 - (z * z) / (c * c)))
        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, 0.0, z], [0.0, y, z])
        assertAlmostEqualList(self, centre, [0.0, 0.0, z], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, y, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis2, [x, 0.0, 0.0], delta=TOL)
        mag = (axis2[0] * axis2[0]) / aa + (z * z) / cc
        self.assertAlmostEqual(mag, 1.0, delta=TOL)

        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, 0.0, 0.0], [0.0, y, z])
        assertAlmostEqualList(self, centre, [0.0, 0.0, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, y, z], delta=TOL)
        assertAlmostEqualList(self, axis2, [a, 0.0, 0.0], delta=TOL)

        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, 0.1, 0.1], [0.0, y, z])
        assertAlmostEqualList(self, centre, [0.0, 0.009782267266570388, 0.1436792247347149], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, -0.735966644680461, 0.3563207752652851], delta=TOL)
        assertAlmostEqualList(self, axis2, [0.9973309128094611, 0.0, 0.0], delta=TOL)
        mag = (axis2[0] * axis2[0]) / aa + (centre[1] * centre[1]) / bb + (centre[2] * centre[2]) / cc
        self.assertAlmostEqual(mag, 1.0, delta=TOL)
        # check original midx is on axis1
        dir1 = normalize(sub(centre, [0.0, 0.1, 0.1]))
        dir2 = normalize(axis1)
        dir3 = normalize(sub([0.0, y, z], centre))
        assertAlmostEqualList(self, dir1, dir2, delta=TOL)
        assertAlmostEqualList(self, dir1, dir3, delta=TOL)

        z = 0.8 * c
        y = math.sqrt((b * b) * (1.0 - (z * z) / (c * c)))
        x = math.sqrt((a * a) * (1.0 - (y * y) / (b * b)))
        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, y, 0.0], [0.0, y, z])
        assertAlmostEqualList(self, centre, [0.0, y, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, 0.0, z], delta=TOL)
        assertAlmostEqualList(self, axis2, [x, 0.0, 0.0], delta=TOL)
        mag = (axis2[0] * axis2[0]) / aa + (y * y) / bb
        self.assertAlmostEqual(mag, 1.0, delta=TOL)

        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, 0.0, 0.0], [0.0, y, z])
        assertAlmostEqualList(self, centre, [0.0, 0.0, 0.0], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, y, z], delta=TOL)
        assertAlmostEqualList(self, axis2, [a, 0.0, 0.0], delta=TOL)

        centre, axis1, axis2 = getEllipsoidPlaneA(a, b, c, [0.0, -0.3, -0.1], [0.0, y, z])
        assertAlmostEqualList(self, centre, [0.0, -0.10732946298984034, 0.3367198838896952], delta=TOL)
        assertAlmostEqualList(self, axis1, [0.0, 0.5573294629898402, 1.2632801161103049], delta=TOL)
        assertAlmostEqualList(self, axis2, [0.9752823267321079, 0.0, 0.0], delta=TOL)
        mag = (axis2[0] * axis2[0]) / aa + (centre[1] * centre[1]) / bb + (centre[2] * centre[2]) / cc
        self.assertAlmostEqual(mag, 1.0, delta=TOL)
        # check original midx is on axis1
        dir1 = normalize(sub(centre, [0.0, -0.3, -0.1]))
        dir2 = normalize(axis1)
        dir3 = normalize(sub([0.0, y, z], centre))
        assertAlmostEqualList(self, dir1, dir2, delta=TOL)
        assertAlmostEqualList(self, dir1, dir3, delta=TOL)

    def test_curve_nearest_intersections(self):
        """
        Test finding nearest points and intersections for curves.
        """
        curve1_x = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]
        curve1_d1 = [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
        curve2_x = [[0.0, -0.2, 0.7], [1.0, -0.3, 0.3]]
        curve2_d1 = [[1.0, -0.1, -0.6], [1.0, -0.1, 0.0]]
        curve3_x = [[0.0, 0.3, 0.5], [1.0, 0.7, 0.5]]
        curve3_d1 = [[1.0, -0.2, 0.0], [1.0, -0.2, 0.0]]
        curve4_x = [[0.4, 0.4, 0.0], [0.3, 0.3, 0.4]]
        curve4_d1 = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]
        loop1_x = [[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]]
        loop1_d1 = [[0.0, -1.5, 0.0], [0.0, 1.5, 0.0]]

        XI_TOL = 1.0E-6
        X_TOL = 1.0E-6

        t1x = [0.1, 0.2, 0.3]
        p1, p1x = getNearestLocationOnCurve(curve1_x, curve1_d1, t1x)
        self.assertEqual(p1[0], 0)
        self.assertAlmostEqual(p1[1], 0.2, delta=XI_TOL)
        assertAlmostEqualList(self, [0.2, 0.0, 0.2], p1x, delta=X_TOL)
        p1t = evaluateCoordinatesOnCurve(curve1_x, curve1_d1, p1, derivative=True)[1]
        # nearest projection is normal to curve
        self.assertAlmostEqual(dot(sub(t1x, p1x), p1t), 0.0, delta=X_TOL)

        t2x = [0.1, 0.2, 0.7]
        p2, p2x = getNearestLocationOnCurve(loop1_x, loop1_d1, t2x, loop=True)
        self.assertEqual(p2[0], 0)
        self.assertAlmostEqual(p2[1], 0.19457868521558017, delta=XI_TOL)
        assertAlmostEqualList(self, [0.1790790431027201, 0.26492322619542114, 0.5], p2x, delta=X_TOL)
        p2t = evaluateCoordinatesOnCurve(loop1_x, loop1_d1, p2, loop=True, derivative=True)[1]
        # nearest projection is normal to curve
        self.assertAlmostEqual(dot(sub(t2x, p2x), p2t), 0.0, delta=X_TOL)

        t3x = [-0.1, 0.2, 0.4]
        p3, p3x = getNearestLocationOnCurve(curve3_x, curve3_d1, t3x, startLocation=(0, 0.7))
        self.assertEqual(p3[0], 0)
        self.assertAlmostEqual(p3[1], 0.0, delta=XI_TOL)
        assertAlmostEqualList(self, curve3_x[0], p3x, delta=X_TOL)

        p4, op4, p4intersects = getNearestLocationBetweenCurves(curve1_x, curve1_d1, curve2_x, curve2_d1)
        p4x, p4t = evaluateCoordinatesOnCurve(curve1_x, curve1_d1, p4, derivative=True)
        op4x = evaluateCoordinatesOnCurve(curve2_x, curve2_d1, op4)
        self.assertFalse(p4intersects)
        self.assertEqual(p4[0], 0)
        self.assertAlmostEqual(p4[1], 0.44315296477065264, delta=XI_TOL)
        # nearest projection is normal to curve
        self.assertAlmostEqual(dot(sub(op4x, p4x), p4t), 0.0, delta=X_TOL)

        p5, op5, p5intersects = getNearestLocationBetweenCurves(
            loop1_x, loop1_d1, curve1_x, curve1_d1, startLocation=(1, 0.9), nLoop=True)
        p5x, p5t = evaluateCoordinatesOnCurve(loop1_x, loop1_d1, p5, loop=True, derivative=True)
        op5x = evaluateCoordinatesOnCurve(curve1_x, curve1_d1, op5)
        self.assertFalse(p5intersects)
        self.assertEqual(p5[0], 0)
        self.assertAlmostEqual(p5[1], 0.5, delta=XI_TOL)
        # nearest projection is normal to curve
        self.assertAlmostEqual(dot(sub(op5x, p5x), p5t), 0.0, delta=X_TOL)

        p6, op6, p6intersects = getNearestLocationBetweenCurves(
            curve3_x, curve3_d1, loop1_x, loop1_d1, startLocation=(0, 0.0), oLoop=True)
        p6x = evaluateCoordinatesOnCurve(curve3_x, curve3_d1, p6)
        op6x = evaluateCoordinatesOnCurve(loop1_x, loop1_d1, op6, loop=True)
        self.assertTrue(p6intersects)
        self.assertEqual(p6[0], 0)
        self.assertAlmostEqual(p6[1], 0.1499585051371609, delta=XI_TOL)
        assertAlmostEqualList(self, p6x, op6x, delta=X_TOL)

        p7, op7, p7intersects = getNearestLocationBetweenCurves(
            loop1_x, loop1_d1, curve3_x, curve3_d1, startLocation=(0, 0.9), nLoop=True)
        p7x = evaluateCoordinatesOnCurve(loop1_x, loop1_d1, p7, loop=True)
        op7x = evaluateCoordinatesOnCurve(curve3_x, curve3_d1, op7)
        self.assertTrue(p7intersects)
        self.assertEqual(p7[0], 1)
        self.assertAlmostEqual(p7[1], 0.152207831857098, delta=XI_TOL)
        assertAlmostEqualList(self, p7x, op7x, delta=X_TOL)

        p8, op8, p8intersects = getNearestLocationBetweenCurves(
            curve4_x, curve4_d1, loop1_x, loop1_d1, startLocation=(0, 0.5), oLoop=True)
        p8x = evaluateCoordinatesOnCurve(curve4_x, curve4_d1, p8, loop=True)
        op8x, op8t = evaluateCoordinatesOnCurve(loop1_x, loop1_d1, op8, derivative=True)
        self.assertFalse(p8intersects)
        self.assertEqual(p8[0], 0)
        self.assertAlmostEqual(p8[1], 1.0, delta=XI_TOL)
        # nearest projection is normal to loop1
        self.assertAlmostEqual(dot(sub(p8x, op8x), op8t), 0.0, delta=X_TOL)

        # context = Context("Curve nearest and intersection")
        # region = context.getDefaultRegion()
        # generate_curve_mesh(region, curve1_x, curve1_d1)
        # generate_curve_mesh(region, curve2_x, curve2_d1)
        # generate_curve_mesh(region, curve3_x, curve3_d1)
        # generate_curve_mesh(region, curve4_x, curve4_d1)
        # generate_curve_mesh(region, loop1_x, loop1_d1, loop=True)
        # fieldmodule = region.getFieldmodule()
        # nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # nodetemplate = nodes.createNodetemplate()
        # pointCoordinates = create_field_coordinates(fieldmodule, "point_coordinates", managed=True)
        # nodetemplate.defineField(pointCoordinates)
        # fieldcache = fieldmodule.createFieldcache()
        # px = [t1x, p1x, t2x, p2x, t3x, p3x, p4x, op4x, p5x, op5x, p6x, p7x, p8x, op8x]
        # for n in range(len(px)):
        #     node = nodes.createNode(-1, nodetemplate)
        #     fieldcache.setNode(node)
        #     pointCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, px[n])

    def test_curve_track_surface_nearest_intersection(self):
        """
        Test finding nearest/intersection points on a curve and a track surface.
        """
        curve3_x = [[0.0, 0.3, 0.5], [1.0, 0.7, 0.5]]
        curve3_d1 = [[1.0, -0.2, 0.0], [1.0, -0.2, 0.0]]
        curve4_x = [[0.4, 0.4, 0.0], [0.3, 0.3, 0.4]]
        curve4_d1 = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]
        surf2_x = [[0.1, 0.0, -0.2], [0.9, 0.0, 0.5], [0.1, 1.0, 0.6], [0.9, 1.0, -0.3]]
        surf2_d1 = [[0.9, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        surf2_d2 = [[0.0, 1.0, 0.0], [0.0, 1.0, -0.8], [0.0, 1.0, 0.0], [0.0, 1.0, -0.8]]
        surf2 = TrackSurface(1, 1, surf2_x, surf2_d1, surf2_d2)

        XI_TOL = 1.0E-6
        X_TOL = 1.0E-6

        p1, cp1, p1intersects = surf2.findNearestPositionOnCurve(curve3_x, curve3_d1)
        self.assertFalse(p1intersects)
        self.assertEqual(cp1[0], 0)
        self.assertAlmostEqual(cp1[1], 0.4432272731754388, delta=XI_TOL)
        p1x, p1d1, p1d2 = surf2.evaluateCoordinates(p1, derivatives=True)
        cp1x = evaluateCoordinatesOnCurve(curve3_x, curve3_d1, cp1)
        # nearest projection is normal to surf2
        delta = sub(cp1x, p1x)
        self.assertAlmostEqual(dot(delta, p1d1), 0.0, delta=X_TOL)
        self.assertAlmostEqual(dot(delta, p1d2), 0.0, delta=X_TOL)

        p2, cp2, p2intersects = surf2.findNearestPositionOnCurve(curve4_x, curve4_d1)
        self.assertTrue(p2intersects)
        self.assertEqual(cp2[0], 0)
        self.assertAlmostEqual(cp2[1], 0.23080643876585413, delta=XI_TOL)
        p2x = surf2.evaluateCoordinates(p2)
        cp2x = evaluateCoordinatesOnCurve(curve4_x, curve4_d1, cp2)
        assertAlmostEqualList(self, p2x, cp2x, delta=X_TOL)

        # context = Context("TrackSurface")
        # region = context.getDefaultRegion()
        # generate_curve_mesh(region, curve3_x, curve3_d1)
        # generate_curve_mesh(region, curve4_x, curve4_d1)
        # surf2.generateMesh(region)
        # fieldmodule = region.getFieldmodule()
        # nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # nodetemplate = nodes.createNodetemplate()
        # pointCoordinates = find_or_create_field_coordinates(fieldmodule, "point_coordinates", managed=True)
        # nodetemplate.defineField(pointCoordinates)
        # fieldcache = fieldmodule.createFieldcache()
        # px = [p1x, cp1x, cp2x]
        # for n in range(len(px)):
        #     node = nodes.createNode(-1, nodetemplate)
        #     fieldcache.setNode(node)
        #     pointCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, px[n])

    def test_track_surface_intersection(self):
        """
        Test finding points on intersection between 2 track surfaces.
        """
        surf1_x = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        surf1_d1 = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        surf1_d2 = [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        surf1 = TrackSurface(1, 1, surf1_x, surf1_d1, surf1_d2)

        surf2_x = [[0.1, 0.0, -0.2], [0.9, 0.0, 0.5], [0.1, 1.0, 0.6], [0.9, 1.0, -0.3]]
        surf2_d1 = [[0.9, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        surf2_d2 = [[0.0, 1.0, 0.0], [0.0, 1.0, -0.8], [0.0, 1.0, 0.0], [0.0, 1.0, -0.8]]
        surf2 = TrackSurface(1, 1, surf2_x, surf2_d1, surf2_d2)

        XI_TOL = 1.0E-6
        X_TOL = 1.0E-6

        p1, op1, p1x, p1t, p1bdy = surf1.findIntersectionPoint(
            surf2, surf1.createPositionProportion(0.25, 0.1), surf2.createPositionProportion(0.5, 0.5))
        self.assertEqual(p1.e1, 0)
        self.assertEqual(p1.e2, 0)
        self.assertAlmostEqual(p1.xi1, 0.38287186861026046, delta=XI_TOL)
        self.assertAlmostEqual(p1.xi2, 0.14232301624230445, delta=XI_TOL)
        self.assertEqual(op1.e1, 0)
        self.assertEqual(op1.e2, 0)
        self.assertAlmostEqual(op1.xi1, 0.35438003143365127, delta=XI_TOL)
        self.assertAlmostEqual(op1.xi2, 0.14232301622713658, delta=XI_TOL)
        assertAlmostEqualList(self, [0.1700422367869369, -0.9854367750944224, 0.0], p1t, delta=X_TOL)

        p2, op2, p2x, p2t, p2bdy = surf1.findIntersectionPoint(
            surf2, surf1.createPositionProportion(0.7, 0.6), surf2.createPositionProportion(0.5, 0.5))
        self.assertEqual(p2.e1, 0)
        self.assertEqual(p2.e2, 0)
        self.assertAlmostEqual(p2.xi1, 0.7455699485954836, delta=XI_TOL)
        self.assertAlmostEqual(p2.xi2, 0.6652325809349918, delta=XI_TOL)
        self.assertEqual(op2.e1, 0)
        self.assertEqual(op2.e2, 0)
        self.assertAlmostEqual(op2.xi1, 0.830980843977797, delta=XI_TOL)
        self.assertAlmostEqual(op2.xi2, 0.6652325809312423, delta=XI_TOL)
        assertAlmostEqualList(self, [-0.8391083674789601, 0.5439642889260237, 0.0], p2t, delta=X_TOL)

        p3, op3, p3x, p3t, p3bdy = surf1.findIntersectionPoint(
            surf2, surf1.createPositionProportion(0.0, 0.326), surf2.createPositionProportion(0.0, 0.25))
        self.assertEqual(p3.e1, 0)
        self.assertEqual(p3.e2, 0)
        self.assertAlmostEqual(p3.xi1, 0.1, delta=XI_TOL)
        self.assertAlmostEqual(p3.xi2, 0.3263518272375643, delta=XI_TOL)
        self.assertEqual(op3.e1, 0)
        self.assertEqual(op3.e2, 0)
        self.assertAlmostEqual(op3.xi1, 0.0, delta=XI_TOL)
        self.assertAlmostEqual(op3.xi2, 0.3263518246534994, delta=XI_TOL)
        assertAlmostEqualList(self, [1.0, 0.0, 0.0], p3t, delta=X_TOL)

        p4, op4, p4x, p4t, p4bdy = surf1.findIntersectionPoint(
            surf2, TrackSurfacePosition(0, 0, 0.35032948194631897, 0.0),
            TrackSurfacePosition(0, 0, 0.35431492084144983, 0.006191352063889618))
        self.assertEqual(p4.e1, 0)
        self.assertEqual(p4.e2, 0)
        self.assertAlmostEqual(p4.xi1, 0.38097836477185065, delta=XI_TOL)
        self.assertAlmostEqual(p4.xi2, 0.0, delta=XI_TOL)
        self.assertEqual(op4.e1, 0)
        self.assertEqual(op4.e2, 0)
        self.assertAlmostEqual(op4.xi1, 0.3528986153613523, delta=XI_TOL)
        self.assertAlmostEqual(op4.xi2, 0.0, delta=XI_TOL)
        assertAlmostEqualList(self, [-0.17134157499316163, -0.9852116852123014, 0.0], p4t, delta=X_TOL)

        cx, cd1, cprops, loop = surf1.findIntersectionCurve(
            surf2, surf1.createPositionProportion(0.25, 0.1), MAX_MAG_DXI=0.2)
        self.assertEqual(len(cx), 9)
        self.assertFalse(loop)
        clength = getCubicHermiteCurvesLength(cx, cd1)
        self.assertAlmostEqual(clength, 0.5166105006376641, delta=X_TOL)
        assertAlmostEqualList(self, [0.1000000000000001, 0.3263518562674236, 0.0], cx[0], delta=X_TOL)
        assertAlmostEqualList(self, [0.3373116664675114, 0.2471498514765583, 0.0], cx[4], delta=X_TOL)
        assertAlmostEqualList(self, [0.3809786164663428, 0.0, 0.0], cx[8], delta=X_TOL)
        assertAlmostEqualList(self, [0.1, 0.3263518562674236], cprops[0], delta=XI_TOL)
        assertAlmostEqualList(self, [0.3373116664675114, 0.2471498514765583], cprops[4], delta=XI_TOL)
        assertAlmostEqualList(self, [0.3809786164663428, 0.0], cprops[8], delta=XI_TOL)

        dx, dd1, dprops, loop = surf1.findIntersectionCurve(
            surf2, surf1.createPositionProportion(0.85, 0.75), MAX_MAG_DXI=0.2)
        self.assertEqual(len(dx), 9)
        self.assertFalse(loop)
        dlength = getCubicHermiteCurvesLength(dx, dd1)
        self.assertAlmostEqual(dlength, 0.5433209057545304, delta=X_TOL)
        assertAlmostEqualList(self, [0.9, 0.625, 0.0], dx[0], delta=X_TOL)
        assertAlmostEqualList(self, [0.6714699015724952, 0.7450512483881269, 0.0], dx[4], delta=3.0*X_TOL)
        assertAlmostEqualList(self, [0.5797035235769858, 1.0, 0.0], dx[8], delta=3.0*X_TOL)
        assertAlmostEqualList(self, [0.9, 0.625], dprops[0], delta=XI_TOL)
        assertAlmostEqualList(self, [0.6714699015724951, 0.7450512483881269], dprops[4], delta=3.0*XI_TOL)
        assertAlmostEqualList(self, [0.5797035235769857, 1.0], dprops[8], delta=3.0*XI_TOL)

        # context = Context("TrackSurface")
        # region = context.getDefaultRegion()
        # surfaceGroupName = "surface"
        # surf1.generateMesh(region, group_name=surfaceGroupName)
        # surf2.generateMesh(region, group_name=surfaceGroupName)
        # coordinateFieldName = "curve_coordinates"
        # curveGroupName = "curve"
        # fieldmodule = region.getFieldmodule()
        # curveCoordinates = find_or_create_field_coordinates(fieldmodule, coordinateFieldName, managed=True)
        # curveGroup = find_or_create_field_group(fieldmodule, curveGroupName)
        # nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # nodetemplate = nodes.createNodetemplate()
        # nodetemplate.defineField(curveCoordinates)
        # nodetemplate.setValueNumberOfVersions(curveCoordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        # curveNodesetGroup = curveGroup.getOrCreateNodesetGroup(nodes)
        # fieldcache = fieldmodule.createFieldcache()
        # px = [p1x, p2x, p3x, p4x]
        # pd1 = [p1t, p2t, p3t, p4t]
        # for n in range(len(px)):
        #     node = nodes.createNode(-1, nodetemplate)
        #     fieldcache.setNode(node)
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, px[n])
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, pd1[n])
        #     curveNodesetGroup.addNode(node)
        # generate_curve_mesh(region, cx, cd1, coordinate_field_name=coordinateFieldName, group_name=curveGroupName)
        # generate_curve_mesh(region, dx, dd1, coordinate_field_name=coordinateFieldName, group_name=curveGroupName)

    def test_tube_intersections1(self):
        """
        Test tube intersections in a diverging bifurcation with one pair of tubes equal sized and continuous,
        but with one small outward tube which only intersects with the first tube.
        """
        elementsCountAround = 8
        elementsCountAlong = 6
        path1Params = [
            [[0.000, 0.000, 0.000], [1.000, 0.000, 0.000]],
            [[0.979, 0.123, 0.000], [1.010, -0.124, 0.000]],
            [[-0.031, 0.248, 0.000], [0.031, 0.348, 0.000]],
            [[0.063, 0.007, 0.000], [0.060, 0.307, 0.000]],
            [[0.000, 0.000, 0.250], [0.000, 0.000, 0.413]],
            [[0.000, 0.000, 0.126], [0.000, 0.000, 0.601]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path1Params, elementsCountAround, subsample_count=1)
        sx, sd1, sd2, sd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(sx)):
            nx += sx[i]
            nd1 += sd1[i]
            nd2 += sd2[i]
            nd12 += sd12[i]
        tube1Surface = TrackSurface(elementsCountAround, elementsCountAlong, nx, nd1, nd2, nd12, loop1=True)

        path2Params = [
            [[1.000, 0.000, 0.000], [2.000, -0.250, 0.000]],
            [[1.010, -0.124, 0.000], [0.980, -0.372, 0.000]],
            [[0.031, 0.348, 0.000], [0.089, 0.234, 0.000]],
            [[0.060, 0.307, 0.000], [-0.013, -0.101, -0.031]],
            [[0.000, 0.000, 0.413], [-0.000, 0.000, 0.250]],
            [[0.000, 0.000, 0.601], [-0.023, -0.019, -0.111]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path2Params, elementsCountAround, subsample_count=1)
        sx, sd1, sd2, sd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(sx)):
            nx += sx[i]
            nd1 += sd1[i]
            nd2 += sd2[i]
            nd12 += sd12[i]
        tube2Surface = TrackSurface(elementsCountAround, elementsCountAlong, nx, nd1, nd2, nd12, loop1=True)

        path3Params = [
            [[1.000, 0.000, 0.000], [0.488, 0.731, 0.107]],
            [[-0.512, 0.731, 0.107], [-0.512, 0.731, 0.107]],
            [[-0.034, -0.015, -0.055], [-0.052, -0.024, -0.088]],
            [[0.000, 0.000, 0.000], [-0.053, -0.053, -0.046]],
            [[-0.068, -0.056, 0.057], [-0.065, -0.054, 0.053]],
            [[0.000, 0.000, 0.000], [-0.086, -0.042, -0.109]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path3Params, elementsCountAround, subsample_count=1)
        sx, sd1, sd2, sd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(sx)):
            nx += sx[i]
            nd1 += sd1[i]
            nd2 += sd2[i]
            nd12 += sd12[i]
        tube3Surface = TrackSurface(elementsCountAround, elementsCountAlong, nx, nd1, nd2, nd12, loop1=True)

        XI_TOL = 1.0E-6
        X_TOL = 1.0E-6

        targetx = [0.7527511365837611, 0.1054917065647476, 0.2920147530232719]
        startPosition = TrackSurfacePosition(1, 0, 0.5, 0.5)
        nearestPosition = tube3Surface.findNearestPosition(targetx, startPosition)
        self.assertEqual(nearestPosition.e1, 2)
        self.assertEqual(nearestPosition.e2, 1)
        self.assertAlmostEqual(nearestPosition.xi1, 0.44002657999055783, delta=XI_TOL)
        self.assertAlmostEqual(nearestPosition.xi2, 0.7782327545334007, delta=XI_TOL)

        targetx = [0.9745695128243425, -0.28544615442781057, -0.23619538278312255]
        startPosition = TrackSurfacePosition(4, 0, 0.158, 0.0)
        nearestPosition = tube3Surface.findNearestPosition(targetx, startPosition)
        self.assertEqual(nearestPosition.e1, 0)
        self.assertEqual(nearestPosition.e2, 0)
        self.assertAlmostEqual(nearestPosition.xi1, 0.26373820317934693, delta=XI_TOL)
        self.assertAlmostEqual(nearestPosition.xi2, 0.0, delta=XI_TOL)

        # get intersection between fully connected tubes from nearby point
        startPosition = TrackSurfacePosition(12, 5, 0.8242639579553614, 0.999980904098904)
        p1x = tube1Surface.evaluateCoordinates(startPosition)
        otherStartPosition = TrackSurfacePosition(4, 0, 0.1580888007016199, 4.3205057553252083e-16)
        p2, op2, p2x, p2t, p2bdy = tube1Surface.findIntersectionPoint(tube2Surface, startPosition, otherStartPosition)
        self.assertEqual(p2.e1, 12)
        self.assertEqual(p2.e2, 5)
        self.assertAlmostEqual(p2.xi1, 0.8242625516927582, delta=XI_TOL)
        self.assertAlmostEqual(p2.xi2, 1.0, delta=XI_TOL)
        self.assertEqual(op2.e1, 4)
        self.assertEqual(op2.e2, 0)
        self.assertAlmostEqual(op2.xi1, 0.8242625516925104, delta=XI_TOL)
        self.assertAlmostEqual(op2.xi2, 0.0, delta=XI_TOL)
        assertAlmostEqualList(self, [0.04507074682153383, 0.5059554804478983, -0.8613812626158557], p2t, delta=X_TOL)

        # get loop intersection of fully connected tube1 and tube2
        ax, ad1, aprops, aloop = tube1Surface.findIntersectionCurve(tube2Surface, curveElementsCount=12)
        self.assertEqual(len(ax), 12)
        self.assertTrue(aloop)
        aCircumference = getCubicHermiteCurvesLength(ax, ad1, loop=True)
        self.assertAlmostEqual(aCircumference, 2.3973686453086143, delta=X_TOL)
        assertAlmostEqualList(self, [0.9708619461568029, -0.3270981528204049, 0.14086800305222866], ax[0], delta=X_TOL)
        assertAlmostEqualList(self, [1.0050064347290057, 0.05620126728045057, -0.40729362662747776], ax[4], delta=X_TOL)
        assertAlmostEqualList(self, [1.0249960710077815, 0.28060105518412687, 0.2442400184294043], ax[8], delta=X_TOL)
        assertAlmostEqualList(self, [0.4405054359950804, 1.0], aprops[0], delta=XI_TOL)
        assertAlmostEqualList(self, [0.7737970101113247, 1.0], aprops[4], delta=XI_TOL)
        assertAlmostEqualList(self, [1.1069601060105503, 1.0], aprops[8], delta=XI_TOL)

        # get loop intersection of unconnected tube2 and tube3
        bx, bd1, bprops, bloop = tube2Surface.findIntersectionCurve(tube3Surface, curveElementsCount=12)
        self.assertIsNone(bx)
        self.assertIsNone(bd1)
        self.assertIsNone(bprops)
        self.assertFalse(bloop)

        # get loop intersection of large tube1 and small tube3
        cx, cd1, cprops, cloop = tube1Surface.findIntersectionCurve(
             tube3Surface, tube1Surface.createPositionProportion(0.1, 0.7))
        self.assertEqual(len(cx), 8)
        self.assertTrue(cloop)
        cCircumference = getCubicHermiteCurvesLength(cx, cd1, loop=True)
        self.assertAlmostEqual(cCircumference, 0.5878577057061585, delta=X_TOL)
        assertAlmostEqualList(self, [0.7318758087995326, 0.28115396635716217, 0.13238674551352603], cx[0], delta=X_TOL)
        assertAlmostEqualList(self, [0.8408764855242207, 0.3182280446541034, -0.04549439718679245], cx[4], delta=X_TOL)
        assertAlmostEqualList(self, [1.0762363613976254, 0.7158314513549112], cprops[0], delta=XI_TOL)
        assertAlmostEqualList(self, [0.9766132059804884, 0.8198672829760584], cprops[4], delta=XI_TOL)

        # make trimmed tube3 starting at intersection with tube1
        tx, td1, td2, td12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong,
                                                     startSurface=tube1Surface)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(tx)):
            nx += tx[i]
            nd1 += td1[i]
            nd2 += td2[i]
            nd12 += td12[i]
        tube3TrimmedSurface = TrackSurface(elementsCountAround, elementsCountAlong, nx, nd1, nd2, nd12, loop1=True)
        tCircumference = getCubicHermiteCurvesLength(tx[0], td1[0], loop=True)
        self.assertAlmostEqual(tCircumference, 0.5891597254350946, delta=X_TOL)
        tLength = getCubicHermiteCurvesLength([tx[n][0] for n in range(elementsCountAlong + 1)],
                                              [td2[n][0] for n in range(elementsCountAlong + 1)])
        self.assertAlmostEqual(tLength, 0.5004143891824094, delta=X_TOL)

        curveLocation1, curveX1 = getNearestLocationOnCurve(
            cx, cd1, targetx=[1.0307591456989758, 0.3452962162336672, -0.05130331144410176], loop=True)
        self.assertEqual(curveLocation1[0], 3)
        self.assertAlmostEqual(curveLocation1[1], 0.2627396492643106, delta=XI_TOL)

        aCurveLocation, cCurveLocation, acIntersection = getNearestLocationBetweenCurves(
            ax, ad1, cx, cd1, aloop, cloop)
        self.assertFalse(acIntersection)
        p3x = evaluateCoordinatesOnCurve(ax, ad1, aCurveLocation, aloop)
        p4x = evaluateCoordinatesOnCurve(cx, cd1, cCurveLocation, cloop)
        self.assertEqual(aCurveLocation[0], 6)
        self.assertAlmostEqual(aCurveLocation[1], 0.7537941734689295, delta=XI_TOL)
        self.assertEqual(cCurveLocation[0], 3)
        self.assertAlmostEqual(cCurveLocation[1], 0.05064922547377515, delta=XI_TOL)

        # context = Context("TrackSurface")
        # region = context.getDefaultRegion()
        # surfaceGroupName = "surface"
        # tube1Surface.generateMesh(region, group_name=surfaceGroupName)
        # tube2Surface.generateMesh(region, group_name=surfaceGroupName)
        # # tube3Surface.generateMesh(region, group_name=surfaceGroupName)
        # tube3TrimmedSurface.generateMesh(region, group_name=surfaceGroupName)
        # coordinateFieldName = "curve_coordinates"
        # curveGroupName = "curve"
        # fieldmodule = region.getFieldmodule()
        # curveCoordinates = find_or_create_field_coordinates(fieldmodule, coordinateFieldName, managed=True)
        # curveGroup = find_or_create_field_group(fieldmodule, curveGroupName)
        # nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # nodetemplate = nodes.createNodetemplate()
        # nodetemplate.defineField(curveCoordinates)
        # nodetemplate.setValueNumberOfVersions(curveCoordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        # curveNodesetGroup = curveGroup.getOrCreateNodesetGroup(nodes)
        # fieldcache = fieldmodule.createFieldcache()
        # px = [p1x, p2x, p3x, p4x]
        # pd1 = [[0.0, 0.0, 0.0], p2t, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        # for n in range(len(px)):
        #     node = nodes.createNode(-1, nodetemplate)
        #     fieldcache.setNode(node)
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, px[n])
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, pd1[n])
        #     curveNodesetGroup.addNode(node)
        # generate_curve_mesh(region, ax, ad1, aloop, coordinate_field_name=coordinateFieldName, group_name=curveGroupName)
        # generate_curve_mesh(region, cx, cd1, cloop, coordinate_field_name=coordinateFieldName, group_name=curveGroupName)

    def test_tube_intersections1_coarse(self):
        """
        Same as above but fewer along.
        """
        elementsCountAround = 8
        elementsCountAlong = 4
        path1Params = [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.9899453611864429, 0.1306534639262169, 0.1208354837258566], [0.986629229688665, -0.1356318297362336, -0.140819493622476]],
            [[-0.03223527987199958, 0.2478818360588229, -0.003934727904278583], [0.0333737119177766, 0.2477165835464935, -0.004763358991550144]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-0.03029087281311803, 0.0, 0.2481581411604695], [0.03532398595224329, 8.623473827193372e-19, 0.2474918504041006]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path1Params, elementsCountAround, subsample_count=1)
        sx, sd1, sd2, sd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(sx)):
            nx += sx[i]
            nd1 += sd1[i]
            nd2 += sd2[i]
            nd12 += sd12[i]
        tube1Surface = TrackSurface(elementsCountAround, elementsCountAlong, nx, nd1, nd2, nd12, loop1=True)

        path2Params = [
            [[1.0, 0.0, 0.0], [1.938402743610923, -0.5417378094686381, -0.3033747813098204]],
            [[0.7703805269668589, -0.789032983143896, -0.3738697910125267], [1.133060151191206, -0.1265084623478793, 0.2366603646370917]],
            [[0.1645877843296779, 0.1824773036586876, -0.04596623651010114], [0.0291111109020875, 0.2482090649360781, -0.006693527142258089]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0897382556035165, -0.02243456390087916, 0.2322579079898364], [-0.04972023063292857, 0.01243005765823214, 0.2446904009813655]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path2Params, elementsCountAround, subsample_count=1)
        sx, sd1, sd2, sd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(sx)):
            nx += sx[i]
            nd1 += sd1[i]
            nd2 += sd2[i]
            nd12 += sd12[i]
        tube2Surface = TrackSurface(elementsCountAround, elementsCountAlong, nx, nd1, nd2, nd12, loop1=True)

        XI_TOL = 1.0E-6

        startPosition = TrackSurfacePosition(6, 3, 0.5717869946250334, 1.0)
        otherStartPosition = TrackSurfacePosition(6, 0, 0.11046416298376727, 0.07944602603404333)
        p2, op2, p2x, p2t, p2bdy = tube1Surface.findIntersectionPoint(tube2Surface, startPosition, otherStartPosition)
        self.assertEqual(p2.e1, 6)
        self.assertEqual(p2.e2, 3)
        self.assertAlmostEqual(p2.xi1, 0.4639613077782414, delta=XI_TOL)
        self.assertAlmostEqual(p2.xi2, 1.0, delta=XI_TOL)
        self.assertEqual(op2.e1, 6)
        self.assertEqual(op2.e2, 0)
        self.assertAlmostEqual(op2.xi1, 0.4873273111526366, delta=XI_TOL)
        self.assertAlmostEqual(op2.xi2, 0.0032771820253160657, delta=XI_TOL)
        self.assertIsNone(p2t)

    def test_tube_intersections2(self):
        """
        Test tube intersections in a diverging bifurcation case with similar tube sizes.
        """
        elementsCountAround = 8
        elementsCountAlong = 6
        path1Params = [
            [[0.000, 0.000, 0.000], [1.000, 0.000, 0.000]],
            [[0.990, 0.131, 0.121], [0.987, -0.136, -0.141]],
            [[-0.032, 0.248, -0.004], [0.033, 0.248, -0.005]],
            [[0.000, 0.000, 0.000], [0.000, 0.000, 0.000]],
            [[-0.030, 0.000, 0.248], [0.035, 0.000, 0.247]],
            [[0.000, 0.000, 0.000], [0.000, 0.000, 0.000]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path1Params, elementsCountAround, subsample_count=1)
        sx, sd1, sd2, sd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(sx)):
            nx += sx[i]
            nd1 += sd1[i]
            nd2 += sd2[i]
            nd12 += sd12[i]
        tube1Surface = TrackSurface(elementsCountAround, elementsCountAlong, nx, nd1, nd2, nd12, loop1=True)

        path2Params = [
            [[1.000, 0.000, 0.000], [1.938, -0.542, -0.303]],
            [[0.770, -0.789, -0.374], [1.133, -0.127, 0.237]],
            [[0.165, 0.182, -0.046], [0.029, 0.248, -0.007]],
            [[0.000, 0.000, 0.000], [0.000, 0.000, 0.000]],
            [[0.090, -0.022, 0.232], [-0.050, 0.012, 0.245]],
            [[0.000, 0.000, 0.000], [0.000, 0.000, 0.000]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path2Params, elementsCountAround, subsample_count=1)
        sx, sd1, sd2, sd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(sx)):
            nx += sx[i]
            nd1 += sd1[i]
            nd2 += sd2[i]
            nd12 += sd12[i]
        tube2Surface = TrackSurface(elementsCountAround, elementsCountAlong, nx, nd1, nd2, nd12, loop1=True)

        path3Params = [
            [[1.000, 0.000, 0.000], [1.658, 0.634, 0.017]],
            [[0.795, 0.473, -0.005], [0.441, 0.812, -0.048]],
            [[-0.128, 0.215, 0.000], [-0.216, 0.120, 0.037]],
            [[0.000, 0.000, 0.000], [0.000, 0.000, 0.000]],
            [[0.001, 0.000, 0.250], [0.038, -0.006, 0.247]],
            [[0.000, 0.000, 0.000], [0.000, 0.000, 0.000]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path3Params, elementsCountAround, subsample_count=1)
        sx, sd1, sd2, sd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(sx)):
            nx += sx[i]
            nd1 += sd1[i]
            nd2 += sd2[i]
            nd12 += sd12[i]
        tube3Surface = TrackSurface(elementsCountAround, elementsCountAlong, nx, nd1, nd2, nd12, loop1=True)

        XI_TOL = 1.0E-6
        X_TOL = 1.0E-6

        targetx = [0.75, -0.1, 0.0]
        # pax = targetx
        # startPosition =TrackSurfacePosition(7, 0, 0.5447810822226353, 0.0)
        # pbx, d1, d2 = tube3Surface.evaluateCoordinates(startPosition, derivatives=True)
        # pbd2 = sub(pax, pbx)
        # pbd1 = mult(d1, dot(normalize(d1), pbd2))
        startPosition = TrackSurfacePosition(7, 0, 0.5, 0.5)
        nearestPosition = tube3Surface.findNearestPosition(targetx, startPosition)
        # pcx, d1, d2 = tube3Surface.evaluateCoordinates(nearestPosition, derivatives=True)
        # pcd2 = sub(pax, pcx)
        # pcd1 = mult(d1, dot(normalize(d1), pcd2))
        if elementsCountAround == 8:
            self.assertEqual(nearestPosition.e1, 7)
            self.assertEqual(nearestPosition.e2, 0)
            self.assertAlmostEqual(nearestPosition.xi1, 0.9804700562386097, delta=0.03)  # as stops with slow progress
            self.assertAlmostEqual(nearestPosition.xi2, 0.0, delta=XI_TOL)

        p1, op1, p1x, p1t, p1bdy = tube1Surface.findIntersectionPoint(
            tube2Surface, tube1Surface.createPositionProportion(0.25, 1.0),
            tube2Surface.createPositionProportion(0.25, 0.0))
        self.assertIsNotNone(p1)
        if elementsCountAround == 8:
            self.assertEqual(p1.e1, 2)
            self.assertEqual(p1.e2, 5)
            self.assertEqual(p1bdy, 2)
            self.assertAlmostEqual(p1.xi1, 0.5513062130611348, delta=XI_TOL)
            self.assertAlmostEqual(p1.xi2, 1.0, delta=XI_TOL)
            self.assertEqual(op1.e1, 2)
            self.assertEqual(op1.e2, 0)
            self.assertAlmostEqual(op1.xi1, 0.5566789508466949, delta=XI_TOL)
            self.assertAlmostEqual(op1.xi2, 0.052793493995319365, delta=XI_TOL)
            assertAlmostEqualList(self, [-0.6607956311504671, -0.7020866361852254, -0.26537424355534356], p1t, delta=X_TOL)

            startPosition = TrackSurfacePosition(6, 5, 0.0, 1.0)
            p2, op2, p2x, p2t, p2bdy = tube1Surface.findIntersectionPoint(
                tube3Surface, startPosition, TrackSurfacePosition(6, 0, 0.0, 0.0))
            self.assertEqual(p2.e1, 6)
            self.assertEqual(p2.e2, 5)
            self.assertAlmostEqual(p2.xi1, 0.33908360205688126, delta=XI_TOL)
            self.assertAlmostEqual(p2.xi2, 0.899262977189931, delta=XI_TOL)
            self.assertEqual(op2.e1, 6)
            self.assertEqual(op2.e2, 0)
            self.assertAlmostEqual(op2.xi1, 0.4059635095925822, delta=XI_TOL)
            self.assertAlmostEqual(op2.xi2, 0.0, delta=XI_TOL)
            assertAlmostEqualList(self, [0.14121807356718433, 0.9590457759554315, 0.2455374825154349], p2t, delta=X_TOL)

            startPosition = TrackSurfacePosition(6, 5, 0.053597696166878706, 1.0)
            p3, op3, p3x, p3t, p3bdy = tube1Surface.findIntersectionPoint(
                tube3Surface, startPosition, TrackSurfacePosition(6, 0, 0.5777545444894798, 0.1230948585475792))
            self.assertEqual(p3.e1, 6)
            self.assertEqual(p3.e2, 5)
            self.assertAlmostEqual(p3.xi1, 0.33909643724125527, delta=XI_TOL)
            self.assertAlmostEqual(p3.xi2, 0.8992504028823927, delta=XI_TOL)

            startPosition = TrackSurfacePosition(5, 5, 0.9639270978611663, 0.7487688426600227)
            p4, op4, p4x, p4t, p4bdy = tube1Surface.findIntersectionPoint(
                tube2Surface, startPosition, TrackSurfacePosition(5, 0, 0.49953899323872, 0.27331994814227073))
            self.assertEqual(p4.e1, 5)
            self.assertEqual(p4.e2, 5)
            self.assertAlmostEqual(p4.xi1, 0.9585925958894155, delta=XI_TOL)
            self.assertAlmostEqual(p4.xi2, 0.7720200484294564, delta=XI_TOL)

            startPosition = TrackSurfacePosition(6, 0, 0.27152105049897823,0.0)
            p5, op5, p5x, p5t, p5bdy = tube2Surface.findIntersectionPoint(
                tube3Surface, startPosition, TrackSurfacePosition(6, 0, 0.1648182942983345,0.4437057086482662))
            self.assertEqual(p5.e1, 6)
            self.assertEqual(p5.e2, 0)
            self.assertAlmostEqual(p5.xi1, 0.3265731651257928, delta=XI_TOL)
            self.assertAlmostEqual(p5.xi2, 0.0690595773802155, delta=XI_TOL)

            startPosition = TrackSurfacePosition(5, 5, 0.8414377968474449, 0.9143600731749943)
            otherStartPosition = TrackSurfacePosition(6, 0, 0.4076509361724341, 0.0017373594850997659)
            p6, op6, p6x, p6t, p6bdy = tube1Surface.findIntersectionPoint(tube3Surface, startPosition, otherStartPosition)
            self.assertEqual(p6.e1, 6)
            self.assertEqual(p6.e2, 5)
            self.assertAlmostEqual(p6.xi1, 0.33909205890943017, delta=XI_TOL)
            self.assertAlmostEqual(p6.xi2, 0.8992546834014119, delta=XI_TOL)

            startPosition = TrackSurfacePosition(2, 0, 0.6590628632288293, 0.0)
            otherStartPosition = TrackSurfacePosition(2, 0, 0.5297149091022884, 0.09186286116765788)
            p7, op7, p7x, p7t, p7bdy = tube2Surface.findIntersectionPoint(tube3Surface, startPosition, otherStartPosition)
            self.assertEqual(p7.e1, 2)
            self.assertEqual(p7.e2, 0)
            self.assertAlmostEqual(p7.xi1, 0.34884854620107486, delta=XI_TOL)
            self.assertAlmostEqual(p7.xi2, 0.1017114317529832, delta=XI_TOL)

        if elementsCountAround == 4:
            # startPosition = TrackSurfacePosition(3, 5, 0.36148219241126434, 1.0)
            # p1x = tube1Surface.evaluateCoordinates(startPosition)
            # p1t = [0.0, 0.0, 0.0]
            # p2, op2, p2x, p2t, p2bdy = tube1Surface.findIntersectionPoint(
            #     tube2Surface, startPosition, TrackSurfacePosition(2, 0, 0.9794494698544911, 0.1562967852059411))
            # self.assertIsNotNone(p2)
            # self.assertEqual(p2.e1, 6)
            # self.assertEqual(p2.e2, 5)
            # self.assertAlmostEqual(p2.xi1, 0.33908360205244037, delta=XI_TOL)
            # self.assertAlmostEqual(p2.xi2, 0.899262977194283, delta=XI_TOL)
            # self.assertEqual(op2.e1, 6)
            # self.assertEqual(op2.e2, 0)
            # self.assertAlmostEqual(op2.xi1, 0.4059622427354084, delta=XI_TOL)
            # self.assertAlmostEqual(op2.xi2, 0.0, delta=XI_TOL)
            # assertAlmostEqualList(self, [0.1412180735660879, 0.9590457759564066, 0.24553748251225724], p2t, delta=X_TOL)

            # startPosition = TrackSurfacePosition(1, 5, 0.3631033157903245, 1.0)
            # p1x = tube1Surface.evaluateCoordinates(startPosition)
            # p1t = [0.0, 0.0, 0.0]
            # p2, op2, p2x, p2t, p2bdy = tube1Surface.findIntersectionPoint(
            #     tube2Surface, startPosition, TrackSurfacePosition(1, 0, 0.33228274550722303,0.0))
            # self.assertIsNotNone(p2)

            # startPosition = TrackSurfacePosition(1, 5, 0.3631033157903245, 1.0)
            # p1x = tube1Surface.evaluateCoordinates(startPosition)
            # p1t = [0.0, 0.0, 0.0]
            # p2, op2, p2x, p2t, p2bdy = tube1Surface.findIntersectionPoint(
            #     tube3Surface, startPosition, TrackSurfacePosition(1, 0, 0.33228274550722303, 0.0))
            # self.assertIsNotNone(p2)

            startPosition = TrackSurfacePosition(3, 0, 0.0865879388733597, 0.0)
            p1x = tube1Surface.evaluateCoordinates(startPosition)
            p1t = [0.0, 0.0, 0.0]
            p2, op2, p2x, p2t, p2bdy = tube2Surface.findIntersectionPoint(
                tube3Surface, startPosition, TrackSurfacePosition(3, 0, 0.19438867305821184, 0.0))
            self.assertIsNotNone(p2)

        # get non-loop intersection of tube1 and tube2
        ax, ad1, aprops, aloop = tube1Surface.findIntersectionCurve(tube2Surface)
        self.assertEqual(len(ax), 9)
        self.assertFalse(aloop)
        aLength = getCubicHermiteCurvesLength(ax, ad1)
        if elementsCountAround == 8:
            self.assertAlmostEqual(aLength, 0.7867534776273488, delta=X_TOL)
            assertAlmostEqualList(self, [1.0179538858287902, -0.10372009294261042, 0.22619317098715946], ax[0], delta=X_TOL)
            assertAlmostEqualList(self, [0.8745386899334123, -0.21894738318303653, -0.07777014486867816], ax[4], delta=X_TOL)
            assertAlmostEqualList(self, [0.9779282561711389, 0.0820751155699032, -0.23449126177600493], ax[8], delta=X_TOL)
            assertAlmostEqualList(self, [0.31891164977233744, 1.0], aprops[0], delta=XI_TOL)
            assertAlmostEqualList(self, [0.5615258901138819, 0.9099206438856747], aprops[4], delta=XI_TOL)
            assertAlmostEqualList(self, [0.8039061628699322, 1.0], aprops[8], delta=XI_TOL)
        elif elementsCountAround == 6:
            self.assertAlmostEqual(aLength, 0.7881053324748947, delta=X_TOL)
        elif elementsCountAround == 4:
            self.assertAlmostEqual(aLength, 0.7881053324748947, delta=X_TOL)

        # # get non-loop intersection of tube1 and tube3
        bx, bd1, bprops, bloop = tube1Surface.findIntersectionCurve(tube3Surface)
        self.assertEqual(len(bx), 9)
        self.assertFalse(bloop)
        bLength = getCubicHermiteCurvesLength(bx, bd1)
        if elementsCountAround == 8:
            self.assertAlmostEqual(bLength, 0.7608142851170996, delta=X_TOL)
            assertAlmostEqualList(self, [0.959094526660969, 0.06711431057004068, -0.2372654694905775], bx[0], delta=X_TOL)
            assertAlmostEqualList(self, [0.9486154683235082, 0.2545005003987398, 0.05281998774919236], bx[4], delta=X_TOL)
            assertAlmostEqualList(self, [1.0147890884040247, -0.023171620406177137, 0.24848738902991338], bx[8], delta=X_TOL)
            assertAlmostEqualList(self, [0.7923873358701699, 0.9832080394698671], bprops[0], delta=XI_TOL)
            assertAlmostEqualList(self, [1.0300414330440506, 0.9150675749144951], bprops[4], delta=XI_TOL)
            assertAlmostEqualList(self, [1.2664762695247593, 0.9841126335511698], bprops[8], delta=XI_TOL)
        elif elementsCountAround == 6:
            self.assertAlmostEqual(bLength, 0.7569291346493467, delta=X_TOL)
        elif elementsCountAround == 4:
            self.assertAlmostEqual(bLength, 0.7568936438693603, delta=X_TOL)

        # get non-loop intersection of tube2 and tube3
        cx, cd1, cprops, cloop = tube2Surface.findIntersectionCurve(tube3Surface)
        self.assertEqual(len(cx), 9)
        self.assertFalse(cloop)
        cLength = getCubicHermiteCurvesLength(cx, cd1)
        if elementsCountAround == 8:
            self.assertAlmostEqual(cLength, 0.9966002517592469, delta=X_TOL)
            assertAlmostEqualList(self, [1.0515358634986118, -0.08502266290179061, 0.2293525653558907], cx[0], delta=X_TOL)
            assertAlmostEqualList(self, [1.3713633820298987, -0.024409957536963414, -0.08284306983602606], cx[4], delta=X_TOL)
            assertAlmostEqualList(self, [0.9646346791047065, 0.057787366383426825, -0.24058971254661396], cx[8], delta=X_TOL)
            assertAlmostEqualList(self, [1.29558913002461046, 0.016955772619840683], cprops[0], delta=XI_TOL)
            assertAlmostEqualList(self, [1.051351610846633235, 0.2953763298880561], cprops[4], delta=XI_TOL)
            assertAlmostEqualList(self, [0.7907961771250209, 0.011505780523740128], cprops[8], delta=XI_TOL)
        elif elementsCountAround == 6:
            self.assertAlmostEqual(cLength, 0.991440388450334, delta=X_TOL)
        elif elementsCountAround == 4:
            self.assertAlmostEqual(cLength, 0.9914369553370298, delta=X_TOL)

        # context = Context("TrackSurface")
        # region = context.getDefaultRegion()
        # surfaceGroupName = "surface"
        # tube1Surface.generateMesh(region, group_name=surfaceGroupName)
        # tube2Surface.generateMesh(region, group_name=surfaceGroupName)
        # tube3Surface.generateMesh(region, group_name=surfaceGroupName)
        # coordinateFieldName = "curve_coordinates"
        # curveGroupName = "curve"
        # fieldmodule = region.getFieldmodule()
        # curveCoordinates = find_or_create_field_coordinates(fieldmodule, coordinateFieldName, managed=True)
        # curveGroup = find_or_create_field_group(fieldmodule, curveGroupName)
        # nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # nodetemplate = nodes.createNodetemplate()
        # nodetemplate.defineField(curveCoordinates)
        # nodetemplate.setValueNumberOfVersions(curveCoordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        # nodetemplate.setValueNumberOfVersions(curveCoordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        # curveNodesetGroup = curveGroup.getOrCreateNodesetGroup(nodes)
        # fieldcache = fieldmodule.createFieldcache()
        # zero = [0.0, 0.0, 0.0]
        # px = [pax, pbx, pcx]  # , p1x, p2x, p3x, p4x, p5x, p6x, p7x]
        # pd1 = [zero, pbd1, pcd1]  # , p1t, p2t, p3t, p4t, p5t, p6t, p7t]
        # pd2 = [zero, pbd2, pcd2]
        # for n in range(len(px)):
        #     node = nodes.createNode(-1, nodetemplate)
        #     fieldcache.setNode(node)
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, px[n])
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, pd1[n])
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, pd2[n])
        #     curveNodesetGroup.addNode(node)
        # generate_curve_mesh(region, ax, ad1, coordinate_field_name=coordinateFieldName, group_name=curveGroupName)
        # generate_curve_mesh(region, bx, bd1, coordinate_field_name=coordinateFieldName, group_name=curveGroupName)
        # generate_curve_mesh(region, cx, cd1, coordinate_field_name=coordinateFieldName, group_name=curveGroupName)

    def test_tube_intersections3(self):
        """
        Regular converging bifurcation failed to find boundary intersection point.
        """
        elementsCountAround = 10
        # elementsCountAlong = 4
        path1Params = [
            [[0.0, -0.25, 0.0], [1.0, 0.0, 0.0]],
            [[0.9701425001453319, 0.24253562503633297, 0.0], [0.9701425001453319, 0.24253562503633297, 0.0]],
            [[-0.06063390625908324, 0.24253562503633297, 0.0], [-0.06063390625908324, 0.24253562503633297, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, -0.0, 0.25], [0.0, -0.0, 0.25]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path1Params, elementsCountAround, subsample_count=1)
        # px, pd1, pd2, pd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(px)):
            nx += px[i]
            nd1 += pd1[i]
            nd2 += pd2[i]
            nd12 += pd12[i]
        tube1Surface = TrackSurface(len(px[0]), len(px) - 1, nx, nd1, nd2, nd12, loop1=True)

        path2Params = [
            [[0.0, 0.25, 0.0], [1.0, 0.0, 0.0]],
            [[0.9701425001453319, -0.24253562503633297, 0.0], [0.9701425001453319, -0.24253562503633297, 0.0]],
            [[0.06063390625908324, 0.24253562503633297, 0.0], [0.06063390625908324, 0.24253562503633297, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-0.0, 0.0, 0.25], [-0.0, 0.0, 0.25]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        px, pd1, pd2, pd12 = getPathRawTubeCoordinates(path2Params, elementsCountAround, subsample_count=1)
        # px, pd1, pd2, pd12 = resampleTubeCoordinates((px, pd1, pd2, pd12), elementsCountAlong)
        nx = []
        nd1 = []
        nd2 = []
        nd12 = []
        for i in range(len(px)):
            nx += px[i]
            nd1 += pd1[i]
            nd2 += pd2[i]
            nd12 += pd12[i]
        tube2Surface = TrackSurface(len(px[0]), len(px) - 1, nx, nd1, nd2, nd12, loop1=True)

        # path3Params = [
        #     [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        #     [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        #     [[-0.0, 0.25, 0.0], [-0.0, 0.25, 0.0]],
        #     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #     [[0.0, -0.0, 0.25], [0.0, -0.0, 0.25]],
        #     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]

        XI_TOL = 1.0E-6
        X_TOL = 1.0E-6

        startPosition = TrackSurfacePosition(12, 0, 0.7886706262895231, 1.0)
        p1x = tube1Surface.evaluateCoordinates(startPosition)
        otherStartPosition = TrackSurfacePosition(2, 0, 0.636882069585452, 0.9147548823734029)
        p2, op2, p2x, p2t, p2bdy = tube1Surface.findIntersectionPoint(
            tube2Surface, startPosition, otherStartPosition)
        self.assertEqual(p2.e1, 12)
        self.assertEqual(p2.e2, 0)
        self.assertAlmostEqual(p2.xi1, 0.4973473706811795, delta=XI_TOL)
        self.assertAlmostEqual(p2.xi2, 1.0, delta=XI_TOL)
        self.assertEqual(op2.e1, 2)
        self.assertEqual(op2.e2, 0)
        self.assertAlmostEqual(op2.xi1, 0.49765944425751174, delta=XI_TOL)
        self.assertAlmostEqual(op2.xi2, 0.9998039549126785, delta=XI_TOL)
        self.assertIsNone(p2t)

        startPosition = TrackSurfacePosition(0, 0, 0.08468387605632358, 0.0)
        p3x = tube1Surface.evaluateCoordinates(startPosition)
        otherStartPosition = TrackSurfacePosition(4, 0, 0.42675444143171326, 0.09458268381308738)
        p4, op4, p4x, p4t, p4bdy = tube1Surface.findIntersectionPoint(
            tube2Surface, startPosition, otherStartPosition)
        self.assertEqual(p4.e1, 0)
        self.assertEqual(p4.e2, 0)
        self.assertAlmostEqual(p4.xi1, 0.001112953121053552, delta=XI_TOL)
        self.assertAlmostEqual(p4.xi2, 0.030691469925266602, delta=XI_TOL)
        assertAlmostEqualList(self, [0.002978246340503309, 1.6957770277963287e-12, 0.999995565014533], p4t, delta=X_TOL)

        # get non-loop intersection of tube1 and tube2
        ax, ad1, aprops, aloop = tube1Surface.findIntersectionCurve(tube2Surface)
        self.assertEqual(len(ax), 9)
        self.assertFalse(aloop)
        aLength = getCubicHermiteCurvesLength(ax, ad1)
        self.assertAlmostEqual(aLength, 2.214117090392043, delta=X_TOL)
        assertAlmostEqualList(self, [0.9999027578036812, 0.00038896878527516776, -0.24987413008984302], ax[0], delta=X_TOL)
        assertAlmostEqualList(self, [-0.03077634623170676, 1.5966942528521302e-09, 7.945917568834134e-05], ax[4], delta=X_TOL)
        assertAlmostEqualList(self, [0.9997557619375088, 0.0009769522499643996, 0.24987245334458275], ax[8], delta=X_TOL)
        assertAlmostEqualList(self, [0.7502552895299374, 1.0], aprops[0], delta=XI_TOL)
        assertAlmostEqualList(self, [1.0000505870312695, 0.030691274736817955], aprops[4], delta=XI_TOL)
        assertAlmostEqualList(self, [1.249358801390615, 1.0], aprops[8], delta=XI_TOL)

        # context = Context("TrackSurface")
        # region = context.getDefaultRegion()
        # surfaceGroupName = "surface"
        # tube1Surface.generateMesh(region, group_name=surfaceGroupName)
        # tube2Surface.generateMesh(region, group_name=surfaceGroupName)
        # coordinateFieldName = "curve_coordinates"
        # curveGroupName = "curve"
        # fieldmodule = region.getFieldmodule()
        # curveCoordinates = find_or_create_field_coordinates(fieldmodule, coordinateFieldName, managed=True)
        # curveGroup = find_or_create_field_group(fieldmodule, curveGroupName)
        # nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # nodetemplate = nodes.createNodetemplate()
        # nodetemplate.defineField(curveCoordinates)
        # nodetemplate.setValueNumberOfVersions(curveCoordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        # curveNodesetGroup = curveGroup.getOrCreateNodesetGroup(nodes)
        # fieldcache = fieldmodule.createFieldcache()
        # px = [p1x, p2x, p3x, p4x]
        # pd1 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], p4t]
        # for n in range(len(px)):
        #     node = nodes.createNode(-1, nodetemplate)
        #     fieldcache.setNode(node)
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, px[n])
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, pd1[n])
        #     curveNodesetGroup.addNode(node)
        # generate_curve_mesh(region, ax, ad1, coordinate_field_name=coordinateFieldName, group_name=curveGroupName)

    def test_2d_tube_intersections_bifurcation(self):
        """
        Test 2D bifurcation tube intersections.
        """
        scaffoldPackage = ScaffoldPackage(MeshType_1d_network_layout1, defaultParameterSetName="Bifurcation")

        context = Context("Test")
        region = context.getDefaultRegion()
        self.assertTrue(region.isValid())
        scaffoldPackage.generate(region)

        fieldmodule = region.getFieldmodule()
        mesh1d = fieldmodule.findMeshByDimension(1)
        self.assertEqual(3, mesh1d.getSize())
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.assertEqual(4, nodes.getSize())
        coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
        self.assertTrue(coordinates.isValid())

        networkMesh = scaffoldPackage.getConstructionObject()
        valueLabels = [
            Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
            Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
            Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3]
        tubeSegments = []
        trackSurfaces = []
        networkSegments = networkMesh.getNetworkSegments()
        elementsCountAround = 8
        for networkSegment in networkSegments:
            pathParameters = get_nodeset_path_ordered_field_parameters(
                nodes, coordinates, valueLabels, networkSegment.getNodeIdentifiers(), networkSegment.getNodeVersions())
            tubeSegment = TubeNetworkMeshSegment(networkSegment, [pathParameters], elementsCountAround, shell_count=0)
            tubeSegments.append(tubeSegment)
            trackSurfaces.append(tubeSegment.getRawTrackSurface())

        XI_TOL = 1.0E-6
        X_TOL = 1.0E-6

        p1x = targetx = [0.5, 0.1, 0.6]
        startPosition = TrackSurfacePosition(2, 0, 0.5, 0.5)
        p2x = trackSurfaces[0].evaluateCoordinates(startPosition)
        nearestPosition = trackSurfaces[0].findNearestPosition(targetx, startPosition)
        p3x = trackSurfaces[0].evaluateCoordinates(nearestPosition)
        self.assertEqual(nearestPosition.e1, 1)
        self.assertEqual(nearestPosition.e2, 0)
        self.assertAlmostEqual(nearestPosition.xi1, 0.7937236157191407, delta=XI_TOL)
        self.assertAlmostEqual(nearestPosition.xi2, 1.0, delta=XI_TOL)

        px, pd1, pd2, pd12 = tubeSegments[0].getRawTubeCoordinates()
        cx = [px[0][4], px[1][4], px[2][4]]
        cd1 = [pd2[0][4], pd2[1][4], pd2[2][4]]
        nearestPosition, nearestCurveLocation, isIntersection = \
            trackSurfaces[1].findNearestPositionOnCurve(cx, cd1, loop=False, sampleEnds=False)
        p4x = evaluateCoordinatesOnCurve(cx, cd1, nearestCurveLocation)
        p5x = trackSurfaces[1].evaluateCoordinates(nearestPosition)
        self.assertTrue(isIntersection)
        self.assertEqual(nearestCurveLocation[0], 1)
        self.assertAlmostEqual(nearestCurveLocation[1], 0.9527864045000324, delta=XI_TOL)
        self.assertEqual(nearestPosition.e1, 4)
        self.assertEqual(nearestPosition.e2, 0)
        self.assertAlmostEqual(nearestPosition.xi1, 0.0, delta=XI_TOL)
        self.assertAlmostEqual(nearestPosition.xi2, 0.042229123600025925, delta=XI_TOL)

        # distant point
        p6x = targetx = [0.8187820665733468, -0.1, 0.0]
        startPosition = TrackSurfacePosition(7, 0, 0.99, 0.0)
        p7x = trackSurfaces[2].evaluateCoordinates(startPosition)
        nearestPosition = trackSurfaces[2].findNearestPosition(targetx, startPosition)
        p8x = trackSurfaces[2].evaluateCoordinates(nearestPosition)
        self.assertEqual(nearestPosition.e1, 3)
        self.assertEqual(nearestPosition.e2, 0)
        self.assertAlmostEqual(nearestPosition.xi1, 1.0, delta=XI_TOL)
        self.assertAlmostEqual(nearestPosition.xi2, 0.0, delta=XI_TOL)

        # non-intersecting curve and surface
        px, _, pd2, _ = tubeSegments[0].getRawTubeCoordinates()
        cx = [px[0][0], px[1][0], px[2][0]]
        cd1 = [pd2[0][0], pd2[1][0], pd2[2][0]]
        nearestPosition, nearestCurveLocation, isIntersection = \
            trackSurfaces[1].findNearestPositionOnCurve(cx, cd1, loop=False, sampleEnds=False)
        p9x = evaluateCoordinatesOnCurve(cx, cd1, nearestCurveLocation)
        p10x = trackSurfaces[1].evaluateCoordinates(nearestPosition)
        self.assertFalse(isIntersection)
        self.assertEqual(nearestCurveLocation[0], 1)
        self.assertAlmostEqual(nearestCurveLocation[1], 1.0, delta=XI_TOL)
        self.assertEqual(nearestPosition.e1, 15)
        self.assertEqual(nearestPosition.e2, 0)
        self.assertAlmostEqual(nearestPosition.xi1, 1.0, delta=XI_TOL)
        self.assertAlmostEqual(nearestPosition.xi2, 0.0, delta=XI_TOL)

        # context = Context("TrackSurface")
        # region = context.getDefaultRegion()
        # fieldmodule = region.getFieldmodule()
        # surfaceGroupName = "surface"
        # for i in range(3):
        #     trackSurfaces[i].generateMesh(region, group_name=surfaceGroupName)
        # coordinateFieldName = "curve_coordinates"
        # curveGroupName = "curve"
        # curveCoordinates = find_or_create_field_coordinates(fieldmodule, coordinateFieldName, managed=True)
        # curveGroup = find_or_create_field_group(fieldmodule, curveGroupName)
        # nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # nodetemplate = nodes.createNodetemplate()
        # nodetemplate.defineField(curveCoordinates)
        # nodetemplate.setValueNumberOfVersions(curveCoordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        # curveNodesetGroup = curveGroup.getOrCreateNodesetGroup(nodes)
        # fieldcache = fieldmodule.createFieldcache()
        # px = [p1x, p2x, p3x, cx[0], cx[1], p4x, p5x, p6x, p7x, p8x, p9x, p10x]
        # zero = [0.0, 0.0, 0.0]
        # pd1 = [zero, zero, zero, cd1[0], cd1[1], zero, zero, zero, zero, zero, zero, zero]
        # for n in range(len(px)):
        #     node = nodes.createNode(-1, nodetemplate)
        #     fieldcache.setNode(node)
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, px[n])
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, pd1[n])
        #     curveNodesetGroup.addNode(node)

    def test_2d_tube_intersections_sphere_cube(self):
        """
        Test 2D tube intersections on a section of the sphere cube.
        """
        cx = [[0.5002720342505129, 0.049999999999999996, -0.08501700857389402],
              [0.2645697738549972, -0.35824829046386303, 0.2483163247594392]]
        cd1 = [[0.10257995144506454, -0.533021063222401, 0.29013991712236775],
               [-0.5128997572253229, -0.17767368774080045, 0.29013991712236775]]
        nx = [[-0.2934372873144786, -0.40824829046386296, -0.08501700857389399],
              [-0.30539112847838207, -0.47895555285030167, -0.15471008125537822],
              [-0.24765910625478924, -0.4789555528487444, -0.23635549006530096],
              [-0.17796723347655352, -0.40824829046386296, -0.24831632475943927],
              [-0.16601339231265005, -0.3375410280774243, -0.178623252077955],
              [-0.22374541453624286, -0.33754102807898145, -0.09697784326803233],
              [0.17796723347655338, -0.408248290463863, 0.2483163247594392],
              [0.24765910625759371, -0.47895555285030167, 0.2363554900632418],
              [0.3053911284793884, -0.4789555528487444, 0.15471008125204752],
              [0.29343728731447843, -0.408248290463863, 0.08501700857389394],
              [0.22374541453343808, -0.3375410280774243, 0.09697784327009142],
              [0.16601339231164347, -0.33754102807898156, 0.17862325208128554]]
        nd1 = [[-0.04936372890878149, -0.08550048652106605, -0.03490542745605377],
               [0.027648784957331453, -0.04277813171414004, -0.0914935840944021],
               [0.07704471667898151, 0.04277813171695018, -0.05656538580855996],
               [0.04936372890878152, 0.08550048652106605, 0.03490542745605376],
               [-0.027648784957331436, 0.04277813171413982, 0.09149358409440221],
               [-0.07704471667898151, -0.042778131716949955, 0.05656538580856002],
               [0.04936372890878148, -0.08550048652106605, 0.03490542745605377],
               [0.07704471667850743, -0.04277813171413998, -0.05656538581133083],
               [0.027648784954560728, 0.04277813171695016, -0.09149358409392554],
               [-0.04936372890878149, 0.0855004865210661, -0.03490542745605379],
               [-0.07704471667850739, 0.04277813171413971, 0.05656538581133116],
               [-0.027648784954560936, -0.04277813171694988, 0.09149358409392555]]
        nd2 = [[0.41031980578025823, -0.3553473754816008, 0.29013991712236753],
               [0.48138584320204963, -0.4168923692351677, 0.3403911940953731],
               [0.4813858432004845, -0.4168923692338123, 0.34039119409426644],
               [0.41031980578025823, -0.3553473754816008, 0.29013991712236753],
               [0.3392537683584668, -0.2938023817280338, 0.23988864014936193],
               [0.33925376836003196, -0.2938023817293892, 0.23988864015046862],
               [0.41031980578025834, 0.35534737548160056, 0.29013991712236753],
               [0.3214872590030191, 0.3245748786048171, 0.34039119409537316],
               [0.3214872590049755, 0.3245748786054948, 0.34039119409426644],
               [0.41031980578025834, 0.35534737548160056, 0.29013991712236753],
               [0.4991523525574976, 0.3861198723583841, 0.23988864014936187],
               [0.49915235255554125, 0.38611987235770634, 0.23988864015046857]]
        nd12 = [[0.08593717887190072, -0.07442378003263339, 0.06076676193636225],
                [0.04299452295280077, -0.03723434910071862, 0.030401718733806066],
                [-0.04299452295562481, 0.03723434910316431, -0.03040171873580296],
                [-0.08593717887190085, 0.0744237800326335, -0.06076676193636234],
                [-0.04299452295280081, 0.037234349100718656, -0.030401718733806093],
                [0.04299452295562462, -0.03723434910316415, 0.03040171873580283],
                [-0.10742147358987579, -0.03721189001631667, 0.06076676193636223],
                [-0.05374315369100107, -0.018617174550359356, 0.03040171873380615],
                [0.053743153694531025, 0.018617174551582166, -0.030401718735802993],
                [0.10742147358987611, 0.03721189001631678, -0.060766761936362414],
                [0.05374315369100105, 0.018617174550359345, -0.03040171873380614],
                [-0.053743153694530844, -0.018617174551582103, 0.03040171873580289]]
        trackSurface = TrackSurface(6, 1, nx, nd1, nd2, nd12, loop1=True)

        XI_TOL = 1.0E-6

        p1x = [0.3374906068069012, -0.3237385667049661, 0.2014021134309831]
        startPosition = TrackSurfacePosition(3, 0, 0.38466571419957285, 1.0)
        p2x = trackSurface.evaluateCoordinates(startPosition)
        nearestPosition = trackSurface.findNearestPosition(p1x, startPosition)
        p3x = trackSurface.evaluateCoordinates(nearestPosition)
        self.assertEqual(nearestPosition.e1, 3)
        self.assertEqual(nearestPosition.e2, 0)
        self.assertAlmostEqual(nearestPosition.xi1, 0.3044617879942786, delta=XI_TOL)
        self.assertAlmostEqual(nearestPosition.xi2, 1.0, delta=XI_TOL)

        p4x = [0.32823354701594526, -0.32925428917806804, 0.2080526547124757]
        startPosition = TrackSurfacePosition(3, 0, 0.6253731101993623, 1.0)
        p5x = trackSurface.evaluateCoordinates(startPosition)
        nearestPosition = trackSurface.findNearestPosition(p4x, startPosition)
        p6x = trackSurface.evaluateCoordinates(nearestPosition)
        self.assertEqual(nearestPosition.e1, 3)
        self.assertEqual(nearestPosition.e2, 0)
        self.assertAlmostEqual(nearestPosition.xi1, 0.3549945895608424, delta=XI_TOL)
        self.assertAlmostEqual(nearestPosition.xi2, 1.0, delta=XI_TOL)

        nearestPosition, nearestCurveLocation, isIntersection = \
            trackSurface.findNearestPositionOnCurve(cx, cd1, sampleEnds=False)
        p7x = trackSurface.evaluateCoordinates(nearestPosition)
        p8x = evaluateCoordinatesOnCurve(cx, cd1, nearestCurveLocation)
        self.assertFalse(isIntersection)
        self.assertEqual(nearestPosition.e1, 3)
        self.assertEqual(nearestPosition.e2, 0)
        self.assertAlmostEqual(nearestPosition.xi1, 0.32559566990023603, delta=XI_TOL)
        self.assertAlmostEqual(nearestPosition.xi2, 1.0, delta=XI_TOL)
        self.assertEqual(nearestCurveLocation[0], 0)
        self.assertAlmostEqual(nearestCurveLocation[1], 0.8587082918561665, delta=XI_TOL)

        # context = Context("TrackSurface")
        # region = context.getDefaultRegion()
        # fieldmodule = region.getFieldmodule()
        # surfaceGroupName = "surface"
        # trackSurface.generateMesh(region, group_name=surfaceGroupName)
        # coordinateFieldName = "curve_coordinates"
        # curveGroupName = "curve"
        # curveCoordinates = find_or_create_field_coordinates(fieldmodule, coordinateFieldName, managed=True)
        # curveGroup = find_or_create_field_group(fieldmodule, curveGroupName)
        # generate_curve_mesh(region, cx, cd1, loop=False, coordinate_field_name=coordinateFieldName, group_name=curveGroupName)
        # nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # nodetemplate = nodes.createNodetemplate()
        # nodetemplate.defineField(curveCoordinates)
        # nodetemplate.setValueNumberOfVersions(curveCoordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        # nodetemplate.setValueNumberOfVersions(curveCoordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        # curveNodesetGroup = curveGroup.getOrCreateNodesetGroup(nodes)
        # fieldcache = fieldmodule.createFieldcache()
        # px = [p1x, p2x, p3x, p4x, p5x, p6x, p7x, p8x]
        # zero = [0.0, 0.0, 0.0]
        # pd1 = [zero, zero, zero, zero, zero, zero, zero, zero]
        # pd2 = [zero, zero, zero, zero, zero, zero, zero, zero]
        # for n in range(len(px)):
        #     node = nodes.createNode(-1, nodetemplate)
        #     fieldcache.setNode(node)
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, px[n])
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, pd1[n])
        #     curveCoordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, pd2[n])
        #     curveNodesetGroup.addNode(node)

    def test_smooth_side_cross_derivatives(self):
        """
        Test algorithm for smoothing side cross derivatives used in network layout.
        """
        loopRadius = 0.5
        tubeRadius1 = 0.05
        tubeRadius2 = 0.15
        deltaRadius = tubeRadius2 - tubeRadius1
        angleAround = 0.5 * math.pi
        d1Mag = loopRadius * angleAround
        x_list = []
        d1_list = []
        d2_list = []
        d3_list = []
        for n in range(2):
            xi = float(n)
            tubeRadius = (1.0 - xi) * tubeRadius1 + xi * tubeRadius2
            angle = angleAround * xi
            cosAngle = math.cos(angle)
            sinAngle = math.sin(angle)
            x_list.append([loopRadius * cosAngle, loopRadius * sinAngle, 0.0])
            d1_list.append([-d1Mag * sinAngle, d1Mag * cosAngle, 0.0])
            d2_list.append([0.0, 0.0, tubeRadius])
            d3_list.append([tubeRadius * cosAngle, tubeRadius * sinAngle, 0.0])

        ascd_list, bscd_list = computeCubicHermiteSideCrossDerivatives(
            x_list[0], d1_list[0], x_list[1], d1_list[1], [d2_list[0], d3_list[0]], [d2_list[1], d3_list[1]])
        d12_list = [ascd_list[0], bscd_list[0]]
        d13_list = [ascd_list[1], bscd_list[1]]

        X_TOL = 1.0E-6
        assertAlmostEqualList(self, [0.0, 0.0, deltaRadius], d12_list[0], delta=X_TOL)
        assertAlmostEqualList(self, [0.0, 0.0, deltaRadius], d12_list[1], delta=X_TOL)
        assertAlmostEqualList(self, [0.09353034063054447, 0.09057565385867462, 0.0], d13_list[0], delta=X_TOL)
        assertAlmostEqualList(self, [-0.27450784324366634, 0.12218985977600845, -0.0], d13_list[1], delta=X_TOL)

        # check rotation around curve for d13
        ANGLE_TOL = 1.6  # degrees
        LENGTH_TOL = 6.0E-4
        for n in range(5):
            xi = 0.25 * n
            d3 = interpolateCubicHermite(d3_list[0], d13_list[0], d3_list[1], d13_list[1], xi)
            targetAngle = math.degrees(xi * angleAround)
            actualAngle = math.degrees(math.atan2(d3[1], d3[0]))
            targetLength = (1.0 - xi) * tubeRadius1 + xi * tubeRadius2
            actualLength = magnitude(d3)
            self.assertAlmostEqual(targetAngle, actualAngle, delta=ANGLE_TOL)
            self.assertAlmostEqual(targetLength, actualLength, delta=LENGTH_TOL)
            # print("xi", xi, "length", actualLength, "angle", actualAngle, targetAngle)

    def test_determineHermiteSerendipityEft(self):
        """
        Test algorithm for determining hermite serendipity eft from node derivative directions.
        """
        context = Context("test_determineHermiteSerendipityEft")
        region = context.getDefaultRegion()
        fieldmodule = region.getFieldmodule()
        mesh3d = fieldmodule.findMeshByDimension(3)

        nodeParameters = [
            [[-1.064, 0.152, 0.035], [0.776, -0.051, -0.204], [0.000, 3.000, 0.000], [0.000, 0.000, 1.000]],
            [[-0.227, -0.113, 0.004], [0.069, -0.003, 0.880], [0.000, 3.000, 0.000], [-0.920, 0.188, 0.028]],
            [[0.000, 3.000, 0.000], [1.000, 0.000, 0.000], [0.000, 3.000, 0.000], [0.000, 0.000, 1.000]],
            [[0.700, 2.967, -0.194], [0.075, -0.196, -0.726], [0.000, 3.000, 0.000], [0.793, 0.064, 0.082]],
            [[-1.033, 0.135, 1.042], [-0.196, 2.329, 0.099], [-0.330, -0.052, 0.698], [0.785, -0.320, -0.149]],
            [[-0.174, 0.001, 0.883], [1.000, 0.000, 0.000], [0.000, 3.000, 0.000], [0.000, 0.000, 1.000]],
            [[-0.904, 2.968, 1.362], [1.657, -0.955, -0.204], [-1.299, 2.712, 0.142], [0.000, 0.000, 1.000]],
            [[0.337, 3.541, 1.461], [1.000, 0.000, 0.000], [-0.681, 2.850, -0.167], [0.000, 0.000, 1.000]]
        ]
        nodeDerivativeFixedWeights = [
            [],
            [],
            [],
            [],
            [],
            [],
            [None, [-1.0, 1.0]],
            []
        ]
        eft, scalefactors = determineTricubicHermiteEft(mesh3d, nodeParameters, nodeDerivativeFixedWeights,
                                                        serendipity=True)
        regularTermExpressions =\
            [[(Node.VALUE_LABEL_D_DS1, [])], [(Node.VALUE_LABEL_D_DS2, [])], [(Node.VALUE_LABEL_D_DS3, [])]]
        expectedNodeTermExpressions = [
            regularTermExpressions,
            [[(Node.VALUE_LABEL_D_DS3, [1])], [(Node.VALUE_LABEL_D_DS2, [])], [(Node.VALUE_LABEL_D_DS1, [])]],
            regularTermExpressions,
            [[(Node.VALUE_LABEL_D_DS3, [])], [(Node.VALUE_LABEL_D_DS2, [])], [(Node.VALUE_LABEL_D_DS1, [1])]],
            [[(Node.VALUE_LABEL_D_DS3, [])], [(Node.VALUE_LABEL_D_DS1, [])], [(Node.VALUE_LABEL_D_DS2, [])]],
            regularTermExpressions,
            [[(Node.VALUE_LABEL_D_DS1, [])], [(Node.VALUE_LABEL_D_DS1, [1]), (Node.VALUE_LABEL_D_DS2, [])],
             [(Node.VALUE_LABEL_D_DS3, [])]],
            regularTermExpressions
        ]
        self.assertEqual(scalefactors, [-1.0])
        self.assertEqual(eft.getNumberOfLocalScaleFactors(), 1)
        self.assertEqual(eft.getScaleFactorType(1), eft.SCALE_FACTOR_TYPE_GLOBAL_GENERAL)
        self.assertEqual(eft.getScaleFactorIdentifier(1), 1)
        for n in range(8):
            ln = n + 1
            for ed in range(3):
                functionNumber = n * 4 + ed + 2
                expectedTermExpression = expectedNodeTermExpressions[n][ed]
                expectedTermCount = len(expectedTermExpression)
                self.assertEqual(eft.getFunctionNumberOfTerms(functionNumber), expectedTermCount)
                for t in range(expectedTermCount):
                    term = t + 1
                    self.assertEqual(eft.getTermLocalNodeIndex(functionNumber, term), ln)
                    self.assertEqual(eft.getTermNodeValueLabel(functionNumber, term), expectedTermExpression[t][0])
                    self.assertEqual(eft.getTermNodeVersion(functionNumber, term), 1)
                    expectedScaleCount = len(expectedTermExpression[t][1])
                    actualScaleCount, scalefactorIndexes = eft.getTermScaling(functionNumber, term, expectedScaleCount)
                    self.assertEqual(actualScaleCount, expectedScaleCount)
                    if expectedScaleCount == 1:
                        self.assertEqual(scalefactorIndexes, 1)
                    else:
                        for s in range(expectedScaleCount):
                            self.assertEqual(scalefactorIndexes[s], 1)

        # test mapping cross derivatives for full tricubic Hermite
        eft, scalefactors = determineTricubicHermiteEft(mesh3d, nodeParameters, nodeDerivativeFixedWeights,
                                                        serendipity=False, mapCrossDerivatives=True)

        regularCrossTermExpressions =\
            [[(Node.VALUE_LABEL_D2_DS1DS2, [])], [(Node.VALUE_LABEL_D2_DS1DS3, [])],
             [(Node.VALUE_LABEL_D2_DS2DS3, [])], [(Node.VALUE_LABEL_D3_DS1DS2DS3, [])]]
        expectedNodeCrossTermExpressions = [
            regularCrossTermExpressions,
            [[(Node.VALUE_LABEL_D2_DS2DS3, [1])], [(Node.VALUE_LABEL_D2_DS1DS3, [1])],
             [(Node.VALUE_LABEL_D2_DS1DS2, [])],  [(Node.VALUE_LABEL_D3_DS1DS2DS3, [1])]],
            regularCrossTermExpressions,
            [[(Node.VALUE_LABEL_D2_DS2DS3, [])], [(Node.VALUE_LABEL_D2_DS1DS3, [1])],
             [(Node.VALUE_LABEL_D2_DS1DS2, [1])], [(Node.VALUE_LABEL_D3_DS1DS2DS3, [1])]],
            [[(Node.VALUE_LABEL_D2_DS1DS3, [])], [(Node.VALUE_LABEL_D2_DS2DS3, [])],
             [(Node.VALUE_LABEL_D2_DS1DS2, [])], [(Node.VALUE_LABEL_D3_DS1DS2DS3, [])]],
            regularCrossTermExpressions,
            [[], [(Node.VALUE_LABEL_D2_DS1DS3, [])], [], []],
            regularCrossTermExpressions
        ]
        crossDerivativeLabels = [Node.VALUE_LABEL_D2_DS1DS2, Node.VALUE_LABEL_D2_DS1DS3,
                                 Node.VALUE_LABEL_D2_DS2DS3, Node.VALUE_LABEL_D3_DS1DS2DS3]
        self.assertEqual(scalefactors, [-1.0])
        self.assertEqual(eft.getNumberOfLocalScaleFactors(), 1)
        self.assertEqual(eft.getScaleFactorType(1), eft.SCALE_FACTOR_TYPE_GLOBAL_GENERAL)
        self.assertEqual(eft.getScaleFactorIdentifier(1), 1)
        for n in range(8):
            ln = n + 1
            for cd in range(4):
                functionNumber = n * 8 + crossDerivativeLabels[cd]
                expectedTermExpression = expectedNodeCrossTermExpressions[n][cd]
                expectedTermCount = len(expectedTermExpression)
                self.assertEqual(eft.getFunctionNumberOfTerms(functionNumber), expectedTermCount)
                for t in range(expectedTermCount):
                    term = t + 1
                    self.assertEqual(eft.getTermLocalNodeIndex(functionNumber, term), ln)
                    self.assertEqual(eft.getTermNodeValueLabel(functionNumber, term), expectedTermExpression[t][0])
                    self.assertEqual(eft.getTermNodeVersion(functionNumber, term), 1)
                    expectedScaleCount = len(expectedTermExpression[t][1])
                    actualScaleCount, scalefactorIndexes = eft.getTermScaling(functionNumber, term, expectedScaleCount)
                    self.assertEqual(actualScaleCount, expectedScaleCount)
                    if expectedScaleCount == 1:
                        self.assertEqual(scalefactorIndexes, 1)
                    else:
                        for s in range(expectedScaleCount):
                            self.assertEqual(scalefactorIndexes[s], 1)


if __name__ == "__main__":
    unittest.main()
