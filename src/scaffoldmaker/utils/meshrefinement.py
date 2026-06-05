"""
Class for refining a mesh from one region to another.
"""
from cmlibs.utils.zinc.field import findOrCreateFieldCoordinates, findOrCreateFieldGroup, \
    findOrCreateFieldStoredMeshLocation, findOrCreateFieldStoredString
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from cmlibs.zinc.result import RESULT_OK
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, findAnnotationGroupByName
from scaffoldmaker.utils.octree import Octree

import copy
import math


class MeshRefinement:
    """
    Class for refining a mesh from one region to another.
    """

    def __init__(self, sourceRegion, targetRegion, sourceAnnotationGroups=[]):
        """
        Assumes targetRegion is empty.
        :param sourceAnnotationGroups: List of AnnotationGroup for source mesh in sourceRegion.
        A copy containing the refined elements is created by the MeshRefinement.
        """
        self._sourceRegion = sourceRegion
        self._sourceFm = sourceRegion.getFieldmodule()
        self._sourceCache = self._sourceFm.createFieldcache()
        self._sourceCoordinates = findOrCreateFieldCoordinates(self._sourceFm)
        # get range of source coordinates for octree range
        self._sourceFm.beginChange()
        sourceNodes = self._sourceFm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        minimumsField = self._sourceFm.createFieldNodesetMinimum(self._sourceCoordinates, sourceNodes)
        result, minimums = minimumsField.evaluateReal(self._sourceCache, 3)
        assert result == RESULT_OK, 'MeshRefinement failed to get minimum coordinates'
        maximumsField = self._sourceFm.createFieldNodesetMaximum(self._sourceCoordinates, sourceNodes)
        result, maximums = maximumsField.evaluateReal(self._sourceCache, 3)
        assert result == RESULT_OK, 'MeshRefinement failed to get maximum coordinates'
        xrange = [(maximums[i] - minimums[i]) for i in range(3)]
        edgeTolerance = 0.5 * (max(xrange))
        if edgeTolerance == 0.0:
            edgeTolerance = 1.0
        minimums = [(minimums[i] - edgeTolerance) for i in range(3)]
        maximums = [(maximums[i] + edgeTolerance) for i in range(3)]
        del minimumsField
        del maximumsField
        self._sourceMesh = self._sourceFm.findMeshByDimension(3)
        self._sourceFaceMesh = self._sourceFm.findMeshByDimension(2)
        self._sourceLineMesh = self._sourceFm.findMeshByDimension(1)
        self._sourceNodes = self._sourceFm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self._sourceElementiterator = self._sourceMesh.createElementiterator()
        self._octree = Octree(minimums, maximums)
        self._tolerance = self._octree.getTolerance()
        self._is_exterior_field = self._sourceFm.createFieldIsExterior()

        self._targetRegion = targetRegion
        self._targetFm = targetRegion.getFieldmodule()
        self._targetFm.beginChange()
        self._targetCache = self._targetFm.createFieldcache()
        self._targetCoordinates = findOrCreateFieldCoordinates(self._targetFm)

        self._targetNodes = self._targetFm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self._nodetemplate = self._targetNodes.createNodetemplate()
        self._nodetemplate.defineField(self._targetCoordinates)

        self._targetMesh = self._targetFm.findMeshByDimension(3)
        self._targetBasis = self._targetFm.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        self._targetEft = self._targetMesh.createElementfieldtemplate(self._targetBasis)
        self._targetElementtemplate = self._targetMesh.createElementtemplate()
        self._targetElementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        result = self._targetElementtemplate.defineField(self._targetCoordinates, -1, self._targetEft)

        self._nodeIdentifier = 1
        self._elementIdentifier = 1
        # prepare annotation group map
        self._sourceAnnotationGroups = sourceAnnotationGroups
        self._annotationGroups = []
        self._sourceAndTargetMeshGroups = []
        for sourceAnnotationGroup in sourceAnnotationGroups:
            targetAnnotationGroup = AnnotationGroup(
                self._targetRegion, sourceAnnotationGroup.getTerm(), isMarker=sourceAnnotationGroup.isMarker())
            self._annotationGroups.append(targetAnnotationGroup)
            # assume have only highest dimension element or node/point annotation groups:
            if sourceAnnotationGroup.hasMeshGroup(self._sourceMesh):
                self._sourceAndTargetMeshGroups.append((sourceAnnotationGroup.getMeshGroup(self._sourceMesh), targetAnnotationGroup.getMeshGroup(self._targetMesh)))

        # prepare element -> marker point list map
        self.elementMarkerMap = {}
        sourceMarkerGroup = findOrCreateFieldGroup(self._sourceFm, "marker")
        sourceMarkerName = findOrCreateFieldStoredString(self._sourceFm, name="marker_name")
        sourceMarkerLocation = findOrCreateFieldStoredMeshLocation(self._sourceFm, self._sourceMesh, name="marker_location")
        sourceMarkerNodes = sourceMarkerGroup.getNodesetGroup(sourceNodes)
        nodeIter = sourceMarkerNodes.createNodeiterator()
        node = nodeIter.next()
        while node.isValid():
            self._sourceCache.setNode(node)
            element, xi = sourceMarkerLocation.evaluateMeshLocation(self._sourceCache, self._sourceMesh.getDimension())
            if element.isValid():
                elementIdentifier = element.getIdentifier()
                markerName = sourceMarkerName.evaluateString(self._sourceCache)
                markerList = self.elementMarkerMap.get(elementIdentifier)
                if not markerList:
                    markerList = []
                    self.elementMarkerMap[elementIdentifier] = markerList
                markerList.append((markerName, xi, node.getIdentifier()))
            node = nodeIter.next()
        if self.elementMarkerMap:
            self._targetMarkerGroup = findOrCreateFieldGroup(self._targetFm, "marker")
            self._targetMarkerName = findOrCreateFieldStoredString(self._targetFm, name="marker_name")
            self._targetMarkerLocation = findOrCreateFieldStoredMeshLocation(self._targetFm, self._targetMesh, name="marker_location")
            self._targetMarkerNodes = self._targetMarkerGroup.getOrCreateNodesetGroup(self._targetNodes)
            self._targetMarkerTemplate = self._targetMarkerNodes.createNodetemplate()
            self._targetMarkerTemplate.defineField(self._targetMarkerName)
            self._targetMarkerTemplate.defineField(self._targetMarkerLocation)

    def __del__(self):
        self._sourceFm.endChange()
        self._targetFm.endChange()

    def getAnnotationGroups(self):
        return self._annotationGroups

    def _face_add_line_ids_ending_in_x(self, face, x, line_ids: set):
        """
        Add identifiers of lines on face element with either end's coordinates within tolerance of x to supplied set.
        :param face: Zinc 2-D face element.
        :param x: 3 component coordinates list.
        :param line_ids: Set of line identifiers.
        """
        for f in range(face.getNumberOfFaces()):
            line = face.getFaceElement(f + 1)
            if line.isValid():
                line_id = line.getIdentifier()
                if line_id not in line_ids:
                    # add line if it has coordinates within tolerance of x at either end
                    for xi in (0.0, 1.0):
                        self._sourceCache.setMeshLocation(line, [xi])
                        result, line_x = self._sourceCoordinates.evaluateReal(self._sourceCache, 3)
                        if result == RESULT_OK:
                            for c, line_c in zip(x, line_x):
                                if math.fabs(c - line_c) > self._tolerance:
                                    break
                            else:
                                line_ids.add(line_id)
                                break

    def _get_connected_exterior_face_ids(self, faces, x):
        """
        Get connected exterior face element identifiers with common lines to any pairs in faces list.
        :param faces: List of at least 2 faces with common lines.
        :param x: Coordinates of point; used only for > 2 faces.
        :return: List of exterior boundary face identifiers from lowest to highest.
        """
        initial_face_count = len(faces)
        assert initial_face_count > 1

        # add faces passed in to set
        # get common lines between them
        face_ids = set()
        face_line_ids = []
        for face in faces:
            face_ids.add(face.getIdentifier())
            tmp_line_ids = []
            for i in range(face.getNumberOfFaces()):
                line = face.getFaceElement(i + 1)
                if line.isValid():
                    tmp_line_ids.append(line.getIdentifier())
            face_line_ids.append(tmp_line_ids)
        new_line_ids = set()
        # if there is a single line between 2 faces can do less work later, but not if there are collapsed faces
        single_line = initial_face_count < 3
        for f1 in range(len(faces) - 1):
            for f2 in range(f1 + 1, len(faces)):
                add_count = 0
                for line_id in face_line_ids[f1]:
                    if line_id in face_line_ids[f2]:
                        new_line_ids.add(line_id)
                        add_count += 1
                if add_count == 0:
                    # assume collapsed face, so add all lines from both faces ending in x at either end
                    single_line = False
                    for fi in (f1, f2):
                        self._face_add_line_ids_ending_in_x(faces[fi], x, new_line_ids)
        line_ids = copy.copy(new_line_ids)

        while True:
            # ensure all parent elements of common lines are in face_ids set
            new_face_ids = []
            for line_id in new_line_ids:
                line = self._sourceLineMesh.findElementByIdentifier(line_id)
                for p in range(line.getNumberOfParents()):
                    face = line.getParentElement(p + 1)
                    face_id = face.getIdentifier()
                    if face_id not in face_ids:
                        new_face_ids.append(face_id)
            face_ids.update(new_face_ids)

            if single_line or (not new_face_ids):
                break

            new_line_ids.clear()
            for face_id in new_face_ids:
                face = self._sourceFaceMesh.findElementByIdentifier(face_id)
                self._face_add_line_ids_ending_in_x(face, x, new_line_ids)
            line_ids.update(new_line_ids)

        exterior_face_ids = []
        for face_id in face_ids:
            face = self._sourceFaceMesh.findElementByIdentifier(face_id)
            self._sourceCache.setElement(face)
            result, value = self._is_exterior_field.evaluateReal(self._sourceCache, 1)
            if (result == RESULT_OK) and (value != 0.0):
                exterior_face_ids.append(face_id)

        exterior_face_ids.sort()
        return exterior_face_ids

    cube_mid_face_xi = [
        [0.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 1.0]
    ]

    square_mid_edge_xi = [
        [0.0, 0.5],
        [1.0, 0.5],
        [0.5, 0.0],
        [0.5, 1.0]
    ]

    def refineElementCubeStandard3d(self, sourceElement, numberInXi1, numberInXi2, numberInXi3):
        """
        Refine cube sourceElement to numberInXi1*numberInXi2*numberInXi3 linear cube
        sub-elements, evenly spaced in xi.
        :return: Node identifiers, node coordinates used in refinement of sourceElement.
        """
        # element_identifier = sourceElement.getIdentifier()
        # print("refine element", element_identifier)
        meshGroups = []
        for sourceAndTargetMeshGroup in self._sourceAndTargetMeshGroups:
            if sourceAndTargetMeshGroup[0].containsElement(sourceElement):
                meshGroups.append(sourceAndTargetMeshGroup[1])

        faces = [None] * 6
        exterior_faces = [False] * 6  # whether face is on exterior boundary of mesh
        null_face_count = 0
        if sourceElement.getNumberOfFaces() == 6:
            for f in range(6):
                face =  sourceElement.getFaceElement(f + 1)
                if face and face.isValid():
                    faces[f] = face
                    self._sourceCache.setElement(face)
                    result, value = self._is_exterior_field.evaluateReal(self._sourceCache, 1)
                    exterior_faces[f] = (result == RESULT_OK) and (value != 0.0)
                else:
                    faces[f] = None
                    null_face_count += 1
        # collapsed elements have no face, so get all faces which are adjacent to collapsed face
        # check there is at least one valid face and one null face
        if 0 < null_face_count < 6:
            for f, face in enumerate(faces):
                if not face:
                    # get coordinates at centre of face
                    self._sourceCache.setMeshLocation(sourceElement, self.cube_mid_face_xi[f])
                    result, mid_face_x = self._sourceCoordinates.evaluateReal(self._sourceCache, 3)
                    if result != RESULT_OK:
                        continue
                    # get list of all faces with mid-edge coordinate within tolerance of mid_face_x
                    adjacent_faces = []
                    for f2, face2 in enumerate(faces):
                        if face2 and not isinstance(face2, list):
                            for xi in self.square_mid_edge_xi:
                                self._sourceCache.setMeshLocation(face2, xi)
                                result, mid_edge_x = self._sourceCoordinates.evaluateReal(self._sourceCache, 3)
                                if result != RESULT_OK:
                                    continue
                                for c in range(3):
                                    if math.fabs(mid_edge_x[c] - mid_face_x[c]) > self._tolerance:
                                        break
                                else:
                                    adjacent_faces.append(face2)
                                    if exterior_faces[f2]:
                                        exterior_faces[f] = True
                                    break
                    if adjacent_faces:
                        faces[f] = adjacent_faces

        # 6 faces above + 1 extra face for not_a_face_index
        not_a_face_index = 6
        faces.append(None)
        exterior_faces.append(False)

        # create nodes
        nids = []
        nx = []
        xi = [0.0, 0.0, 0.0]
        tol = self._octree._tolerance
        for k in range(numberInXi3 + 1):
            k_face_index = 4 if (k == 0) else 5 if (k == numberInXi3) else not_a_face_index
            k_face = faces[k_face_index]
            k_exterior = exterior_faces[k_face_index]
            xi[2] = k / numberInXi3
            for j in range(numberInXi2 + 1):
                j_face_index = 2 if (j == 0) else 3 if (j == numberInXi2) else not_a_face_index
                j_face = faces[j_face_index]
                j_exterior = exterior_faces[j_face_index]
                xi[1] = j / numberInXi2
                for i in range(numberInXi1 + 1):
                    i_face_index = 0 if (i == 0) else 1 if (i == numberInXi1) else not_a_face_index
                    i_face = faces[i_face_index]
                    i_exterior = exterior_faces[i_face_index]
                    xi[0] = i / numberInXi1
                    self._sourceCache.setMeshLocation(sourceElement, xi)
                    result, x = self._sourceCoordinates.evaluateReal(self._sourceCache, 3)
                    shareable = False
                    surface_face_ids = True  # since None is used for no extra data in Octree

                    connected_faces = []
                    for face in (i_face, j_face, k_face):
                        if face:
                            for tmp_face in face if isinstance(face, list) else [face]:
                                if tmp_face not in connected_faces:
                                    connected_faces.append(tmp_face)
                    face_count = len(connected_faces)
                    if face_count > 0:
                        shareable = True
                        exterior_count = (i_exterior, j_exterior, k_exterior).count(True)
                        if face_count == 1:
                            if exterior_count == 1:
                                shareable = False  # nodes only belong to this element
                            # else interior
                        else:
                            surface_face_ids = self._get_connected_exterior_face_ids(connected_faces, x)
                            if not surface_face_ids:
                                surface_face_ids = True
                    nodeId = None
                    if shareable:
                        nodeId, extra_data = self._octree.findObjectByCoordinates(x, surface_face_ids)
                        # if nodeId:
                        #     print("Found existing node", nodeId, extra_data, "at", x)
                    if nodeId is None:
                        node = self._targetNodes.createNode(self._nodeIdentifier, self._nodetemplate)
                        self._targetCache.setNode(node)
                        result = self._targetCoordinates.setNodeParameters(self._targetCache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                        nodeId = self._nodeIdentifier
                        if shareable:
                            # print("Add shareable node", nodeId, surface_face_ids, "at", x)
                            self._octree.addObjectAtCoordinates(x, (nodeId, surface_face_ids))
                        # else:
                        #     print("Add unique node", nodeId, surface_face_ids, "at", x)
                        self._nodeIdentifier += 1
                    nids.append(nodeId)
                    nx.append(x)
        # create elements
        startElementIdentifier = self._elementIdentifier
        for k in range(numberInXi3):
            ok = (numberInXi2 + 1) * (numberInXi1 + 1)
            for j in range(numberInXi2):
                oj = (numberInXi1 + 1)
                for i in range(numberInXi1):
                    bni = k * ok + j * oj + i
                    element = self._targetMesh.createElement(self._elementIdentifier, self._targetElementtemplate)
                    enids = [nids[bni], nids[bni + 1], nids[bni + oj], nids[bni + oj + 1],
                             nids[bni + ok], nids[bni + ok + 1], nids[bni + ok + oj], nids[bni + ok + oj + 1]]
                    result = element.setNodesByIdentifier(self._targetEft, enids)
                    # if result != RESULT_OK:
                    # print('Element', self._elementIdentifier, result, enids)
                    self._elementIdentifier += 1

                    for meshGroup in meshGroups:
                        meshGroup.addElement(element)

        # re-map any markers embedded in the source element
        if self.elementMarkerMap:
            markerList = self.elementMarkerMap.get(sourceElement.getIdentifier())
            if markerList:
                numberInXi = [numberInXi1, numberInXi2, numberInXi3]
                elementOffset = [1, numberInXi1, numberInXi1 * numberInXi2]
                targetXi = [0.0] * 3
                for marker in markerList:
                    markerName, sourceXi, sourceNodeIdentifier = marker
                    annotationGroup = findAnnotationGroupByName(self._annotationGroups, markerName)
                    if not annotationGroup:
                        print("Could not find annotation group", markerName)
                        continue
                    # determine which sub-element, targetXi that sourceXi maps to
                    targetElementIdentifier = startElementIdentifier
                    for i in range(3):
                        targetXi[i] = sourceXi[i] * numberInXi[i]
                        el = int(targetXi[i])
                        if el < numberInXi[i]:
                            targetXi[i] -= el
                        else:
                            el = numberInXi[i] - 1
                            targetXi[i] = 1.0
                        targetElementIdentifier += el * elementOffset[i]
                    targetElement = self._targetMesh.findElementByIdentifier(targetElementIdentifier)
                    markerNode = annotationGroup.createMarkerNode(
                        self._nodeIdentifier, element=targetElement, xi=targetXi)
                    # add marker node to target annotation groups for any non-marker source annotation groups
                    # the source node was in. (Marker annotation groups all share the same 'marker' group.)
                    for sourceAnnotationGroup in self._sourceAnnotationGroups:
                        if not sourceAnnotationGroup.isMarker():
                            if sourceAnnotationGroup.getNodesetGroup(self._sourceNodes).findNodeByIdentifier(
                                    sourceNodeIdentifier).isValid():
                                targetAnnotationGroup = findAnnotationGroupByName(
                                    self._annotationGroups, sourceAnnotationGroup.getName())
                                if targetAnnotationGroup:
                                    targetAnnotationGroup.getNodesetGroup(self._targetNodes).addNode(markerNode)
                    self._nodeIdentifier += 1

        return nids, nx

    def refineAllElementsCubeStandard3d(self, numberInXi1, numberInXi2, numberInXi3):
        element = self._sourceElementiterator.next()
        while element.isValid():
            self.refineElementCubeStandard3d(element, numberInXi1, numberInXi2, numberInXi3)
            element = self._sourceElementiterator.next()
