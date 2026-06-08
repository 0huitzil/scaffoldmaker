import re
import math
import logging
import tempfile

from cmlibs.maths.vectorops import add, distance, magnitude, mult, normalize, sub
from cmlibs.utils.zinc.field import get_group_list
from cmlibs.utils.zinc.finiteelement import get_element_node_identifiers
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.utils.zinc.group import groups_have_same_local_contents
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node

from scaffoldmaker.annotation.vagus_terms import (
    get_vagus_term, marker_name_in_terms, get_left_vagus_marker_locations_list, get_right_vagus_marker_locations_list)
from scaffoldmaker.utils.zinc_utils import get_nodeset_field_parameters, get_mesh_node_identifier_sequences


logger = logging.getLogger(__name__)


class VagusInputData:
    """
    Categorising and storing input data from data region for vagus box scaffold
    """

    def __init__(self, data_region):
        """
        :param data_region Zinc data region with input data
        """

        self._trunk_keywords = ['cervical vagus nerve', 'thoracic vagus nerve',
                                'cervical trunk', 'thoracic trunk', 'vagus x nerve trunk']
        self._branch_keywords = ['branch', 'nerve', 'ganglion', 'fusion']
        self._non_branch_keywords = ['perineurium', 'epineurium']
        self._term_keywords = ['fma:', 'fma_', 'ilx:', 'ilx_', 'uberon:', 'uberon_']
        self._orientation_keywords = ['orientation']

        self._annotation_term_map = {}
        self._branch_coordinates_data = {}
        self._branch_radius_data = {}
        self._branch_connectivity_data = {}
        self._branch_sequences_data = {}
        self._branch_parent_map = {}
        self._branch_common_group_map = {}
        self._datafile_path = None
        self._level_markers = {}
        self._orientation_data = {}
        self._side_label = ""
        self._trunk_group_name = None
        self._trunk_coordinates = []
        self._trunk_radius = []

        fm = data_region.getFieldmodule()
        fc = fm.createFieldcache()

        nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        coordinates = fm.findFieldByName("coordinates").castFiniteElement()
        radius = fm.findFieldByName("radius").castFiniteElement()
        if not radius.isValid():
            radius = None
        datapoints = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        marker_names = fm.findFieldByName("marker_name")
        mesh = fm.findMeshByDimension(1)

        annotation_names = []
        term_annotation_names = []
        group_list = get_group_list(fm)
        for group in group_list:
            group_name = group.getName().strip()
            lower_name = group_name.casefold()
            if any(keyword in lower_name for keyword in self._term_keywords):
                term_annotation_names.append(group_name)
            else:
                annotation_names.append(group_name)
        for annotation_name in annotation_names:
            annotation_group = fm.findFieldByName(annotation_name).castGroup()
            for term_annotation in term_annotation_names:
                term_group = fm.findFieldByName(term_annotation).castGroup()
                if groups_have_same_local_contents(annotation_group, term_group):
                    self._annotation_term_map[annotation_name] = term_annotation
                    break
            else:
                # no matching term is found for annotation group
                self._annotation_term_map[annotation_name] = ""

        found_trunk_group_names = []
        branch_group_names = []
        orientation_group_names = []
        for annotation_name in annotation_names:
            lower_name = annotation_name.casefold()
            if any([keyword in lower_name for keyword in self._trunk_keywords]) and 'branch' not in lower_name:
                found_trunk_group_names.append(annotation_name)
            elif any([keyword in lower_name for keyword in self._branch_keywords]) and \
                    all([keyword not in lower_name for keyword in self._non_branch_keywords]):
                branch_group_names.append(annotation_name)
            elif any([keyword in lower_name for keyword in self._orientation_keywords]):
                orientation_group_names.append(annotation_name)

        # extract marker data - name, coordinates (no marker terms are available)
        # sort markers here?
        marker_group = fm.findFieldByName("marker").castGroup()
        if marker_group:
            marker_nodes = marker_group.getNodesetGroup(datapoints)
            marker_node_iter = marker_nodes.createNodeiterator()
            marker_node = marker_node_iter.next()
            while marker_node.isValid():
                fc.setNode(marker_node)
                _, x = coordinates.evaluateReal(fc, 3)
                marker_name = marker_names.evaluateString(fc).strip()
                if marker_name_in_terms(marker_name):
                    # add anatomical landmark marker if in approved terms
                    self._level_markers[marker_name] = x
                marker_node = marker_node_iter.next()

        # extract orientation data
        orientation_ignore_node_ids = []
        orientation_ignore_group_name = 'orientation ignore'
        if orientation_ignore_group_name in orientation_group_names:
            group = fm.findFieldByName(orientation_ignore_group_name).castGroup()
            nodeset_group = group.getNodesetGroup(nodes)
            _, values = get_nodeset_field_parameters(nodeset_group, coordinates, [Node.VALUE_LABEL_VALUE])
            orientation_ignore_node_ids = [value[0] for value in values]
            orientation_group_names.remove(orientation_ignore_group_name)
        for orientation_group_name in orientation_group_names:
            group = fm.findFieldByName(orientation_group_name).castGroup()
            nodeset_group = group.getNodesetGroup(nodes)
            _, values = get_nodeset_field_parameters(nodeset_group, coordinates, [Node.VALUE_LABEL_VALUE])
            orientation_points = [value[1][0][0] for value in values if (value[0] not in orientation_ignore_node_ids)]
            self._orientation_data[orientation_group_name] = orientation_points

        # extract trunk data - coordinates, nodes, radius
        if len(found_trunk_group_names) > 0:
            if 'left' in found_trunk_group_names[0]:
                self._trunk_group_name = 'left vagus nerve'
                self._annotation_term_map[self._trunk_group_name] = get_vagus_term(self._trunk_group_name)[1]
                self._side_label = 'left'
            elif 'right' in found_trunk_group_names[0]:
                self._trunk_group_name = 'right vagus nerve'
                self._annotation_term_map[self._trunk_group_name] = get_vagus_term(self._trunk_group_name)[1]
                self._side_label = 'right'
        assert self._trunk_group_name

        if self._trunk_group_name:
            trunk_node_ids = []
            trunk_coordinates = []
            trunk_radius = []
            for found_trunk_group_name in found_trunk_group_names:
                group = fm.findFieldByName(found_trunk_group_name).castGroup()
                nodeset_group = group.getNodesetGroup(nodes)
                mesh_group = group.getMeshGroup(mesh)
                node_coordinate_values = get_nodeset_field_parameters(
                    nodeset_group, coordinates, [Node.VALUE_LABEL_VALUE])[1]
                trunk_node_ids += [value[0] for value in node_coordinate_values]
                for value in node_coordinate_values:
                    trunk_coordinates.append(value[1][0])
                if radius:
                    node_radius_values = get_nodeset_field_parameters(nodeset_group, radius, [Node.VALUE_LABEL_VALUE])[1]
                    for value in node_radius_values:
                        trunk_radius.append(value[1][0][0])

            self._segment_trunk_info_list = make_segment_trunk_info(
                fm, fc, coordinates, nodes, mesh, group_list, found_trunk_group_names, self._trunk_group_name)

            self._trunk_coordinates = trunk_coordinates
            self._trunk_radius = trunk_radius

        # extract branch data - name, coordinates, nodes, radius
        for branch_name in branch_group_names:
            group = fm.findFieldByName(branch_name).castGroup()
            nodeset_group = group.getNodesetGroup(nodes)
            if 'xml.ex' in branch_name or nodeset_group.getSize() < 2:
                # xml.ex are temporary regions from segmentation stitcher that might have keywords in their names
                # branch should have at least two nodes to be connected to parent
                continue
            _, values = get_nodeset_field_parameters(nodeset_group, coordinates, [Node.VALUE_LABEL_VALUE])
            # make above into a dict to look up by node id
            node_coordinates_parameters = {}
            for node_id, parameters in values:
                node_coordinates_parameters[node_id] = parameters[0][0]

            mesh_group = group.getMeshGroup(mesh)
            node_ids_list = get_mesh_node_identifier_sequences(mesh_group, coordinates)
            if node_ids_list:
                # check for old data which didn't make an element from the trunk node to the first branch node:
                first_branch_node_id = values[0][0]
                for node_ids in node_ids_list:
                    if first_branch_node_id in node_ids:
                        break
                else:
                    logger.warning('Branch ' + branch_name + ' did not have parent node ' + str(first_branch_node_id) +
                                   ' in branch elements. Including as first node.')
                    node_ids_list[0].insert(0, first_branch_node_id)
            else:
                # if no elements, fall back to branch nodes in order, allows only a single branch
                node_ids_list = [[value[0] for value in values]]
            # get node coordinates in order
            branch_coordinates = []
            for node_ids in node_ids_list:
                for node_id in node_ids:
                    branch_coordinates.append(node_coordinates_parameters[node_id])
            self._branch_coordinates_data[branch_name] = branch_coordinates
            self._branch_connectivity_data[branch_name] = node_ids_list
            self._branch_sequences_data[branch_name] = [len(node_ids) for node_ids in node_ids_list]

            if radius:
                _, values = get_nodeset_field_parameters(nodeset_group, radius, [Node.VALUE_LABEL_VALUE])
                if not all((value[1][0][0] == 0.0) for value in values):
                    # make above into a dict to look up by node id
                    node_radius_parameters = {}
                    for node_id, parameters in values:
                        node_radius_parameters[node_id] = parameters[0][0]
                    branch_radius = []
                    for node_ids in node_ids_list:
                        for node_id in node_ids:
                            branch_radius.append(node_radius_parameters[node_id])
                    self._branch_radius_data[branch_name] = branch_radius

        # find parent branches where each branch connects to
        # limitaton: parent is the same for each separate branch with the same name
        for branch_name, node_ids_list in self._branch_connectivity_data.items():
            parent_name = None
            parameters_index = 0
            for node_ids in node_ids_list:
                next_parent_name = None
                for start_node_index in (0, -1):  # in case reversed order
                    branch_first_node_id = node_ids[start_node_index]
                    #  first check if trunk is a parent by searching for a common node
                    if branch_first_node_id in trunk_node_ids:
                        next_parent_name = self._trunk_group_name
                        break
                if not next_parent_name:
                    # check other branches if a common node exists
                    # issue: likely to have problems if branch is processed before child branch
                    for start_node_index in (0, -1):  # in case reversed order
                        branch_first_node_id = node_ids[start_node_index]
                        for parent_branch_name, parent_node_ids_list in self._branch_connectivity_data.items():
                            if parent_branch_name != branch_name:
                                for parent_node_ids in parent_node_ids_list:
                                    if branch_first_node_id in parent_node_ids:
                                        next_parent_name = parent_branch_name
                                        break
                            if next_parent_name:
                                break
                        if next_parent_name:
                            break
                if next_parent_name:
                    if parent_name:
                        if next_parent_name != parent_name:
                            logger.warning('Branches with name ' + branch_name + ' have both ' + parent_name + ' and ' +
                                           next_parent_name + ' as parents. Using ' + parent_name)
                    else:
                        parent_name = next_parent_name
                    if start_node_index == -1:
                        # reverse order of branch nodes, parameters etc. so always heading away from parent
                        count = len(node_ids)
                        branch_x = self._branch_coordinates_data[branch_name]
                        branch_r = self._branch_radius_data.get(branch_name)
                        for i in range(count // 2):
                            i1 = parameters_index + i
                            i2 = parameters_index + count - i - 1
                            branch_x[i1], branch_x[i2] = branch_x[i2], branch_x[i1]
                            if branch_r:
                                branch_r[i1], branch_r[i2] = branch_r[i2], branch_r[i1]
                        node_ids.reverse()  # reverse in place
                parameters_index += len(node_ids)
            if not parent_name:
                # assume trunk is a parent by default, if no other is found
                parent_name = self._trunk_group_name
            self._branch_parent_map[branch_name] = parent_name
            # print(branch_name, ' -> ', parent_name)

        # group common branches by names
        branch_common_map = group_common_branches(branch_group_names)
        self._branch_common_group_map = branch_common_map

        # write all data in a file for geometry fitter
        sir = data_region.createStreaminformationRegion()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            datafile_path = temp_file.name
            srf = sir.createStreamresourceFile(datafile_path)
            data_region.write(sir)
        self._datafile_path = datafile_path

    def get_level_markers(self):
        """
        Get all level marker names and coordinates from the data.
        :return: Dict mapping marker name to x,y,z coordinates.
        """
        return self._level_markers

    def get_orientation_data(self):
        """
        Get all orientations and coordinates from the data.
        :return Dict mapping 8 possible orientations (anterior, left, right, etc.) to list of x, y, z coordinates.
        """
        return self._orientation_data

    def get_trunk_group_name(self):
        """
        Get the name used for the trunk group in the data.
        return: String with trunk group name.
        """
        return self._trunk_group_name

    def get_trunk_coordinates(self):
        """
        Get the x, y, z coordinates of the trunk in the data.
        return: List of coordinates for the trunk group.
        """
        return self._trunk_coordinates

    def get_trunk_radius(self):
        """
        Get the radius values of the trunk in the data. The values are in the same order as trunk coordinates
        and each radius point is associated with a trunk coordinate.
        return: List of radius values for the trunk group.
        """
        return self._trunk_radius

    def get_branch_coordinates_data(self):
        """
        Get all branch names and coordinates from the data.
        return: Dict mapping branch name to x, y, z data.
        """
        return self._branch_coordinates_data

    def get_branch_radius_data(self):
        """
        Get the radius values of the trunk in the data. The values are in the same order as trunk coordinates
        and each radius point is associated with a trunk coordinate.
        return: List of radius values for the trunk group.
        """
        return self._branch_radius_data

    def get_branch_sequences_data(self):
        """
        Get information about how many distinct branches there are for each branch name.
        return: Dict mapping branch name to list of numbers of nodes in each branch sequence.
        """
        return self._branch_sequences_data

    def get_annotation_term_map(self):
        """
        Get all annotation names and terms.
        return: Dict mapping annotation name to term annotation name.
        """
        return self._annotation_term_map

    def get_branch_common_group_map(self):
        """
        Get common branch names from the data.
        return: Dict mapping common branch name to list of branches with common names.
        """
        return self._branch_common_group_map

    def get_branch_parent_map(self):
        """
        Get all branch names and their parent branch names (for first level branches trunk is the parent, etc.).
        return: Dict mapping branch name to the parent branch or trunk name.
        """
        return self._branch_parent_map

    def get_side_label(self):
        """
        Get label indicating side of the vagus (left or right or '')
        """
        return self._side_label

    def get_datafile_path(self):
        """
        Get the path to the temporary file with the data.
        return: directory name of the temporary data file.
        """
        return self._datafile_path

    def reset_datafile_path(self):
        """
        Reset the path to the file with the data.
        """
        self._datafile_path = None

    def get_segment_trunk_info_list(self):
        """
        Get segment trunk information gleaned from each .exf file read into segmentations stitcher, recognized by
        group names ending in '.exf', but not containing '.exf' multiple times as for connection groups.
        These are in order down the nerve.
        :return: list of segment info dict with at least fields: 'name', 'unordered_coordinates', 'ordered_points',
        'centroid', 'range'. 'ordered_coordinates' is only present if a single polyline crosses the segment.
        'ordered_points' is the same as 'ordered_coordinates' (if present) otherwise it is 2 points along the mean
        direction of the segment relative to its neighbours (for multi-path segments).
        """
        return self._segment_trunk_info_list


def group_common_branches(branch_names):
    """
    Groups branches with the same annotations and destinations, only different by A, B, C variant character.
    :param branch_names: List with supplied branch names.
    :return branch_common_map: Dictionary mapping common branch name to list of branches with common names.
    Only contains entries for branches with variant names.
    """

    branch_common_map = {}
    for branch_name in branch_names:
        # remove single letters like A, B, C, etc. surrounded by whitespace
        common_key = re.sub(r'\b[A-Z]\b\s?', '', branch_name).strip()
        if common_key != branch_name:
            # branch variant was found
            variant_branch_list = branch_common_map.get(common_key)
            if variant_branch_list:
                variant_branch_list.append(branch_name)
            else:
                branch_common_map[common_key] = [branch_name]
    return branch_common_map


def load_vagus_data(region):
    """
    :param region: Zinc region for model definition.
    return: Provided the input file is supplied, it returns a data region with input data, otherwise None.
    """
    data_region = region.getParent().findChildByName('data')
    if not data_region.isValid():
        logger.warning("Missing input data.")
        return None

    vagus_data = VagusInputData(data_region)
    return vagus_data


def bfs_to_furthest(graph, start, trunk_path_ids):
    """
    Breadth first search.
    :param graph: Map of  node identifiers to node identifiers they are connected to.
    :param start: Start node identifier, a graph end point.
    :param trunk_path_ids: List of previously added node identifiers down path.
    return: List of node identifiers from start to the furthest connected end.
    """

    visited = set()
    parent = {start: None}
    queue = [start]
    last = start

    while queue:
        current = queue.pop(0)
        visited.add(current)
        last = current
        for neighbor in graph[current]:
            if (neighbor not in visited) and (neighbor not in trunk_path_ids) and (neighbor not in queue):
                parent[neighbor] = current
                queue.append(neighbor)

    # Trace path from the furthest node back to start
    path = []
    while last is not None:
        path.append(last)
        last = parent[last]
    return list(reversed(path))


def make_segment_trunk_info(fieldmodule, fieldcache, coordinates, nodes, mesh1d, group_list, trunk_group_names,
                            trunk_group_name):
    """
    Make ordered (from top to bottom of nerve) segment trunk information to help get an initial guess of path.
    Segment data is in groups with names ending in .exf, but not containing .exf more than once as that is
    used for connection groups from Segmentation Stitcher.
    :param fieldmodule: Zinc Fieldmodule for region.
    :param fieldcache: Zinc Fieldcache.
    :param coordinates: Coordinate field.
    :param nodes: Nodes in region.
    :param mesh1d: 1-D mesh in region.
    :param group_list: List of all zinc groups in region
    :param trunk_group_names: Names of trunk groups in source data.
    :param trunk_group_name: Name of whole trunk group.
    :return: List of segment trunk information in order down nerve. Information is a dict with at least fields
    'name': segment name
    'unordered_coordinates': List of all node coordinates in segment.
    'ordered_coordinates': Optional ordered list of raw coordinates from top to bottom. Only present if there is
    single polyline in segment; absent if not so.
    'ordered_points': Same as 'ordered_coordinates' if present, otherwise exactly 2 points in the mean direction
    of trunk in segment; nerve ends have these points right to the end, but interior segments have 2 points at
    0.25, 0.75 proportion along. This is used to give initial path down trunk.
    """
    segment_trunk_info_list = []
    with ChangeManager(fieldmodule):
        fieldcache.clearLocation()
        is_trunk = None
        for trunk_group_name in trunk_group_names:
            trunk_group = fieldmodule.findFieldByName(trunk_group_name).castGroup()
            is_trunk = fieldmodule.createFieldOr(is_trunk, trunk_group) if is_trunk else trunk_group

        # get raw segment information
        for group in group_list:
            group_name = group.getName()
            if (group_name[-4:] == '.exf') and (1 == group_name.count('.exf')):
                segment_trunk_group = fieldmodule.createFieldGroup()
                segment_trunk_group.setName(group_name + ' ' + trunk_group_name)
                segment_trunk_nodeset_group = segment_trunk_group.createNodesetGroup(nodes)
                is_segment_trunk = fieldmodule.createFieldAnd(group, is_trunk)
                segment_trunk_nodeset_group.addNodesConditional(is_segment_trunk)
                segment_trunk_mesh_group = segment_trunk_group.createMeshGroup(mesh1d)
                segment_trunk_mesh_group.addElementsConditional(is_segment_trunk)
                del is_segment_trunk
                first_node_id = segment_trunk_nodeset_group.createNodeiterator().next().getIdentifier()
                if first_node_id < 0:
                    continue  # empty segment
                unordered_coordinates = []
                mean_coordinates = fieldmodule.createFieldNodesetMean(coordinates, segment_trunk_nodeset_group)
                result, centroid = mean_coordinates.evaluateReal(fieldcache, 3)
                del mean_coordinates
                minimum_coordinates = fieldmodule.createFieldNodesetMinimum(coordinates, segment_trunk_nodeset_group)
                result, min_x = minimum_coordinates.evaluateReal(fieldcache, 3)
                del minimum_coordinates
                maximum_coordinates = fieldmodule.createFieldNodesetMaximum(coordinates, segment_trunk_nodeset_group)
                result, max_x = maximum_coordinates.evaluateReal(fieldcache, 3)
                del maximum_coordinates
                segment_trunk_info = {
                    'name': group_name,
                    'first_node_id': first_node_id,
                    'nodeset_group': segment_trunk_nodeset_group,
                    'unordered_coordinates': unordered_coordinates,
                    'centroid': centroid,
                    'range': [min_x, max_x]
                }
                # ensure in order of lowest node in segment which should be from top to bottom of nerve
                for i, other_segment_info in enumerate(segment_trunk_info_list):
                    if first_node_id < other_segment_info['first_node_id']:
                        segment_trunk_info_list.insert(i, segment_trunk_info)
                        break
                else:
                    segment_trunk_info_list.append(segment_trunk_info)
                node_coordinate_values = get_nodeset_field_parameters(
                    segment_trunk_nodeset_group, coordinates, [Node.VALUE_LABEL_VALUE])[1]
                # build segment trunk graph: map of node identifiers to node identifiers they are connected to
                node_id_coordinates = {}
                trunk_graph = {}
                for node_id, values in node_coordinate_values:
                    x = values[0][0]
                    node_id_coordinates[node_id] = x
                    trunk_graph[node_id] = []
                    unordered_coordinates.append(x)
                element_iterator = segment_trunk_mesh_group.createElementiterator()
                element = element_iterator.next()
                while element.isValid():
                    eft = element.getElementfieldtemplate(coordinates, -1)
                    node_ids = get_element_node_identifiers(element, eft)
                    for n in range(len(node_ids) - 1):
                        trunk_graph[node_ids[n]].append(node_ids[n + 1])
                        trunk_graph[node_ids[n + 1]].append(node_ids[n])
                    element = element_iterator.next()
                # check if single polyline from one end to the other
                count0 = 0
                count1 = 0
                count3plus = 0
                start_node_id = None
                for node_id, connected_node_ids in trunk_graph.items():
                    count = len(connected_node_ids)
                    if count == 0:
                        count0 += 1
                    elif count == 1:
                        count1 += 1
                        if start_node_id is None:
                            start_node_id = node_id
                    elif count > 2:
                        count3plus += 1
                if (count0 == 0) and (count1 == 2) and (count3plus == 0):
                    ordered_coordinates = []
                    node_id = start_node_id
                    last_node_id = None
                    while True:
                        x = node_id_coordinates[node_id]
                        ordered_coordinates.append(x)
                        connected_node_ids = trunk_graph[node_id]
                        for connected_node_id in connected_node_ids:
                            if connected_node_id != last_node_id:
                                break
                        else:
                            break
                        last_node_id = node_id
                        node_id = connected_node_id
                    segment_trunk_info['ordered_coordinates'] = ordered_coordinates
                del segment_trunk_mesh_group
                del segment_trunk_nodeset_group
                del segment_trunk_group
        del is_trunk

        # add segment coordinates/point order
        s_count = len(segment_trunk_info_list)
        prev_centroid = None
        for s, segment_trunk_info in enumerate(segment_trunk_info_list):
            unordered_coordinates = segment_trunk_info['unordered_coordinates']
            centroid = segment_trunk_info['centroid']
            next_centroid = segment_trunk_info_list[s + 1]['centroid'] if s < (s_count - 1) else None
            nodeset_group = segment_trunk_info['nodeset_group']
            ordered_coordinates = segment_trunk_info.get('ordered_coordinates')
            # print(segment_trunk_info['name'], 'single path' if ordered_coordinates else 'multi-path')
            if not ordered_coordinates:
                logger.info('Segment ' + segment_trunk_info['name'] + ' is multi-path')
            direction = None
            if s_count == 1:
                if ordered_coordinates:
                    pass  # assume in correct order
                elif len(unordered_coordinates) == 1:
                    ordered_coordinates = unordered_coordinates
                else:
                    direction = normalize(sub(centroid, unordered_coordinates[0]))
            else:
                if ordered_coordinates:
                    # reverse ordered coordinates if wrong end is closer to prev/next_centroid
                    if prev_centroid:
                        far_distance = magnitude(sub(ordered_coordinates[-1], prev_centroid))
                        near_distance = magnitude(sub(ordered_coordinates[0], prev_centroid))
                    else:
                        far_distance = magnitude(sub(next_centroid, ordered_coordinates[0]))
                        near_distance = magnitude(sub(next_centroid, ordered_coordinates[-1]))
                    if far_distance < near_distance:
                        ordered_coordinates.reverse()
                else:
                    next_point = next_centroid if next_centroid else centroid
                    prev_point = prev_centroid if prev_centroid else centroid
                    direction = normalize(sub(next_point, prev_point))
            if ordered_coordinates:
                ordered_points = ordered_coordinates
            else:
                # get range of coordinates in direction, sample 2 points for straight line
                direction_coordinate = fieldmodule.createFieldDotProduct(
                    coordinates - fieldmodule.createFieldConstant(centroid),
                    fieldmodule.createFieldConstant(direction))
                result, min_d = fieldmodule.createFieldNodesetMinimum(
                    direction_coordinate, nodeset_group).evaluateReal(fieldcache, 1)
                result, max_d = fieldmodule.createFieldNodesetMaximum(
                    direction_coordinate, nodeset_group).evaluateReal(fieldcache, 1)
                del direction_coordinate
                min_x = add(centroid, mult(direction, min_d))
                max_x = add(centroid, mult(direction, max_d))
                # only go to the full min_x / max_x range for the first and last segments
                if s_count == 1:
                    xi_list = [0.0, 1.0]
                elif s == 0:
                    xi_list = [0.0, 0.75]
                elif s < (s_count - 1):
                    xi_list = [0.25, 0.75]
                else:
                    xi_list = [0.25, 1.0]
                ordered_points = [add(mult(min_x, 1.0 - xi), mult(max_x, xi)) for xi in xi_list]
            # don't want these left behind
            del segment_trunk_info['nodeset_group']
            del nodeset_group
            segment_trunk_info['ordered_points'] = ordered_points
            prev_centroid = centroid

    return segment_trunk_info_list
