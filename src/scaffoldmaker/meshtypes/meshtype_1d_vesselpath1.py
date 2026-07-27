"""
Generates a 1-D vessel path mesh.
"""


import pandas as pd
from cmlibs.utils.zinc.field import (
    find_or_create_field_coordinates,
    findOrCreateFieldGroup,
    findOrCreateFieldStoredMeshLocation,
    findOrCreateFieldStoredString,
)
from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Element, Elementbasis
from cmlibs.zinc.field import Field
from cmlibs.zinc.fieldcache import Fieldcache
from cmlibs.zinc.fieldmodule import Fieldmodule
from cmlibs.zinc.node import Node
from cmlibs.zinc.region import Region

from scaffoldmaker.annotation.annotationgroup import AnnotationGroup
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base


class MeshType_1d_vesselpath1(Scaffold_base):
    '''
    classdocs
    '''
    @staticmethod
    def getName():
        return '1D Vessel Path 1'

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        :return: [] empty list of AnnotationGroup, None
        """
        try:
            region
        except NameError:
            context = Context('scaffold')
            region = context.getDefaultRegion()
        vessel_landmark_connections_filename =  \
            'C:\\Users\\fcas077\\Documents\\ABI\\VITAL\\vessel_lengths\\' \
        + 'Vessel_path_information\\vessel_landmark_connections.json'
        vessel_landmark_positions_filename = \
            'C:\\Users\\fcas077\\Documents\\ABI\\VITAL\\vessel_lengths\\' \
        + 'Vessel_path_information\\vessel_landmark_positions.json'
        vessel_connections = pd.read_json(vessel_landmark_connections_filename)
        vessel_positions = pd.read_json(vessel_landmark_positions_filename)
        fieldmodule = region.getFieldmodule()
        fieldcache = fieldmodule.createFieldcache()
        node_identifier = 1
        element_identifier = 1
        n_vessels = len(vessel_connections) + 1
        node_network = Node_network(region, fieldcache)
        element_matrix = Vertex_matrix(n_vessels)
        annotation_groups = []
        # Generate nodes
        for row in vessel_positions.itertuples():
            node_network.set_node_parameters(row._2, row._1, node_identifier)
            node_identifier += 1
        node_network.set_zinc_nodes()
        # Generate vertices
        for row in vessel_connections.itertuples():
            origin_id = node_network.get_node_id_from_name(row._1)
            termination_id = node_network.get_node_id_from_name(row._2)
            if origin_id != 0 and termination_id != 0:
                elem_dic = {
                    'name': row._3, 
                    'origin_id': origin_id, 
                    'termination_id': termination_id, 
                    'element_identifier': element_identifier
                }
                element_matrix.set_element_to_index(elem_dic, element_identifier)
                element_identifier += 1
        # Generate annotation groups
        mesh = fieldmodule.findMeshByDimension(1)
        # Generate elements
        node_matrix = element_matrix.get_matrix()
        for elem in node_matrix:
            if elem is not None:
                element_identifier = elem.get('element_identifier')
                node_ids = [elem.get('origin_id'), elem.get('termination_id')]
                element = create_cube_element(
                    fieldmodule, element_identifier, node_ids)
                vessel_name = elem.get('name')
                annotation_group = AnnotationGroup(region, (vessel_name, ""))
                annotation_groups.append(annotation_group)
                mesh_group = annotation_group.getMeshGroup(mesh)
                mesh_group.addElement(element)
        return annotation_groups, None


class Node_network():

    def __init__(self, region: Region, fieldcache: Fieldcache):
        self.fieldmodule = region.getFieldmodule()
        self.fieldcache = fieldcache
        self.coordinates = find_or_create_field_coordinates(self.fieldmodule, 'coordinates')
        self.nodes = self.fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self.node_identifier = 1
        self.nodes_dict = {}
        self.names_to_ids = {}
        value_labels = [
            Node.VALUE_LABEL_VALUE,
        ]
        nodes = self.nodes
        coordinates = self.coordinates
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        for value_label in value_labels[1:]:
            nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)
        self.nodetemplate = nodetemplate

    def set_node_parameters(self, node_position: list, name: str,
                            node_identifier:int = 0, is_linear = True) -> None :
        nodes_dict = self.nodes_dict
        names_to_ids = self.names_to_ids
        assert len(node_position) == 3
        nodes_dict[node_identifier] = {
            'parameters': node_position,
            'is_linear': is_linear,
            'name': name
        }
        names_to_ids[name] = node_identifier

    def set_zinc_nodes(self):
        node_dict = self.nodes_dict
        coordinates = self.coordinates
        fieldcache = self.fieldcache
        nodetemplate = self.nodetemplate
        for node_identifier, node_params in node_dict.items():
            node_coordinates = node_params.get('parameters')
            node = self.nodes.createNode(node_identifier, nodetemplate)
            fieldcache.setNode(node)
            coordinates.setNodeParameters(
                fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, node_coordinates)

    def get_node_parameters(self, node_identifier):
        nodes_dict = self.nodes_dict
        node_dict = nodes_dict[node_identifier].get('parameters')
        return node_dict

    def get_node_identifier(self)-> int:
        return self.node_identifier

    def remove_node(self, node_identifier):
        nodes_dict = self.nodes_dict
        nodes_dict.pop(node_identifier)

    def get_node_id_from_name(self, name: str):
        names_to_ids = self.names_to_ids
        node_id = names_to_ids.get(name, 0)
        return node_id

class Vertex_matrix():

    def __init__(self, n_rows: int):
        self.node_matrix = [None for n in range(n_rows)]

    def set_element_to_index(self, vertex: dict, index: int):
        node_matrix = self.node_matrix
        node_matrix[index] = vertex

    def get_matrix(self) -> list:
        return self.node_matrix

    def get_vertex(self, index: list) -> dict:
        node_matrix = self.node_matrix
        vertex = node_matrix[index]
        return vertex

def create_cube_element(fieldmodule: Fieldmodule, element_identifier: int,
                        node_ids: list,) -> list:
    # Zinc setup
    mesh = fieldmodule.findMeshByDimension(1)
    coordinates = find_or_create_field_coordinates(fieldmodule)
    linearBasis = fieldmodule.createElementbasis(1, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
    eft = mesh.createElementfieldtemplate(linearBasis)
    elementtemplate = mesh.createElementtemplate()
    elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
    result = elementtemplate.defineField(coordinates, -1, eft)
    element = mesh.createElement(element_identifier, elementtemplate)
    element.setNodesByIdentifier(eft, node_ids)
    return element

