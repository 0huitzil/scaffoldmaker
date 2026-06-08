"""
Utility functions for easing use of Zinc API.
"""
from cmlibs.maths.vectorops import add, cross, div, magnitude, mult, normalize, rejection, set_magnitude, sub
from cmlibs.utils.zinc.field import (
    find_or_create_field_coordinates, find_or_create_field_finite_element, find_or_create_field_group,
    find_or_create_field_stored_mesh_location, find_or_create_field_stored_string)
from cmlibs.utils.zinc.finiteelement import (
    get_element_node_identifiers, get_maximum_element_identifier, get_maximum_node_identifier)
from cmlibs.utils.zinc.general import ChangeManager, HierarchicalChangeManager
from cmlibs.zinc.context import Context
from cmlibs.zinc.element import Element, Elementbasis, MeshGroup
from cmlibs.zinc.field import Field, FieldGroup, FieldFindMeshLocation
from cmlibs.zinc.fieldmodule import Fieldmodule
from cmlibs.zinc.node import Node, NodesetGroup
from cmlibs.zinc.result import RESULT_OK
from fieldfitter.fitter import Fitter as FieldFitter
from scaffoldfitter.fitter import Fitter as GeometryFitter
from scaffoldfitter.fitterstepconfig import FitterStepConfig
from scaffoldfitter.fitterstepfit import FitterStepFit
from scaffoldmaker.utils import interpolation as interp
import copy
import logging
import math


logger = logging.getLogger(__name__)


def interpolateNodesCubicHermite(cache, coordinates, xi, normal_scale,
        node1, derivative1, scale1, cross_derivative1, cross_scale1,
        node2, derivative2, scale2, cross_derivative2, cross_scale2):
    """
    Interpolates position and first derivative with cubic Hermite basis.
    Interpolates cross derivative linearly.
    :param cache: Field cache to evaluate in.
    :param coordinates: Coordinates field.
    :param xi: Element coordinate to interpolate at.
    :param normal_scale: Magnitude of normal derivative to return.
    :param node1, node2: Start and end nodes.
    :param derivative1, derivative2: Node value label for derivatives.
    :param scale1, scale2: Real value scaling derivatives, to reverse if needed.
    :param cross_derivative1, cross_derivative2: Node value label for cross derivatives.
    :param cross_scale1, cross_scale2: Real value scaling cross_derivatives, to reverse if needed.
    :return: x, dx_ds, dx_ds_cross, dx_ds_normal
    """
    cache.setNode(node1)
    result, v1 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, 3 )
    result, d1 = coordinates.getNodeParameters(cache, -1, derivative1, 1, 3 )
    result, d1c = coordinates.getNodeParameters(cache, -1, cross_derivative1, 1, 3 )
    d1 = [ scale1*d for d in d1 ]
    d1c = [ cross_scale1*d for d in d1c ]
    cache.setNode(node2)
    result, v2 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, 3 )
    result, d2 = coordinates.getNodeParameters(cache, -1, derivative2, 1, 3 )
    result, d2c = coordinates.getNodeParameters(cache, -1, cross_derivative2, 1, 3 )
    d2 = [ scale2*d for d in d2 ]
    d2c = [ cross_scale2*d for d in d2c ]

    arcLength = interp.computeCubicHermiteArcLength(v1, d1, v2, d2, True)
    mag = arcLength/magnitude(d1)
    d1 = [ mag*d for d in d1 ]
    mag = arcLength/magnitude(d2)
    d2 = [ mag*d for d in d2 ]

    xr = 1.0 - xi
    x = interp.interpolateCubicHermite(v1, d1, v2, d2, xi)
    dx_ds = interp.interpolateCubicHermiteDerivative(v1, d1, v2, d2, xi)
    scale = min(xi, xr)
    dx_ds = [ scale*d for d in dx_ds ]
    dx_ds_cross = [ (xr*d1c[c] + xi*d2c[c]) for c in range(3) ]

    radialVector = normalize(cross(dx_ds_cross, dx_ds))
    dx_ds_normal = [ normal_scale*d for d in radialVector ]

    return x, dx_ds, dx_ds_cross, dx_ds_normal


def computeNodeDerivativeHermiteLagrange(cache, coordinates, node1, derivative1, scale1, node2, scale2):
    """
    Computes the derivative at node2 from quadratic Hermite-Lagrange interpolation of
    node1 value and derivative1 to node2 value.
    :param cache: Field cache to evaluate in.
    :param coordinates: Coordinates field.
    :param node1, node2: Start and end nodes.
    :param derivative1: Node value label for derivative at node1.
    :param scale1, scale2: Scaling to apply to derivatives at nodes, e.g. -1.0 to reverse.
    :return: dx_dxi at node2
    """
    cache.setNode(node1)
    result, v1 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, 3 )
    result, d1 = coordinates.getNodeParameters(cache, -1, derivative1, 1, 3 )
    d1 = [ d*scale1 for d in d1 ]
    cache.setNode(node2)
    result, v2 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, 3 )
    d2 = interp.interpolateHermiteLagrangeDerivative(v1, d1, v2, 1.0)
    d2 = [ d*scale2 for d in d2 ]
    return d2


def createFaceMeshGroupExteriorOnFace(fieldmodule : Fieldmodule, elementFaceType) -> MeshGroup:
    """
    Returns mesh group for the exterior surface on the face described
    by elementFaceType.
    """
    with ChangeManager(fieldmodule):
        isExterior = fieldmodule.createFieldIsExterior()
        isOnFace = fieldmodule.createFieldIsOnFace(elementFaceType)
        mesh2d = fieldmodule.findMeshByDimension(2)
        faceGroup = fieldmodule.createFieldGroup()
        faceMeshGroup = faceGroup.createMeshGroup(mesh2d)
        faceMeshGroup.addElementsConditional(fieldmodule.createFieldAnd(isExterior, isOnFace))
        del isExterior
        del isOnFace
    return faceMeshGroup


def mesh_destroy_elements_and_nodes_by_identifiers(mesh, element_identifiers):
    '''
    Deletes elements and related nodes using element identifiers.
    :param element_identifiers: Element identifiers for elements to be deleted.
    '''
    fm = mesh.getFieldmodule()
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    with ChangeManager(fm):
        # put the elements in a group and use subelement handling to get nodes in use by it
        destroyGroup = fm.createFieldGroup()
        destroyGroup.setSubelementHandlingMode(FieldGroup.SUBELEMENT_HANDLING_MODE_FULL)
        destroyMesh = destroyGroup.createMeshGroup(mesh)
        for elementIdentifier in element_identifiers:
            element = mesh.findElementByIdentifier(elementIdentifier)
            destroyMesh.addElement(element)
        if destroyMesh.getSize() > 0:
            # must destroy elements first as Zinc won't destroy nodes that are in use
            mesh.destroyElementsConditional(destroyGroup)
            nodes.destroyNodesConditional(destroyGroup)
        del destroyMesh
        del destroyGroup
    return


def get_mesh_first_element_with_node(mesh, field, node):
    """
    Assumes all components of field have the same Elementfieldtemplate.
    :param mesh: A Zinc Mesh or MeshGroup.
    :param field: Field to query, since nodes are mapped by field.
    :param node: A Zinc Node
    :return: Zinc Element or None
    """
    elementiterator = mesh.createElementiterator()
    element = elementiterator.next()
    while element.isValid():
        eft = element.getElementfieldtemplate(field, -1)
        if eft.isValid():
            node_count = eft.getNumberOfLocalNodes()
            for n in range(1, node_count + 1):
                if element.getNode(eft, n) == node:
                    return element
        element = elementiterator.next()
    return None


def get_mesh_node_identifier_sequences(mesh1d, field):
    """
    Get sequences of connected nodes in 1D mesh which field is directly defined on.
    Implementation expects mesh to consist of only polylines i.e. not a network and for elements to be
    consecutive and in the same orientation along each sequence.
    :param mesh1d: 1-D mesh or mesh group which field is defined on in every element.
    :param field: The field to get connectivity for. Must be finite element type.
    :return: List of lists of node identifiers in each sequence.
    """
    node_ids_list = []
    elementiterator = mesh1d.createElementiterator()
    element = elementiterator.next()
    while element.isValid():
        eft = element.getElementfieldtemplate(field, -1)
        if not eft.isValid():
            logger.error("mesh_get_connected_node_identifier_sequences. Field not defined on element")
            return []
        node_ids = get_element_node_identifiers(element, eft)
        if node_ids_list and (node_ids_list[-1][-1] == node_ids[0]):
            node_ids_list[-1] += node_ids[1:]
        else:
            node_ids_list.append(node_ids)
        element = elementiterator.next()
    return node_ids_list


def get_nodeset_field_parameters(nodeset, field, only_value_labels=None):
    """
    Returns parameters of field from nodes in nodeset in identifier order.
    Assumes all components have the same labels and versions.
    :param nodeset: Owning nodeset nodes are from.
    :param field: The field to get parameters for. Must be finite element type.
    :param only_value_labels: Optional list of node value labels to limit extraction from
    e.g. [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1].
    :return: list of valueLabels returned, list of node field parameters.
    Parameters are a tuple of (node identifier, node parameters),
    where node parameters are a list over value labels of a list of versions of parameters.
    Value labels without any parameters are removed before returning.
    """
    fieldmodule = nodeset.getFieldmodule()
    finite_element_field = field.castFiniteElement()
    assert finite_element_field.isValid(), "get_nodeset_field_parameters:  Field is not finite element type"
    components_count = field.getNumberOfComponents()
    fieldcache = fieldmodule.createFieldcache()
    value_labels = only_value_labels if only_value_labels else [
        Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
        Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3, Node.VALUE_LABEL_D2_DS2DS3, Node.VALUE_LABEL_D3_DS1DS2DS3]
    value_labels_count = len(value_labels)
    value_labels_parameter_counts = [0 for value_label in value_labels]
    node_field_parameters = []
    nodeiterator = nodeset.createNodeiterator()
    node = nodeiterator.next()
    while node.isValid():
        fieldcache.setNode(node)
        node_parameters = []
        field_defined_at_node = False
        for i in range(value_labels_count):
            value_parameters = []
            version = 1
            while True:
                result, parameters = finite_element_field.getNodeParameters(
                    fieldcache, -1, value_labels[i], version, components_count)
                if result != RESULT_OK:
                    break
                field_defined_at_node = True
                value_parameters.append(parameters)
                value_labels_parameter_counts[i] += 1
                version += 1
            node_parameters.append(value_parameters)
        if field_defined_at_node:
            node_field_parameters.append((node.getIdentifier(), node_parameters))
        node = nodeiterator.next()
    for i in range(value_labels_count - 1, -1, -1):
        if value_labels_parameter_counts[i] == 0:
            value_labels.pop(i)
            for node_parameters in node_field_parameters:
                node_parameters[1].pop(i)
    return value_labels, node_field_parameters


def set_nodeset_field_parameters(nodeset, field, value_labels, node_field_parameters, edit_group_name=None):
    """
    Set node parameters for coordinates field in path from listed values.
    :param nodeset: Owning nodeset nodes are from.
    :param field: The field to set parameters for. Must be finite element type.
    :param value_labels: List of node values/derivatives to set e.g. [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1]
    :param node_field_parameters: List of tuple of (node identifier, node parameters),
    where node parameters are a list over value labels of a list of versions of parameters,
    as returned by get_nodeset_field_parameters().
    If a particular node / value label / version is None or an empty list, no assignment is made.
    :param edit_group_name: Optional name of group to get or create and put modified nodes in the
    respective nodeset group.
    """
    fieldmodule = nodeset.getFieldmodule()
    finite_element_field = field.castFiniteElement()
    assert finite_element_field.isValid(), "set_nodeset_field_parameters:  Field is not finite element type"
    value_labels_count = len(value_labels)
    edit_nodeset_group = None
    with ChangeManager(fieldmodule):
        fieldcache = fieldmodule.createFieldcache()
        for node_identifier, node_parameters in node_field_parameters:
            node = nodeset.findNodeByIdentifier(node_identifier)
            assert(node.isValid()), "set_nodeset_field_parameters: Missing node " + str(node_identifier)
            fieldcache.setNode(node)
            changed_node_parameters = False
            for d in range(value_labels_count):
                node_value_parameters = node_parameters[d]
                versions_count = len(node_value_parameters)
                for v in range(versions_count):
                    if node_value_parameters[v]:
                        changed_node_parameters = True
                        finite_element_field.setNodeParameters(
                            fieldcache, -1, value_labels[d], v + 1, node_value_parameters[v])
            if edit_group_name and changed_node_parameters:
                if not edit_nodeset_group:
                    edit_group = find_or_create_field_group(fieldmodule, edit_group_name, managed=True)
                    edit_nodeset_group = edit_group.getOrCreateNodesetGroup(nodeset)
                edit_nodeset_group.addNode(node)


def make_nodeset_derivatives_orthogonal(nodeset, field, make_d2_normal: bool=True, make_d3_normal:bool=True,
                                        edit_group_name=None):
    """
    Make d2 and/or d3 normal to d1 for all nodes and versions of field parameters in nodeset.
    Side derivatives keep their current magnitudes.
    All derivatives must have the same number of versions.
    :param nodeset: Owning nodeset containing nodes to modify.
    :param field: The field to modify parameters for. Must be finite element type with 1-3 real components.
    :param make_d2_normal: Set to true if d2 is to be made normal to d1.
    :param make_d3_normal: Set to true if d3 is to be made normal to d1 and d2.
    :param edit_group_name: Optional name of group to get or create and put modified nodes in the
    respective nodeset group.
    """
    assert make_d2_normal or make_d3_normal
    finite_element_field = field.castFiniteElement()
    assert finite_element_field.isValid(), "make_nodeset_derivatives_orthogonal:  Field is not finite element type"
    assert field.getNumberOfComponents() <= 3, "make_nodeset_derivatives_orthogonal:  Field has more than 3 components"
    value_labels_in = [Node.VALUE_LABEL_D_DS1]
    if make_d2_normal or make_d3_normal:
        value_labels_in.append(Node.VALUE_LABEL_D_DS2)
    if make_d3_normal:
        value_labels_in.append(Node.VALUE_LABEL_D_DS3)
    value_labels, node_field_parameters = get_nodeset_field_parameters(nodeset, field, value_labels_in)
    assert Node.VALUE_LABEL_D_DS1 in value_labels, "make_nodeset_derivatives_orthogonal. Missing d/ds1 parameters"
    assert (not make_d2_normal) or Node.VALUE_LABEL_D_DS2 in value_labels, \
        "make_nodeset_derivatives_orthogonal. Missing d/ds2 parameters"
    assert (not make_d3_normal) or Node.VALUE_LABEL_D_DS3 in value_labels, \
        "make_nodeset_derivatives_orthogonal. Missing d/ds3 parameters"
    d1_index = value_labels.index(Node.VALUE_LABEL_D_DS1)
    d2_index = value_labels.index(Node.VALUE_LABEL_D_DS2) if (Node.VALUE_LABEL_D_DS2 in value_labels) else None
    d3_index = value_labels.index(Node.VALUE_LABEL_D_DS3) if (Node.VALUE_LABEL_D_DS3 in value_labels) else None
    for node_identifier, node_parameters in node_field_parameters:
        versions_count = len(node_parameters[d1_index])
        assert (((d2_index is None) or (len(node_parameters[d2_index]) == versions_count)) and
                ((d3_index is None) or (len(node_parameters[d3_index]) == versions_count))), \
            "make_nodeset_derivatives_orthogonal. Mismatched numbers of derivative versions at node " \
            + str(node_identifier)
        for v in range(versions_count):
            d1 = node_parameters[d1_index][v]
            d2 = node_parameters[d2_index][v] if (d2_index is not None) else None
            if make_d2_normal:
                td2 = rejection(d2, d1)
                td2 = set_magnitude(td2, magnitude(d2))
                d2 = node_parameters[d2_index][v] = td2
            if make_d3_normal:
                d3 = node_parameters[d3_index][v]
                if d2:
                    td3 = cross(d1, d2)
                else:
                    td3 = rejection(d3, d1)
                td3 = set_magnitude(td3, magnitude(d3))
                d3 = node_parameters[d3_index][v] = td3
    remove_indexes = [d1_index]
    if (d2_index is not None) and (not make_d2_normal):
        # order from highest index to lowest
        if d1_index < d2_index:
            remove_indexes.insert(0, d2_index)
        else:
            remove_indexes.append(d2_index)
    for i in remove_indexes:
        value_labels.pop(i)
    for node_identifier, node_parameters in node_field_parameters:
        for i in remove_indexes:
            node_parameters.pop(i)
    set_nodeset_field_parameters(nodeset, field, value_labels, node_field_parameters, edit_group_name)


def get_nodeset_path_field_parameters(nodeset, field, value_labels):
    """
    Get vectors of field parameters of nodes from nodeset in identifier order for
    each specified value_labels. Only first version of each parameter is obtained per node.
    Fails if any value_labels or version is not defined for all nodes and field components.
    :param nodeset: Owning nodeset.
    :param field: The field to get parameters for. Must be finite element type.
    :param value_labels: List of parameters required as list of node value labels.
    e.g. [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1].
    :return: List of value_label_parameters in order of value_labels e.g x[], d1[].
    """
    fieldmodule = nodeset.getFieldmodule()
    finite_element_field = field.castFiniteElement()
    assert finite_element_field.isValid(), "get_nodeset_path_field_parameters:  Field is not finite element type"
    components_count = field.getNumberOfComponents()
    value_labels_count = len(value_labels)
    assert value_labels_count > 0
    fieldcache = fieldmodule.createFieldcache()
    value_label_parameters = [[] for i in range(value_labels_count)]
    nodeiterator = nodeset.createNodeiterator()
    node = nodeiterator.next()
    while node.isValid():
        fieldcache.setNode(node)
        version = 1
        for i in range(value_labels_count):
            result, parameters = \
                finite_element_field.getNodeParameters(fieldcache, -1, value_labels[i], version, components_count)
            assert result == RESULT_OK, "get_nodeset_path_field_parameters. Node value/version not defined"
            value_label_parameters[i].append(parameters)
        node = nodeiterator.next()
    return value_label_parameters


def get_nodeset_path_ordered_field_parameters(nodeset, field, value_labels, node_identifiers, versions):
    """
    Get vectors of field parameters of nodes from nodeset in specified identifier order
    for a single version (version 1 for VALUE, versions[n] for derivatives at the n'th node),
    for each specified value_labels.
    Fails if any value_labels or versions are not defined for all nodes and field components.
    :param nodeset: Owning nodeset.
    :param field: The field to get parameters for. Must be finite element type.
    :param value_labels: List of parameters required as list of node value labels.
    e.g. [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1].
    :param node_identifiers: List of node identifiers to only get parameters for.
    :param versions: List of node parameter versions for the above node identifiers, starting at 1.
    :return: List of value_label_parameters in order of value_labels e.g x[], d1[].
    """
    fieldmodule = nodeset.getFieldmodule()
    finite_element_field = field.castFiniteElement()
    assert finite_element_field.isValid(), \
        "get_nodeset_path_ordered_field_parameters:  Field is not finite element type"
    components_count = field.getNumberOfComponents()
    value_labels_count = len(value_labels)
    assert value_labels_count > 0
    node_identifiers_count = len(node_identifiers)
    assert node_identifiers_count == len(versions)
    fieldcache = fieldmodule.createFieldcache()
    value_label_parameters = [[] for i in range(value_labels_count)]
    for n in range(len(node_identifiers)):
        node = nodeset.findNodeByIdentifier(node_identifiers[n])
        assert node.isValid(), "get_nodeset_path_field_parameters. Missing node"
        fieldcache.setNode(node)
        for i in range(value_labels_count):
            value_label = value_labels[i]
            version = 1 if (value_label == Node.VALUE_LABEL_VALUE) else versions[n]
            result, parameters = \
                finite_element_field.getNodeParameters(fieldcache, -1, value_label, version, components_count)
            assert result == RESULT_OK, "get_nodeset_path_ordered_field_parameters. Node value/version not defined"
            value_label_parameters[i].append(parameters)
    return value_label_parameters


def setPathParameters(region, nodeValueLabels, nodeValues, editGroupName=None):
    '''
    Set node parameters for coordinates field in path from listed values.
    Only handles version 1.
    :param nodeValueLabels: List of nodeValueLabels to set e.g. [ Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1 ]
    :param nodeValues: List of values for each type e.g. [ xlist, d1list ]
    :param editGroupName: Optional name of existing or new Zinc group to record modified nodes in.
    '''
    fieldmodule = region.getFieldmodule()
    coordinates = fieldmodule.findFieldByName('coordinates').castFiniteElement()
    componentsCount = coordinates.getNumberOfComponents()
    # following requires at least one value label and node, assumes consistent values and components counts
    nodeValueLabelsCount = len(nodeValueLabels)
    nodesCount = len(nodeValues[0])
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    assert nodesCount == nodes.getSize()
    with ChangeManager(fieldmodule):
        if editGroupName:
            editGroup = find_or_create_field_group(fieldmodule, editGroupName, managed=True)
            editNodesetGroup = editGroup.getOrCreateNodesetGroup(nodes)
        cache = fieldmodule.createFieldcache()
        nodeiterator = nodes.createNodeiterator()
        node = nodeiterator.next()
        n = 0
        while node.isValid():
            cache.setNode(node)
            for v in range(nodeValueLabelsCount):
                coordinates.setNodeParameters(cache, -1, nodeValueLabels[v], 1, nodeValues[v][n])
            if editGroupName:
                editNodesetGroup.addNode(node)
            node = nodeiterator.next()
            n += 1


def parameter_lists_to_string(values_list, format_string):
    '''
    :return: 'None' if values is an empty list, the values in the first item if only one, otherwise the lists of values.
    '''
    if not values_list:
        return 'None'
    if len(values_list) == 1:
        return '[' + ','.join(format_string.format(value) for value in values_list[0]) + ']'
    return '[' + ','.join(
        '[' + ','.join(format_string.format(value) for value in sub_values_list) + ']'
        for sub_values_list in values_list) + ']'


def print_node_field_parameters(value_labels, node_field_parameters, format_string='{: 11e}'):
    """
    Print value labels and node parameters returned by get_nodeset_field_parameters ready for
    pasting into python code. Example:
    [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1]
    [
    (1, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    (2, [[0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ]
    where node 2 has 2 versions of D_DS1.
    :param format_string: Default format uses exponential notation with 7 significant digits = single precision.
    """
    print('[' + ', '.join('Node.VALUE_LABEL_' + Node.ValueLabelEnumToString(value_label)
                          for value_label in value_labels) + ']')
    print('[')
    last_node_identifier = node_field_parameters[-1][0]
    for node_identifier, node_parameters in node_field_parameters:
        print('(' + str(node_identifier) + ', [' + ', '.join(parameter_lists_to_string(valueParameters, format_string)
                                                             for valueParameters in node_parameters) + '])' +
              (', ' if (node_identifier != last_node_identifier) else ''))
    print(']\n')


def exnode_string_from_nodeset_field_parameters(
        field_names=["coordinates"],
        value_labels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1],
        node_field_parameters = [[
            (1, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            (2, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]],
        group_name = 'meshEdits'):
    """
    Return a string in Zinc EX format defining nodes with the supplied identifiers
    and coordinate field parameters.
    Works in a private zinc context.
    :param field_names: List of fields defining fields for node_field_parameters.
    :param value_labels: List of Node.ValueLabels supplied for each node.
    :param node_field_parameters: List of tuples of node identifier and parameters for
    each of the node values supplied. Versions may be supplied for any node/value
    by supplying a list of coordinate vectors. A single version can be supplied
    either as a single vector, or as a list containing a single vector.
    e.g. (1, [[0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
    defines node 1 with 2 versions of DS1 using the default value_labels.
    Note: If node identifier is -1 it uses the first available identifier starting at 1;
    if used, this should be used for all node identifiers.
    :param group_name:  Name of group to put nodes in.
    """
    # following requires at least one value label and node, assumes consistent values and components counts
    value_labels_count = len(value_labels)
    assert len(node_field_parameters) > 0
    node_parameters = node_field_parameters[0][0][1]
    components_count = \
        len(node_parameters[0][0]) if isinstance(node_parameters[0][0], list) else len(node_parameters[0])
    context = Context('exnode_string_from_nodeset_field_parameters')
    region = context.getDefaultRegion()
    fieldmodule = region.getFieldmodule()
    allCoordinates = []
    with ChangeManager(fieldmodule):
        fieldcache = fieldmodule.createFieldcache()
        for i in range(len(field_names)):
            allCoordinates.append(find_or_create_field_coordinates(fieldmodule, field_names[i],
                                                                   components_count=components_count))

        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        group = fieldmodule.createFieldGroup()
        group.setName(group_name)
        nodeset_group = group.createNodesetGroup(nodes)

        # create nodes
        for i in range(len(node_field_parameters)):
            nodetemplates = {} # dict mapping from tuple of derivative versions to nodetemplate
            for node_identifier, node_parameters in node_field_parameters[i]:
                value_labels_versions = []
                for d in range(value_labels_count):
                    if isinstance(node_parameters[d][0], list):
                        value_labels_versions.append(len(node_parameters[d]))
                    else:
                        value_labels_versions.append(1)
                value_labels_versions = tuple(value_labels_versions)  # must be tuple to use as dict key
                nodetemplate = nodetemplates.get(value_labels_versions)

                if not nodetemplate:
                    nodetemplate = nodes.createNodetemplate()
                    nodetemplate.defineField(allCoordinates[i])
                    if not Node.VALUE_LABEL_VALUE in value_labels:
                        nodetemplate.setValueNumberOfVersions(allCoordinates[i], -1, Node.VALUE_LABEL_VALUE, 0)
                    for d in range(value_labels_count):
                        nodetemplate.setValueNumberOfVersions(allCoordinates[i], -1, value_labels[d], value_labels_versions[d])
                    nodetemplates[value_labels_versions] = nodetemplate
                if i == 0:
                    node = nodeset_group.createNode(node_identifier, nodetemplate)
                else:
                    node = nodes.findNodeByIdentifier(node_identifier)
                    node.merge(nodetemplate)
                fieldcache.setNode(node)
                for d in range(value_labels_count):
                    if isinstance(node_parameters[d][0], list):
                        for v in range(value_labels_versions[d]):
                            allCoordinates[i].setNodeParameters(fieldcache, -1, value_labels[d], v + 1, node_parameters[d][v])
                    else:
                        allCoordinates[i].setNodeParameters(fieldcache, -1, value_labels[d], 1, node_parameters[d])

        # serialise to string
        sir = region.createStreaminformationRegion()
        srm = sir.createStreamresourceMemory()
        sir.setResourceGroupName(srm, group_name)
        region.write(sir)
        result, exString = srm.getBuffer()

    return exString

def disconnectFieldMeshGroupBoundaryNodes(coordinateFields, meshGroup1, meshGroup2, nextNodeIdentifier):
    """
    Duplicate nodes in use by coordinateField on meshGroup1 and meshGroup2 and use these exclusively in meshGroup2.
    :param coordinateFields: The sequence of field to disconnect, type Zinc FieldFiniteElement. These are expected to
    be define with the same Elementfieldtemplate for all components in any element.
    :param meshGroup1: A Zinc MeshGroup containing elements in first group.
    :param meshGroup2: A Zinc MeshGroup containing elements in second group, which will be changed to use the new copies of nodes.
    :param nextNodeIdentifier: First available node identifier.
    :return: Final nextNodeIdentifier to use after this call, list of created node identifiers.
    """
    elemiter = meshGroup1.createElementiterator()
    element = elemiter.next()
    nodeIdentifiers1 = set()
    while element.isValid():
        eft = element.getElementfieldtemplate(coordinateFields[0], -1)
        nodeCount = eft.getNumberOfLocalNodes()
        for n in range(1, nodeCount + 1):
            node = element.getNode(eft, n)
            nodeIdentifiers1.add(node.getIdentifier())
        element = elemiter.next()
    copyIdentifiersMap = {}
    fieldmodule = coordinateFields[0].getFieldmodule()
    fieldcache = fieldmodule.createFieldcache()
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = nodes.createNodetemplate()

    allValueLabels = [
        Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
        Node.VALUE_LABEL_D2_DS1DS3, Node.VALUE_LABEL_D2_DS2DS3, Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D3_DS1DS2DS3]

    with ChangeManager(fieldmodule):
        elemiter = meshGroup2.createElementiterator()
        element = elemiter.next()
        while element.isValid():
            eft = element.getElementfieldtemplate(coordinateFields[0], -1)
            nodeCount = eft.getNumberOfLocalNodes()
            for n in range(1, nodeCount + 1):
                existingNode = element.getNode(eft, n)
                existingNodeIdentifier = existingNode.getIdentifier()
                copyNodeIdentifier = copyIdentifiersMap.get(existingNodeIdentifier)
                copyNode = None

                if copyNodeIdentifier:
                    copyNode = nodes.findNodeByIdentifier(copyNodeIdentifier)
                else:
                    if existingNodeIdentifier in nodeIdentifiers1:
                        copyNodeIdentifier = nextNodeIdentifier
                        for coordinateField in coordinateFields:
                            nodetemplate.defineFieldFromNode(coordinateField, existingNode)
                        copyNode = nodes.createNode(nextNodeIdentifier, nodetemplate)
                        copyIdentifiersMap[existingNodeIdentifier] = copyNodeIdentifier
                        nextNodeIdentifier += 1

                        # copy field parameters from existingNode to copyNode
                        for coordinateField in coordinateFields:
                            components_count = coordinateField.getNumberOfComponents()

                            for valueLabel in allValueLabels:
                                versionCount = nodetemplate.getValueNumberOfVersions(coordinateField, -1, valueLabel)

                                for version in range(1, versionCount + 1):
                                    fieldcache.setNode(existingNode)
                                    result, values = coordinateField.getNodeParameters(
                                        fieldcache, -1, valueLabel, version, components_count)
                                    fieldcache.setNode(copyNode)
                                    coordinateField.setNodeParameters(
                                        fieldcache, -1, valueLabel, version, values)

                if copyNode:
                    result = element.setNode(eft, n, copyNode)
                    assert result == 1

            element = elemiter.next()

    return nextNodeIdentifier, list(copyIdentifiersMap.values())


def clearRegion(region):
    """
    Destroy all elements, nodes, datapoints in region and unmanage all fields.
    Remove all child regions.
    Some fields may remain due to external references being held.
    :param region: Region to clear.
    """
    with HierarchicalChangeManager(region):
        child = region.getFirstChild()
        while child.isValid():
            nextSibling = child.getNextSibling()
            region.removeChild(child)
            child = nextSibling
        fieldmodule = region.getFieldmodule()
        for dimension in range(3, 0, -1):
            mesh = fieldmodule.findMeshByDimension(dimension)
            mesh.destroyAllElements()
        for fieldDomainType in [Field.DOMAIN_TYPE_NODES, Field.DOMAIN_TYPE_DATAPOINTS]:
            nodeset = fieldmodule.findNodesetByFieldDomainType(fieldDomainType)
            nodeset.destroyAllNodes()
        fieldIter = fieldmodule.createFielditerator()
        field = fieldIter.next()
        while field.isValid():
            field.setManaged(False)
            field = fieldIter.next()


def generate_curve_mesh(region, nx, nd1, loop=False, start_node_identifier=None, start_element_identifier=None,
                        coordinate_field_name="coordinates", group_name=None):
    """
    Generate a set of 1-D elements with Hermite basis
    :param region: Zinc Region.
    :param nx: Coordinates along curve.
    :param nd1: Derivatives along curve.
    :param loop: True if curve loops back to first point, False if not.
    :param start_node_identifier: Optional first node identifier to use.
    :param start_element_identifier: Optional first 1D element identifier to use.
    :param coordinate_field_name: Optional name of coordinate field to define, if omitted use "coordinates".
    :param group_name: Optional name of group to put new nodes and elements in.
    :return: next node identifier, next 1D element identifier
    """
    fieldmodule = region.getFieldmodule()
    with ChangeManager(fieldmodule):
        coordinates = find_or_create_field_coordinates(fieldmodule, name=coordinate_field_name)
        group = find_or_create_field_group(fieldmodule, group_name) if group_name else None

        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        node_identifier = start_node_identifier if start_node_identifier is not None else \
            max(get_maximum_node_identifier(nodes), 0) + 1

        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)

        mesh = fieldmodule.findMeshByDimension(1)
        element_identifier = start_element_identifier if start_element_identifier is not None else \
            max(get_maximum_element_identifier(mesh), 0) + 1
        elementtemplate = mesh.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
        cubicHermiteBasis = fieldmodule.createElementbasis(
            1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        eft = mesh.createElementfieldtemplate(cubicHermiteBasis)
        elementtemplate.defineField(coordinates, -1, eft)

        fieldcache = fieldmodule.createFieldcache()
        nids = []
        nCount = len(nx)
        nodeset_group = group.getOrCreateNodesetGroup(nodes) if group else None
        for n in range(nCount):
            node = nodes.createNode(node_identifier, nodetemplate)
            nids.append(node_identifier)
            fieldcache.setNode(node)
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, nx[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, nd1[n])
            if nodeset_group:
                nodeset_group.addNode(node)
            node_identifier += 1
        eCount = nCount if loop else nCount - 1
        if loop:
            nids.append(nids[0])
        mesh_group = group.getOrCreateMeshGroup(mesh) if group else None
        for e in range(eCount):
            element = mesh.createElement(element_identifier, elementtemplate)
            element.setNodesByIdentifier(eft, nids[e:e + 2])
            if mesh_group:
                mesh_group.addElement(element)
            element_identifier += 1

    return node_identifier, element_identifier


def generate_datapoints(region, px, start_data_identifier=None, coordinate_field_name="coordinates",
                        field_names_and_values=[], group_name=None):
    """
    Generate a set of datapoints in the region.
    :param region: Zinc Region.
    :param px: Coordinates of data points.
    :param start_data_identifier: Optional first datapoint identifier to use.
    :param coordinate_field_name: Optional name of coordinate field to define, if omitted use "coordinates".
    :param field_names_and_values: Optional lists of (field_name, list of values) for additional fields to
    define on the datapoints. Values may be scalar or vector (list of lists) real, or string.
    Must be same number of values as number of points.
    :param group_name: Optional name of group to put new datapoints in.
    :return: next datapoint identifier
    """
    fieldmodule = region.getFieldmodule()
    with ChangeManager(fieldmodule):
        coordinates = find_or_create_field_coordinates(fieldmodule, name=coordinate_field_name)
        group = find_or_create_field_group(fieldmodule, group_name) if group_name else None

        datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        data_identifier = start_data_identifier if (start_data_identifier is not None) else \
            max(get_maximum_node_identifier(datapoints), 0) + 1
        data_group = group.getOrCreateNodesetGroup(datapoints) if group else datapoints

        nodetemplate = datapoints.createNodetemplate()
        nodetemplate.defineField(coordinates)
        fields_values = []  # (field, is_string, values)
        for field_name, field_values in field_names_and_values:
            is_string = isinstance(field_values[0], str)
            if is_string:
                field = find_or_create_field_stored_string(fieldmodule, field_name, managed=True)
            else:
                components_count = len(field_values[0]) if isinstance(field_values[0], list) else 1
                field = find_or_create_field_finite_element(fieldmodule, field_name, components_count, managed=True)
            nodetemplate.defineField(field)
            fields_values.append((field, is_string, field_values))

        fieldcache = fieldmodule.createFieldcache()
        for n, x in enumerate(px):
            node = data_group.createNode(data_identifier, nodetemplate)
            fieldcache.setNode(node)
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
            for field, is_string, values in fields_values:
                if is_string:
                    field.assignString(fieldcache, values[n])
                else:
                    field.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, values[n])
            data_identifier += 1

    return data_identifier


def generate_mesh_marker_points(mesh, marker_data, start_node_identifier=None):
    """
    Create marker points in the "marker" group with supplied names and mesh locations.
    :param mesh: Zinc region to create marker points in.
    :param marker_data: list of marker (name, (element_identifier, xi)). Xi must have same size as mesh dimension.
    :return: Next node identifier
    """
    fieldmodule = mesh.getFieldmodule()
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    node_identifier = start_node_identifier if start_node_identifier is not None else \
        max(get_maximum_node_identifier(nodes), 0) + 1
    with ChangeManager(fieldmodule):
        marker_group = find_or_create_field_group(fieldmodule, "marker")
        marker_nodeset_group = marker_group.getOrCreateNodesetGroup(nodes)
        marker_location = find_or_create_field_stored_mesh_location(
            fieldmodule, mesh, name="marker_location", managed=True)
        marker_name = find_or_create_field_stored_string(fieldmodule, name="marker_name", managed=True)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(marker_name)
        nodetemplate.defineField(marker_location)
        fieldcache = fieldmodule.createFieldcache()
        for name, location in marker_data:
            node = marker_nodeset_group.createNode(node_identifier, nodetemplate)
            fieldcache.setNode(node)
            marker_name.assignString(fieldcache, name)
            element = mesh.findElementByIdentifier(location[0])
            xi = location[1]
            if element.isValid():
                marker_location.assignMeshLocation(fieldcache, element, xi)
            node_identifier += 1
    return node_identifier


def mesh_get_element_nodes_map(mesh):
    """
    Get the nodes used by each element in mesh, in no particular order.
    Supports face and line meshes which inherit field from higher-level elements.
    This is quite expensive for large meshes as relies on group sub-element handling.
    Zinc issue: nodes list is only based on the first coordinate field.
    :param mesh: A Zinc mesh or mesh group containing the elements to query.
    :return: dict element identifier -> list(node identifiers)
    """
    fieldmodule = mesh.getFieldmodule()
    elementid_to_nodeids = {}
    with ChangeManager(fieldmodule):
        group = fieldmodule.createFieldGroup()
        group.setSubelementHandlingMode(FieldGroup.SUBELEMENT_HANDLING_MODE_FULL)
        mesh_group = group.createMeshGroup(mesh)
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodeset_group = group.createNodesetGroup(nodes)
        elementiterator = mesh.createElementiterator()
        element = elementiterator.next()
        while element.isValid():
            mesh_group.addElement(element)
            nodeiterator = nodeset_group.createNodeiterator()
            node = nodeiterator.next()
            nodeIds = []
            while node.isValid():
                nodeIds.append(node.getIdentifier())
                node = nodeiterator.next()
            nodeset_group.removeAllNodes()
            del nodeiterator
            elementid_to_nodeids[element.getIdentifier()] = nodeIds
            element = elementiterator.next()
        del elementiterator
        del mesh_group
        del nodeset_group
        del group
    return elementid_to_nodeids


def group_add_connected_elements(group: FieldGroup, other_mesh_group: MeshGroup):
    """
    Add to group the elements from other_mesh_group which use a node from the group's
    node group, or are connected to an element which does.
    Note this can be quite expensive for large meshes.
    :param group: Zinc FieldGroup to add elements to. The group's NodeGroup must
    contain at least one node to connect to, and nodes from connected elements are
    added to it
    :param other_mesh_group: Other mesh group containing candidate elements to add.
    """
    fieldmodule = group.getFieldmodule()
    with ChangeManager(fieldmodule):
        old_subelement_mode = group.getSubelementHandlingMode()
        group.setSubelementHandlingMode(FieldGroup.SUBELEMENT_HANDLING_MODE_FULL)
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        mesh_group = group.getOrCreateMeshGroup(other_mesh_group)
        assert mesh_group.isValid()
        nodeset_group = group.getNodesetGroup(nodes)
        assert nodeset_group.isValid()
        elementid_to_nodeids = mesh_get_element_nodes_map(other_mesh_group)
        connected = True
        while connected:
            connected = False
            for elementid, nodeids in elementid_to_nodeids.items():
                for nodeid in nodeids:
                    node = nodeset_group.findNodeByIdentifier(nodeid)
                    if node.isValid():
                        connected = True
                        break
                if connected:
                    mesh_group.addElement(other_mesh_group.findElementByIdentifier(elementid))
                    del elementid_to_nodeids[elementid]
                    break
        group.setSubelementHandlingMode(old_subelement_mode)


def blendDerivativeBetweenDifferentSizeElements(coordinates, cache, nodes, prevNodeID, transitionNodeID, nextNodeID):
    """
    Blends derivatives for nodes between two elements of different lengths. Calculates an end derivative using
    upstream coordinates and derivative, and a start derivative using downstream coordinates and derivative. The
    derivative at the junction is then rescaled with the harmonic mean of start and end derivative magnitudes.
    :param coordinates: coordinate field
    :param cache: field cache
    :param nodes: nodeset
    :param prevNodeID: node identifier of upstream node
    :param transitionNodeID: node identifier of node between the two elements of different lengths
    :param nextNodeID: node identifier of downstream node
    """
    node = nodes.findNodeByIdentifier(prevNodeID)
    cache.setNode(node)
    xPrev = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, 3)[1]
    d2Prev = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)[1]

    node = nodes.findNodeByIdentifier(nextNodeID)
    cache.setNode(node)
    xNext = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, 3)[1]
    d2Next = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)[1]

    node = nodes.findNodeByIdentifier(transitionNodeID)
    cache.setNode(node)
    xTransition = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, 3)[1]
    d2Transition = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)[1]

    d2_in = interp.computeCubicHermiteEndDerivative(xPrev, d2Prev, xTransition, d2Transition)
    d2_out = interp.computeCubicHermiteStartDerivative(xTransition, d2Transition, xNext, d2Next)
    d2MeanMag = 2.0 / ((1.0 / magnitude(d2_in)) + (1.0 / magnitude(d2_out)))
    d2Mean = set_magnitude(d2Transition, d2MeanMag)
    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, d2Mean)


def find_or_create_field_zero_fibres(fieldmodule, name="zero fibres", managed=True):
    """
    Find or create a constant field representing 3 zero fibre angles.
    :param fieldmodule: Owning field module to find or create field in.
    :param name: Name of zero fibres field.
    :param managed: Whether
    :return: Zinc Field.
    """
    zero_fibres = fieldmodule.findFieldByName(name)
    if zero_fibres.isValid():
        return zero_fibres
    with ChangeManager(fieldmodule):
        zero_fibres = fieldmodule.createFieldConstant([0.0, 0.0, 0.0])
        zero_fibres.setName(name)
        zero_fibres.setManaged(managed)
    return zero_fibres


def fit_hermite_curve(bx, bd1, px, outlier_length=0.0, region=None, group_name=None):
    """
    Fit 1-D multi-element hermite curve to list of data point coordinates.
    Uses scaffoldfitter/Zinc to perform fit.
    :param bx: Initial/before curve coordinates, close to data.
    :param bd1: Initial/before curve derivatives, close to data.
    :param px: List of data points [x, y, z] or [x, y] if 2-D to fit to.
    :param outlier_length: Absolute outlier length for data if positive, or relative outlier length if negative e.g.
    -0.1 removes data points with projections lengths within 10% of largest projection length. Not used in the first
    fit iteration. Ignored if zero.
    :param region: Optional Zinc Region to perform fit in, so available for re-use after call.
    Region is expected to be empty.
    :param group_name: Optional name of group to put fit elements and data in, or None to use default "curve".
    :return: cx, cd1 (lists of hermite coordinates and derivatives)
    """
    points_count = len(px)
    elements_count = len(bx) - 1
    components_count = len(bx[0])
    curve_length = interp.getCubicHermiteCurvesLength(bx, bd1)
    curve_group_name = group_name if group_name else "curve"

    if region:
        fit_region = region
    else:
        # perform the fit in a Zinc Region in a private Context
        context = Context("fit_hermite_curve")
        fit_region = context.getDefaultRegion()
    fieldmodule = fit_region.getFieldmodule()
    with ChangeManager(fieldmodule):
        coordinates = find_or_create_field_coordinates(fieldmodule)
        generate_curve_mesh(fit_region, bx, bd1, group_name=curve_group_name)
        generate_datapoints(fit_region, px, group_name=curve_group_name)
        # need a zero fibre field to apply strain/curvature penalties on 1-D model
        zero_fibres = find_or_create_field_zero_fibres(fieldmodule)

    fitter = GeometryFitter(region=fit_region)
    # fitter.setDiagnosticLevel(1)
    fitter.setModelCoordinatesField(coordinates)
    fitter.setFibreField(zero_fibres)
    fitter.defineCommonMeshFields()
    fitter.setDataCoordinatesField(coordinates)
    fitter.defineDataProjectionFields()
    fitter.initializeFit()

    # calibrated for 25 points / element
    data_weight = (25.0 * elements_count) / points_count
    strain_penalty = 1.0E-6 * curve_length
    # calibrated by scaling the model: a power of 3 relationship
    curvature_penalty = 1.0E-6 * (curve_length ** 3)
    fit1 = FitterStepFit()
    fitter.addFitterStep(fit1)
    fit1.setGroupDataWeight(None, data_weight)
    fit1.setGroupStrainPenalty(None, [strain_penalty])
    fit1.setGroupCurvaturePenalty(None, [curvature_penalty])
    fit1.run()
    del fit1

    if outlier_length != 0.0:
        config2 = FitterStepConfig()
        fitter.addFitterStep(config2)
        config2.setGroupOutlierLength(None, outlier_length)
        config2.run()
        del config2

        fit3 = FitterStepFit()
        fitter.addFitterStep(fit3)
        fit3.run()
        del fit3

    fitter.cleanup()
    del fitter

    # extract fitted curve parameters from nodes
    cx = []
    cd1 = []
    fieldcache = fieldmodule.createFieldcache()
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodeiterator = nodes.createNodeiterator()
    node = nodeiterator.next()
    while node.isValid():
        fieldcache.setNode(node)
        result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, components_count)
        result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, components_count)
        cx.append(x)
        cd1.append(d1)
        node = nodeiterator.next()

    return cx, cd1


def define_and_fit_field(region, coordinate_field_name, data_coordinate_field_name, fit_field_name,
                         gradient1_penalty, gradient2_penalty, group_name=None):
    """
    Define and fit field over mesh in region. Components of field are defined identically to first
    component of the coordinate field.
    :param region: Region containing geometric model to define and fit field to, and data points containing
    field values
    :param coordinate_field_name: Name of coordinate field on the model. The first component of this field is used
    as a template for defining all components of the fit field over the model.
    :param data_coordinate_field_name: Name of coordinate field on the data, used to find nearest mesh location on
    model.
    :param fit_field_name: Name of field to fit; must be defined over data points.
    :param gradient1_penalty: Penalty factor for first gradient of field w.r.t. coordinates.
    :param gradient2_penalty: Penalty factor for second gradient of field w.r.t. coordinates.
    :param group_name: Optional name of group to limit fitting over.
    :return: rms_error, max_error
    """
    fitter = FieldFitter(region=region)
    # fitter.setDiagnosticLevel(1)
    fieldmodule = region.getFieldmodule()
    coordinates = fieldmodule.findFieldByName(coordinate_field_name).castFiniteElement()
    fitter.setModelCoordinatesField(coordinates)
    data_coordinates = fieldmodule.findFieldByName(data_coordinate_field_name).castFiniteElement()
    fitter.setDataCoordinatesField(data_coordinates)
    if group_name:
        model_fit_group = fieldmodule.findFieldByName(group_name).castGroup()
        assert model_fit_group.isValid()
        fitter.setModelFitGroup(model_fit_group)
    fitter.updateFitFields()
    if fieldmodule.findMeshByDimension(coordinates.getNumberOfComponents()).getSize() == 0:
        # need a zero fibre field to apply strain/curvature penalties on 1-D model
        fitter.setFibreField(find_or_create_field_zero_fibres(fieldmodule))
    fitter.setGradient1Penalty([gradient1_penalty])
    fitter.setGradient2Penalty([gradient2_penalty])
    fitter.setFitField(fit_field_name, True)
    fitter.fitField(fit_field_name)
    rms_error, max_error = fitter.getFieldDataRMSAndMaximumErrors(fit_field_name)
    fitter.cleanup()
    del fitter
    return rms_error, max_error

def find_first_node_conditional(nodeset, conditional_field):
    """
    Returns the first node in the conditional field.
    :param nodeset: set of nodes.
    :param conditional_field: Field containing nodes that satisfy conditions passed to the function.
    """
    fieldmodule = nodeset.getFieldmodule()
    with ChangeManager(fieldmodule):
        group = fieldmodule.createFieldGroup()
        nodeset_group = group.createNodesetGroup(nodeset)
        nodeset_group.addNodesConditional(conditional_field)
        junction_node = nodeset_group.createNodeiterator().next()

        del nodeset_group
        del group

    return junction_node

def get_node_mesh_location(node, node_coordinates, mesh, mesh_coordinates, search_mesh=None):
    """
    Returns the element and xi value of the node in the host mesh.
    :param node: node to find mesh location.
    :param node_coordinates: field coordinates
    :param mesh: mesh where node sits in.
    :param mesh_coordinates: coordinates of mesh. Could be geometric or material coordinates
    :param search_mesh: mesh for conducting search for node location within the mesh. Can be used to reduce search
    time if a partial mesh containing the node is supplied.
    """
    fieldmodule = mesh.getFieldmodule()
    with ChangeManager(fieldmodule):
        find_mesh_location = fieldmodule.createFieldFindMeshLocation(node_coordinates, mesh_coordinates, mesh)
        if search_mesh:
            find_mesh_location.setSearchMesh(search_mesh)
        find_mesh_location.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
        fieldcache = fieldmodule.createFieldcache()
        fieldcache.setNode(node)
        element, xi = find_mesh_location.evaluateMeshLocation(fieldcache, mesh.getDimension())
        del fieldcache
        del find_mesh_location

    return element, xi

def translate_nodeset_coordinates(nodeset, coordinates, translation):
    """
    Translate nodes by applying a translation vector to their coordinates.
    :param nodeset: Set of nodes.
    :param coordinates: Coordinates field.
    :param translation:
    """
    fieldmodule = nodeset.getFieldmodule()
    fieldcache = fieldmodule.createFieldcache()
    components_count = coordinates.getNumberOfComponents()
    with ChangeManager(fieldmodule):
        nodeiterator = nodeset.createNodeiterator()
        node = nodeiterator.next()
        while node.isValid():
            fieldcache.setNode(node)
            result, x = coordinates.evaluateReal(fieldcache, components_count)
            result = coordinates.assignReal(fieldcache, add(x, translation))
            node = nodeiterator.next()
