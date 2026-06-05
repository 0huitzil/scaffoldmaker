from cmlibs.maths.vectorops import (
    add, cross, distance, dot, magnitude, matrix_mult, matrix_inv, mult, normalize, rejection, set_magnitude, sub)
from cmlibs.utils.zinc.field import find_or_create_field_group, find_or_create_field_coordinates
from cmlibs.utils.zinc.finiteelement import evaluate_field_nodeset_range
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.element import Element, Elementbasis, Elementfieldtemplate
from cmlibs.zinc.field import Field, FieldFindMeshLocation, FieldGroup
from cmlibs.zinc.node import Node
from cmlibs.zinc.result import RESULT_OK
from scaffoldfitter.fitter import Fitter as GeometryFitter
from scaffoldfitter.fitterstepfit import FitterStepFit
from scaffoldmaker.annotation.annotationgroup import AnnotationGroup, findOrCreateAnnotationGroupForTerm, \
    findAnnotationGroupByName
from scaffoldmaker.annotation.vagus_terms import get_vagus_term, get_vagus_marker_term, \
    get_left_vagus_marker_locations_list, get_right_vagus_marker_locations_list
from scaffoldmaker.meshtypes.scaffold_base import Scaffold_base
from scaffoldmaker.utils.constructionobject import ConstructionObject
from scaffoldmaker.utils.eft_utils import remapEftLocalNodes, remapEftNodeValueLabel, remapEftNodeValueLabelWithNodes, \
    setEftScaleFactorIds
from scaffoldmaker.utils.interpolation import (
    evaluateCoordinatesOnCurve, evaluateScalarOnCurve, getCubicHermiteBasis, getCubicHermiteBasisDerivatives,
    getCubicHermiteArcLength, getCubicHermiteCurvature, getCubicHermiteCurvesLength,
    getCubicHermiteTrimmedCurvesLengths, getNearestLocationOnCurve, get_curve_from_points,
    interpolateCubicHermiteDerivative, sampleCubicHermiteCurves, sampleCubicHermiteCurvesSmooth,
    smoothCurveSideCrossDerivatives, track_curve_side_direction)
from scaffoldmaker.utils.read_vagus_data import load_vagus_data
from scaffoldmaker.utils.zinc_utils import (
    define_and_fit_field, find_or_create_field_zero_fibres, fit_hermite_curve, generate_curve_mesh,
    generate_datapoints, generate_mesh_marker_points)
import copy
import logging
import math


logger = logging.getLogger(__name__)


class MeshType_3d_nerve1(Scaffold_base):
    """
    Generates a hermite x bilinear 3-D box network mesh based on data supplied by an input file.
    """

    @classmethod
    def getName(cls):
        return "3D Nerve 1"

    @classmethod
    def getParameterSetNames(cls):
        return [
            'Default',
            'Human Left Vagus 1',
            'Human Right Vagus 1']

    @classmethod
    def getDefaultOptions(cls, parameterSetName="Default"):
        baseParameterSetName = 'Human Left Vagus 1' if (parameterSetName == 'Default') else parameterSetName
        options = {
            'Base parameter set': baseParameterSetName,
            'Number of elements along the trunk pre-fit': 30,
            'Number of elements along the trunk': 50,
            'Trunk proportion': 1.0,
            'Trunk fit number of iterations': 5,
            'Default anterior direction': [0.0, 1.0, 0.0],
            'Default trunk diameter': 3000.0,
            'Branch diameter trunk proportion': 0.5
        }
        return options

    @classmethod
    def getOrderedOptionNames(cls):
        return [
            'Number of elements along the trunk pre-fit',
            'Number of elements along the trunk',
            'Trunk proportion',
            'Trunk fit number of iterations',
            'Default anterior direction',
            'Default trunk diameter',
            'Branch diameter trunk proportion'
        ]

    @classmethod
    def checkOptions(cls, options):
        dependent_changes = False
        for key in [
            'Number of elements along the trunk',
            'Number of elements along the trunk pre-fit'
        ]:
            if options[key] < 1:
                options[key] = 1
        for key in [
            'Trunk proportion',
            'Branch diameter trunk proportion'
        ]:
            if options[key] < 0.1:
                options[key] = 0.1
            elif options[key] > 1.0:
                options[key] = 1.0
        for key in [
            'Default trunk diameter'
        ]:
            if options[key] <= 0.0:
                options[key] = 1.0
        if options['Trunk fit number of iterations'] < 0:
            options['Trunk fit number of iterations'] = 0
        # enforce direction is 3-component non-zero vector
        default_anterior_direction = options['Default anterior direction']
        if len(default_anterior_direction) == 0:
            default_anterior_direction = [0.0, 1.0, 0.0]
        else:
            count = len(default_anterior_direction)
            if count < 3:
                default_anterior_direction += [0.0] * (3 - count)
            elif count > 3:
                default_anterior_direction = default_anterior_direction[:3]
            if magnitude(default_anterior_direction) == 0.0:
                default_anterior_direction = [0.0, 1.0, 0.0]
        options['Default anterior direction'] = default_anterior_direction
        return dependent_changes

    @classmethod
    def generateBaseMesh(cls, region, options):
        """
        Generate the 3d mesh for a vagus with branches, incorporating 1d central line,
        2d epineurium and 3d box based on 1D hermite curve parameters with side axes.
        :param region: Zinc region to define model in. Must be empty.
        :param options: Dict containing options. See getDefaultOptions().
        return: list of AnnotationGroup, Nerve construction/metadata object
        """
        trunk_elements_count_prefit = options['Number of elements along the trunk pre-fit']
        trunk_elements_count = options['Number of elements along the trunk']
        trunk_proportion = options['Trunk proportion']
        trunk_fit_iterations = options['Trunk fit number of iterations']
        default_anterior_direction = options['Default anterior direction']
        default_trunk_diameter = options['Default trunk diameter']
        branch_diameter_trunk_proportion = options['Branch diameter trunk proportion']

        nerve_metadata = NerveMetadata("vagus nerve")

        # Zinc setup for vagus scaffold
        fieldmodule = region.getFieldmodule()
        fieldcache = fieldmodule.createFieldcache()

        # node - geometric coordinates
        value_labels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                        Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
                        Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3]

        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        coordinates = find_or_create_field_coordinates(fieldmodule)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        for value_label in value_labels[1:]:
            nodetemplate.setValueNumberOfVersions(coordinates, -1, value_label, 1)

        # 3D box
        mesh3d = fieldmodule.findMeshByDimension(3)
        hermite_bilinear_basis = fieldmodule.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        hermite_bilinear_basis.setFunctionType(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        eft3d = mesh3d.createElementfieldtemplate(hermite_bilinear_basis)
        setEftScaleFactorIds(eft3d, [1], [])
        ln = 1
        for n3 in range(2):
            s3 = [1] if (n3 == 0) else []
            for n2 in range(2):
                s2 = [1] if (n2 == 0) else []
                for n1 in range(2):
                    remapEftNodeValueLabel(eft3d, [ln], Node.VALUE_LABEL_VALUE,
                                           [(Node.VALUE_LABEL_VALUE, []),
                                            (Node.VALUE_LABEL_D_DS2, s2),
                                            (Node.VALUE_LABEL_D_DS3, s3)])
                    remapEftNodeValueLabel(eft3d, [ln], Node.VALUE_LABEL_D_DS1,
                                           [(Node.VALUE_LABEL_D_DS1, []),
                                            (Node.VALUE_LABEL_D2_DS1DS2, s2),
                                            (Node.VALUE_LABEL_D2_DS1DS3, s3)])
                    ln += 1
        remapEftLocalNodes(eft3d, 2, [1, 2, 1, 2, 1, 2, 1, 2])
        elementtemplate = mesh3d.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        elementtemplate.defineField(coordinates, -1, eft3d)

        # 3D box branch root
        # scale factors:
        # + 1 global scale factor (-1.0)
        # + 2 offsets of centroid in parent xi2, xi3
        # + 4 element scale factors for parent Hermite basis functions
        # + 4 element scale factors for parent Hermite basis function derivatives
        # + 9 coefficients of dB/dT, derivatives of branch derivatives w.r.t. trunk derivatives at root point
        eft3dBR = mesh3d.createElementfieldtemplate(hermite_bilinear_basis)
        setEftScaleFactorIds(eft3dBR, [1], [], 19)
        fns = [2, 3, 4, 5]
        dfns = [6, 7, 8, 9]
        # these are values in [-1.0, 1.0] which scale d2 and d3 to add to centroid x for non-centroid branch start
        cxd2 = 10
        cxd3 = 11
        # si_db_dt = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
        c11 = 12
        c12 = 13
        c13 = 14
        c21 = 15
        c22 = 16
        c23 = 17
        c31 = 18
        c32 = 19
        c33 = 20
        pd2 = [[None, None, fns[0], fns[1], None, None],
               [None, None, fns[2], fns[3], None, None]]
        pd3 = [[None, None, None, None, fns[0], fns[1]],
               [None, None, None, None, fns[2], fns[3]]]
        bx = [[[fns[0]], [fns[1]], None, None, None, None],
              [[fns[2]], [fns[3]], None, None, None, None]]
        bd1 = [[[c11, dfns[0]], [c11, dfns[1]], [c12, fns[0]], [c12, fns[1]], [c13, fns[0]], [c13, fns[1]]],
               [[c11, dfns[2]], [c11, dfns[3]], [c12, fns[2]], [c12, fns[3]], [c13, fns[2]], [c13, fns[3]]]]
        bd2 = [[[c21, dfns[0]], [c21, dfns[1]], [c22, fns[0]], [c22, fns[1]], [c23, fns[0]], [c23, fns[1]]],
               [[c21, dfns[2]], [c21, dfns[3]], [c22, fns[2]], [c22, fns[3]], [c23, fns[2]], [c23, fns[3]]]]
        bd3 = [[[c31, dfns[0]], [c31, dfns[1]], [c32, fns[0]], [c32, fns[1]], [c33, fns[0]], [c33, fns[1]]],
               [[c31, dfns[2]], [c31, dfns[3]], [c32, fns[2]], [c32, fns[3]], [c33, fns[2]], [c33, fns[3]]]]

        local_nodes = [1, 3]
        value_labels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1,
                       Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
                       Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3]
        ln = 1
        for n3 in range(2):
            s3 = [1] if (n3 == 0) else []
            for n2 in range(2):
                s2 = [1] if (n2 == 0) else []
                for fv in range(2):
                    old_value_label = value_labels[fv]
                    expression_terms = []
                    ubx, ubd2, ubd3 = (bx, bd2, bd3) if (fv == 0) else (bd1, None, None)
                    for tn in range(2):
                        tln = local_nodes[tn]
                        for pv in range(6):
                            value_label = value_labels[pv]
                            if ubx[tn][pv] is not None:
                                expression_terms.append((tln, value_label, ubx[tn][pv]))
                            if ubx is bx:
                                # offset from parent centroid
                                if pd2[tn][pv]:
                                    expression_terms.append((tln, value_label, [cxd2, pd2[tn][pv]]))
                                if pd3[tn][pv]:
                                    expression_terms.append((tln, value_label, [cxd3, pd3[tn][pv]]))
                            if ubd2:
                                expression_terms.append((tln, value_label, s2 + ubd2[tn][pv]))
                            if ubd3:
                                expression_terms.append((tln, value_label, s3 + ubd3[tn][pv]))
                    remapEftNodeValueLabelWithNodes(eft3dBR, ln, old_value_label, expression_terms)
                ln += 2
        ln = 2
        for n3 in range(2):
            s3 = [1] if (n3 == 0) else []
            for n2 in range(2):
                s2 = [1] if (n2 == 0) else []
                remapEftNodeValueLabel(eft3dBR, [ln], Node.VALUE_LABEL_VALUE,
                                       [(Node.VALUE_LABEL_VALUE, []),
                                        (Node.VALUE_LABEL_D_DS2, s2),
                                        (Node.VALUE_LABEL_D_DS3, s3)])
                remapEftNodeValueLabel(eft3dBR, [ln], Node.VALUE_LABEL_D_DS1,
                                       [(Node.VALUE_LABEL_D_DS1, []),
                                        (Node.VALUE_LABEL_D2_DS1DS2, s2),
                                        (Node.VALUE_LABEL_D2_DS1DS3, s3)])
                ln += 2
        # remap nodes so that parent nodes are local nodes 1 and 2, last regular node is local node 3
        remapEftLocalNodes(eft3dBR, 3, [1, 3, 2, 3, 3, 3, 3, 3])
        elementtemplate_branch_root = mesh3d.createElementtemplate()
        elementtemplate_branch_root.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        elementtemplate_branch_root.defineField(coordinates, -1, eft3dBR)

        # 1D centroid
        mesh1d = fieldmodule.findMeshByDimension(1)
        hermite_basis = fieldmodule.createElementbasis(1, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        eft1d = mesh1d.createElementfieldtemplate(hermite_basis)
        linetemplate = mesh1d.createElementtemplate()
        linetemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
        linetemplate.defineField(coordinates, -1, eft1d)

        # 1D centroid branch root
        # scale factors:
        # + 1 global scale factor (-1.0)
        # + 2 offsets of centroid in parent xi2, xi3
        # + 4 element scale factors for parent Hermite basis functions
        # + 4 element scale factors for parent Hermite basis function derivatives
        # + 3 coefficients of dB/dT for derivatives of branch derivatives w.r.t. trunk derivatives at root, d1 only
        eft1dBR = mesh1d.createElementfieldtemplate(hermite_basis)
        eft1dBR.setNumberOfLocalNodes(3)
        setEftScaleFactorIds(eft1dBR, [], [], 13)
        ln = 1
        for fv in range(2):
            old_value_label = value_labels[fv]
            expression_terms = []
            ubx = bx if (fv == 0) else bd1
            for tn in range(2):
                tln = local_nodes[tn]
                for pv in range(6):
                    value_label = value_labels[pv]
                    if ubx[tn][pv] is not None:
                        expression_terms.append((tln, value_label, [c - 1 for c in ubx[tn][pv]]))
                    if ubx is bx:
                        # offset from parent centroid
                        if pd2[tn][pv]:
                            expression_terms.append((tln, value_label, [cxd2 - 1, pd2[tn][pv] - 1]))
                        if pd3[tn][pv]:
                            expression_terms.append((tln, value_label, [cxd3 - 1, pd3[tn][pv] - 1]))
            remapEftNodeValueLabelWithNodes(eft1dBR, ln, old_value_label, expression_terms)
        ln += 2
        # remap nodes so that parent nodes are local nodes 1 and 2, last regular node is local node 3
        remapEftLocalNodes(eft1dBR, 3, [1, 3, 2])
        linetemplate_branch_root = mesh1d.createElementtemplate()
        linetemplate_branch_root.setElementShapeType(Element.SHAPE_TYPE_LINE)
        linetemplate_branch_root.defineField(coordinates, -1, eft1dBR)

        # 2D epineurium - 4 elements around circle
        # scale factors:
        # + 3 global scale factor (-1.0, 0.5, pi/4)
        scalefactors2d = [-1.0, 0.5, 0.25 * math.pi]
        c_minus1 = 1
        c_half = 2
        c_pi__4 = 3
        mesh2d = fieldmodule.findMeshByDimension(2)
        bicubic_hermite_serendipity_basis = (
            fieldmodule.createElementbasis(2, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE_SERENDIPITY))
        facetemplate_and_eft_list = []
        for f in range(4):
            eft2d = mesh2d.createElementfieldtemplate(bicubic_hermite_serendipity_basis)
            setEftScaleFactorIds(eft2d, [1, 2, 3], [])
            ln = 1
            for n2 in range(2):
                for n1 in range(2):
                    value_expression_terms = [(Node.VALUE_LABEL_VALUE, [])]
                    d_ds1_expression_terms = [(Node.VALUE_LABEL_D_DS1, [])]
                    d_ds2_expression_terms = []
                    pole = (f + n2) % 4
                    if pole == 0:
                        value_expression_terms.append((Node.VALUE_LABEL_D_DS2, [c_half]))
                        d_ds1_expression_terms.append((Node.VALUE_LABEL_D2_DS1DS2, [c_half]))
                        d_ds2_expression_terms.append((Node.VALUE_LABEL_D_DS3, [c_pi__4]))
                    elif pole == 1:
                        value_expression_terms.append((Node.VALUE_LABEL_D_DS3, [c_half]))
                        d_ds1_expression_terms.append((Node.VALUE_LABEL_D2_DS1DS3, [c_half]))
                        d_ds2_expression_terms.append((Node.VALUE_LABEL_D_DS2, [c_minus1, c_pi__4]))
                    elif pole == 2:
                        value_expression_terms.append((Node.VALUE_LABEL_D_DS2, [c_minus1, c_half]))
                        d_ds1_expression_terms.append((Node.VALUE_LABEL_D2_DS1DS2, [c_minus1, c_half]))
                        d_ds2_expression_terms.append((Node.VALUE_LABEL_D_DS3, [c_minus1, c_pi__4]))
                    else:  # pole == 3:
                        value_expression_terms.append((Node.VALUE_LABEL_D_DS3, [c_minus1, c_half]))
                        d_ds1_expression_terms.append((Node.VALUE_LABEL_D2_DS1DS3, [c_minus1, c_half]))
                        d_ds2_expression_terms.append((Node.VALUE_LABEL_D_DS2, [c_pi__4]))
                    remapEftNodeValueLabel(eft2d, [ln], Node.VALUE_LABEL_VALUE, value_expression_terms)
                    remapEftNodeValueLabel(eft2d, [ln], Node.VALUE_LABEL_D_DS1, d_ds1_expression_terms)
                    remapEftNodeValueLabel(eft2d, [ln], Node.VALUE_LABEL_D_DS2, d_ds2_expression_terms)
                    ln += 1
            remapEftLocalNodes(eft2d, 2, [1, 2, 1, 2])
            facetemplate = mesh2d.createElementtemplate()
            facetemplate.setElementShapeType(Element.SHAPE_TYPE_SQUARE)
            facetemplate.defineField(coordinates, -1, eft2d)
            facetemplate_and_eft_list.append((facetemplate, eft2d))

        # 2D epineurium branch root - 4 elements around circle
        # scale factors:
        # + 3 global scale factor (-1.0, 0.5, pi/4)
        # + 2 offsets of centroid in parent xi2, xi3
        # + 4 element scale factors for parent Hermite basis functions
        # + 4 element scale factors for parent Hermite basis function derivatives
        # + 9 coefficients of dB/dT, derivatives of branch derivatives w.r.t. trunk derivatives at root point
        # need to offset the following by 2 to fit 2 more global scale factors
        fns = [4, 5, 6, 7]
        dfns = [8, 9, 10, 11]
        # these are values in [-1.0, 1.0] which scale d2 and d3 to add to centroid x for non-centroid branch start
        cxd2 = 12
        cxd3 = 13
        # si_db_dt = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
        c11 = 14
        c12 = 15
        c13 = 16
        c21 = 17
        c22 = 18
        c23 = 19
        c31 = 20
        c32 = 21
        c33 = 22
        pd2 = [[None, None, fns[0], fns[1], None, None],
               [None, None, fns[2], fns[3], None, None]]
        pd3 = [[None, None, None, None, fns[0], fns[1]],
               [None, None, None, None, fns[2], fns[3]]]
        bx = [[[fns[0]], [fns[1]], None, None, None, None],
              [[fns[2]], [fns[3]], None, None, None, None]]
        bd1 = [[[c11, dfns[0]], [c11, dfns[1]], [c12, fns[0]], [c12, fns[1]], [c13, fns[0]], [c13, fns[1]]],
               [[c11, dfns[2]], [c11, dfns[3]], [c12, fns[2]], [c12, fns[3]], [c13, fns[2]], [c13, fns[3]]]]
        bd2 = [[[c21, dfns[0]], [c21, dfns[1]], [c22, fns[0]], [c22, fns[1]], [c23, fns[0]], [c23, fns[1]]],
               [[c21, dfns[2]], [c21, dfns[3]], [c22, fns[2]], [c22, fns[3]], [c23, fns[2]], [c23, fns[3]]]]
        bd3 = [[[c31, dfns[0]], [c31, dfns[1]], [c32, fns[0]], [c32, fns[1]], [c33, fns[0]], [c33, fns[1]]],
               [[c31, dfns[2]], [c31, dfns[3]], [c32, fns[2]], [c32, fns[3]], [c33, fns[2]], [c33, fns[3]]]]
        bicubic_value_labels = [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2]
        facetemplate_and_eft_list_branch_root = []
        for f in range(4):
            eft2dBR = mesh2d.createElementfieldtemplate(bicubic_hermite_serendipity_basis)
            setEftScaleFactorIds(eft2dBR, [1, 2, 3], [], 19)
            ln = 1
            for n2 in range(2):
                ubx = bx  # common centroid for all branch root nodes
                ubd1 = bd1  # all d1 derivatives are parallel at branch root
                pole = (f + n2) % 4
                if pole == 0:
                    ubxo, ubxos = bd2, [c_half]
                    ubd2, ubd2s = bd3, [c_pi__4]
                elif pole == 1:
                    ubxo, ubxos = bd3, [c_half]
                    ubd2, ubd2s = bd2, [c_minus1, c_pi__4]
                elif pole == 2:
                    ubxo, ubxos = bd2, [c_minus1, c_half]
                    ubd2, ubd2s = bd3, [c_minus1, c_pi__4]
                else:  # pole == 3:
                    ubxo, ubxos = bd3, [c_minus1, c_half]
                    ubd2, ubd2s = bd2, [c_pi__4]
                for fv in range(3):
                    old_value_label = bicubic_value_labels[fv]
                    expression_terms = []
                    for tn in range(2):
                        tln = local_nodes[tn]
                        for pv in range(6):
                            value_label = value_labels[pv]
                            if old_value_label == Node.VALUE_LABEL_VALUE:
                                # start at parent centroid
                                if ubx[tn][pv] is not None:
                                    expression_terms.append((tln, value_label, ubx[tn][pv]))
                                # offset from parent centroid
                                if pd2[tn][pv]:
                                    expression_terms.append((tln, value_label, [cxd2, pd2[tn][pv]]))
                                if pd3[tn][pv]:
                                    expression_terms.append((tln, value_label, [cxd3, pd3[tn][pv]]))
                                # offset from branch centroid
                                expression_terms.append((tln, value_label, ubxos + ubxo[tn][pv]))
                            elif old_value_label == Node.VALUE_LABEL_D_DS1:
                                expression_terms.append((tln, value_label, ubd1[tn][pv]))
                            else:  # old_value_label == Node.VALUE_LABEL_D_DS2:
                                expression_terms.append((tln, value_label, ubd2s + ubd2[tn][pv]))
                    remapEftNodeValueLabelWithNodes(eft2dBR, ln, old_value_label, expression_terms)
                ln += 2
            ln = 2
            for n2 in range(2):
                value_expression_terms = [(Node.VALUE_LABEL_VALUE, [])]
                d_ds1_expression_terms = [(Node.VALUE_LABEL_D_DS1, [])]
                d_ds2_expression_terms = []
                pole = (f + n2) % 4
                if pole == 0:
                    value_expression_terms.append((Node.VALUE_LABEL_D_DS2, [2]))
                    d_ds1_expression_terms.append((Node.VALUE_LABEL_D2_DS1DS2, [2]))
                    d_ds2_expression_terms.append((Node.VALUE_LABEL_D_DS3, [3]))
                elif pole == 1:
                    value_expression_terms.append((Node.VALUE_LABEL_D_DS3, [2]))
                    d_ds1_expression_terms.append((Node.VALUE_LABEL_D2_DS1DS3, [2]))
                    d_ds2_expression_terms.append((Node.VALUE_LABEL_D_DS2, [1, 3]))
                elif pole == 2:
                    value_expression_terms.append((Node.VALUE_LABEL_D_DS2, [1, 2]))
                    d_ds1_expression_terms.append((Node.VALUE_LABEL_D2_DS1DS2, [1, 2]))
                    d_ds2_expression_terms.append((Node.VALUE_LABEL_D_DS3, [1, 3]))
                else:  # pole == 3:
                    value_expression_terms.append((Node.VALUE_LABEL_D_DS3, [1, 2]))
                    d_ds1_expression_terms.append((Node.VALUE_LABEL_D2_DS1DS3, [1, 2]))
                    d_ds2_expression_terms.append((Node.VALUE_LABEL_D_DS2, [3]))
                remapEftNodeValueLabel(eft2dBR, [ln], Node.VALUE_LABEL_VALUE, value_expression_terms)
                remapEftNodeValueLabel(eft2dBR, [ln], Node.VALUE_LABEL_D_DS1, d_ds1_expression_terms)
                remapEftNodeValueLabel(eft2dBR, [ln], Node.VALUE_LABEL_D_DS2, d_ds2_expression_terms)
                ln += 2

            # remap nodes so that parent nodes are local nodes 1 and 2, last regular node is local node 3
            remapEftLocalNodes(eft2dBR, 3, [1, 3, 2, 3])
            facetemplate_branch_root = mesh2d.createElementtemplate()
            facetemplate_branch_root.setElementShapeType(Element.SHAPE_TYPE_SQUARE)
            facetemplate_branch_root.defineField(coordinates, -1, eft2dBR)
            facetemplate_and_eft_list_branch_root.append((facetemplate_branch_root, eft2dBR))

        # =========
        # Load Data
        # =========

        vagus_data = load_vagus_data(region)
        invalid_data = not vagus_data
        if not invalid_data:
            marker_data = vagus_data.get_level_markers()
            trunk_data = vagus_data.get_trunk_coordinates()
            if len(marker_data) < 2:
                logger.error("Missing or incomplete data. At least two landmarks are expected in the data.")
                invalid_data = True
            if len(trunk_data) < 2:
                logger.error("Missing or incomplete data. At least two trunk points are required.")
                invalid_data = True
            if vagus_data.get_side_label() not in ['left', 'right']:
                logger.error("Missing left or right marker side indication.")
                invalid_data = True
        if invalid_data:
            return [], None

        # ===========
        # Build Trunk
        # ===========

        # fit trunk with radius and orientation
        only_1d_trunk = False
        region1d = region if only_1d_trunk else region.createRegion()
        tx, td1, td2, td12, td3, td13 = generate_trunk_1d(
            vagus_data, trunk_proportion, trunk_elements_count_prefit, trunk_elements_count,
            trunk_fit_iterations, default_anterior_direction, default_trunk_diameter, region1d, nerve_metadata)
        trunk_length = getCubicHermiteCurvesLength(tx, td1)
        trunk_mean_element_length = trunk_length / trunk_elements_count

        default_branch_diameter = branch_diameter_trunk_proportion * default_trunk_diameter

        annotation_groups = []
        annotation_term_map = vagus_data.get_annotation_term_map()
        trunk_group_name = vagus_data.get_trunk_group_name()
        trunk_group = AnnotationGroup(region, (trunk_group_name, annotation_term_map[trunk_group_name]))
        annotation_groups.append(trunk_group)

        if only_1d_trunk:
            return annotation_groups, None

        # vagus annotation groups
        centroid_annotation_group = AnnotationGroup(region, get_vagus_term("vagus centroid"))
        annotation_groups.append(centroid_annotation_group)
        centroid_mesh_group = centroid_annotation_group.getMeshGroup(mesh1d)

        epineurium_annotation_group = AnnotationGroup(region, get_vagus_term("epineurium"))
        annotation_groups.append(epineurium_annotation_group)
        epineurium_mesh_group = epineurium_annotation_group.getMeshGroup(mesh2d)

        # trunk annotation groups
        trunk_mesh_group = trunk_group.getMeshGroup(mesh3d)
        trunk_face_mesh_group = trunk_group.getMeshGroup(mesh2d)
        trunk_line_mesh_group = trunk_group.getMeshGroup(mesh1d)

        node_identifier = 1
        element_identifier = 1
        face_identifier = 1
        line_identifier = 1

        # create trunk nodes
        tnid = []
        for n in range(trunk_elements_count + 1):
            node = nodes.createNode(node_identifier, nodetemplate)
            fieldcache.setNode(node)
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, tx[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, td1[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, td2[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, td12[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, td3[n])
            coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, td13[n])
            tnid.append(node_identifier)
            node_identifier += 1

        # create trunk elements
        for e in range(trunk_elements_count):
            nids = [e + 1, e + 2]

            element = mesh3d.createElement(element_identifier, elementtemplate)
            element.setNodesByIdentifier(eft3d, nids)
            element.setScaleFactors(eft3d, [-1.0])
            trunk_mesh_group.addElement(element)
            element_identifier += 1

            line = mesh1d.createElement(line_identifier, linetemplate)
            line.setNodesByIdentifier(eft1d, nids)
            centroid_mesh_group.addElement(line)
            trunk_line_mesh_group.addElement(line)
            line_identifier += 1

            for f in range(4):
                facetemplate, eft2d = facetemplate_and_eft_list[f]
                face = mesh2d.createElement(face_identifier, facetemplate)
                face.setNodesByIdentifier(eft2d, nids)
                face.setScaleFactors(eft2d, scalefactors2d)
                epineurium_mesh_group.addElement(face)
                trunk_face_mesh_group.addElement(face)
                face_identifier += 1

        # trunk material coordinates span from 0.0 to 1.0 at trunk proportion 1.0
        # aspect ratio 3mm diameter over 500mm length
        trunk_material_diameter = 3.0 / 500.0
        branch_material_diameter = branch_diameter_trunk_proportion * trunk_material_diameter

        # ==============
        # Build Branches
        # ==============

        parent_parameters = {}
        parent_parameters[trunk_group_name] = (tx, td1, td2, td12, td3, td13, tnid)
        parent_elements_counts = {}  # map of parent branch name to list of numbers of elements in branches
        parent_elements_counts[trunk_group_name] = [trunk_elements_count]
        # the first branch node mainly marks connection on the parent/trunk centroid,
        # hence the following code starts branches from a proportion of parent radius
        # out, up to an upper limit on the proportion of the first segment:
        PARENT_RADIUS_PROPORTION = 0.5
        FIRST_SEGMENT_MAX_CUT_XI = 0.5
        branch_max_element_length = trunk_mean_element_length

        derivative_xi1 = mesh3d.getChartDifferentialoperator(1, 1)
        derivative_xi2 = mesh3d.getChartDifferentialoperator(1, 2)
        derivative_xi3 = mesh3d.getChartDifferentialoperator(1, 3)

        # following is assigned to branch coordinates before finding trunk location
        branch_start_coordinates = fieldmodule.createFieldConstant([0.0, 0.0, 0.0])
        # note zinc will not use the fast find cache due to field modifications with change caching on
        find_trunk_location = fieldmodule.createFieldFindMeshLocation(
            branch_start_coordinates, coordinates, trunk_mesh_group)
        find_trunk_location.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)

        # iterate over branches off trunk, and branches of branches
        visited_branches_order = []
        branch_root_parameters = {}
        branch_coordinates_data = vagus_data.get_branch_coordinates_data()
        branch_sequences_data = vagus_data.get_branch_sequences_data()
        branch_parent_map = vagus_data.get_branch_parent_map()
        queue = [branch for branch in branch_parent_map.keys() if branch_parent_map[branch] == trunk_group_name]
        branch_name = None
        branch_parent_name = None
        trunk_is_parent = False
        branch_coordinates = []
        branch_data_nodes_counts = []
        branch_box_group = None
        branch_box_mesh_group = None
        branch_box_face_mesh_group = None
        branch_box_line_mesh_group = None

        # iterate over branch names and distinct branches from the branch_sequences_data
        while True:
            if not branch_coordinates:
                # get the next branch in queue
                if not queue:
                    break
                branch_name = queue.pop(0)
                if branch_name in visited_branches_order:
                    logger.warning("already processed branch " + branch_name)
                    continue
                visited_branches_order.append(branch_name)
                branch_parent_name = branch_parent_map[branch_name]
                trunk_is_parent = branch_parent_name == trunk_group_name
                # print(branch_name, '<--', branch_parent_name)
                branch_coordinates = copy.copy(branch_coordinates_data[branch_name])
                branch_data_nodes_counts = copy.copy(branch_sequences_data[branch_name])

                # branch annotation groups
                branch_box_group = AnnotationGroup(region, (branch_name, annotation_term_map[branch_name]))
                annotation_groups.append(branch_box_group)
                branch_box_mesh_group = branch_box_group.getMeshGroup(mesh3d)
                branch_box_face_mesh_group = branch_box_group.getMeshGroup(mesh2d)
                branch_box_line_mesh_group = branch_box_group.getMeshGroup(mesh1d)

            tx, td1, td2, td12, td3, td13, tnid = parent_parameters[branch_parent_name]
            branch_nodes_count = branch_data_nodes_counts[0]
            branch_px = branch_coordinates[:branch_nodes_count]
            branch_coordinates = branch_coordinates[branch_nodes_count:]
            branch_data_nodes_counts.pop(0)

            # get point in trunk volume closest to first point in branch data
            # parent_group = trunk_group
            parent_mesh_group = trunk_mesh_group
            find_parent_location = find_trunk_location
            if not trunk_is_parent:
                parent_group = fieldmodule.findFieldByName(branch_parent_name).castGroup()
                parent_mesh_group = parent_group.getMeshGroup(mesh3d)
                find_parent_location = fieldmodule.createFieldFindMeshLocation(
                    branch_start_coordinates, coordinates, parent_mesh_group)
                find_parent_location.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
                del parent_group
            fieldcache.clearLocation()
            branch_start_coordinates.assignReal(fieldcache, branch_px[0])
            parent_element, parent_xi = find_parent_location.evaluateMeshLocation(fieldcache, 3)
            if not parent_element.isValid():
                logger.error("Nerve: branch " + branch_name + " start point could not be found in parent nerve")
                continue
            # get radius at parent_location
            fieldcache.setMeshLocation(parent_element, parent_xi)
            _, d2 = coordinates.evaluateDerivative(derivative_xi2, fieldcache, 3)
            _, d3 = coordinates.evaluateDerivative(derivative_xi3, fieldcache, 3)
            parent_radius = 0.25 * (magnitude(d2) + magnitude(d3))
            first_distance = distance(branch_px[1], branch_px[0])
            xi = min(FIRST_SEGMENT_MAX_CUT_XI, PARENT_RADIUS_PROPORTION * parent_radius / first_distance)
            new_start_x = add(mult(branch_px[0], 1.0 - xi), mult(branch_px[1], xi))

            # cut the first part of the branch:
            px = [new_start_x] + branch_px[1:]
            ax, ad1 = get_curve_from_points(px, maximum_element_length=branch_max_element_length)
            bx, bd1 = fit_hermite_curve(ax, ad1, px)
            branch_length = getCubicHermiteCurvesLength(bx, bd1)
            branch_elements_count = math.ceil(branch_length / branch_max_element_length)
            # previously had minimum of 2 elements along branch as can't attach sub-branches from first element
            # branch_elements_count = max(2, branch_elements_count)
            cx, cd1 = sampleCubicHermiteCurves(bx, bd1, branch_elements_count)[0:2]

            # find the parent location at the fitted branch start location
            fieldcache.clearLocation()
            branch_start_coordinates.assignReal(fieldcache, cx[0])
            parent_element, parent_xi = find_parent_location.evaluateMeshLocation(fieldcache, 3)
            if not parent_element.isValid():
                logger.error("Nerve: branch " + branch_name + " fitted start point could not be found in parent nerve")
                continue
            parent_first_element = parent_mesh_group.createElementiterator().next()
            parent_index = parent_element.getIdentifier() - parent_first_element.getIdentifier()
            branch_in_first_parent_element = False
            # handle element indexes when there are several branches of the same name
            # because the parent parameters are just appended after each branch = there isn't an element in between.
            parent_start_index = 0
            for elements_count in parent_elements_counts[branch_parent_name]:
                if parent_index < (parent_start_index + elements_count):
                    break
                parent_index += 1
                parent_start_index += elements_count + 1

            parent_location = (parent_index, parent_xi[0])
            if (not trunk_is_parent) and (parent_index - parent_start_index == 0):
                if parent_location[0] < 0.99:
                    logger.warning("Nerve: attaching branch " + branch_name +
                                   " at end of first element of parent branch " + branch_parent_name +
                                   " instead of calculated proportion " + str(parent_location[0]))
                parent_location = (0, 1.0)
            cxd2 = 2.0 * (parent_xi[1] - 0.5)
            cxd3 = 2.0 * (parent_xi[2] - 0.5)

            # parent interpolation
            pn1 = parent_location[0]
            pn2 = pn1 + 1
            pxi = parent_location[1]
            fns = list(getCubicHermiteBasis(pxi))  # for x, d2, d3
            dfns = list(getCubicHermiteBasisDerivatives(pxi))  # for d1
            # first derivatives interpolated on parent:
            pd1 = [dot(dfns, [tx[pn1][c], td1[pn1][c], tx[pn2][c], td1[pn2][c]]) for c in range(3)]
            pd2 = [dot(fns, [td2[pn1][c], td12[pn1][c], td2[pn2][c], td12[pn2][c]]) for c in range(3)]
            pd3 = [dot(fns, [td3[pn1][c], td13[pn1][c], td3[pn2][c], td13[pn2][c]]) for c in range(3)]
            # first derivatives required on branch:
            bd1 = cd1[0]
            branch_root_diameter = \
                branch_diameter_trunk_proportion * magnitude(pd2) if trunk_is_parent else default_branch_diameter
            bd3 = set_magnitude(cross(pd1, bd1), branch_root_diameter)
            # scale bd2 to fit expected aspect ratio for material coordinates, depending on angle
            cos_angle = dot(normalize(pd1), normalize(bd1))
            # cos2_angle = cos_angle * cos_angle
            # sin2_angle = 1.0 - cos2_angle
            angle = math.acos(cos_angle)
            sin_angle = math.sin(angle)
            m1 = cos_angle * branch_root_diameter  # at 0 radians
            branch_material_diameter_proportion = branch_material_diameter * trunk_elements_count / trunk_proportion
            m2 = sin_angle * magnitude(pd1) * branch_material_diameter_proportion  # at PI/2 radians
            mag_bd2 = magnitude([m1, m2])
            bd2 = set_magnitude(cross(bd3, bd1), mag_bd2)

            basis_from = [pd1, pd2, pd3]
            basis_to = [bd1, bd2, bd3]
            coefs = matrix_mult(basis_to, matrix_inv(basis_from))

            # get side derivatives, minimising rotation from trunk
            # dir2 = normalize(bd2)
            dir3 = normalize(bd3)
            cd2 = [bd2]
            cd3 = [bd3]
            for e in range(len(cx) - 1):
                dir1, dir2, dir3 = track_curve_side_direction(cx, cd1, dir3, (e, 0.0), (e, 1.0))
                cd2.append(set_magnitude(dir2, default_branch_diameter))
                cd3.append(set_magnitude(dir3, default_branch_diameter))
            cd12, cd13 = smoothCurveSideCrossDerivatives(cx, cd1, [cd2, cd3])

            # create branch elements and nodes past root
            cnid = [None]  # no first node on branch
            for e in range(len(cx) - 1):
                n = e + 1
                node = nodes.createNode(node_identifier, nodetemplate)
                fieldcache.setNode(node)
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, cx[n])
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, cd1[n])
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, cd2[n])
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, cd12[n])
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, cd3[n])
                coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, cd13[n])

                if e == 0:
                    # branch root 3D element
                    if parent_location[0] == 0:
                        # special case of branch off first parent element
                        # doesn't use the first local node as parent_location[1] == 1.0
                        nids = [tnid[pn2], tnid[pn2], node_identifier]
                    else:
                        nids = [tnid[pn1], tnid[pn2], node_identifier]
                    scalefactors = [-1] + fns + dfns + [cxd2, cxd3] + coefs[0] + coefs[1] + coefs[2]
                    element = mesh3d.createElement(element_identifier, elementtemplate_branch_root)
                    element.setNodesByIdentifier(eft3dBR, nids)
                    element.setScaleFactors(eft3dBR, scalefactors)
                    # branch root 1D line
                    scalefactors = fns + dfns + [cxd2, cxd3] + coefs[0]
                    line = mesh1d.createElement(line_identifier, linetemplate_branch_root)
                    line.setNodesByIdentifier(eft1dBR, nids)
                    line.setScaleFactors(eft1dBR, scalefactors)
                else:
                    # branch regular 3D element
                    nids = [node_identifier - 1, node_identifier]
                    element = mesh3d.createElement(element_identifier, elementtemplate)
                    element.setNodesByIdentifier(eft3d, nids)
                    element.setScaleFactors(eft3d, [-1.0])
                    # branch regular 1D line
                    line = mesh1d.createElement(line_identifier, linetemplate)
                    line.setNodesByIdentifier(eft1d, nids)
                    line.setScaleFactors(eft1d, [-1.0])
                branch_box_mesh_group.addElement(element)
                centroid_mesh_group.addElement(line)
                branch_box_line_mesh_group.addElement(line)
                cnid.append(node_identifier)
                element_identifier += 1
                line_identifier += 1

                # 2D epineurium
                for f in range(4):
                    if e == 0:
                        # branch root 2D face
                        facetemplate_branch_root, eft2dBR = facetemplate_and_eft_list_branch_root[f]
                        if parent_location[0] == 0:
                            # special case of branch off first parent element
                            # doesn't use the first local node as parent_location[1] == 1.0
                            nids = [tnid[pn2], tnid[pn2], node_identifier]
                        else:
                            nids = [tnid[pn1], tnid[pn2], node_identifier]
                        scalefactors = scalefactors2d + fns + dfns + [cxd2, cxd3] + coefs[0] + coefs[1] + coefs[2]
                        face = mesh3d.createElement(face_identifier, facetemplate_branch_root)
                        face.setNodesByIdentifier(eft2dBR, nids)
                        face.setScaleFactors(eft2dBR, scalefactors)
                    else:
                        # branch regular 2D face
                        facetemplate, eft2d = facetemplate_and_eft_list[f]
                        nids = [node_identifier - 1, node_identifier]
                        face = mesh2d.createElement(face_identifier, facetemplate)
                        face.setNodesByIdentifier(eft2d, nids)
                        face.setScaleFactors(eft2d, scalefactors2d)
                    epineurium_mesh_group.addElement(face)
                    branch_box_face_mesh_group.addElement(face)
                    face_identifier += 1

                node_identifier += 1

            # add branches of branches, storing parameters for embedding sub-branch root
            child_branches = [branch for branch in branch_parent_map.keys() if branch_parent_map[branch] == branch_name]
            if child_branches:
                queue = child_branches + queue
                existing_parent_parameters = parent_parameters.get(branch_name)
                if existing_parent_parameters:
                    for dst, src in zip(existing_parent_parameters, (cx, cd1, cd2, cd12, cd3, cd13, cnid)):
                        dst += src
                    parent_elements_counts[branch_name].append(len(cx) - 1)
                else:
                    parent_parameters[branch_name] = existing_parent_parameters = (cx, cd1, cd2, cd12, cd3, cd13, cnid)
                    parent_elements_counts[branch_name] = [len(cx) - 1]

        # =================================================
        # Add material coordinates and straight coordinates
        # =================================================

        coordinates.setName("other coordinates")  # temporarily rename
        sir = region.createStreaminformationRegion()
        srm = sir.createStreamresourceMemory()
        region.write(sir)
        result, buffer = srm.getBuffer()
        coordinates.setName("coordinates")  # restore name before reading other coordinates back in
        sir = region.createStreaminformationRegion()
        sir.createStreamresourceMemoryBuffer(buffer)
        # read the other coordinates in twice and rename to define vagus coordinates and straight coordinates
        # vagus coordinates are unit length along trunk (1st component) with corresponding aspect ratio
        # trunk goes along z-axis with left d2 in x-direction, anterior d3 in y-direction, top centre at origin
        region.read(sir)
        vagus_coordinates = fieldmodule.findFieldByName("other coordinates").castFiniteElement()
        vagus_coordinates.setName("vagus coordinates")
        for c in range(1, 4):
            vagus_coordinates.setComponentName(c, str(c))
        # straight coordinates have same lengths, radii and branch angles as geometric coordinates, but are straight
        # trunk goes along z-axis with left d2 in x-direction, anterior d3 in y-direction, top centre at origin
        region.read(sir)
        straight_coordinates = fieldmodule.findFieldByName("other coordinates").castFiniteElement()
        straight_coordinates.setName("straight coordinates")

        zero = [0.0, 0.0, 0.0]

        # assign vagus coordinates
        x = [0.0, 0.0, 0.0]
        d1 = [0.0, 0.0, trunk_proportion / trunk_elements_count]
        d2 = [trunk_material_diameter, 0.0, 0.0]
        d3 = [0.0, trunk_material_diameter, 0.0]
        elem_iter = mesh3d.createElementiterator()
        element = elem_iter.next()
        while element.isValid():
            # trunk elements first, followed by branch elements (with first element having more than 2 local nodes)
            element_id = element.getIdentifier()
            eft = element.getElementfieldtemplate(vagus_coordinates, -1)
            local_nodes_count = eft.getNumberOfLocalNodes()
            ln = 2
            if element_id == 1:
                # first trunk element
                node = element.getNode(eft, 1)
                fieldcache.setNode(node)
                vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, d1)
                vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, d2)
                vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)
                vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, zero)
                vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, zero)
            elif local_nodes_count > 2:
                # first branch element
                fieldcache.setMeshLocation(element, [0.0, 0.5, 0.5])  # as xi1 is along the branch
                _, x = vagus_coordinates.evaluateReal(fieldcache, 3)
                _, d1 = vagus_coordinates.evaluateDerivative(derivative_xi1, fieldcache, 3)
                _, d2 = vagus_coordinates.evaluateDerivative(derivative_xi2, fieldcache, 3)
                _, d3 = vagus_coordinates.evaluateDerivative(derivative_xi3, fieldcache, 3)
                # remove shear on remainder of branch:
                d2 = set_magnitude(cross(d3, d1), branch_material_diameter)
                d3 = set_magnitude(cross(d1, d2), branch_material_diameter)
                ln = 3

            x = add(x, d1)
            node = element.getNode(eft, ln)
            fieldcache.setNode(node)
            vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
            vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, d1)
            vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, d2)
            vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)
            vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, zero)
            vagus_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, zero)
            element = elem_iter.next()

        # assign straight coordinates
        x = [0.0, 0.0, 0.0]
        dir1 = [0.0, 0.0, 1.0]
        dir2 = [1.0, 0.0, 0.0]
        dir3 = [0.0, 1.0, 0.0]
        bx, bd1 = zero, zero
        elem_iter = mesh3d.createElementiterator()
        element = elem_iter.next()
        while element.isValid():
            # trunk elements first, followed by branch elements (with first element having more than 2 local nodes)
            element_id = element.getIdentifier()
            eft = element.getElementfieldtemplate(straight_coordinates, -1)
            local_nodes_count = eft.getNumberOfLocalNodes()
            ln = 2
            if element_id == 1:
                # first trunk element
                node = element.getNode(eft, 1)
                fieldcache.setNode(node)
                _, ax = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
                _, ad1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
                _, ad2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
                _, ad3 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)
                _, ad12 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, 3)
                _, ad13 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, 3)
                d1 = mult(dir1, magnitude(ad1))
                d2 = mult(dir2, magnitude(ad2))
                d3 = mult(dir3, magnitude(ad3))
                d12 = mult(dir2, dot(normalize(ad2), ad12))
                d13 = mult(dir3, dot(normalize(ad3), ad13))
                straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
                straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, d1)
                straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, d2)
                straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)
                straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, d12)
                straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, d13)
            elif local_nodes_count > 2:
                # first branch element
                fieldcache.setMeshLocation(element, [0.0, 0.5, 0.5])  # as xi1 is along the branch
                _, ax = coordinates.evaluateReal(fieldcache, 3)
                _, ad1 = coordinates.evaluateDerivative(derivative_xi1, fieldcache, 3)
                _, x = straight_coordinates.evaluateReal(fieldcache, 3)
                _, d1 = straight_coordinates.evaluateDerivative(derivative_xi1, fieldcache, 3)
                _, d3 = straight_coordinates.evaluateDerivative(derivative_xi3, fieldcache, 3)
                # remove shear on remainder of branch:
                dir1 = normalize(d1)
                dir2 = normalize(cross(d3, dir1))
                dir3 = cross(dir1, dir2)
                ln = 3
            else:
                ax, ad1 = bx, bd1

            node = element.getNode(eft, ln)
            fieldcache.setNode(node)
            _, bx = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
            _, bd1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
            _, bd2 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, 3)
            _, bd3 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, 3)
            _, bd12 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, 3)
            _, bd13 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, 3)

            element_length = getCubicHermiteArcLength(ax, ad1, bx, bd1)
            x = add(x, mult(dir1, element_length))
            d1 = mult(dir1, magnitude(bd1))
            d2 = mult(dir2, magnitude(bd2))
            d3 = mult(dir3, magnitude(bd3))
            d12 = mult(dir2, dot(normalize(bd2), bd12))
            d13 = mult(dir3, dot(normalize(bd3), bd13))

            node = element.getNode(eft, ln)
            fieldcache.setNode(node)
            straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, x)
            straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, d1)
            straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS2, 1, d2)
            straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS3, 1, d3)
            straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, d12)
            straight_coordinates.setNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, d13)
            element = elem_iter.next()

        # ====================
        # Create marker points
        # ====================

        side_label = vagus_data.get_side_label()
        is_left = side_label == 'left'
        vagus_level_terms = get_left_vagus_marker_locations_list() if is_left \
            else get_right_vagus_marker_locations_list()
        ordered_marker_data = []  # list from top to bottom of nerve of (name, material_coordinate)
        for marker_term_name, material_coordinate in vagus_level_terms.items():
            for idx, data in enumerate(ordered_marker_data):
                if material_coordinate < data[1]:
                    break
            else:
                idx = len(ordered_marker_data)
            ordered_marker_data.insert(idx, (marker_term_name, material_coordinate))
        # start at 10001 unless node already in use
        start_marker_node_identifier = 10001
        if node_identifier < start_marker_node_identifier:
            node_identifier = start_marker_node_identifier
        for marker_name, material_coordinate in ordered_marker_data:
            if material_coordinate > trunk_proportion:
                break
            annotation_group = findOrCreateAnnotationGroupForTerm(
                annotation_groups, region, get_vagus_marker_term(marker_name), isMarker=True)
            annotation_group.createMarkerNode(
                node_identifier, materialCoordinatesField=vagus_coordinates,
                materialCoordinates=[0.0, 0.0, material_coordinate])
            node_identifier += 1

        # ==========================
        # Add combined branch groups
        # ==========================

        branch_common_groups = vagus_data.get_branch_common_group_map()
        for branch_common_name, branch_names in branch_common_groups.items():
            term = get_vagus_term(branch_common_name)
            branch_common_group = findOrCreateAnnotationGroupForTerm(annotation_groups, region, term)

            for branch_name in branch_names:
                branch_group = findAnnotationGroupByName(annotation_groups, branch_name)
                if not branch_group:
                    logger.warning("Nerve: Could not find annotation for branch " + branch_name +
                                   ". Can't add to common branch group")
                    continue
                for meshnd in [mesh1d, mesh2d, mesh3d]:
                    branch_common_mesh_group = branch_common_group.getMeshGroup(meshnd)
                    branch_mesh_group = branch_group.getMeshGroup(meshnd)
                    el_iter = branch_mesh_group.createElementiterator()
                    element = el_iter.next()
                    while element.isValid():
                        branch_common_mesh_group.addElement(element)
                        element = el_iter.next()

        # ============================================
        # Add trunk section groups: cervical, thoracic
        # ============================================

        cervical_trunk_group = findOrCreateAnnotationGroupForTerm(
            annotation_groups, region, get_vagus_term(side_label + ' cervical vagus nerve'))
        thoracic_trunk_group = findOrCreateAnnotationGroupForTerm(
            annotation_groups, region, get_vagus_term(side_label + ' thoracic vagus nerve'))
        cervical_thoracic_boundary_marker_name = \
            side_label + ' level of superior border of the clavicle on the vagus nerve'
        cervical_thoracic_boundary_material_coordinate = vagus_level_terms[cervical_thoracic_boundary_marker_name]

        trunk_group = findAnnotationGroupByName(annotation_groups, trunk_group_name)
        for dimension in range(3, 0, -1):
            mesh = fieldmodule.findMeshByDimension(dimension)
            trunk_mesh_group = trunk_group.getMeshGroup(mesh)
            cervical_trunk_mesh_group = cervical_trunk_group.getMeshGroup(mesh)
            thoracic_trunk_mesh_group = thoracic_trunk_group.getMeshGroup(mesh)
            el_iter = trunk_mesh_group.createElementiterator()
            element = el_iter.next()
            while element.isValid():
                fieldcache.setMeshLocation(element, [0.5, 0.5, 0.5])
                _, material_coordinate = vagus_coordinates.evaluateReal(fieldcache, 3)
                if material_coordinate[2] < cervical_thoracic_boundary_material_coordinate:
                    cervical_trunk_mesh_group.addElement(element)
                else:
                    thoracic_trunk_mesh_group.addElement(element)
                element = el_iter.next()

        return annotation_groups, nerve_metadata

    @classmethod
    def defineFaceAnnotations(cls, region, options, annotationGroups):
        """
        Add orientation anterior 1-D annotation group.
        :param region: Zinc region containing model.
        :param options: Dict containing options. See getDefaultOptions().
        :param annotationGroups: List of annotation groups for elements created in generateBaseMesh().
        New face/line annotation groups are appended to this list.
        """
        fieldmodule = region.getFieldmodule()
        mesh2d = fieldmodule.findMeshByDimension(2)
        mesh1d = fieldmodule.findMeshByDimension(1)

        epineurium_annotation_group = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_vagus_term("epineurium"))
        epineurium_mesh_group = epineurium_annotation_group.getMeshGroup(mesh2d)
        vagusAnteriorLineAnnotationGroup = findOrCreateAnnotationGroupForTerm(
            annotationGroups, region, get_vagus_term("orientation anterior"))
        vagusAnteriorLineMeshGroup = vagusAnteriorLineAnnotationGroup.getMeshGroup(mesh1d)
        # only want orientation on the trunk
        trunk_group = None
        for trunk_group_name in ["left vagus nerve", "right vagus nerve"]:
            trunk_group_field = fieldmodule.findFieldByName(trunk_group_name)
            if trunk_group_field.isValid():
                trunk_group = trunk_group_field.castGroup()
                break
        trunk_mesh_group = trunk_group.getMeshGroup(mesh2d) if trunk_group else None

        faceIterator = epineurium_mesh_group.createElementiterator()
        quadrant = 0
        face = faceIterator.next()
        while face.isValid():
            # trunk elements are the first consecutive block
            if trunk_mesh_group and not trunk_mesh_group.containsElement(face):
                break
            if quadrant == 0:
                line = face.getFaceElement(4)
                vagusAnteriorLineMeshGroup.addElement(line)
            quadrant = (quadrant + 1) % 4
            face = faceIterator.next()


def generate_trunk_1d(vagus_data, trunk_proportion, trunk_elements_count_prefit, trunk_elements_count,
                      trunk_fit_iterations, default_anterior_direction, default_trunk_diameter, region, nerve_metadata):
    """
    Build and fit a 1-D trunk curve to trunk data, calibrated to marker point positions.
    :param vagus_data: Vagus data extracted from input data region.
    :param trunk_proportion: Proportion up to 1.0 of whole vagus to build.
    :param trunk_elements_count_prefit: Number of elements in pre-fit mesh to trunk data.
    :param trunk_elements_count: Number of elements in final 1-D mesh.
    :param trunk_fit_iterations: Number of iterations in main trunk fit >= 1.
    :param default_anterior_direction: Vector direction to use as anterior if no orientotion data is supplied.
    Normalized in this function.
    :param default_trunk_diameter: Diameter in final units to use if no radius parameters.
    :param region: Region to put the fitted 1-D geometry including marker points in.
    :param nerve_metadata: Construction object to put fitting quality metadata into.
    :return: tx, td1, td2, td12, td3, td13 (parameters for 1-D fitted trunk geometry, left and anterior side
    directions and rates of change w.r.t. d1).
    """

    # 1. pre-fit to range of trunk data

    trunk_data_coordinates = vagus_data.get_trunk_coordinates()
    is_left = vagus_data.get_side_label() == 'left'
    raw_marker_data = vagus_data.get_level_markers()
    px = [e[0] for e in trunk_data_coordinates]
    segment_trunk_info_list = vagus_data.get_segment_trunk_info_list()
    if segment_trunk_info_list:
        ax = []
        for segment_trunk_info in segment_trunk_info_list:
            ax += segment_trunk_info['ordered_points']
    else:
        ax = px
    bx, bd1 = get_curve_from_points(ax, number_of_elements=trunk_elements_count_prefit)
    length = getCubicHermiteCurvesLength(bx, bd1)
    # outlier_length = 0.025 * length
    # # needs to be bigger if fewer elements:
    # if trunk_elements_count_prefit < 20:
    #     outlier_length += 0.075 * (20 - trunk_elements_count_prefit) / 19.0
    cx, cd1 = fit_hermite_curve(bx, bd1, px)  # , outlier_length=outlier_length)
    # resample to even size
    dx, dd1 = sampleCubicHermiteCurvesSmooth(cx, cd1, trunk_elements_count_prefit)[0:2]
    # generate_curve_mesh(region, dx, dd1, group_name=vagus_data.get_trunk_group_name())[0]
    # return dx, dd1, [], []

    # 2. make initial full trunk extending pre-fit based on furthest marker material coordinates

    vagus_level_terms = get_left_vagus_marker_locations_list() if is_left \
        else get_right_vagus_marker_locations_list()
    marker_data = []  # list from top to bottom of nerve of (name, material_coordinate, data_coordinates)
    for marker_term_name, material_coordinate in vagus_level_terms.items():
        if marker_term_name in raw_marker_data.keys():
            data_coordinates = raw_marker_data[marker_term_name]
            for idx, data in enumerate(marker_data):
                if material_coordinate < data[1]:
                    break
            else:
                idx = len(marker_data)
            marker_data.insert(idx, (marker_term_name, material_coordinate, data_coordinates))

    start_marker_material_coordinate = marker_data[0][1]
    start_marker_curve_location = getNearestLocationOnCurve(dx, dd1, marker_data[0][2])[0]
    start_marker_extra_length = 0.0
    if (start_marker_curve_location[0] == 0) and (start_marker_curve_location[1] < 1.0E-4):
        # case when start marker point is before start of curve
        start_marker_extra_length = max(0.0, dot(sub(dx[0], marker_data[0][2]), normalize(dd1[0])))
    end_marker_material_coordinate = marker_data[-1][1]
    end_marker_curve_location = getNearestLocationOnCurve(dx, dd1, marker_data[-1][2])[0]
    end_marker_extra_length = 0.0
    if (end_marker_curve_location[0] == (trunk_elements_count_prefit - 1)) and (
            end_marker_curve_location[1] > 0.9999):
        # case when end marker point is after end of curve
        end_marker_extra_length = max(0.0, dot(sub(marker_data[-1][2], dx[-1]), normalize(dd1[-1])))
    start_length, length, end_length = getCubicHermiteTrimmedCurvesLengths(
        dx, dd1, start_marker_curve_location, end_marker_curve_location)[0:3]

    # determine material coordinates of start and end of curve, from range of marker material coordinates
    marker_delta_material_length = end_marker_material_coordinate - start_marker_material_coordinate
    marker_delta_length = length + start_marker_extra_length + end_marker_extra_length
    material_length_per_length = marker_delta_material_length / marker_delta_length
    start_curve_material_coordinate = (start_marker_material_coordinate -
                                       (start_length - start_marker_extra_length) * material_length_per_length)
    end_curve_material_coordinate = (end_marker_material_coordinate +
                                     (end_length - end_marker_extra_length) * material_length_per_length)
    # extend curves at each end, by moving end node if short extension, or adding node if large
    start_extra_length = start_curve_material_coordinate / material_length_per_length
    if start_extra_length < 0.0:
        logger.warning("Projected start material coordinate " + str(start_curve_material_coordinate) +
                       " is < 0; using full range of data.")
        start_extra_length = 0.0
    start_direction = normalize(dd1[0])
    mag_d1 = magnitude(dd1[0])
    start_dx = sub(dx[0], mult(start_direction, start_extra_length))
    if start_extra_length < (0.51 * mag_d1):
        dx[0] = start_dx
        dd1[0] = set_magnitude(dd1[0], mag_d1 + 2.0 * start_extra_length)
    else:
        dx.insert(0, start_dx)
        dd1.insert(0, mult(start_direction, 2.0 * start_extra_length - mag_d1))
    end_extra_length = (trunk_proportion - end_curve_material_coordinate) / material_length_per_length
    if end_extra_length < 0.0:
        logger.warning("Projected end material coordinate " + str(end_curve_material_coordinate) +
                       " is > trunk proportion " + str(trunk_proportion) +  "; using full range of data.")
        end_extra_length = 0.0
    end_direction = normalize(dd1[-1])
    mag_d1 = magnitude(dd1[-1])
    end_dx = add(dx[-1], mult(end_direction, end_extra_length))
    if start_extra_length < (0.51 * mag_d1):
        dx[-1] = end_dx
        dd1[-1] = set_magnitude(dd1[-1], mag_d1 + 2.0 * end_extra_length)
    else:
        dx.append(end_dx)
        dd1.append(mult(end_direction, 2.0 * end_extra_length - mag_d1))

    # resample to even size and final elements count - this is the initial state for full trunk fit with markers
    ex, ed1 = sampleCubicHermiteCurvesSmooth(dx, dd1, trunk_elements_count)[0:2]

    # 3. full trunk centroid fit

    fit_region = region.createRegion()
    fieldmodule = fit_region.getFieldmodule()
    coordinates = find_or_create_field_coordinates(fieldmodule)
    components_count = coordinates.getNumberOfComponents()
    mesh1d = fieldmodule.findMeshByDimension(1)
    trunk_group_name = vagus_data.get_trunk_group_name()
    with ChangeManager(fieldmodule):
        node_identifier = generate_curve_mesh(fit_region, ex, ed1, group_name=trunk_group_name)[0]
        trunk_group = find_or_create_field_group(fieldmodule, trunk_group_name)
        pr = vagus_data.get_trunk_radius()
        field_names_and_values = [("radius", pr)] if pr else []
        data_identifier = 1
        data_identifier = generate_datapoints(
            fit_region, px, data_identifier, field_names_and_values=field_names_and_values, group_name=trunk_group_name)
        radius = fieldmodule.findFieldByName("radius").castFiniteElement()

        # add marker points in order along trunk
        ordered_marker_data = []  # list of (name, curve_location)
        ordered_material_coordinate = []  # list of material coordinate
        for marker_term_name, material_coordinate in vagus_level_terms.items():
            for idx, prior_material_coordinate in enumerate(ordered_material_coordinate):
                if material_coordinate < prior_material_coordinate:
                    break
            else:
                idx = len(ordered_marker_data)
            real_element_xi = (material_coordinate / trunk_proportion) * trunk_elements_count
            element_index = min(int(real_element_xi), trunk_elements_count - 1)
            element_xi = (element_index + 1, real_element_xi - element_index)
            ordered_marker_data.insert(idx, (marker_term_name, element_xi))
            ordered_material_coordinate.insert(idx, material_coordinate)
        node_identifier = generate_mesh_marker_points(mesh1d, ordered_marker_data, node_identifier)

        # add data marker points:
        data_identifier = generate_datapoints(
            fit_region, list(raw_marker_data.values()), data_identifier,
            field_names_and_values=[("marker_name", list(raw_marker_data.keys()))], group_name="marker")

        zero_fibres = find_or_create_field_zero_fibres(fieldmodule)

        # create group for applying higher stiffness for multi-path segments
        # elements are in this group if either end nodes are in the range of a multi-path segment
        multi_path_group = fieldmodule.createFieldGroup()
        multi_path_group.setName('multi-path')
        multi_path_group.setManaged(True)  # if not managed, group settings are ignored
        multi_path_group_name = multi_path_group.getName()  # in case name was in use
        multi_path_mesh_group = multi_path_group.createMeshGroup(mesh1d)
        data_multi_path_ranges = []
        for segment_trunk_info in segment_trunk_info_list:
            if not segment_trunk_info.get('ordered_coordinates'):
                data_multi_path_ranges.append(segment_trunk_info['range'])
                # print('segment', segment_trunk_info['name'], 'is multi-path')

        node_in_data_multi_path_range = []
        for x in ex:
            in_data_multi_path_range = False
            for data_multi_path_range in data_multi_path_ranges:
                for c in range(3):
                    if (x[c] < data_multi_path_range[0][c]) or (x[c] > data_multi_path_range[1][c]):
                        break
                else:
                    in_data_multi_path_range = True
                    break
            node_in_data_multi_path_range.append(in_data_multi_path_range)
        element_in_data_multi_path_range = []
        for element_identifier in range(1, trunk_elements_count + 1):
            if (node_in_data_multi_path_range[element_identifier - 1] or
                    node_in_data_multi_path_range[element_identifier]):
                multi_path_mesh_group.addElement(mesh1d.findElementByIdentifier(element_identifier))
                # print("Element", element_identifier,"is in multi-path range")

    # note that fitting is very slow if done within ChangeManager as find mesh location is slow
    # this includes working with the user-supplied region which is called with ChangeManager on.
    fitter = GeometryFitter(region=fit_region)
    length = getCubicHermiteCurvesLength(ex, ed1)
    outlier_length = 0.05 * length
    config_step = fitter.getInitialFitterStepConfig()
    config_step.setGroupOutlierLength(None, outlierLength=outlier_length)
    fitter.setDiagnosticLevel(1)
    fitter.setModelCoordinatesField(coordinates)
    fitter.setFibreField(zero_fibres)
    del zero_fibres
    fitter.defineCommonMeshFields()
    fitter.setDataCoordinatesField(coordinates)
    fitter.defineDataProjectionFields()
    fitter.setMarkerGroupByName("marker")
    fitter.initializeFit()

    # calibration_points_count = 23954
    points_count_calibration_factor = len(px) / 25000
    # calibration_length = 27840.0
    length_calibration_factor = length / 25000.0
    strain_penalty = 1.0E+4 * points_count_calibration_factor * length_calibration_factor
    curvature_penalty = 1.0E+9 * points_count_calibration_factor * (length_calibration_factor ** 3)
    marker_weight = 20.0 * points_count_calibration_factor
    sliding_factor = 0.01

    for step in range(0, min(2, trunk_fit_iterations)):
        fit_step = FitterStepFit()
        fitter.addFitterStep(fit_step)
        if step == 0:
            fit_step.setGroupDataWeight("marker", marker_weight)
            fit_step.setUpdateReferenceState(True)
        weight = 1.0 if (step == 0) else 0.1
        fit_step.setGroupStrainPenalty(None, [weight * strain_penalty])
        fit_step.setGroupCurvaturePenalty(None, [weight * curvature_penalty])
        fit_step.setGroupDataSlidingFactor(None, weight * sliding_factor)

        fit_step.setGroupStrainPenalty(multi_path_group_name, [5.0 * weight * strain_penalty])
        fit_step.setGroupCurvaturePenalty(multi_path_group_name, [50.0 * weight * curvature_penalty])

        if step > 0:
            fit_step.setNumberOfIterations(trunk_fit_iterations - 1)
        fit_step.run()
        del fit_step

    datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    rms_error, max_error = fitter.getDataRMSAndMaximumProjectionError(trunk_group.getNodesetGroup(datapoints))
    nerve_metadata.set_name_rms_max_error("trunk centroid fit error", rms_error, max_error)

    with ChangeManager(fieldmodule):
        # make a real field which increases down the trunk proportional to vagus coordinates
        trunk_distance = (fieldmodule.findFieldByName("cmiss_number") +
                          fieldmodule.createFieldComponent(fieldmodule.findFieldByName("xi"), 1))
        host_trunk_distance = fieldmodule.createFieldEmbedded(trunk_distance, fitter.getDataHostLocationField())
        minimums, maximums = evaluate_field_nodeset_range(host_trunk_distance, trunk_group.getNodesetGroup(datapoints))
        min_projection_node_number = math.ceil(minimums)
        max_projection_node_number = math.floor(maximums)
        del host_trunk_distance

    fitter.cleanup()
    del fitter

    # fit radius
    if pr:
        # add default radius points at trunk nodes outside range of radius data projections
        tx = []
        fieldcache = fieldmodule.createFieldcache()
        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodeiterator = nodes.createNodeiterator()
        node = nodeiterator.next()
        for n in range(trunk_elements_count + 1):
            nodeIdentifier = node.getIdentifier()
            if (nodeIdentifier < min_projection_node_number) or (nodeIdentifier > max_projection_node_number):
                fieldcache.setNode(node)
                result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, components_count)
                tx.append(x)
            node = nodeiterator.next()
        if tx:
            field_names_and_values = [("radius", [default_trunk_diameter * 0.5] * len(tx))]
            data_identifier = generate_datapoints(
                fit_region, tx, data_identifier, field_names_and_values=field_names_and_values, group_name=trunk_group_name)

        # add projection distance to radius
        trunk_location = fieldmodule.createFieldFindMeshLocation(coordinates, coordinates, mesh1d)
        trunk_location.setSearchMode(trunk_location.SEARCH_MODE_NEAREST)
        projected_coordinates = fieldmodule.createFieldEmbedded(coordinates, trunk_location)
        projection_distance = fieldmodule.createFieldMagnitude(projected_coordinates - coordinates)
        new_radius = radius + projection_distance
        trunk_datapoints = trunk_group.getNodesetGroup(datapoints)
        fieldassignment = radius.createFieldassignment(new_radius)
        fieldassignment.setNodeset(trunk_datapoints)
        fieldassignment.assign()
        gradient1_penalty = 10000.0 * points_count_calibration_factor * length_calibration_factor
        gradient2_penalty = 1.0E+8 * points_count_calibration_factor * (length_calibration_factor ** 3)
        rms_error, max_error = define_and_fit_field(
            fit_region, "coordinates", "coordinates", "radius",
            gradient1_penalty, gradient2_penalty, group_name=trunk_group_name)
        nerve_metadata.set_name_rms_max_error("trunk radius fit error", rms_error, max_error)

    # extract fitted trunk parameters from trunk nodes (not marker points)
    length = getCubicHermiteCurvesLength(ex, ed1)
    default_trunk_radius = 0.5 * default_trunk_diameter
    tx = []
    td1 = []
    rx = []
    rd1 = []
    fieldcache = fieldmodule.createFieldcache()
    nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodeiterator = nodes.createNodeiterator()
    node = nodeiterator.next()
    for n in range(trunk_elements_count + 1):
        fieldcache.setNode(node)
        result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, components_count)
        result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, components_count)
        tx.append(x)
        td1.append(d1)
        if pr:
            result, x = radius.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 1)
            result, d1 = radius.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 1)
        else:
            x = default_trunk_radius
            d1 = 0.0
        rx.append(x)
        rd1.append(d1)
        node = nodeiterator.next()
    # get mean_radius to eliminate outlier orientation points
    mean_radius = sum(pr) / len(pr) if pr else default_trunk_radius
    max_orientation_projection_error = 8.0 * mean_radius

    # remove all datapoints from previous fits
    datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    datapoints.destroyAllNodes()

    segments_metadata = {}
    with ChangeManager(fieldmodule):
        # make a real field which increases down the trunk proportional to vagus coordinates
        trunk_distance = (fieldmodule.findFieldByName("cmiss_number") +
                          fieldmodule.createFieldComponent(fieldmodule.findFieldByName("xi"), 1))
        find_mesh_location = fieldmodule.createFieldFindMeshLocation(coordinates, coordinates, mesh1d)
        find_mesh_location.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
        host_trunk_distance = fieldmodule.createFieldEmbedded(trunk_distance, find_mesh_location)
        datapoints_min_trunk_distance = fieldmodule.createFieldNodesetMinimum(host_trunk_distance, datapoints)
        datapoints_max_trunk_distance = fieldmodule.createFieldNodesetMaximum(host_trunk_distance, datapoints)
    fieldcache.clearLocation()
    distance_to_material = trunk_proportion / trunk_elements_count
    for segment_trunk_info in segment_trunk_info_list:
        segment_name = segment_trunk_info['name']
        sx = segment_trunk_info['unordered_coordinates']
        generate_datapoints(fit_region, sx, start_data_identifier=1)
        min_result, segment_min_trunk_distance = datapoints_min_trunk_distance.evaluateReal(fieldcache, 1)
        max_result, segment_max_trunk_distance = datapoints_max_trunk_distance.evaluateReal(fieldcache, 1)
        if (min_result == RESULT_OK) and (max_result == RESULT_OK):
            segments_metadata[segment_name] = {
                "minimum vagus coordinate": segment_min_trunk_distance * distance_to_material,
                "maximum vagus coordinate": segment_max_trunk_distance * distance_to_material
            }
        else:
            logger.warning("Could not calculate material coordinates range for segment " + segment_name)
        datapoints.destroyAllNodes()
    nerve_metadata.set_name_value("segments", segments_metadata)

    # fit orientation
    debug_print = False
    orientation_dct = vagus_data.get_orientation_data()
    orientation_x = []
    orientation_twist_angles = []
    if not orientation_dct:
        # generate 'orientation anterior' points using the default_anterior_direction and default_trunk_radius
        # :return Dict mapping 8 possible orientations (anterior, left, right, etc.) to list of x, y, z coordinates.
        dira = normalize(default_anterior_direction)
        anterior_x = []
        for e in range(trunk_elements_count):
            x, d1 = evaluateCoordinatesOnCurve(tx, td1, (e, 0.5), derivative=True)
            dir1 = normalize(d1)
            if abs(dot(dir1, dira)) > 0.9:
                continue  # skip as trunk is too in-line with anterior direction
            dir2 = cross(dira, dir1)
            dir3 = normalize(cross(dir1, dir2))
            anterior_x.append(add(x, mult(dir3, default_trunk_radius)))
        if anterior_x:
            orientation_dct = {'orientation anterior': anterior_x}
    if orientation_dct:
        # convert to list in order down the trunk
        length = getCubicHermiteCurvesLength(tx, td1)
        orientation_directions = []  # list of anterior directions
        orientation_names = []  # list of orientation names
        orientation_locations = []  # list of curve location (element index, xi) for orientation points
        one_sqrt2 = 1.0 / math.sqrt(2.0)
        # weights of d2, d3 to give anterior direction for a given orientation direction name
        orientation_anterior_weights = {
            "left": (-1.0, 0.0),
            "left anterior": (-one_sqrt2, one_sqrt2),
            "anterior": (0.0, 1.0),
            "right anterior": (one_sqrt2, one_sqrt2),
            "right": (1.0, 0.0),
            "right posterior": (one_sqrt2, -one_sqrt2),
            "posterior": (0.0, -1.0),
            "left posterior": (-one_sqrt2, -one_sqrt2)
        }
        for name, x_list in orientation_dct.items():
            direction_name = name.split("orientation ", 1)[1]
            weights = orientation_anterior_weights.get(direction_name)
            if not weights:
                logger.warning("Nerve: Ignoring unrecognized orientation points with name '" + name + "'")
                continue
            wt2, wt3 = weights
            for data_x in x_list:
                curve_location, x = getNearestLocationOnCurve(tx, td1, data_x)
                e1 = curve_location[0]
                e2 = curve_location[0] + 1
                xi = curve_location[1]
                d1 = interpolateCubicHermiteDerivative(tx[e1], td1[e1], tx[e2], td1[e2], xi)
                dir1 = normalize(d1)
                projection = sub(data_x, x)
                normal_projection = rejection(projection, dir1)
                projection_error = magnitude(normal_projection)
                if projection_error < rms_error:
                    # skip orientation points within rmsError of trunk centroid as inaccurate
                    logger.warning("Nerve: Ignoring orientation point '" + name + "' at location " +
                                   str(curve_location) + " as projection error " + str(projection_error) +
                                   " is less than trunk RMS error " + str(rms_error))
                    continue
                dirp = normalize(projection)
                if math.fabs(dot(dir1, dirp)) > 0.5:
                    # skip orientation points with severely non-normal projection e.g. past end of curve
                    logger.warning("Nerve: Ignoring orientation point '" + name + "' at location " +
                                   str(curve_location) + " as projection is oblique or co-linear with centroid curve")
                    continue
                if projection_error > max_orientation_projection_error:
                    # skip orientation points too far from centroid
                    logger.warning("Nerve: Ignoring orientation point '" + name + "' at location " +
                                   str(curve_location) + " with coordinates " + str(data_x) +
                                   " as projection error " + str(projection_error) + " exceeds " +
                                   str(max_orientation_projection_error / mean_radius) + "x mean radius " +
                                   str(mean_radius))
                    continue
                dir2 = normalize(cross(dirp, dir1))
                dir3 = cross(dir1, dir2)
                anterior_direction = add(mult(dir2, wt2), mult(dir3, wt3))
                for idx, orientation_location in enumerate(orientation_locations):
                    if curve_location < orientation_location:
                        break
                else:
                    idx = len(orientation_locations)
                orientation_x.insert(idx, x)
                orientation_directions.insert(idx, anterior_direction)
                orientation_names.insert(idx, name)
                orientation_locations.insert(idx, curve_location)
        if not orientation_directions:
            logger.warning("Nerve: All orientation points ignored, using default orientation")
        else:
            # twist angles are positive in right hand sense around direction down trunk
            twist_angle = 0.0
            orientation_twist_angles.append(twist_angle)  # top orientation point is at 0 radians
            direction = orientation_directions[0]
            for i in range(1, len(orientation_locations)):
                dir1, dir2, dir3 = track_curve_side_direction(
                    tx, td1, direction, orientation_locations[i - 1], orientation_locations[i])
                direction = orientation_directions[i]
                y = dot(direction, dir2)
                x = dot(direction, dir3)
                delta_twist_angle = -math.atan2(y, x)
                twist_angle += delta_twist_angle
                orientation_twist_angles.append(twist_angle)
            if debug_print:
                print("Orientation data:")
                for location, direction, twist_angle, name, in zip(
                        orientation_locations, orientation_directions, orientation_twist_angles, orientation_names):
                    print(location, direction, twist_angle, name)
    if orientation_twist_angles:
        data_identifier = 1
        twist_angle_field_name = "twist angle"
        twist_points_count_calibration_factor = len(orientation_twist_angles) / 27.0
        data_identifier = generate_datapoints(
            fit_region, orientation_x, data_identifier,
            field_names_and_values=[(twist_angle_field_name, orientation_twist_angles)])

        gradient1_penalty = 100.0 * twist_points_count_calibration_factor * length_calibration_factor
        gradient2_penalty = 1.0E+8 * twist_points_count_calibration_factor * (length_calibration_factor ** 3)
        rms_error, max_error = define_and_fit_field(
            fit_region, "coordinates", "coordinates", twist_angle_field_name,
            gradient1_penalty, gradient2_penalty, group_name=trunk_group_name)
        nerve_metadata.set_name_rms_max_error(
            "trunk twist angle fit error degrees", math.degrees(rms_error), math.degrees(max_error))
        twist_angle = fieldmodule.findFieldByName(twist_angle_field_name).castFiniteElement()
        # extract fitted twist angle parameters from nodes
        ax = []
        ad1 = []
        for n in range(trunk_elements_count + 1):
            node = nodes.findNodeByIdentifier(n + 1)
            fieldcache.setNode(node)
            result, x = twist_angle.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 1)
            result, d1 = twist_angle.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 1)
            ax.append(x)
            ad1.append(d1)
        if debug_print:
            print("Orientation fit:")
            for i in range(len(orientation_locations)):
                fit_twist_angle = evaluateScalarOnCurve(ax, ad1, orientation_locations[i])
                print(orientation_locations[i], orientation_twist_angles[i], "-->", fit_twist_angle)
            print("node, ax, ad1:")
            for i in range(len(ax)):
                print(i + 1, ax[i], ad1[i])

        # get best-fit anterior direction at first orientation point
        first_point = 0
        first_location = orientation_locations[first_point]
        dir1, dir2, dir3 = track_curve_side_direction(
            tx, td1, orientation_directions[first_point], first_location, first_location)
        first_twist_data = orientation_twist_angles[first_point]
        first_twist_radians = evaluateScalarOnCurve(ax, ad1, first_location)
        delta_twist_radians = first_twist_radians - first_twist_data
        first_direction = normalize(add(mult(dir3, math.cos(delta_twist_radians)),
                                        mult(dir2, -math.sin(delta_twist_radians))))
    else:
        # this case is unlikely now due to creation of anterior points above
        logger.warning("Nerve: Default anterior direction is in-line with top of nerve; expect poor results.")
        ad1 = ax = [0.0] * (trunk_elements_count + 1)
        first_location = (0, 0.0)
        first_twist_radians = 0.0
        dira = normalize(default_anterior_direction)
        dir1 = normalize(td1[0])
        if abs(dot(dir1, dira)) > 0.999:
            first_direction = dira
        else:
            dir2 = cross(dira, dir1)
            first_direction = normalize(cross(dir1, dir2))

    # compute side/anterior directions and their rates of change
    td2 = []
    td12 = []
    td3 = []
    td13 = []
    # track from first orientation point along nodes back to start, then forward from first orientation point to end
    prev_twist_radians = first_twist_radians
    prev_direction = first_direction
    prev_location = first_location
    next_location = (prev_location[0], 0.0)
    forward = False
    node_indexes = []
    tmp_ta = []
    tmp_dta = []
    for n in range(trunk_elements_count + 1):
        dir1, dir2, dir3 = track_curve_side_direction(
            tx, td1, prev_direction, prev_location, next_location, forward)
        node_index = (next_location[0] + 1) if forward else next_location[0]
        x = tx[node_index]
        d1 = td1[node_index]
        rv = 2.0 * rx[node_index]  # span of box is double radius
        rd = 2.0 * rd1[node_index]  # as above
        next_twist_radians = ax[node_index]
        ad = ad1[node_index]
        if forward:
            delta_twist_radians = next_twist_radians - prev_twist_radians
        else:
            delta_twist_radians = prev_twist_radians - next_twist_radians
        ante = normalize(add(mult(dir3, math.cos(delta_twist_radians)),
                             mult(dir2, -math.sin(delta_twist_radians))))
        left = normalize(cross(ante, dir1))

        # calculate centroid curvature in d2, d3 directions
        curvatures = []
        for dirn in [left, ante]:
            curvature = 0.0
            count = 0
            if node_index > 0:
                curvature += getCubicHermiteCurvature(
                    tx[node_index - 1], td1[node_index - 1], x, d1, dirn, 1.0)
                count += 1
            if n < trunk_elements_count:
                curvature += getCubicHermiteCurvature(
                    x, d1, tx[node_index + 1], td1[node_index + 1], dirn, 0.0)
                count += 1
            curvature /= count
            curvatures.append(curvature)

        d2 = set_magnitude(left, rv)
        d12 = add(add(mult(left, rd), mult(ante, rv * ad)), mult(d1, -rv * curvatures[0]))
        d3 = set_magnitude(ante, rv)
        d13 = add(add(mult(ante, rd), mult(left, -rv * ad)), mult(d1, -rv * curvatures[1]))
        ix = n if forward else 0
        node_indexes.insert(ix, node_index)
        td2.insert(ix, d2)
        td12.insert(ix, d12)
        td3.insert(ix, d3)
        td13.insert(ix, d13)
        tmp_ta.insert(ix, next_twist_radians)
        tmp_dta.insert(ix, delta_twist_radians)

        prev_twist_radians = next_twist_radians
        prev_direction = ante
        prev_location = next_location
        if forward:
            next_location = (next_location[0] + 1, 1.0)
        elif next_location[0] > 0:
            next_location = (next_location[0] - 1, 0.0)
        else:
            prev_twist_radians = first_twist_radians
            prev_direction = first_direction
            prev_location = first_location
            next_location = (prev_location[0], 1.0)
            forward = True
    if debug_print:
        print("node, ta, dta, d3:")
        for i in range(len(tmp_ta)):
            print(i + 1, tmp_ta[i], tmp_dta[i], td3[i])

    # copy model to user-supplied region
    sir = fit_region.createStreaminformationRegion()
    srm = sir.createStreamresourceMemory()
    sir.setResourceDomainTypes(srm, Field.DOMAIN_TYPE_NODES | Field.DOMAIN_TYPE_MESH1D)
    fit_region.write(sir)
    result, buffer = srm.getBuffer()
    sir = region.createStreaminformationRegion()
    srm = sir.createStreamresourceMemoryBuffer(buffer)
    region.read(sir)

    return tx, td1, td2, td12, td3, td13


class NerveMetadata(ConstructionObject):

    def __init__(self, top_level_name):
        """
        :param top_level_name: Unique name for top level of metadata dict
        """
        self._top_level_name = top_level_name
        self._metadata = {}

    def getMetadata(self):
        return {self._top_level_name: self._metadata}

    def set_name_rms_max_error(self, quantity_name, rms_error, max_error):
        """
        :param quantity_name: Stem name of metadata e.g. "trunk centroid fit error"
        to which " max" and " rms" are added.
        :param rms_error: RMS error value, scalar real or list/vector.
        :param max_error: Maximum error value, scalar real or list/vector.
        """
        self._metadata[quantity_name + " rms"] = rms_error
        self._metadata[quantity_name + " max"] = max_error

    def set_name_value(self, quantity_name, value):
        """
        Add a name value pair to the metadata.
        :param quantity_name: Name of quantity to add a value for.
        :param value: Any value serialisable by json.dumps().
        """
        self._metadata[quantity_name] = value
