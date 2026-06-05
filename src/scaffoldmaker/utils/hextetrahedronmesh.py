"""
Utilities for building 3-D tetrahedron-shaped meshes out of hexahedral elements with cubic Hermite serendipity
interpolation.
"""
from scaffoldmaker.utils.interpolation import (
    DerivativeScalingMode, get_nway_point, linearlyInterpolateVectors, sampleHermiteCurve,
    smoothCubicHermiteDerivativesLine)
from scaffoldmaker.utils.quadtrianglemesh import QuadTriangleMesh
import copy


class HexTetrahedronMesh:
    """
    Generates a tetrahedron mesh from c1-continuous hex (cube) elements, with a 3-way or 4-way points inside.
    Tetrahedron is defined from 4 corner points 0-3, origin and other axis ends as right-handed axes.
    Parameters on faces are set from QuadTriangleMesh objects.
    3-D parameters are interpolated from face parameters.
    Some limitations on currently supported number of elements in directions are asserted in the constructor.
    """

    def __init__(self, axis_counts, diag_counts, nway_d_factor=0.6):
        """
        :param axis_counts: Number of elements along 0-1, 0-2, 0-3 axes.
        :param diag_counts: Number of elements along 1-2, 1-3, 2-3 diagonals.
        Coordinates nx are indexed in 1, 2, 3 directions from origin at index 0, 0, 0
        with holes around the corners and 3-way or 4-way points.
        :param nway_d_factor: Value, normally from 0.5 to 1.0 giving n-way derivative magnitude as a proportion
        of the minimum regular magnitude sampled to the n-way point. This reflects that distances from the mid-side
        of a triangle to the centre are shorter, so the derivative in the middle must be smaller.
        """
        assert all((count >= 2) for count in axis_counts)
        assert all((count >= 2) for count in diag_counts)
        # check the faces have valid element counts around them
        max_diag_count0 = axis_counts[0] + axis_counts[1] - 2
        assert any((diag_counts[0] == diag_count) for diag_count in range(max_diag_count0, 1, -2))
        max_diag_count1 = axis_counts[0] + axis_counts[2] - 2
        assert any((diag_counts[1] == diag_count) for diag_count in range(max_diag_count1, 1, -2))
        max_diag_count2 = axis_counts[1] + axis_counts[2] - 2
        assert any((diag_counts[2] == diag_count) for diag_count in range(max_diag_count2, 1, -2))
        max_diag_count3 = diag_counts[0] + diag_counts[1] - 2
        assert any((diag_counts[2] == diag_count) for diag_count in range(max_diag_count3, 1, -2))
        self._axis_counts = copy.copy(axis_counts)
        self._diag_counts = copy.copy(diag_counts)
        self._nway_d_factor = nway_d_factor

        # current limitation:
        # only supports 4-way point for now, consistent with constant transition count away from origin
        trans_count0 = (axis_counts[0] + axis_counts[1] - diag_counts[0]) // 2
        trans_count1 = (axis_counts[0] + axis_counts[2] - diag_counts[1]) // 2
        trans_count2 = (axis_counts[1] + axis_counts[2] - diag_counts[2]) // 2
        assert trans_count0 == trans_count1 == trans_count2
        self._trans_count = trans_count0

        # counts of elements to 3-way point opposite to 3 node at axis 1, axis 2, axis 3
        self._box_counts = [self._axis_counts[i] - self._trans_count for i in range(3)]
        none_parameters = [None] * 4  # x, d1, d2, d3
        self._nx = []  # shield mesh with holes over n3, n2, n1, d
        for n3 in range(axis_counts[2] + 1):
            # index into transition zone
            trans3 = self._trans_count + n3 - axis_counts[2]
            nx_layer = []
            for n2 in range(axis_counts[1] + 1):
                # index into transition zone
                trans2 = self._trans_count + n2 - axis_counts[1]
                nx_row = []
                # s = ""
                for n1 in range(axis_counts[0] + 1):
                    # index into transition zone
                    trans1 = self._trans_count + n1 - axis_counts[0]
                    if (((trans1 <= 0) and (trans2 <= 0) and (trans3 <= 0)) or
                            (trans1 == trans2 == trans3) or
                            ((trans1 < 0) and ((trans2 == trans3) or (trans2 < 0))) or
                            ((trans2 < 0) and ((trans3 == trans1) or (trans3 < 0))) or
                            ((trans3 < 0) and ((trans1 == trans2) or (trans1 < 0)))):
                        parameters = copy.copy(none_parameters)
                        # s += "[]"
                    else:
                        parameters = None
                        # s += "  "
                    nx_row.append(parameters)
                nx_layer.append(nx_row)
                # print(s)
            self._nx.append(nx_layer)

    def get_axis_counts(self):
        return self._axis_counts

    def get_box_counts(self):
        return self._box_counts

    def get_parameters(self):
        """
        Get parameters array e.g. for copying to ellipsoid.
        :return: Internal parameters array self._nx. Not to be modified.
        """
        return self._nx

    def set_triangle_abc(self, trimesh: QuadTriangleMesh):
        """
        Set parameters on the outer abc surface triangle of octant.
        :param trimesh: Coordinates to set on outer surface.
        """
        assert trimesh.get_element_count12() == self._diag_counts[0]
        assert trimesh.get_element_count13() == self._diag_counts[1]
        assert trimesh.get_element_count23() == self._diag_counts[2]
        start_indexes = [self._axis_counts[0], 0, 0]
        for n3 in range(self._box_counts[2]):
            px, pd1, pd2, pd3 = trimesh.get_parameters12(n3)
            self._set_coordinates_across([px, pd1, pd2, pd3], [[0, 1, 2, 3]], start_indexes, [[0, 1, 0], [-1, 0, 0]])
            start_indexes[2] += 1
        start_indexes = [0, 0, self._axis_counts[2]]
        for n2 in range(self._box_counts[1]):
            px, pd1, pd2, pd3 = trimesh.get_parameters31(n2, self._box_counts[0] + 1)
            self._set_coordinates_across([px, pd1, pd2, pd3], [[0, 1, 2, 3]], start_indexes, [[1, 0, 0]])
            start_indexes[1] += 1
        start_indexes = [0, self._axis_counts[1], self._axis_counts[2]]
        px, pd1, pd2, pd3 = trimesh.get_parameters_diagonal()
        self._set_coordinates_across([px, pd1, pd2, pd3], [[0, 1, 2, 3]], start_indexes, [[1, 0, 0]])

    def set_triangle_abo(self, trimesh: QuadTriangleMesh):
        """
        Set parameters on triangle 1-2-origin, an inner surface of octant.
        :param trimesh: Triangle coordinate data with x, d1, d2, optional d3.
        """
        assert trimesh.get_element_count12() == self._diag_counts[0]
        assert trimesh.get_element_count13() == self._axis_counts[0]
        assert trimesh.get_element_count23() == self._axis_counts[1]
        start_indexes = [self._axis_counts[0], 0, 0]
        for n0 in range(self._trans_count):
            px, pd1, pd2, pd3 = trimesh.get_parameters12(n0)
            self._set_coordinates_across([px, pd1, pd2, pd3], [[0, 1, -3, 2]], start_indexes, [[0, 1, 0], [-1, 0, 0]])
            start_indexes[0] -= 1
        start_indexes = [0, 0, 0]
        for n2 in range(self._box_counts[1] + 1):
            px, pd1, pd2, pd3 = (trimesh.get_parameters31(n2, self._box_counts[0] + 1) if (n2 < self._box_counts[1])
                                 else trimesh.get_parameters_diagonal())
            self._set_coordinates_across([px, pd1, pd2, pd3], [[0, 1, 2, 3]], start_indexes, [[1, 0, 0]])
            start_indexes[1] += 1

    def set_triangle_aco(self, trimesh: QuadTriangleMesh):
        """
        Set parameters on triangle 1-3-origin, an inner surface of octant.
        :param trimesh: Triangle coordinate data with x, d1, d2, optional d3.
        """
        assert trimesh.get_element_count12() == self._diag_counts[1]
        assert trimesh.get_element_count13() == self._axis_counts[0]
        assert trimesh.get_element_count23() == self._axis_counts[2]
        start_indexes = [self._axis_counts[0], 0, 0]
        for n0 in range(self._trans_count):
            px, pd1, pd2, pd3 = trimesh.get_parameters12(n0)
            self._set_coordinates_across(
                [px, pd1, pd2, pd3], [[0, 2, -3, -1], [0, -1, -3, -2]], start_indexes, [[0, 0, 1], [-1, 0, 0]])
            start_indexes[0] -= 1
        start_indexes = [0, 0, 0]
        for n3 in range(self._box_counts[2] + 1):
            px, pd1, pd2, pd3 = (trimesh.get_parameters31(n3, self._box_counts[0] + 1) if (n3 < self._box_counts[2])
                                 else trimesh.get_parameters_diagonal())
            self._set_coordinates_across([px, pd1, pd2, pd3], [[0, 1, 3, -2]], start_indexes, [[1, 0, 0]])
            start_indexes[2] += 1

    def set_triangle_bco(self, trimesh: QuadTriangleMesh):
        """
        Set parameters on triangle 2-3-origin, an inner surface of octant.
        :param trimesh: Triangle coordinate data with x, d1, d2, optional d3.
        """
        assert trimesh.get_element_count12() == self._diag_counts[2]
        assert trimesh.get_element_count13() == self._axis_counts[1]
        assert trimesh.get_element_count23() == self._axis_counts[2]
        start_indexes = [0, self._axis_counts[1], 0]
        for n0 in range(self._trans_count):
            px, pd1, pd2, pd3 = trimesh.get_parameters12(n0)
            self._set_coordinates_across(
                [px, pd1, pd2, pd3], [[0, 2, -3, -1], [0, -2, -3, 1]], start_indexes, [[0, 0, 1], [0, -1, 0]])
            start_indexes[1] -= 1
        start_indexes = [0, 0, 0]
        for n3 in range(self._box_counts[2] + 1):
            px, pd1, pd2, pd3 = (trimesh.get_parameters31(n3, self._box_counts[1] + 1) if (n3 < self._box_counts[2])
                                 else trimesh.get_parameters_diagonal())
            self._set_coordinates_across([px, pd1, pd2, pd3], [[0, 2, 3, 1]], start_indexes, [[0, 1, 0]])
            start_indexes[2] += 1

    def _get_transitions(self, indexes):
        """
        For each index direction, get False if in core or True if in transition zone.
        :param indexes: Location indexes in 1, 2, 3 directions.
        :return: Transition 1, 2, 3 directions.
        """
        return [(indexes[i] - self._box_counts[i]) > 0 for i in range(3)]

    def _set_coordinates_across(self, parameters, parameter_indexes, start_indexes, index_increments,
                                skip_start=False, skip_end=False, blend=False):
        """
        Insert parameters across the coordinates array.
        :param parameters: List of lists of N node parameters e.g. [px, pd1, pd2, pd3]
        :param parameter_indexes: Lists of parameter indexes where x=0, d1=1, d2=2, d3=3. Starts with first and
        advances at transitions change, then stays on the last. Can be negative to invert vector.
        e.g. [[0, 1, 2], [0, -2, 1]] for [x, d1, d2] then [x, -d2, d1] from first corner.
        :param start_indexes: Indexes into nx array for start point.
        :param index_increments: List of increments in indexes. Starts with first and uses next at each transition
        change, then stays on the last.
        :param skip_start: Set to True to skip the first value.
        :param skip_end: Set to True to skip the last value.
        :param blend: Set to True to blend parameters with any old parameters at locations.
        """
        indexes = start_indexes
        parameter_number = 0
        parameter_index = parameter_indexes[0]
        increment_number = 0
        index_increment = index_increments[0]
        start_n = 1 if skip_start else 0
        last_n = len(parameters[0]) - 1
        limit_n = len(parameters[0]) - (1 if skip_end else 0)
        last_trans = self._get_transitions(indexes)
        for n in range(start_n, limit_n):
            if n > 0:
                while True:
                    indexes = [indexes[c] + index_increment[c] for c in range(3)]
                    # skip over blank transition coordinates
                    if self._nx[indexes[2]][indexes[1]][indexes[0]]:
                        break
            trans = self._get_transitions(indexes)
            if last_trans and (trans != last_trans):
                if parameter_number < (len(parameter_indexes) - 1):
                    parameter_number += 1
                    parameter_index = parameter_indexes[parameter_number]
                if increment_number < (len(index_increments) - 1):
                    increment_number += 1
                    index_increment = index_increments[increment_number]
            nx = self._nx[indexes[2]][indexes[1]][indexes[0]]
            for parameter, spix in zip(parameters, parameter_index):
                if not parameter[n]:
                    continue
                new_parameter = [-d for d in parameter[n]] if (spix < 0) else copy.copy(parameter[n])
                pix = abs(spix)
                if blend and nx[pix]:
                    if pix == 0:
                        new_parameter = [0.5 * (nx[pix][c] + new_parameter[c]) for c in range(3)]
                    else:
                        # harmonic mean to cope with significant element size differences on boundary
                        new_parameter = linearlyInterpolateVectors(
                            nx[pix], new_parameter, 0.5, magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
                nx[pix] = new_parameter
            last_trans = trans

    def _smooth_derivative_across(self, start_indexes, end_indexes, index_increments, derivative_indexes,
                                  fix_start_direction=True, fix_end_direction=True):
        """
        Smooth derivatives across octant.
        :param start_indexes: Indexes of first point.
        :param end_indexes: Indexes of last point.
        :param index_increments: List of increments in indexes. Starts with first and advances at transitions change,
        then stays on the last.
        :param derivative_indexes: List of signed derivative parameter index to set along where 1=d1, 2=d2, 3=d3.
        Starts with first and advances at transitions change, then stays on the last. Can be negative to invert vector.
        e.g. [1, -2] for d1 then -d2 from first transition change.
        :param fix_start_direction: Set to True to keep the start direction but scale its magnitude.
        :param fix_end_direction: Set to True to keep the end direction but scale its magnitude.
        """
        indexes = start_indexes
        derivative_number = 0
        derivative_index = derivative_indexes[0]
        increment_number = 0
        index_increment = index_increments[0]
        indexes_list = []
        derivative_index_list = []
        px = []
        pd = []
        n = 0
        last_trans = self._get_transitions(indexes)
        while True:
            if n > 0:
                if indexes == end_indexes:
                    break
                while True:
                    indexes = [indexes[i] + index_increment[i] for i in range(3)]
                    # skip over blank coordinates in transition zone
                    if self._nx[indexes[2]][indexes[1]][indexes[0]]:
                        break
            trans = self._get_transitions(indexes)
            if last_trans and (trans != last_trans):
                if derivative_number < (len(derivative_indexes) - 1):
                    derivative_number += 1
                    derivative_index = derivative_indexes[derivative_number]
                if increment_number < (len(index_increments) - 1):
                    increment_number += 1
                    index_increment = index_increments[increment_number]
            parameters = self._nx[indexes[2]][indexes[1]][indexes[0]]
            x = parameters[0]
            indexes_list.append(copy.copy(indexes))
            spix = derivative_index
            derivative_index_list.append(spix)
            pix = abs(spix)
            if parameters[pix]:
                d = [-ad for ad in parameters[pix]] if (spix < 0) else parameters[pix]
            else:
                d = [0.0, 0.0, 0.0]
            px.append(x)
            pd.append(d)
            n += 1
            last_trans = trans
        sd = smoothCubicHermiteDerivativesLine(
            px, pd, fixStartDirection=fix_start_direction, fixEndDirection=fix_end_direction)
        for n in range(len(sd)):
            indexes = indexes_list[n]
            spix = derivative_index_list[n]
            new_derivative = [-d for d in sd[n]] if (spix < 0) else sd[n]
            pix = abs(spix)
            self._nx[indexes[2]][indexes[1]][indexes[0]][pix] = new_derivative

    def build_interior(self):
        """
        Determine interior coordinates from surface coordinates.
        """
        # determine 4-way point location from mean curves between side points linking to it
        point12 = self._nx[0][self._box_counts[1]][self._box_counts[0]]
        point13 = self._nx[self._box_counts[2]][0][self._box_counts[0]]
        point23 = self._nx[self._box_counts[2]][self._box_counts[1]][0]
        point123 = self._nx[self._axis_counts[2]][self._axis_counts[1]][self._axis_counts[0]]

        x_4way, d_4way = get_nway_point(
            [point23[0], point13[0], point12[0], point123[0]],
            [point23[1], point13[2], point12[3], [-d for d in point123[3]]],
            [self._box_counts[0], self._box_counts[1], self._box_counts[2], self._trans_count],
            sampleHermiteCurve, nway_d_factor=self._nway_d_factor)

        # smooth sample from sides to 3-way points using end derivatives
        min_weight = 1  # GRC revisit, remove?
        ax, ad1 = sampleHermiteCurve(
            point23[0], point23[1], None, x_4way, d_4way[0], None, self._box_counts[0],
            start_weight=self._box_counts[0] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
        bx, bd2 = sampleHermiteCurve(
            point13[0], point13[2], None, x_4way, d_4way[1], None, self._box_counts[1],
            start_weight=self._box_counts[1] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
        cx, cd3 = sampleHermiteCurve(
            point12[0], point12[3], None, x_4way, d_4way[2], None, self._box_counts[2],
            start_weight=self._box_counts[2] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
        tx, td3 = sampleHermiteCurve(
            point123[0], [-d for d in point123[3]], None, x_4way, d_4way[3], None, self._trans_count,
            start_weight=self._trans_count + min_weight, end_weight=1.0 + min_weight, end_transition=True)

        self._set_coordinates_across([ax, ad1], [[0, 1]], [0, self._box_counts[1], self._box_counts[2]], [[1, 0, 0]])
        self._set_coordinates_across([bx, bd2], [[0, 2]], [self._box_counts[0], 0, self._box_counts[2]], [[0, 1, 0]])
        self._set_coordinates_across([cx, cd3], [[0, 3]], [self._box_counts[0], self._box_counts[1], 0], [[0, 0, 1]])
        self._set_coordinates_across([tx, td3], [[0, -3]],
                                     [self._axis_counts[0], self._axis_counts[1], self._axis_counts[2]],
                                     [[-1, -1, -1]], skip_end=True)

        # sample up to 3-way lines connecting to 4-way point
        for n3 in range(1, self._box_counts[2]):
            point13 = self._nx[n3][0][self._box_counts[0]]
            point23 = self._nx[n3][self._box_counts[1]][0]
            point123 = self._nx[n3][self._axis_counts[1]][self._axis_counts[0]]
            point_3way = self._nx[n3][self._box_counts[1]][self._box_counts[0]]

            x_3way, d_3way = get_nway_point(
                [point23[0], point13[0], point123[0]],
                [point23[1], point13[2], [-d for d in point123[3]]],
                [self._box_counts[0], self._box_counts[1], self._trans_count],
                sampleHermiteCurve, prescribed_x_nway=point_3way[0], nway_d_factor=self._nway_d_factor)

            ax, ad1, ad2 = sampleHermiteCurve(
                point23[0], point23[1], point23[2], x_3way, d_3way[0], None, self._box_counts[0],
                start_weight=self._box_counts[0] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
            ad1[-1] = d_3way[0]
            ad2[-1] = d_3way[1]
            bx, bd2, bd1 = sampleHermiteCurve(
                point13[0], point13[2], point13[1], x_3way, d_3way[1], None, self._box_counts[1],
                start_weight=self._box_counts[1] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
            bd1[-1] = d_3way[0]
            bd2[-1] = d_3way[1]
            tx, td3, td1 = sampleHermiteCurve(
                point123[0], [-d for d in point123[3]], point123[1], x_3way, d_3way[2], None,
                self._trans_count, start_weight=self._trans_count + min_weight,
                end_weight=1.0 + min_weight, end_transition=True)

            self._set_coordinates_across([ax, ad1, ad2], [[0, 1, 2]], [0, self._box_counts[1], n3], [[1, 0, 0]])
            self._set_coordinates_across([bx, bd1, bd2], [[0, 1, 2]], [self._box_counts[0], 0, n3], [[0, 1, 0]])
            self._set_coordinates_across(
                [tx, td1, td3], [[0, 1, -3]], [self._axis_counts[0], self._axis_counts[1], n3], [[-1, -1, 0]],
                skip_end=True)

        for n2 in range(1, self._box_counts[1]):
            point12 = self._nx[0][n2][self._box_counts[0]]
            point23 = self._nx[self._box_counts[2]][n2][0]
            point123 = self._nx[self._axis_counts[2]][n2][self._axis_counts[0]]
            point_3way = self._nx[self._box_counts[2]][n2][self._box_counts[0]]

            x_3way, d_3way = get_nway_point(
                [point23[0], point12[0], point123[0]],
                [point23[1], point12[3], [-d for d in point123[3]]],
                [self._box_counts[0], self._box_counts[2], self._trans_count],
                sampleHermiteCurve, prescribed_x_nway=point_3way[0], nway_d_factor=self._nway_d_factor)

            ax, ad1, ad3 = sampleHermiteCurve(
                point23[0], point23[1], point23[3], x_3way, d_3way[0], None, self._box_counts[0],
                start_weight=self._box_counts[0] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
            ad1[-1] = d_3way[0]
            ad3[-1] = d_3way[1]
            bx, bd3, bd1 = sampleHermiteCurve(
                point12[0], point12[3], point12[1], x_3way, d_3way[1], None, self._box_counts[2],
                start_weight=self._box_counts[2] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
            bd1[-1] = d_3way[0]
            bd3[-1] = d_3way[1]
            tx, td3, td1 = sampleHermiteCurve(
                point123[0], [-d for d in point123[3]], point123[1], x_3way, d_3way[2], None,
                self._trans_count, start_weight=self._trans_count + min_weight,
                end_weight=1.0 + min_weight, end_transition=True)

            self._set_coordinates_across(
                [ax, ad1, ad3], [[0, 1, 3]], [0, n2, self._box_counts[2]], [[1, 0, 0]], blend=True)
            self._set_coordinates_across(
                [bx, bd1, bd3], [[0, 1, 3]], [self._box_counts[0], n2, 0], [[0, 0, 1]], blend=True)
            self._set_coordinates_across(
                [tx, td1, td3], [[0, 1, -3]], [self._axis_counts[0], n2, self._axis_counts[2]], [[-1, 0, -1]],
                skip_end=True, blend=True)

        for n1 in range(1, self._box_counts[0]):
            point12 = self._nx[0][self._box_counts[1]][n1]
            point13 = self._nx[self._box_counts[2]][0][n1]
            point123 = self._nx[self._axis_counts[2]][self._axis_counts[1]][n1]
            point_3way = self._nx[self._box_counts[2]][self._box_counts[1]][n1]

            x_3way, d_3way = get_nway_point(
                [point13[0], point12[0], point123[0]],
                [point13[2], point12[3], [-d for d in point123[3]]],
                [self._box_counts[0], self._box_counts[2], self._trans_count],
                sampleHermiteCurve, prescribed_x_nway=point_3way[0], nway_d_factor=self._nway_d_factor)

            ax, ad2, ad3 = sampleHermiteCurve(
                point13[0], point13[2], point13[3], x_3way, d_3way[0], None, self._box_counts[1],
                start_weight=self._box_counts[1] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
            ad2[-1] = d_3way[0]
            ad3[-1] = d_3way[1]
            bx, bd3, bd2 = sampleHermiteCurve(
                point12[0], point12[3], point12[2], x_3way, d_3way[1], None, self._box_counts[2],
                start_weight=self._box_counts[2] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
            bd2[-1] = d_3way[0]
            bd3[-1] = d_3way[1]
            tx, td3, td2 = sampleHermiteCurve(
                point123[0], [-d for d in point123[3]], point123[2], x_3way, d_3way[2], None,
                self._trans_count, start_weight=self._trans_count + min_weight,
                end_weight=1.0 + min_weight, end_transition=True)

            self._set_coordinates_across(
                [ax, ad2, ad3], [[0, 2, 3]], [n1, 0, self._box_counts[2]], [[0, 1, 0]], blend=True)
            self._set_coordinates_across(
                [bx, bd2, bd3], [[0, 2, 3]], [n1, self._box_counts[1], 0], [[0, 0, 1]], blend=True)
            self._set_coordinates_across(
                [tx, td2, td3], [[0, 2, -3]], [n1, self._axis_counts[1], self._axis_counts[2]],
                [[0, -1, -1]], skip_end=True, blend=True)

        for nt in range(1, self._trans_count):
            point12 = self._nx[0][self._axis_counts[1] - nt][self._axis_counts[0] - nt]
            point13 = self._nx[self._axis_counts[2] - nt][0][self._axis_counts[0] - nt]
            point23 = self._nx[self._axis_counts[2] - nt][self._axis_counts[1] - nt][0]
            point_3way = \
                self._nx[self._axis_counts[2] - nt][self._axis_counts[1] - nt][self._axis_counts[0] - nt]

            x_3way, d_3way = get_nway_point(
                [point23[0], point13[0], point12[0]],
                [point23[1], point13[2], point12[2]],
                [self._box_counts[0], self._box_counts[1], self._box_counts[2]],
                sampleHermiteCurve, prescribed_x_nway=point_3way[0], nway_d_factor=self._nway_d_factor)

            ax, ad1, ad2 = sampleHermiteCurve(
                point23[0], point23[1], point23[2], x_3way, d_3way[0], None, self._box_counts[0],
                start_weight=self._box_counts[0] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
            ad1[-1] = d_3way[0]
            ad2[-1] = d_3way[1]
            bx, bd2, bd1 = sampleHermiteCurve(
                point13[0], point13[2], point13[1], x_3way, d_3way[1], None, self._box_counts[1],
                start_weight=self._box_counts[1] + min_weight, end_weight=1.0 + min_weight, end_transition=True)
            bd1[-1] = d_3way[0]
            bd2[-1] = d_3way[1]
            cx, cd2, cd1 = sampleHermiteCurve(
                point12[0], point12[2], point12[1], x_3way, d_3way[2], None, self._box_counts[2],
                start_weight=self._box_counts[2] + min_weight, end_weight=1.0 + min_weight,
                end_transition=True)

            self._set_coordinates_across(
                [ax, ad1, ad2], [[0, 1, 2]], [0, self._axis_counts[1] - nt, self._axis_counts[2] - nt],
                [[1, 0, 0]], blend=True)
            self._set_coordinates_across(
                [bx, bd1, bd2], [[0, 1, 2]], [self._axis_counts[0] - nt, 0, self._axis_counts[2] - nt],
                [[0, 1, 0]], blend=True)
            self._set_coordinates_across(
                [cx, cd1, cd2], [[0, 1, 2]], [self._axis_counts[0] - nt, self._axis_counts[1] - nt, 0],
                [[0, 0, 1]], skip_end=True, blend=True)

        # average point coordinates across 3 directions between side faces and surfaces to 4 3-way lines.
        min_weight = 1  # GRC revisit, remove?
        # 1-direction
        for n2 in range(1, self._box_counts[1]):
            for n3 in range(1, self._box_counts[2]):
                start_indexes = [0, n2, n3]
                corner_indexes = [self._box_counts[0], n2, n3]
                end_indexes = [self._axis_counts[0], n2, n3]
                start = self._nx[start_indexes[2]][start_indexes[1]][start_indexes[0]]
                corner = self._nx[corner_indexes[2]][corner_indexes[1]][corner_indexes[0]]
                end = self._nx[end_indexes[2]][end_indexes[1]][end_indexes[0]]
                px, _ = sampleHermiteCurve(
                    start[0], start[1], None, corner[0], corner[1], None, self._box_counts[0],
                    start_weight=self._box_counts[0] + min_weight, end_weight=1.0 + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], start_indexes, [[1, 0, 0]], skip_start=True, skip_end=True, blend=True)
                px, _ = sampleHermiteCurve(
                    corner[0], corner[1], None, end[0], end[3], None, self._trans_count,
                    start_weight=1.0 + min_weight, end_weight=self._trans_count + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], corner_indexes, [[1, 0, 0]], skip_start=True, skip_end=True, blend=True)
            for nt in range(1, self._trans_count):
                start_indexes = [0, n2, self._box_counts[2] + nt]
                corner_indexes = [self._box_counts[0] + nt, n2, self._box_counts[2] + nt]
                end_indexes = [self._box_counts[0] + nt, n2, 0]
                start = self._nx[start_indexes[2]][start_indexes[1]][start_indexes[0]]
                corner = self._nx[corner_indexes[2]][corner_indexes[1]][corner_indexes[0]]
                end = self._nx[end_indexes[2]][end_indexes[1]][end_indexes[0]]
                px, _ = sampleHermiteCurve(
                    start[0], start[1], None, corner[0], corner[1], None, self._box_counts[0],
                    start_weight=self._box_counts[0] + min_weight, end_weight=1.0 + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], start_indexes, [[1, 0, 0]], skip_start=True, skip_end=True, blend=True)
                px, _ = sampleHermiteCurve(
                    corner[0], corner[1], None, end[0], [-d for d in end[2]], None, self._box_counts[2],
                    start_weight=1.0 + min_weight, end_weight=self._box_counts[2] + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], corner_indexes, [[0, 0, -1]], skip_start=True, skip_end=True, blend=True)
        # 2-direction
        for n3 in range(1, self._box_counts[2]):
            for n1 in range(1, self._box_counts[0]):
                start_indexes = [n1, 0, n3]
                corner_indexes = [n1, self._box_counts[1], n3]
                end_indexes = [n1, self._axis_counts[1], n3]
                start = self._nx[start_indexes[2]][start_indexes[1]][start_indexes[0]]
                corner = self._nx[corner_indexes[2]][corner_indexes[1]][corner_indexes[0]]
                end = self._nx[end_indexes[2]][end_indexes[1]][end_indexes[0]]
                px, _ = sampleHermiteCurve(
                    start[0], start[2], None, corner[0], corner[2], None, self._box_counts[1],
                    start_weight=self._box_counts[1] + min_weight, end_weight=1.0 + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], start_indexes, [[0, 1, 0]], skip_start=True, skip_end=True, blend=True)
                px, _ = sampleHermiteCurve(
                    corner[0], corner[2], None, end[0], end[3], None, self._trans_count,
                    start_weight=1.0 + min_weight, end_weight=self._trans_count + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], corner_indexes, [[0, 1, 0]], skip_start=True, skip_end=True, blend=True)
            for nt in range(1, self._trans_count):
                start_indexes = [self._box_counts[0] + nt, 0, n3]
                corner_indexes = [self._box_counts[0] + nt, self._box_counts[1] + nt, n3]
                end_indexes = [0, self._box_counts[1] + nt, n3]
                start = self._nx[start_indexes[2]][start_indexes[1]][start_indexes[0]]
                corner = self._nx[corner_indexes[2]][corner_indexes[1]][corner_indexes[0]]
                end = self._nx[end_indexes[2]][end_indexes[1]][end_indexes[0]]
                px, _ = sampleHermiteCurve(
                    start[0], start[1], None, corner[0], corner[1], None, self._box_counts[1],
                    start_weight=self._box_counts[1] + min_weight, end_weight=1.0 + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], start_indexes, [[0, 1, 0]], skip_start=True, skip_end=True, blend=True)
                px, _ = sampleHermiteCurve(
                    corner[0], corner[1], None, end[0], end[1], None, self._box_counts[0],
                    start_weight=1.0 + min_weight, end_weight=self._box_counts[0] + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], corner_indexes, [[-1, 0, 0]], skip_start=True, skip_end=True, blend=True)
        # 3-direction
        for n1 in range(1, self._box_counts[0]):
            for n2 in range(1, self._box_counts[1]):
                start_indexes = [n1, n2, 0]
                corner_indexes = [n1, n2, self._box_counts[2]]
                end_indexes = [n1, n2, self._axis_counts[2]]
                start = self._nx[start_indexes[2]][start_indexes[1]][start_indexes[0]]
                corner = self._nx[corner_indexes[2]][corner_indexes[1]][corner_indexes[0]]
                end = self._nx[end_indexes[2]][end_indexes[1]][end_indexes[0]]
                px, _ = sampleHermiteCurve(
                    start[0], start[3], None, corner[0], corner[3], None, self._box_counts[2],
                    start_weight=self._box_counts[2] + min_weight, end_weight=1.0 + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], start_indexes, [[0, 0, 1]], skip_start=True, skip_end=True, blend=True)
                px, _ = sampleHermiteCurve(
                    corner[0], corner[3], None, end[0], end[3], None, self._trans_count,
                    start_weight=1.0 + min_weight, end_weight=self._trans_count + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], corner_indexes, [[0, 0, 1]], skip_start=True, skip_end=True, blend=True)
            for nt in range(1, self._trans_count):
                start_indexes = [n1, self._box_counts[1] + nt, 0]
                corner_indexes = [n1, self._box_counts[1] + nt, self._box_counts[2] + nt]
                end_indexes = [n1, 0, self._box_counts[2] + nt]
                start = self._nx[start_indexes[2]][start_indexes[1]][start_indexes[0]]
                corner = self._nx[corner_indexes[2]][corner_indexes[1]][corner_indexes[0]]
                end = self._nx[end_indexes[2]][end_indexes[1]][end_indexes[0]]
                px, _ = sampleHermiteCurve(
                    start[0], start[2], None, corner[0], [-d for d in corner[2]], None, self._box_counts[2],
                    start_weight=self._box_counts[2] + min_weight, end_weight=1.0 + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], start_indexes, [[0, 0, 1]], skip_start=True, skip_end=True, blend=True)
                px, _ = sampleHermiteCurve(
                    corner[0], [-d for d in corner[2]], None, end[0], [-d for d in end[2]], None, self._box_counts[1],
                    start_weight=1.0 + min_weight, end_weight=self._box_counts[1] + min_weight)
                self._set_coordinates_across(
                    [px], [[0]], corner_indexes, [[0, -1, 0]], skip_start=True, skip_end=True, blend=True)

        # smooth 1-direction
        for n2 in range(1, self._box_counts[1]):
            for n3 in range(1, self._box_counts[2]):
                self._smooth_derivative_across(
                    [0, n2, n3], [self._axis_counts[0], n2, n3],
                    [[1, 0, 0]], [1, 3], fix_start_direction=True, fix_end_direction=True)
            for nt in range(1, self._trans_count):
                self._smooth_derivative_across(
                    [0, n2, self._box_counts[2] + nt], [self._box_counts[0] + nt, n2, 0],
                    [[1, 0, 0], [0, 0, -1]], [1, 1, -2], fix_start_direction=True, fix_end_direction=True)
        # smooth 2-direction
        for n3 in range(1, self._box_counts[2]):
            for n1 in range(1, self._box_counts[0]):
                self._smooth_derivative_across(
                    [n1, 0, n3], [n1, self._axis_counts[1], n3],
                    [[0, 1, 0]], [2, 3], fix_start_direction=True, fix_end_direction=True)
            for nt in range(1, self._trans_count):
                self._smooth_derivative_across(
                    [self._box_counts[0] + nt, 0, n3], [0, self._box_counts[1] + nt, n3],
                    [[0, 1, 0], [-1, 0, 0]], [1], fix_start_direction=True, fix_end_direction=True)
        # smooth 3-direction
        for n1 in range(1, self._box_counts[0]):
            for n2 in range(1, self._box_counts[1]):
                self._smooth_derivative_across(
                    [n1, n2, 0], [n1, n2, self._axis_counts[2]],
                    [[0, 0, 1]], [3], fix_start_direction=True, fix_end_direction=True)
            for nt in range(1, self._trans_count):
                self._smooth_derivative_across(
                    [n1, self._box_counts[1] + nt, 0], [n1, 0, self._box_counts[2] + nt],
                    [[0, 0, 1], [0, -1, 0]], [2, -2], fix_start_direction=True, fix_end_direction=True)

    def mirror_yz(self):
        """
        Mirror coordinates and derivatives about both y = 0 and z = 0 planes.
        """
        for n3 in range(self._axis_counts[2] + 1):
            nx_layer = self._nx[n3]
            for n2 in range(self._axis_counts[1] + 1):
                nx_row = nx_layer[n2]
                for n1 in range(self._axis_counts[0] + 1):
                    nx = nx_row[n1]
                    if nx:
                        for d in nx:
                            if d:
                                d[1] = -d[1]
                                d[2] = -d[2]
