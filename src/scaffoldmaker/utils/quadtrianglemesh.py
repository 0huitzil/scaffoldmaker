"""
Utilities for building 2-D triangle-shaped meshes out of quad elements with cubic Hermite serendipity interpolation.
"""
from scaffoldmaker.utils.interpolation import (
    DerivativeScalingMode, get_nway_point, linearlyInterpolateVectors, smoothCubicHermiteDerivativesLine)
import copy


class QuadTriangleMesh:
    """
    Generates a triangular mesh for interpolating with c1-continuous quad elements, with a 3-way point in the middle.
    3 corner points and 3 side points are on the triangle as follows:
             3
            / \
          13  23
         /      \
        1---12---2
    Derivatives at point 1 are d1 from 1-2, d2 from 1-3.
    Derivatives at point 2 are d1 from 1-2, d2 from 2-3.
    Derivatives at point 3 are d1 from 3-1, d2 from 3-2.
    Side point12 is box_count2 from point1, box_count1 from point2.
    Side point13 is box_count3 from point1, box_count1 from point3.
    Side point23 is box_count3 from point2, box_count2 from point3.
    Parameters on 12, 13 rows to 3-way point use derivatives as for point3.
    """

    def __init__(self, box_count1, box_count2, box_count3, sample_curve,
                 move_x_to_surface=None, move_d_to_surface=None, nway_d_factor=0.6):
        """
        :param box_count1: Number of elements from opposite points to side points by point1.
        :param box_count2: Number of elements from opposite points to side points by point2.
        :param box_count3: Number of elements from opposite points to side points by point3.
        :param sample_curve: Callable sampling a curve between 2 edge points.
        :param move_x_to_surface: Optional callable taking (x) and moving it to a surface.
        :param move_d_to_surface: Optional callable taking (x, d) and adjusting d to be tangential to a surface.
        :param nway_d_factor: Value, normally from 0.5 to 1.0 giving n-way derivative magnitude as a proportion
        of the minimum regular magnitude sampled to the n-way point. This reflects that distances from the mid-side
        of a triangle to the centre are shorter, so the derivative in the middle must be smaller.
        """
        self._box_count1 = box_count1
        self._box_count2 = box_count2
        self._box_count3 = box_count3
        self._sample_curve = sample_curve
        self._move_x_to_surface = move_x_to_surface
        self._move_d_to_surface = move_d_to_surface
        self._3_way_d_factor = nway_d_factor
        self._element_count12 = box_count1 + box_count2
        self._element_count13 = box_count1 + box_count3
        self._element_count23 = box_count2 + box_count3
        self._node_count12 = self._element_count12 + 1
        self._node_count13 = self._element_count13 + 1
        self._node_count23 = self._element_count23 + 1
        none_parameters = [None] * 4  # x, d1, d2
        self._nx = []
        for n13 in range(self._node_count13):
            nx_row = []
            # s = ""
            for n12 in range(self._node_count12):
                if ((n12 < box_count2) or (n13 < box_count3) or
                        ((n12 - box_count2) == (n13 - box_count3))):
                    parameters = copy.copy(none_parameters)
                    # s += "[]"
                else:
                    parameters = None
                    # s += "  "
                nx_row.append(parameters)
            # print(s)
            self._nx.append(nx_row)

    def get_element_count12(self):
        return self._element_count12

    def get_element_count13(self):
        return self._element_count13

    def get_element_count23(self):
        return self._element_count23

    def set_edge_parameters12(self, px, pd1, pd2, pd3=None):
        """
        Set parameters along 1-2 edge of triangle.
        :param px: Coordinates x.
        :param pd1: Derivatives in direction from point1 towards point2.
        :param pd2: Derivatives in direction towards point3.
        :param pd3: Optional derivatives in 3rd direction.
        """
        assert len(px) == self._node_count12
        for n12 in range(self._node_count12):
            d3 = copy.copy(pd3[n12]) if pd3 else None
            self._nx[0][n12] = [copy.copy(px[n12]), copy.copy(pd1[n12]), copy.copy(pd2[n12]), d3]

    def get_edge_parameters12(self):
        """
        Get parameters along 1-2 edge of triangle, as would be passed into set_edge_parameters12().
        :return: px, pd1, pd2, pd3
        """
        px = []
        pd1 = []
        pd2 = []
        pd3 = []
        for n12 in range(self._node_count12):
            x, d1, d2, d3 = self._nx[0][n12]
            px.append(x)
            pd1.append(d1)
            pd2.append(d2)
            pd3.append(d3)
        return copy.deepcopy((px, pd1, pd2, pd3))

    def set_edge_parameters13(self, px, pd1, pd2, pd3=None):
        """
        Set parameters along 1-3 edge of triangle.
        :param px: Coordinates x.
        :param pd1: Derivatives in direction towards point2.
        :param pd2: Derivatives in direction from point1 towards point3.
        :param pd3: Optional derivatives in 3rd direction.
        """
        assert len(px) == self._node_count13
        for n13 in range(self._node_count13):
            d3 = copy.copy(pd3[n13]) if pd3 else None
            if n13 < self._box_count3:
                parameters = [copy.copy(px[n13]), copy.copy(pd1[n13]), copy.copy(pd2[n13]), d3]
            else:
                parameters = [copy.copy(px[n13]), [-d for d in pd2[n13]], copy.copy(pd1[n13]), d3]
            self._nx[n13][0] = parameters

    def get_edge_parameters13(self):
        """
        Get parameters along 1-3 edge of triangle, as would be passed into set_edge_parameters13().
        :return: px, pd1, pd2, pd3
        """
        px = []
        pd1 = []
        pd2 = []
        pd3 = []
        for n13 in range(self._node_count13):
            x, d1, d2, d3 = self._nx[n13][0]
            if n13 >= self._box_count3:
                d1, d2 = d2, [-d for d in d1]
            px.append(x)
            pd1.append(d1)
            pd2.append(d2)
            pd3.append(d3)
        return copy.deepcopy((px, pd1, pd2, pd3))

    def set_edge_parameters23(self, px, pd1, pd2, pd3=None):
        """
        Set parameters along 123 edge of triangle.
        :param px: Coordinates x.
        :param pd1: Derivatives in direction away from point1.
        :param pd2: Derivatives in direction from point2 towards point3.
        :param pd3: Optional derivatives in 3rd direction.
        """
        assert len(px) == self._node_count23
        for n23 in range(self._node_count23):
            d3 = copy.copy(pd3[n23]) if pd3 else None
            if n23 < self._box_count3:
                parameters = [copy.copy(px[n23]), copy.copy(pd1[n23]), copy.copy(pd2[n23]), d3]
                n13 = n23
                n12 = self._element_count12
            else:
                parameters = [copy.copy(px[n23]), [-d for d in pd1[n23]], [-d for d in pd2[n23]], d3]
                n13 = self._element_count13
                n12 = (self._element_count12 if n23 == self._box_count3 else
                       self._box_count2 - (n23 - self._box_count3))
            self._nx[n13][n12] = parameters

    def get_edge_parameters23(self):
        """
        Get parameters along 2-3 edge of triangle, as would be passed into set_edge_parameters23().
        :return: px, pd1, pd2, pd3
        """
        px = []
        pd1 = []
        pd2 = []
        pd3 = []
        for n23 in range(self._node_count23):
            if n23 < self._box_count3:
                n13 = n23
                n12 = self._element_count12
            else:
                n13 = self._element_count13
                n12 = (self._element_count12 if n23 == self._box_count3 else
                       self._box_count2 - (n23 - self._box_count3))
            x, d1, d2, d3 = self._nx[n13][n12]
            if n23 >= self._box_count3:
                d1, d2 = [-d for d in d1], [-d for d in d2]
            px.append(x)
            pd1.append(d1)
            pd2.append(d2)
            pd3.append(d3)
        return copy.deepcopy((px, pd1, pd2, pd3))

    def get_parameters12(self, n13, count=None):
        """
        Get parameters along 1-2 direction of triangle.
        :param n13: Index from 0 at point1 toward point3, up to element_count13.
        :param count: Optional lower number to get, or None for all in row.
        :return: px, pd1, pd2, pd3
        """
        assert 0 <= n13 < self._node_count13
        n_limit = count if count else self._node_count12
        assert 1 <= n_limit <= self._node_count12
        px = []
        pd1 = []
        pd2 = []
        pd3 = []
        nx_row = self._nx[n13]
        n12 = 0
        for n in range(n_limit):
            while not nx_row[n12]:
                n12 += 1
            nx = nx_row[n12]
            px.append(nx[0])
            pd1.append(nx[1])
            pd2.append(nx[2])
            pd3.append(nx[3])
            n12 += 1
        return px, pd1, pd2, pd3

    def get_parameters31(self, n12, count=None):
        """
        Get parameters along 3-1 direction of triangle.
        :param n12: Index from 0 at point3 toward point2, up to element_count12.
        :param count: Optional lower number to get, or None for all in row.
        :return: px, pd1, pd2, pd3
        """
        assert 0 <= n12 < self._node_count12
        n_limit = count if count else self._node_count13
        assert 1 <= n_limit <= self._node_count13
        px = []
        pd1 = []
        pd2 = []
        pd3 = []
        n13 = self._element_count13
        for n in range(n_limit):
            nx_row = self._nx[n13]
            while not nx_row[n12]:
                n13 -= 1
            nx = nx_row[n12]
            px.append(nx[0])
            pd1.append(nx[1])
            pd2.append(nx[2])
            pd3.append(nx[3])
            n13 -= 1
        return px, pd1, pd2, pd3

    def get_parameters_diagonal(self):
        """
        Get parameters along diagonal from point23 to 3-way point
        :return: px, pd1, pd2, pd3
        """
        px = []
        pd1 = []
        pd2 = []
        pd3 = []
        for n in range(self._box_count1 + 1):
            n13 = self._element_count13 - n
            n12 = self._element_count12 - n
            nx = self._nx[n13][n12]
            px.append(nx[0])
            pd1.append(nx[1])
            pd2.append(nx[2])
            pd3.append(nx[3])
        return px, pd1, pd2, pd3

    def _get_corner(self, indexes):
        """
        Get integer indicating which corner if any the indexes are on.
        :param indexes: Index across 1-2 direction, index across 1-3 direction.
        :return: 0 if not a corner, 1 if 2-3 corner, 2 if 1-3 corner, 3 if 1-2 corner, 4 if 1-2-3 corner.
        """
        n12, n13 = indexes
        if n12 < self._box_count2:
            if n13 == self._box_count3:
                return 2
        elif n12 == self._box_count2:
            if n13 < self._box_count3:
                return 3
            elif n13 == self._box_count3:
                return 4
        elif (n12 - self._box_count2) == (n13 - self._box_count3):
            return 1
        return 0

    def _set_coordinates_across(self, parameters, parameter_indexes, start_indexes, index_increments,
                                skip_start=False, skip_end=False, blend=False):
        """
        Insert parameters in a row across triangle into the coordinates array.
        :param parameters: List of lists of N node parameters e.g. [px, pd1, pd2]
        :param parameter_indexes: Lists of parameter indexes where 0=x, 1=d1, 2=d2, 3=d3. Starts with first and
        advances at a 3-way line, then cycles back to first. Can be negative to invert vector.
        e.g. [[0, 1, 2], [0, -2, 1]] for [x, d1, d2] then [x, -d2, d1] from 3-way line.
        :param start_indexes: Index of first point, a list of 2-integers within the range of self._nx.
        :param index_increments: List of increments in indexes. Starts with first and moves to next at each 3-way line,
        then cycles back to first.
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
        limit_n = len(parameters[0]) - (1 if skip_end else 0)
        for n in range(start_n, limit_n):
            if n > 0:
                while True:
                    indexes = [indexes[0] + index_increment[0], indexes[1] + index_increment[1]]
                    # skip over blank coordinates around 3-way line
                    if self._nx[indexes[1]][indexes[0]]:
                        break
            corner = self._get_corner(indexes)
            if corner and (
                    corner != self._get_corner([indexes[0] + index_increment[0],  indexes[1] + index_increment[1]])):
                parameter_number += 1
                if parameter_number == len(parameter_indexes):
                    parameter_number = 0
                parameter_index = parameter_indexes[parameter_number]
                increment_number += 1
                if increment_number == len(index_increments):
                    increment_number = 0
                index_increment = index_increments[increment_number]
            nx = self._nx[indexes[1]][indexes[0]]
            for parameter, spix in zip(parameters, parameter_index):
                new_parameter = [-d for d in parameter[n]] if (spix < 0) else copy.copy(parameter[n])
                pix = abs(spix)
                if blend and nx[pix]:
                    if pix == 0:
                        # for fairness, move to surface before blending
                        if self._move_x_to_surface:
                            new_parameter = self._move_x_to_surface(new_parameter)
                        new_parameter = [0.5 * (nx[pix][c] + new_parameter[c]) for c in range(3)]
                    else:
                        # harmonic mean to cope with significant element size differences on boundary
                        new_parameter = linearlyInterpolateVectors(
                            nx[pix], new_parameter, 0.5, magnitudeScalingMode=DerivativeScalingMode.HARMONIC_MEAN)
                nx[pix] = new_parameter

    def _smooth_derivative_across(self, start_indexes, end_indexes, index_increments, derivative_indexes,
                                  fix_start_direction=True, fix_end_direction=True):
        """
        Smooth derivatives across triangle.
        :param start_indexes: Indexes of first point.
        :param end_indexes: Indexes of last point.
        :param index_increments: List of increments in indexes. Starts with first and advances at each corner, then
        cycles back to first.
        :param derivative_indexes: List of signed derivative parameter index to set along where 1=d1, 2=d2, 3=d3.
        Starts with first and advances at each corner, then cycles back to first. Can be negative to invert vector.
        e.g. [1, -2] for d1 then -d2 from first corner.
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
        while True:
            if n > 0:
                if indexes == end_indexes:
                    break
                while True:
                    indexes = [indexes[0] + index_increment[0], indexes[1] + index_increment[1]]
                    # skip over blank coordinates around 3-way line
                    if self._nx[indexes[1]][indexes[0]]:
                        break
            corner = self._get_corner(indexes)
            if corner and (
                    corner != self._get_corner([indexes[0] + index_increment[0],  indexes[1] + index_increment[1]])):
                derivative_number += 1
                if derivative_number == len(derivative_indexes):
                    derivative_number = 0
                derivative_index = derivative_indexes[derivative_number]
                increment_number += 1
                if increment_number == len(index_increments):
                    increment_number = 0
                index_increment = index_increments[increment_number]
            parameters = self._nx[indexes[1]][indexes[0]]
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
        sd = smoothCubicHermiteDerivativesLine(
            px, pd, fixStartDirection=fix_start_direction, fixEndDirection=fix_end_direction)
        if self._move_d_to_surface:
            for n in range(1, len(sd) - 1):
                sd[n] = self._move_d_to_surface(px[n], sd[n])
            sd = smoothCubicHermiteDerivativesLine(px, sd, fixAllDirections=True)
        for n in range(len(sd)):
            indexes = indexes_list[n]
            spix = derivative_index_list[n]
            new_derivative = [-d for d in sd[n]] if (spix < 0) else sd[n]
            pix = abs(spix)
            self._nx[indexes[1]][indexes[0]][pix] = new_derivative

    def build(self, regular_count2=0):
        """
        Determine interior coordinates from edge coordinates.
        :param regular_count2: Optional number of rows of regular box elements in 2 direction.
        """
        assert 0 <= regular_count2 <= (self._box_count2 - 1)
        if regular_count2:
            # sample regular row on common boundary with triangle
            pointr1 = self._nx[0][regular_count2]
            pointr2 = self._nx[self._element_count13][regular_count2]
            rx, rd2, rd1 = self._sample_curve(
                pointr1[0], pointr1[2], pointr1[1],
                pointr2[0], [-d for d in pointr2[1]], pointr2[2],
                self._element_count13)
            self._set_coordinates_across([rx, rd1, rd2], [[0, 1, 2], [0, 2, 1]], [regular_count2, 0], [[0, 1]],
                                         skip_start=True, skip_end=True)
            # sample point coordinates of regular rows parallel to triangle
            for r in range(1, regular_count2):
                pointr1 = self._nx[0][r]
                pointr2 = self._nx[self._element_count13][r]
                px, _ = self._sample_curve(
                    pointr1[0], pointr1[2], None, pointr2[0], [-d for d in pointr2[1]], None, self._element_count13)
                self._set_coordinates_across([px], [[0]], [r, 0], [[0, 1]], skip_start=True, skip_end=True)

        # determine 3-way point location from mean curves between side points linking to it
        point12 = self._nx[0][self._box_count2]
        point13 = self._nx[self._box_count3][regular_count2]
        point23 = self._nx[self._element_count13][self._element_count12]
        x_3way, d_3way = get_nway_point(
            [point23[0], point13[0], point12[0]],
            [point23[1], point13[2], point12[2]],
            [self._box_count1, self._box_count2 - regular_count2, self._box_count3],
            self._sample_curve, self._move_x_to_surface,
            nway_d_factor=self._3_way_d_factor)

        # smooth sample from sides to 3-way points using end derivatives
        min_weight = 1.0  # arbitrary but looks good
        ax, ad1, ad2 = self._sample_curve(
            point23[0], point23[1], point23[2], x_3way, d_3way[0], None, self._box_count1,
            start_weight=self._box_count1 + min_weight, end_weight=1.0 + min_weight, end_transition=True)
        bx, bd2, bd1 = self._sample_curve(
            point13[0], point13[2], point13[1], x_3way, d_3way[1], None, self._box_count2 - regular_count2,
            start_weight=self._box_count2 - regular_count2 + min_weight, end_weight=1.0 + min_weight,
            end_transition=True)
        cx, cd2, cd1 = self._sample_curve(
            point12[0], point12[2], point12[1], x_3way, d_3way[2], None, self._box_count3,
            start_weight=self._box_count3 + min_weight, end_weight=1.0 + min_weight, end_transition=True)
        ad2[-1] = bd2[-1]
        bd1[-1] = ad1[-1]
        self._set_coordinates_across([ax, ad1, ad2], [[0, 1, 2]], [self._element_count12, self._element_count13],
                                     [[-1, -1]])
        self._set_coordinates_across([bx, bd1, bd2], [[0, 1, 2]], [regular_count2, self._box_count3], [[1, 0]])
        self._set_coordinates_across([cx, cd1, cd2], [[0, 1, 2]], [self._box_count2, 0], [[0, 1]], skip_end=True)

        # average point coordinates across 2 directions between triangle edges and 3-way lines.
        # 1-2 curves
        min_weight = 1.0  # arbitrary but looks good
        start_indexes = [regular_count2, 0]
        corner_indexes = [self._box_count2, 0]
        end_indexes = [self._element_count12, 0]
        for i in range(1, self._box_count3):
            start_indexes[1] += 1
            corner_indexes[1] += 1
            end_indexes[1] += 1
            start = self._nx[start_indexes[1]][start_indexes[0]]
            corner = self._nx[corner_indexes[1]][corner_indexes[0]]
            end = self._nx[end_indexes[1]][end_indexes[0]]
            px, _ = self._sample_curve(
                start[0], start[1], None, corner[0], corner[1], None, self._box_count2 - regular_count2,
                start_weight=self._box_count2 - regular_count2 + min_weight, end_weight=1.0 + min_weight)
            self._set_coordinates_across(
                [px], [[0]], start_indexes, [[1, 0]], skip_start=True, skip_end=True)
            px, _ = self._sample_curve(
                corner[0], corner[1], None, end[0], end[1], None, self._box_count1,
                start_weight=1.0 + min_weight, end_weight=self._box_count1 + min_weight)
            self._set_coordinates_across(
                [px], [[0]], corner_indexes, [[1, 0]], skip_start=True, skip_end=True)
        # 1-3 curves
        start_indexes = [regular_count2, 0]
        corner_indexes = [regular_count2, self._box_count3]
        end_indexes = [regular_count2, self._element_count13]
        for i in range(regular_count2 + 1, self._box_count2):
            start_indexes[0] += 1
            corner_indexes[0] += 1
            end_indexes[0] += 1
            start = self._nx[start_indexes[1]][start_indexes[0]]
            corner = self._nx[corner_indexes[1]][corner_indexes[0]]
            end = self._nx[end_indexes[1]][end_indexes[0]]
            px, _ = self._sample_curve(
                start[0], start[2], None, corner[0], [-d for d in corner[1]], None, self._box_count3,
                start_weight=self._box_count3 + min_weight, end_weight=1.0 + min_weight)
            self._set_coordinates_across(
                [px], [[0]], start_indexes, [[0, 1]], skip_start=True, skip_end=True, blend=True)
            px, _ = self._sample_curve(
                corner[0], [-d for d in corner[1]], None, end[0], [-d for d in end[1]], None, self._box_count1,
                    start_weight=1.0 + min_weight, end_weight=self._box_count1 + min_weight)
            self._set_coordinates_across(
                [px], [[0]], corner_indexes, [[0, 1]], skip_start=True, skip_end=True, blend=True)
        # 2-3 curves
        start_indexes = [self._element_count12, 0]
        corner_indexes = [self._element_count12, self._element_count13]
        end_indexes = [regular_count2, self._element_count13]
        for i in range(1, self._box_count1):
            start_indexes[0] -= 1
            corner_indexes[0] -= 1
            corner_indexes[1] -= 1
            end_indexes[1] -= 1
            start = self._nx[start_indexes[1]][start_indexes[0]]
            corner = self._nx[corner_indexes[1]][corner_indexes[0]]
            end = self._nx[end_indexes[1]][end_indexes[0]]
            px, _ = self._sample_curve(
                start[0], start[2], None, corner[0], [-d for d in corner[2]], None, self._box_count3,
                start_weight=self._box_count3 + min_weight, end_weight=1.0 + min_weight)
            self._set_coordinates_across(
                [px], [[0]], start_indexes, [[0, 1]], skip_start=True, skip_end=True, blend=True)
            px, _ = self._sample_curve(
                corner[0], [-d for d in corner[2]], None, end[0], [-d for d in end[2]], None, self._box_count2 - regular_count2,
                start_weight=1.0 + min_weight, end_weight=self._box_count2 - regular_count2 + min_weight)
            self._set_coordinates_across(
                [px], [[0]], corner_indexes, [[-1, 0]], skip_start=True, skip_end=True, blend=True)

        # smooth 1-2 curves
        start_indexes = [0, 0]
        end_indexes = [self._element_count12, 0]
        for i in range(1, self._box_count3):
            start_indexes[1] += 1
            end_indexes[1] += 1
            self._smooth_derivative_across(start_indexes, end_indexes, [[1, 0]], [1],
                fix_start_direction=True, fix_end_direction=True)
        if regular_count2:
            # need to smooth d2 through regular elements to 3-way point
            start_indexes[1] += 1
            end_indexes[0] = regular_count2
            end_indexes[1] += 1
            self._smooth_derivative_across(start_indexes, end_indexes, [[1, 0]], [2],
                fix_start_direction=True, fix_end_direction=True)
        # smooth 1-3 curves
        start_indexes = [0, 0]
        end_indexes = [0, self._element_count13]
        for i in range(1, self._box_count2):
            start_indexes[0] += 1
            end_indexes[0] += 1
            self._smooth_derivative_across(start_indexes, end_indexes, [[0, 1]], [2, -1],
                fix_start_direction=True, fix_end_direction=True)
        # smooth 2-3 curves
        start_indexes = [self._element_count12, 0]
        end_indexes = [0, self._element_count13]
        for i in range(1, self._box_count1):
            start_indexes[0] -= 1
            end_indexes[1] -= 1
            self._smooth_derivative_across(start_indexes, end_indexes, [[0, 1], [-1, 0]], [2, -2],
                fix_start_direction=True, fix_end_direction=True)

    def assign_d3(self, evaluate_d3):
        """
        Assigns value of d3 for all coordinates with a location, from value given by user-supplied function.
        Does not assign d3 if it is already set at a point.
        :param evaluate_d3: Callable taking current values of x, d1, d2 and returning d3.
        """
        for n13 in range(self._node_count13):
            nx_row = self._nx[n13]
            for n12 in range(self._node_count12):
                nx = nx_row[n12]
                if nx and nx[0] and not nx[3]:
                    nx[3] = evaluate_d3(nx[0], nx[1], nx[2])
