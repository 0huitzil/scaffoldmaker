"""
Octree for searching for objects by coordinates
"""
from __future__ import division

import copy
import math


class Octree:
    """
    Octree for searching for objects by coordinates
    """

    def __init__(self, minimums, maximums, tolerance = None):
        """
        :param minimums: List of 3 minimum coordinate values. Caller to include any edge allowance.
        :param maximums: List of 3 maximum coordinate values. Caller to include any edge allowance.
        :param tolerance: If supplied, tolerance to use, or None to compute as 1.0E-6*diagonal.
        """
        self._dimension = 3
        self._dimensionPower2 = 1 << self._dimension
        self._maxObjects = 20
        assert len(minimums) == self._dimension, 'Octree minimums is invalid length'
        assert len(maximums) == self._dimension, 'Octree maximums is invalid length'
        if tolerance is None:
            self._tolerance = 1.0E-6*math.sqrt(sum(((maximums[i] - minimums[i])*(maximums[i] - minimums[i])) for i in range(self._dimension)))
        else:
            self._tolerance = tolerance
        self._minimums = copy.deepcopy(minimums)
        self._maximums = copy.deepcopy(maximums)
        # Octree is either leaf with _coordinatesObjects, or has 2**self._dimension children
        self._coordinatesObjects = []
        # exactly 2^self._dimension children, cycling in lowest x index fastest
        self._children = None

    def _findObjectByCoordinates(self, x, extra_data):
        """
        Find closest existing object with |x - ox| < tolerance.
        :param x: 3 coordinates in a list.
        :param extra_data: Extra data to compare with 2nd component of stored tuple (object, extra_data) or None if
        not a tuple with extra data. Value must resolve to True, or be None.
        :return: nearest distance, nearest object or None, None if none found.
        """
        nearestDistance = None
        nearestObject = (None, None) if extra_data else None
        if self._coordinatesObjects is not None:
            for coordinatesObject in self._coordinatesObjects:
                # cheaply determine if in 2*tolerance sized box around object
                cox = coordinatesObject[0]
                for c in range(self._dimension):
                    if math.fabs(x[c] - cox[c]) > self._tolerance:
                        break
                else:
                    if extra_data:
                        if extra_data != coordinatesObject[1][1]:
                            continue  # extra data does not match
                    # now test exact distance
                    distance = math.sqrt(sum((x[c] - cox[c])*(x[c] - cox[c]) for c in range(self._dimension)))
                    if (distance < self._tolerance) and ((nearestDistance is None) or (distance < nearestDistance)):
                        nearestDistance = distance
                        nearestObject = coordinatesObject[1]
        else:
            centre = self._children[0]._maximums
            for i in range(self._dimensionPower2):
                inBoundsPlusTolerance = True
                for c in range(self._dimension):
                    if i & (1 << c):
                        if x[c] < (centre[c] - self._tolerance):
                            inBoundsPlusTolerance = False
                            break
                    elif x[c] > (centre[c] + self._tolerance):
                        inBoundsPlusTolerance = False
                        break
                if inBoundsPlusTolerance:
                    distance, obj = self._children[i]._findObjectByCoordinates(x, extra_data)
                    if (distance is not None) and ((nearestDistance is None) or (distance < nearestDistance)):
                        nearestDistance = distance
                        nearestObject = obj
        return nearestDistance, nearestObject

    def findObjectByCoordinates(self, x, extra_data=None):
        """
        Find closest existing object with |x - ox| < tolerance.
        :param x: 3 coordinates in a list.
        :param extra_data: Optional extra data to compare with 2nd component of stored tuple (object, extra_data).
        Default/None means no tuple, no extra data.
        :return: nearest object (or object tuple) or None (or (None, None) if extra_data) if not found.
        """
        nearestDistance, nearestObject = self._findObjectByCoordinates(x, extra_data)
        return nearestObject


    def addObjectAtCoordinates(self, x, obj):
        """
        Add object at coordinates to octree.
        Caller must have received None result for findObjectByCoordinates() first!
        Assumes caller has verified x is within range of Octree.
        :param x: 3 coordinates in a list.
        :param obj: object to store with coordinates. Must be a tuple of (object, extra data) if needing to match
        extra data when searching octree.
        """
        if self._coordinatesObjects is not None:
            if len(self._coordinatesObjects) < self._maxObjects:
                self._coordinatesObjects.append( (copy.deepcopy(x), obj) )
                return
            else:
                # subdivide and add coordinatesObjects plus new object to new children
                coordinatesObjects = self._coordinatesObjects
                self._coordinatesObjects = None
                self._children = []
                for i in range(self._dimensionPower2):
                    childMinimums = copy.deepcopy(self._minimums)
                    childMaximums = copy.deepcopy(self._maximums)
                    for c in range(self._dimension):
                        if i & (1 << c):
                            childMinimums[c] = 0.5*(self._minimums[c] + self._maximums[c])
                        else:
                            childMaximums[c] = 0.5*(self._minimums[c] + self._maximums[c])
                    child = Octree(childMinimums, childMaximums, self._tolerance)
                    self._children.append(child)
                # add coordinatesObjects to children
                for coordinatesObject in coordinatesObjects:
                    self.addObjectAtCoordinates(coordinatesObject[0], coordinatesObject[1])
        # add the new object to the first child it fits in, using efficient octree search
        i = 0
        centre = self._children[0]._maximums
        for c in range(self._dimension):
            if x[c] > centre[c]:
                i += 1 << c
        self._children[i].addObjectAtCoordinates(x, obj)

    def getTolerance(self):
        return self._tolerance
