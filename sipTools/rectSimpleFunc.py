""" This module takes a rectangle mesh and
partions it into m_x X m_y rectangles in 2D or
m_x X m_y X m_z rectangles in 3D, then creates a
"""

from dolfin import *
import numpy as np


class rectSimpleFunc(object):
    """takes a rectangle, partitions it into smaller
    rectangles. A field is taken as peicewise constant
    in each of these rectangle. A rectangular mesh with start point at
    the origin (0,0,0).
    """
    def __init__(self, mesh, list_partition, list_dimension):
        """ takes a list/array of partition v where v[0] is the partitions
        in x dimension, v[1] in y dimension and so on.
        list_dimension is a list/array u where u[0] is the
        range of mesh in x direction, u[1] in y direction and so on.
        """

        self.partitions = list_partition
        self.dimensions = list_dimension
        self.num_basis = 1
        self._mesh = mesh
        self.V0 = FunctionSpace(mesh, 'DG', 0)

        for num in self.partitions:
            self.num_basis *= num

        self.char_funcs = np.empty((numBasis), dtype=object)

    def _buildCharFuncs(self):
        """ builds the characteristic functions. Private Functions
        TODO: modify this for 3D.
        """
        x_array = np.linspace(0, self.dimensions[0], self.partitions[0] + 1)
        y_array = np.linspace(0, self.dimensions[1], self.partitions[1] + 1)

        for i in range(0, numBasis):
        m = self.partitions[0]
        char_funcs[i] = interpolate(CharFunc([x_array[i % m],
                                              x_array[(i % m)+1],
                                              y_array[i/m],
                                              y_array[(i/m)+1]]),
                                    V0)





