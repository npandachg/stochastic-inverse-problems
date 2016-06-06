#!/usr/bin/en python

from dolfin import *
from numpy import *


class meshDS(object):

    """Docstring for meshDS. """

    def __init__(self, mesh):
        """ meshDS constructor.

        :mesh: reads a fenics mesh object

        """
        self._mesh = mesh
        # empty dictionary of node to elements connectivity
        self.node_elem = {}
        # empty dictionary of edges to elements connectivity
        self.edges_elem = {}

        # initialize the mesh and read in the values
        self._mesh.init()
        self._dim = self._mesh.topology().dim()
        self.num_nodes = self._mesh.num_vertices()
        self.num_elements = self._mesh.num_cells()
        self.num_edges = self._mesh.num_edges()

        # node to element connectivity
        for nodes in entities(self._mesh, 0):
            self.node_elem[nodes.index()] = nodes.entities(self._dim)

        # data structures for elem volume and centroid
        self.elem_vol_array = empty((self.num_elements), dtype=float)
        self.elem_centroid_array = empty((self.num_elements), dtype=object)
        self.vc_array_cache = False

    def getNodes(self):
        """
        :returns: num of nodes in the mesh
        """
        return self.num_nodes

    def getElements(self):
        """
        :returns: number of elements in the mesh
        """
        return self.num_elements

    def getEdges(self):
        """
        :returns: number of elements in the mesh
        """
        return self.num_edges

    def getElemToNodes(self):
        """
        :returns: Elements - Nodes Connectivity array of array
        """
        return self._mesh.cells()

    def getNodesToElem(self):
        """
        :returns: returns Nodes to Element connectivity as a dictionary
        where nodes_elem[i] is an array of all the elements attached
        to node i.
        """

        return self.node_elem

    def getElemVCArray(self):

        """
        :returns: array of element volume and and an array of element
        centroid object.
        elem_vol_array[i] is the volume of the cell i.
        elem_centroid_array[i][0] is the x co-ordinate of the centroid
        for element number i.
        elem_centroid_array[i][1] is the y co-ordinate of the centroid
        for element number i.
        elem_centroid_array[i][2] is the y co-ordinate of the centroid
        for element number i.
        """

        if self.vc_array_cache:
            return self.elem_vol_array, self.elem_centroid_array

        for cell_indx in range(self.num_elements):
            # First get the cell object corresponding to the cell_indx
            cell_obj = Cell(self._mesh, cell_indx)
            # Find the cell volume and cell centroid
            self.elem_vol_array[cell_indx] = cell_obj.volume()
            self.elem_centroid_array[cell_indx] = cell_obj.midpoint()

        self.vc_array_cache = True
        return self.elem_vol_array, self.elem_centroid_array

