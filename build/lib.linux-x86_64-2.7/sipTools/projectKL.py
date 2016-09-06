from dolfin import *
import numpy as np
import petsc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from meshDS import*
# initialize petsc
petsc4py.init()


class projectKL(object):

    """ This class takes a mesh object and projects a covariance
    function onto the space of "hat" functions defined on the mesh.
    """

    def __init__(self, mesh):
        """projectKL constructor
        :mesh: takes a fenics mesh
        """
        # create meshDS obect
        self._mesh = mesh
        self.domain = meshDS(mesh)
        self.c_volume_array, self.c_centroid_array = \
            self.domain.getElemVCArray()
        self.node_to_elem = self.domain.getNodesToElem()
        self.flag = False

    def _getCovMat(self, cov_expr):
        """ Builds the covariance matrix. Private function.
        TODO: modify for 3D.

        :cov_expr: Expression (dolfin) covariance function
        :returns:  PETSC matrix covariance matrix cov_mat

        """
        # store the expression
        self.expr = cov_expr
        # create a PETSC matrix for cov_mat
        cov_mat = PETSc.Mat().create()
        cov_mat.setType('aij')
        cov_mat.setSizes(self.domain.getNodes(), self.domain.getNodes())
        cov_mat.setUp()

        # scalar valued function is evaluated in this variable
        cov_ij = np.empty((1), dtype=float)
        # the points to evalute the expression
        xycor = np.empty((4), dtype=float)

        print '---------------------------'
        print '---------------------------'
        print ' Building Covariance Matrix'
        print '---------------------------'
        print '---------------------------'
        # Loop through global nodes and build the matrix for i < j because of
        # symmetric nature.
        for node_i in range(0, self.domain.getNodes()):
            # global node node_i
            for node_j in range(node_i, self.domain.getNodes()):
                # global node node_j
                temp_cov_ij = 0
                for elem_i in self.node_to_elem[node_i]:
                    # elem_i : element attached to node_i
                    # x1 : x co-ordinate of the centroid of element elem_i
                    x1 = self.c_centroid_array[elem_i].x()
                    # y1 : x co-ordinate of the centroid of element elem_i
                    y1 = self.c_centroid_array[elem_i].y()
                    for elem_j in self.node_to_elem[node_j]:
                        # elem_j : element attached to node_j
                        # x2 : x co-ordinate for the centroid of element elem_j
                        x2 = self.c_centroid_array[elem_j].x()
                        # y2 : y co-ordinate for the centroid of element elem_j
                        y2 = self.c_centroid_array[elem_j].y()
                        xycor[0] = x1
                        xycor[1] = x2
                        xycor[2] = y1
                        xycor[3] = y2
                        # evaluate the expression
                        cov_expr.eval(cov_ij, xycor)
                        if cov_ij[0] > 0:
                            temp_cov_ij += (1.0 / 3) * (1.0 / 3) * \
                                cov_ij[0] * \
                                self.c_volume_array[elem_i] * \
                                self.c_volume_array[elem_j]

                cov_mat.setValue(node_i, node_j, temp_cov_ij)
                cov_mat.setValue(node_j, node_i, temp_cov_ij)
        cov_mat.assemblyBegin()
        cov_mat.assemblyEnd()
        print '---------------------------'
        print '---------------------------'
        print ' Finished Covariance Matrix'
        print '---------------------------'
        print '---------------------------'

        return cov_mat

    def _getBMat(self):
        """Forms the B matrix in CX = BX solve where C is the
        covariance matrix and B is just a mass matrix.
        This is a private function. DO NOT call this
        unless debuging.

        :returns: PETScMatrix B
        """

        """B matrix is just a mass matrix, can be easily assembled
        through fenics. However, the ordering in Fenics is not the
        mesh ordering. So we build a temp matrix and then use the
        vertex to dof map to get the right ordering interms of our
        mesh nodes.
        """

        # create function space of order 1. For KL, we only restrict
        # to first order spaces.
        V = FunctionSpace(self._mesh, "CG", 1)
        # Define basis and bilinear form
        u = TrialFunction(V)
        v = TestFunction(V)
        a = u * v * dx
        # assemble in a temp matrix
        B_temp = assemble(a)

        # create petsc matrix B
        B = PETSc.Mat().create()
        B.setType('aij')
        B.setSizes(self.domain.getNodes(), self.domain.getNodes())
        B.setUp()

        # store the value in a a temp array B_ij
        B_ij = B_temp.array()

        # get the vertex to dof map
        v_to_d_map = vertex_to_dof_map(V)

        print '---------------------------'
        print '---------------------------'
        print ' Building Mass Matrix '
        print '---------------------------'
        print '---------------------------'
        for node_i in range(0, self.domain.getNodes()):
            for node_j in range(node_i, self.domain.getNodes()):
                B_ij_nodes = B_ij[v_to_d_map[node_i], v_to_d_map[node_j]]
                if B_ij_nodes > 0:
                    B.setValue(node_i, node_j, B_ij_nodes)
                    B.setValue(node_j, node_i, B_ij_nodes)

        B.assemblyBegin()
        B.assemblyEnd()
        print '---------------------------'
        print '---------------------------'
        print ' Finished Mass Matrix '
        print '---------------------------'
        print '---------------------------'
        return B

    def projectCovToMesh(self, num_kl, cov_expr):
        """Solves CX = BX where C is the covariance matrix
        :num_kl : number of kl exapansion terms needed
        :returns: TODO

        """
        # Make sure num_kl is not greater than number of nodes
        if num_kl > self.domain.getNodes():
            num_kl = self.domain.getNodes()

        # turn the flag to true
        self.flag = True
        # get C,B matrices
        C = PETScMatrix(self._getCovMat(cov_expr))
        B = PETScMatrix(self._getBMat())
        # Solve the generalized eigenvalue problem
        eigensolver = SLEPcEigenSolver(C, B)
        eigensolver.solve(num_kl)

        """ Diagnostic:
        Get the number of eigen values that converged.
        nconv = eigensolver.get_number_converged()
        Get N eigenpairs where N is the number of KL expansion and
        check if N < nconv otherwise you had
        really shitty matrix
        """

        # create numpy array of vectors and eigenvalues
        self.eigen_funcs = np.empty((num_kl), dtype=object)
        self.eigen_vals = np.empty((num_kl), dtype=float)

        # store the eigenvalues and eigen functions
        V = FunctionSpace(self._mesh, "CG", 1)
        for eigen_pairs in range(0, num_kl):
            lambda_r, lambda_c, x_real, x_complex = eigensolver.get_eigenpair(
                eigen_pairs)
            self.eigen_funcs[eigen_pairs] = Function(V)
            # use dof_to_vertex map to map values to the function space
            self.eigen_funcs[eigen_pairs].vector()[:] = \
                x_real[dof_to_vertex_map(V)]  # *np.sqrt(lambda_r)
            # divide by norm to make the unit norm again
            self.eigen_funcs[eigen_pairs].vector()[:] = \
                self.eigen_funcs[eigen_pairs].vector()[:] / \
                norm(self.eigen_funcs[eigen_pairs])
            self.eigen_vals[eigen_pairs] = lambda_r

    def truncate(self, num_kl, tol=1e-1, flag="default"):
        """ Returns the truncated KL expansion. 95% variability
        happens at the index N when
        the N partial sums of the eigenvalues account for about 95%
        of the total infinite sum.
        Since we don't have an estimate for the infinite sum,
        we check when the partial sums are within tol.
        By default tol = 1e-1.

        :num_kl : number of kl exapansion terms needed
        :returns: the number of terms
        """

        # if num_kl > np.size(self.eigen_vals):
        #    num_kl = np.size(self.eigen_vals)

        if flag == "partial":
            previous = 0
            for i in range(0, num_kl):
                current = self.eigen_vals[i] + previous
                indx = i
                if abs(current - previous) <= tol:
                    break
                previous = current
            return indx

        total = np.sum(self.eigen_vals)
        sum_val = 0.0
        for i in range(0, num_kl):
            sum_val = sum_val + self.eigen_vals[i]
            if sum_val / total >= 0.95:
                break
        return i

    def expandRandField(self, num_kl, sample_point, attr, read_in={}):
        """Expands the random field as a finite linear combination
        of the KL eigen functions with coefficients sampled
        from a distribution given by the array
        sample_point.
        If attr is log, then the random field is the exponentiated.

        :num_kl : number of kl exapansion terms
        :sample_point : coefficient of each KL term
        :attr : attribute of the random field eg log, etc
        :returns: the random field on the function space

        """
        self.build_field = True

        # By default it is CG space of order 1
        function_space = FunctionSpace(self._mesh, "CG", 1)
        # Instantiate the random field as an element of the function space
        rand_field = Function(function_space)
        # create a temp field array
        temp_field = np.zeros((self._mesh.num_vertices()), dtype=float)
        # compute the linear combination

        if read_in:
            KL_eigen_funcs = read_in['KL_eigen_funcs']
            KL_eigen_vals = read_in['KL_eigen_vals']
            for kl in range(0, num_kl):
                temp_field += sqrt(KL_eigen_vals[0, kl]) * \
                    sample_point[kl] * KL_eigen_funcs[kl, :]
        else:
            for kl in range(0, num_kl):
                temp_field += sqrt(self.eigen_vals[kl]) * sample_point[
                    kl] * self.eigen_funcs[kl].vector().array()

        if attr == "log":
            rand_field_array = np.exp(temp_field)
        else:
            rand_field_array = temp_field

        # use dof_to_vertex map to map values of the array to the function
        # space
        rand_field.vector()[:] = rand_field_array[
            dof_to_vertex_map(function_space)]

        return rand_field

