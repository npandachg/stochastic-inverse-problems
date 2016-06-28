from dolfin import *

print "Loading all data"

# Create a unit square mesh
nx = 31
ny = 31
p0 = Point(0.0, 0.0)
p1 = Point(10.0, 10.0)
mesh = RectangleMesh(p0, p1, nx, ny)
# initialize the mesh to generate connectivity
mesh.init()

# Random field is projected on the space of Hat functions in the mesh
V = FunctionSpace(mesh, "CG", 1)


def left_boundary(x, on_boundary):
    """TODO: Docstring for left_boundary.

    :x: TODO
    :on_boundary: TODO
    :returns: TODO

    """
    tol = 1e-14
    return on_boundary and abs(x[0]) < tol
Gamma_0 = DirichletBC(V, Constant(5.0), left_boundary)


def bottom_boundary(x, on_boundary):
    """TODO: Docstring for left_boundary.

    :x: TODO
    :on_boundary: TODO
    :returns: TODO

    """
    tol = 1e-14
    return on_boundary and abs(x[1]) < tol

Gamma_1 = DirichletBC(V, Constant(10.0), bottom_boundary)


# boundary condition
bcs = [Gamma_0, Gamma_1]
# Setup adjoint boundary conditions.
Gamma_adj_0 = DirichletBC(V, Constant(0.0), left_boundary)
Gamma_adj_1 = DirichletBC(V, Constant(0.0), bottom_boundary)
bcs_adj = [Gamma_adj_0, Gamma_adj_1]

# Setup the QoI class


class CharFunc(Expression):
    def __init__(self, region):
        self.a = region[0]
        self.b = region[1]
        self.c = region[2]
        self.d = region[3]

    def eval(self, v, x):
        v[0] = 0
        if (x[0] >= self.a) & (x[0] <= self.b) & (x[1] >= self.c) & \
                (x[1] <= self.d):
            v[0] = 1
        return v

# Define the QoI maps
# First the characteristic functions
Chi_1 = CharFunc([0.75, 1.25, 7.75, 8.25])
Chi_2 = CharFunc([7.75, 8.25, 0.75, 1.25])

# number of KL samples
numSamplesKL = 500
activeSamples = 1000

# set up the variance and correlation length
var = np.linspace(0.5, 3.5, 5)
eta = np.linspace(4, 10, 7)

var_ref = 2.5
eta_ref = 6.25

