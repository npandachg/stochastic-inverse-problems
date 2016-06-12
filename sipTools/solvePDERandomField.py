""" This module has helper functions to solve
PDEs with random Fields
"""

from dolfin import*


def solvePoissonRandomField(rand_field, mesh, f, bcs, poly_order=1):
    """
    Solves the poisson equation with a random field :
    (\grad \dot (rand_field \grad(u)) = -f)
    """
    # create the function space
    V = FunctionSpace(mesh, "CG", poly_order)
    u = TrialFunction(V)
    v = TestFunction(V)
    L = f * v * dx
    a = rand_field * inner(nabla_grad(u), nabla_grad(v)) * dx
    u = Function(V)
    solve(a == L, u, bcs)
    return u

