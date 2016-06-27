from dolfin import *
import numpy as np
from sipTools.solvePDERandomField import solvePoissonRandomField
from sipTools.rectSimpleFunc import rectSimpleFunc
import scipy.io as sio
import bet.calculateP.calculateP as calculateP

'''
This script solves a poisson equation with a random field.
  (\grad \dot (rand_field \grad(u)) = -f)
The random field is approximated using simple functions
'''

# Load all the data that are fixed for this problem
execfile("examples/Porous_Media_KL/loadData.py")

# create a simple function object
rect_simple_func = rectSimpleFunc(mesh, partition, dimension)

param_dim = rect_simple_func.num_basis
lam_domain = np.repeat([[1.0, 2.0]], param_dim, axis=0)

coeff_samples = calculateP.emulate_iid_lebesgue(lam_domain=lam_domain,
                                                num_l_emulate=numSamplesSimple)
coeff_samples = coeff_samples.transpose()

"""
for each sample of the coefficients, generate the simple function
field and solve the pde.
"""

print " Start sampling simple functions "
for i in range(0, numSamplesSimple):

    print " Sampling field "
    sample_point = coeff_samples[:, i]
    perm_k = rect_simple_func.expandRandField(sample_point)

    # solve Poisson with this random field using FEM
    print " Solving PDE"
    u = solvePoissonRandomField(perm_k, mesh, Constant(-1.0 * 0), bcs)

    # compute QoI and derivatives using adjoints

    QoI[i, 0] = u(pt_1)
    QoI[i, 1] = u(pt_2)

sio.savemat('Subsurface_1x2', {'QoI': QoI, 'samples': coeff_samples})

