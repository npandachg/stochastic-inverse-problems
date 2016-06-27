from dolfin import *
from sipTools.solvePDERandomField import solvePoissonRandomField
import scipy.io as sio

'''
This script solves a poisson equation with a random field.
  (\grad \dot (perm_field \grad(u)) = -f)
The perm_field is fixed here and is a linear function in y:
  perm_field = a0 + a1*y
'''
# Load all the data that are fixed for this problem
execfile("examples/Porous_Media_KL/loadData.py")
a0 = 1.25
a1 = 0.04

# create the permiability field
perm_field = Expression("a0+a1*x[1]", a0=a0, a1=a1)

# solve the PDE
print " Solving PDE"
u = solvePoissonRandomField(perm_field, mesh, Constant(-1.0 * 0), bcs)
QoI_ref = np.zeros([1, 2])
# compute QoI
QoI_ref[0, 0] = u(pt_1)
QoI_ref[0, 1] = u(pt_2)

# save the files
sio.savemat('Subsurface_ref', {'QoI_ref': QoI_ref, 'field_params': [a0, a1]})
