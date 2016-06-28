""" This script generates the "reference" data. We will use this
to invert as model selection.
"""

import numpy as np
from dolfin import *
from sipTools.projectKL import projectKL
import scipy.io as sio
from sipTools.samplePdeActiveVar import samplePdeActiveVar
# Load all the data that are fixed for this problem
execfile("examples/Porous_Media_KL/loadData.py")

print "reference eta and var variable ", eta_ref, var_ref

# Initialize the KL object
rect_mesh = projectKL(mesh)
num_KL = 200  # mesh.num_vertices()

count = 0
length_tot = 0
file_name = "reference_data"

# compute the covariance expression
etaX = eta_ref
etaY = etaX
C = var_ref
cov = Expression("C*exp(-(fabs(x[0]-x[1]))/ex- (fabs(x[2]-x[3]))/ey)",
                 ex=etaX, ey=etaY, C=C)

# project Cov to Mesh
rect_mesh.projectCovToMesh(num_KL, cov)  # fine mesh
# truncate KL terms with approx. 95% variability
size_kl = rect_mesh.truncate(num_KL)
print "Number of KL terms have been truncated to, ", size_kl
# sample each KL coefficient : here I have assumed standard normal density
xi_k = np.random.standard_normal(size=(size_kl, numSamplesKL))
# pass all the loaded data to your own function
active_var_samples, qoi_samples = samplePdeActiveVar(numSamplesKL,
                                                     size_kl,
                                                     xi_k,
                                                     mesh,
                                                     rect_mesh,
                                                     bcs,
                                                     bcs_adj,
                                                     Chi_1,
                                                     Chi_2)
print np.shape(active_var_samples)[0]
length_tot += np.shape(active_var_samples)[0]
fname = file_name + "_" + str(etaX) + "_" + str(C)
sio.savemat(fname, {'QoI': qoi_samples, 'samples': active_var_samples})
