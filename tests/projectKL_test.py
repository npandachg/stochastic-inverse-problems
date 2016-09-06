from dolfin import *
from sipTools.projectKL import projectKL
import numpy as np
import scipy.io as sio

# Create a unit square mesh
nx = 11
ny = 11
p0 = Point(0.0, 0.0)
p1 = Point(10.0, 10.0)
mesh = RectangleMesh(p0, p1, nx, ny)

# create a covariance function
C = 1
eta_x = 4.0
eta_y = eta_x
cov = Expression("C*exp(-(fabs(x[0]-x[1]))/ex- (fabs(x[2]-x[3]))/ey)",
                 ex=eta_x, ey=eta_y, C=C)

# initialize a projectKL object
num_kl = 50
rectmesh = projectKL(mesh)
# project the cov. expression on to the mesh
rectmesh.projectCovToMesh(num_kl, cov)
# now truncate the number of KL terms based on 95% variability
size_kl = rectmesh.truncate(num_kl)

# get the number of eigenvalues
print "Number of truncated eigenfuncs ", size_kl

# save the eigen functions and eigen_mat
eigen_func_mat = np.zeros((size_kl,
                          rectmesh.eigen_funcs[0].vector().array().size))
for i in range(0, size_kl):
    eigen_func_mat[i, :] = rectmesh.eigen_funcs[i].vector().array()

    kl_mdat = dict()
    kl_mdat['KL_eigen_funcs'] = eigen_func_mat
    kl_mdat['KL_eigen_vals'] = rectmesh.eigen_vals[0:size_kl]
print np.shape(eigen_func_mat)
sio.savemat("KL_expansion", kl_mdat)

# modify eigen values
KL_mdat = sio.loadmat("KL_expansion")
KL_mdat['KL_eigen_vals'] = 0.5 * KL_mdat['KL_eigen_vals']

size = np.shape(KL_mdat['KL_eigen_funcs'])[0]
print "size_kl is ", np.shape(KL_mdat['KL_eigen_funcs'])[0]
v_temp = np.empty((size), dtype=float)
v_temp = np.ascontiguousarray(KL_mdat['KL_eigen_funcs'][0, :])
print np.size(KL_mdat['KL_eigen_funcs'][0, :])
print np.shape(v_temp)

# expand rand_field
xi_k = np.random.standard_normal(size=(size_kl, 1))
perm_k = rectmesh.expandRandField(size_kl, xi_k[:, 0], "log", read_in=KL_mdat)

# plot the mesh
# plot(mesh, interactive=True)

