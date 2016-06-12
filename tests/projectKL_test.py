from dolfin import *
from sipTools.projectKL import projectKL
import numpy as np

# Create a unit square mesh
nx = 51
ny = 51
p0 = Point(0.0, 0.0)
p1 = Point(1.0, 1.0)
mesh = RectangleMesh(p0, p1, nx, ny)

# create a covariance function
C = 1
eta_x = 4.0
eta_y = 6.0
cov = Expression("C*exp(-(fabs(x[0]-x[1]))/ex- (fabs(x[2]-x[3]))/ey)",
                 ex=eta_x, ey=eta_y, C=C)

# initialize a projectKL object
num_kl = 100
rectmesh = projectKL(mesh)
# project the cov. expression on to the mesh
rectmesh.projectCovToMesh(num_kl, cov)
# now truncate the number of KL terms based on 95% variability
size_kl = rectmesh.truncate(num_kl)

# get the number of eigenvalues
print "Number of eigenfuncs ", np.size(rectmesh.eigen_funcs)

# plot the mesh
plot(mesh, interactive=True)
