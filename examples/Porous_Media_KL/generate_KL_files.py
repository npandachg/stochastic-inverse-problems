import numpy as np
from dolfin import *
from sipTools.projectKL import projectKL
import scipy.io as sio

# Load all the data that are fixed for this problem
execfile("examples/Porous_Media_KL/loadData.py")
out_dir = "outfiles/KL_files/eigenFuncsVals/"


# Initialize the KL object
rect_mesh = projectKL(mesh)
num_KL = 200  # mesh.num_vertices()

count = 0
length_tot = 0
file_name = "KL_eigenFuncsVals"
C = 1.0
for eta_val in eta:
    print "Current sample of eta : ", eta_val
    # compute the covariance expression
    etaX = eta_val
    etaY = etaX

    cov = Expression("C*exp(-(fabs(x[0]-x[1]))/ex- (fabs(x[2]-x[3]))/ey)",
                     ex=etaX, ey=etaY, C=C)

    # project Cov to Mesh
    rect_mesh.projectCovToMesh(num_KL, cov)  # fine mesh
    # truncate KL terms with approx. 95% variability
    size_kl = rect_mesh.truncate(num_KL)
    print "Number of KL terms have been truncated to, ", size_kl

    fname = out_dir + file_name + "_" + str(etaX)
    # save the eigen functions and eigen_mat
    eigen_func_mat = np.zeros((size_kl,
                              rect_mesh.eigen_funcs[0].vector().array().size))
    for i in range(0, size_kl):
        eigen_func_mat[i, :] = rect_mesh.eigen_funcs[i].vector().array()

        kl_mdat = dict()
        kl_mdat['KL_eigen_funcs'] = eigen_func_mat
        kl_mdat['KL_eigen_vals'] = rect_mesh.eigen_vals[0:size_kl]

    sio.savemat(fname, kl_mdat)
