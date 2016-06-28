import numpy as np
from dolfin import *
from sipTools import projectKL
from sipTools import samplePdeActiveVar
import scipy.io as sio

# Load all the data that are fixed for this problem
execfile("examples/Porous_Media_KL/loadData.py")

# define samples for correlation length and variance

# Stack eta, var as in BET
vv1, vv2 = np.meshgrid(var, eta, indexing='ij')
stat_samples = np.vstack((vv1.flat[:], vv2.flat[:])).transpose()

num_stat_samples = np.size(var) * np.size(eta)

print "number of samples in eta and var ", num_stat_samples


# Initialize the KL object
rect_mesh = projectKL(mesh)
num_KL = 200  # mesh.num_vertices()

count = 0
length_tot = 0
file_name = "samples_and_data"
for sample in stat_samples:
    print "Current sample of variance and eta : ", sample
    # compute the covariance expression
    etaX = sample[1]
    etaY = etaX
    C = sample[0]
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
                                                         activeSamples,
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

print "total length is ", length_tot

