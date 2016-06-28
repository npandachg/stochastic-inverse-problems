import numpy as np
import scipy.io as sio
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
from mpi4py import MPI
from sipTools import *
comm = MPI.COMM_WORLD

# Load all the data that are fixed for this problem
execfile("examples/Porous_Media_KL/loadData.py")

# define samples for correlation length and variance

# Stack eta, var as in BET
vv1, vv2 = np.meshgrid(var, eta, indexing='ij')
stat_samples = np.vstack((vv1.flat[:], vv2.flat[:])).transpose()

num_stat_samples = np.size(var) * np.size(eta)

print "number of samples in eta and var ", num_stat_samples


file_name = "samples_and_data"
length_tot = num_stat_samples * activeSamples

'''
stores the entire samples,
sample[:, 0] is beta_1 (active var 1),
sample[:, 1] is beta_2 (active var 2),
sample[:, 2] is etaX   (correlation)
and sample[:, 3] is C  (variance)
'''
samples = np.zeros([length_tot, 2])
'''
data is the entire data sample,
data[:,0] is the QoI number 1,
data[:,1] is the QoI number 2
'''
data = np.zeros([length_tot, 2])
'''

'''
vec_length = []

'''
vec_length : stores the length (number) of samples for each case.
case list  :  each case contains the variance and the correlation length
count      :  index for the number of cases
reference dictionary : gives the dictionary of case number to the
                       variance and correlation length
'''

count = 0
case_list = []
reference_dict = {}
length = 0
stride = 0

for sample in stat_samples:

    C = sample[0]
    etaX = sample[1]
    fname = file_name + "_" + str(etaX) + "_" + str(C)

    # create the case list
    case_list.append(C)
    case_list.append(etaX)

    # read in the data file
    m = sio.loadmat(fname)
    Q = m['QoI'].transpose()
    S = m['samples']

    # print data summary
    print "******************"
    print "Summary statistics for the data"
    print "count:", count
    print "mean QoI[0]", np.mean(Q[:, 0])
    print "min QoI[0]", np.min(Q[:, 0])
    print "max QoI[0]", np.max(Q[:, 0])
    print "25 percentile QoI[0]", np.percentile(Q[:, 0], 25)
    print "75 percentile QoI[0]", np.percentile(Q[:, 0], 75)
    print "mean QoI[1]", np.mean(Q[:, 1])
    print "min QoI[1]", np.min(Q[:, 1])
    print "max QoI[1]", np.max(Q[:, 1])
    print "25 percentile QoI[1]", np.percentile(Q[:, 1], 25)
    print "75 percentile QoI[1]", np.percentile(Q[:, 1], 75)
    print "******************"

    # add to vec_length
    vec_length.append(np.shape(Q)[0])

    # get the current length of samples
    n_samples = np.shape(Q)[0]

    # populate the samples and data array
    samples[stride:(stride + n_samples), 0] = C * np.ones([n_samples, ])
    samples[stride:(stride + n_samples), 1] = etaX * np.ones([n_samples, ])
    data[stride:stride + n_samples, :] = Q

    # create the dictionary
    reference_dict[count] = case_list
    case_list = []

    # update stride
    stride += n_samples

    # update count
    count += 1

# modify this : get this from reference_script
fname = "reference_data" + "_" + str(etaX) + "_" + str(C)
QoI_num = 2
# read in the reference data
m = sio.loadmat(fname)
Q = m['QoI'].transpose()
q0 = np.mean(Q[:, 0])
q1 = np.mean(Q[:, 1])
# make the reference point, mean of the reference QoI
Q_ref = np.zeros(2)
Q_ref = [q0, q1]
# Q_ref = [-1, 2]

deterministic_discretize_D = True

bin_size = [0.1, 0.1]
output_box = np.zeros([QoI_num, 2])
output_box[:, 0] = [Q_ref[0] - bin_size[0] / 2., Q_ref[0] + bin_size[0] / 2.]
output_box[:, 1] = [Q_ref[1] - bin_size[1] / 2., Q_ref[1] + bin_size[1] / 2.]

(d_distr_prob, d_distr_samples, d_Tree) = simpleFunP.uniform_hyperrectangle_user(data=data, domain=output_box, center_pts_per_edge=1)


# calculate probablities
(P, lambda_emulate, io_ptr) = calculateP.prob(samples=samples,
                                              data=data,
                                              rho_D_M=d_distr_prob,
                                              d_distr_samples=d_distr_samples,
                                              d_Tree=d_Tree)


case = []
discrete_prob = np.zeros((np.size(var), np.size(eta)))

'''
 discrete_prob[i][j][k] = prob of i^th variance, j^th correlation length
'''
i = 0
j = 0
for variance in var:
    j = 0
    for corr in eta:
        case.append(variance)
        case.append(corr)
        # check which case number this corresponds to
        for key in reference_dict:
            if reference_dict[key] == case:
                case_num = key
                break
        case = []
        if case_num == 0:
            stride = 0
        else:
            stride = np.sum(vec_length[0:case_num])
        discrete_prob[i][j] = sum(P[stride:stride + vec_length[case_num], ])
        j += 1
    i += 1

# save the matrix as a mat file
sio.savemat('discProbMatrix_case.mat', dict(x=discrete_prob))
