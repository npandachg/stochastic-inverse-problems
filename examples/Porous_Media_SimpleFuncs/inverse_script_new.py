import numpy as np
import scipy.io as sio
import bet.sample as samp
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
from mpi4py import MPI
from sipTools import *
import matplotlib
matplotlib.use('pdf')
matplotlib.use('Agg')
comm = MPI.COMM_WORLD

file_name = "Subsurface"
path_name = "outfiles/SimpleFunc_Files/"
count = 0
case_list = []
reference_dict = {}
length = 0
stride = 0
numSamplesSimple = 500
cases = 3
length_tot = numSamplesSimple * cases
samples = np.zeros([length_tot, 1])
data = np.zeros([length_tot, 2])

mesoscales = [[1, 2], [1, 3], [1, 4]]
for sample in mesoscales:

    m = sample[0]
    n = sample[1]
    meso = m * n
    fname = path_name + file_name + "_" + str(m) + "x" + str(n)

    # create the case list
    case_list.append(m * n)

    # read in the data file
    mat = sio.loadmat(fname)
    Q = mat['QoI']
    S = mat['samples']

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

    # get the current length of samples
    print "number of samples"
    n_samples = np.shape(Q)[0]
    print n_samples

    # populate the samples and data array
    samples[stride:(stride + n_samples), 0] = meso * np.ones([n_samples, ])
    data[stride:stride + n_samples, :] = Q

    # create the dictionary
    reference_dict[count] = case_list
    case_list = []

    # update stride
    stride += n_samples

    # update count
    count += 1

# reference QoI
fname = path_name + "Subsurface_ref.mat"
ref_data = sio.loadmat(fname)
Q_ref = ref_data['QoI_ref']
Q_ref = [Q_ref[0, 0], Q_ref[0, 1]]

# create inupt and output samples
input_samples = samp.sample_set(1)
input_samples.set_domain(np.array([[1.5, 4.5]]))
input_samples.set_values(samples)
# associate volume
input_samples.estimate_volume_mc()

output_samples = samp.sample_set(2)
output_samples.set_values(data)

disc_name_base = "disc_prob_simpleFunc"
# create a discretization object
my_discretization = samp.discretization(input_sample_set=input_samples,
                                        output_sample_set=output_samples)
fact = 4
simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
    data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.2 / fact,
    center_pts_per_edge=1)

# calculate the induced probability
calculateP.prob(my_discretization)
samp.save_discretization(my_discretization, fname)
P = input_samples.get_probabilities()

case = []
discrete_prob = np.zeros(3)
print np.shape(P)

print n_samples
stride = 0
for meso in range(3):
    print stride, stride + n_samples
    print sum(P[meso * n_samples:(meso + 1) * n_samples, ])
    discrete_prob[meso] = sum(P[meso * n_samples:(meso + 1) * n_samples, ])
    stride += n_samples

fname2 = disc_name_base + "_p01d" + str(fact)
sio.savemat(fname2, dict(prob=discrete_prob))
