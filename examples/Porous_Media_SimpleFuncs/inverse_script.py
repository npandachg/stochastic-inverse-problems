#! /usr/bin/env python

# Copyright (C) 2014-2015 The BET Development Team

"""
This example generates uniform samples on a 3D grid
and evaluates a linear map to a 2d space. Probabilities
in the paramter space are calculated using emulated points.
1D and 2D marginals are calculated, smoothed, and plotted.
"""

import numpy as np
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import scipy.io as sio

# parameter domain
param_dim = 2
lam_domain = np.repeat([[1.0, 2.0]], param_dim, axis=0)

# sample data
path_name = "outfiles/SimpleFunc_Files/"
fname = path_name + "Subsurface_1x2.mat"
sample_data = sio.loadmat(fname)
data = sample_data['QoI']
samples = sample_data['samples']
samples = samples.transpose()

random_sample = True

# reference QoI
fname = path_name + "Subsurface_ref.mat"
ref_data = sio.loadmat(fname)
Q_ref = ref_data['QoI_ref']
Q_ref = [Q_ref[0, 0], Q_ref[0, 1]]


deterministic_discretize_D = True

if deterministic_discretize_D is True:
    (d_distr_prob, d_distr_samples, d_Tree) = \
        simpleFunP.uniform_hyperrectangle(data=data,
                                          Q_ref=Q_ref,
                                          bin_ratio=0.2,
                                          center_pts_per_edge=3)
else:
    (d_distr_prob, d_distr_samples, d_Tree) = \
        simpleFunP.unif_unif(data=data,
                             Q_ref=Q_ref,
                             M=50,
                             bin_ratio=0.2,
                             num_d_emulate=1E5)

'''
Suggested changes for user:

If using a regular grid of sampling (if random_sample = False), we set

  lambda_emulate = samples

Otherwise, play around with num_l_emulate. A value of 1E2 will probably
give poor results while results become fairly consistent with values
that are approximately 10x the number of samples.

Note that you can always use

  lambda_emulate = samples

and this simply will imply that a standard Monte Carlo assumption is
being used, which in a measure-theoretic context implies that each
Voronoi cell is assumed to have the same measure. This type of
approximation is more reasonable for large n_samples due to the slow
convergence rate of Monte Carlo (it converges like 1/sqrt(n_samples)).
'''
if random_sample is False:
    lambda_emulate = samples
else:
    lambda_emulate = calculateP.emulate_iid_lebesgue(
        lam_domain=lam_domain, num_l_emulate=1E5)


# calculate probablities
(P, lambda_emulate, io_ptr, emulate_ptr) = \
    calculateP.prob_emulated(samples=samples,
                             data=data,
                             rho_D_M=d_distr_prob,
                             d_distr_samples=d_distr_samples,
                             lambda_emulate=lambda_emulate,
                             d_Tree=d_Tree)

# calculate 2d marginal probs
'''
Suggested changes for user:

At this point, the only thing that should change in the plotP.* inputs
should be either the nbins values or sigma (which influences the kernel
density estimation with smaller values implying a density estimate that
looks more like a histogram and larger values smoothing out the values
more).

There are ways to determine "optimal" smoothing parameters (e.g., see CV, GCV,
and other similar methods), but we have not incorporated these into the code
as lower-dimensional marginal plots have limited value in understanding the
structure of a high dimensional non-parametric probability measure.
'''
(bins, marginals2D) = \
    plotP.calculate_2D_marginal_probs(P_samples=P,
                                      samples=lambda_emulate,
                                      lam_domain=lam_domain,
                                      nbins=10)
# smooth 2d marginals probs (optional)
# marginals2D = plotP.smooth_marginals_2D(marginals2D,bins, sigma=0.1)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins,
                             lam_domain, filename="linearMap",
                             plot_surface=False)

# calculate 1d marginal probs
(bins, marginals1D) = \
    plotP.calculate_1D_marginal_probs(P_samples=P,
                                      samples=lambda_emulate,
                                      lam_domain=lam_domain,
                                      nbins=10)
# smooth 1d marginal probs (optional)
# marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.1)
# plot 2d marginal probs
plotP.plot_1D_marginal_probs(
    marginals1D, bins, lam_domain, filename="linearMap")
