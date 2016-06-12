from dolfin import *
import numpy as np
import numpy.linalg as LA
from sipTools.solvePDERandomField import solvePoissonRandomField

# Straight from internet : finds points in Convex hull


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.
    Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay
    triangulation will be computed.
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def samplePdeActiveVar(numSamples, numKL, xi_k, mesh, KL_obj, bcs,
                       bcs_adj, Chi_1, Chi_2):

    # Get the eigenfunctions and eigenvalues
    eigen_func = KL_obj.eigen_funcs
    eigen_val = KL_obj.eigen_vals

    # Define the QoI array
    QoI = np.zeros([numSamples, 2])

    # Array to hold the QoI derivative w.r.t each KL coefficient
    QoI_deriv_1 = np.zeros([numSamples, numKL])
    QoI_deriv_2 = np.zeros([numSamples, numKL])

    # For each sample solve the PDE
    numPoint = 0
    f = Constant(-1.0)  # forcing of Poisson
    for i in range(0, numSamples):

        print "Sample point number: %g" % numPoint
        samplePoint = []
        for kl in range(0, numKL):
            samplePoint.append(xi_k[kl, i])

        perm_k = KL_obj.expandRandField(numKL, samplePoint, "log")
        # solve Poisson with this random field using FEM
        u = solvePoissonRandomField(perm_k, mesh, f, bcs)

        # solve adjoints
        # (using same order because only getting derivatives
        # not error estimates)
        phi1 = solvePoissonRandomField(perm_k, mesh, Chi_1, bcs_adj)
        phi2 = solvePoissonRandomField(perm_k, mesh, Chi_2, bcs_adj)

        # Compute QoI
        QoI[numPoint, 0] = assemble(u * Chi_1 * dx)
        QoI[numPoint, 1] = assemble(u * Chi_2 * dx)
        print "Q1 = %g" % QoI[numPoint, 0]
        print "Q2 = %g" % QoI[numPoint, 1]

        # compute gradients of the QoI component maps
        for kl in range(0, numKL):
            term1 = -assemble(inner(sqrt(eigen_val[kl]) *
                              eigen_func[kl] * perm_k *
                              nabla_grad(u), nabla_grad(phi1)) * dx)
            term2 = -assemble(inner(sqrt(eigen_val[kl]) *
                              eigen_func[kl] * perm_k *
                              nabla_grad(u), nabla_grad(phi2)) * dx)
            QoI_deriv_1[numPoint, kl] = term1
            QoI_deriv_2[numPoint, kl] = term2

        numPoint += 1

    print "***********************"
    print "Computing the Active subspace"
    print "***********************"
    # Do the active subspaces
    # Form C
    C_1 = (1.0 / numSamples) * (QoI_deriv_1.transpose()).dot(QoI_deriv_1)
    C_2 = (1.0 / numSamples) * (QoI_deriv_2.transpose()).dot(QoI_deriv_2)

    # Get eigenvalue decomposition
    eVal_1, eVec_1 = LA.eig(C_1)
    eVal_2, eVec_2 = LA.eig(C_2)

    # sort the eigenvalues from max to min
    idx_1 = eVal_1.argsort()[::-1]
    idx_2 = eVal_2.argsort()[::-1]

    # update the eigen vectors acording to this sort : i.e largest to smallest
    eVec_1 = eVec_1[idx_1]
    eVec_2 = eVec_2[idx_2]

    # Compute the active subspace. In this case, it is the first one
    # cast them to real, don't want complex types
    beta_1 = np.real((xi_k.transpose()).dot(eVec_1[:, 0]))
    beta_2 = np.real((xi_k.transpose()).dot(eVec_2[:, 0]))

    print "***********************"
    print "Fitting curve to data"
    print "***********************"

    # fit a polynomial
    ftemp_1 = np.polyfit(beta_1, QoI[:, 0], 3)
    # use poly1d object to evaluate polynomial at a point or array
    f1 = np.poly1d(ftemp_1)

    ftemp_2 = np.polyfit(beta_2, QoI[:, 1], 3)
    # use poly1d object to evaluate polynomial at a point or array
    f2 = np.poly1d(ftemp_2)

    # Generate samples in active var space
    # First generate samples of KL coefficients
    num_pts = 100  # 10000
    v1 = np.random.standard_normal(size=(numKL, num_pts))
    # Project KL coefficients to active variables to generate samples
    sample_points = np.zeros([num_pts, 2])
    sample_points[:, 0] = np.real((v1.transpose()).dot(eVec_1[:, 0]))
    sample_points[:, 1] = np.real((v1.transpose()).dot(eVec_2[:, 0]))
    # Evaluate surrogate on active variables
    qoI = [f1(sample_points[:, 0]), f2(sample_points[:, 1])]

    return sample_points, qoI

