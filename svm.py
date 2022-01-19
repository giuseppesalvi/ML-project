"""
    This module contains the implementation of the Support Vector Machines
"""
import numpy as np
import scipy.optimize as op

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def svm_dual_wrapper(DTR, LTR, K):
    """
    """
    def svm_dual(alpha):
        """
        """
        N = DTR.shape[1]

        z = mcol(np.array(2 * LTR - 1))

        D_hat = np.vstack((DTR, np.ones(N) * K))
        G_hat = np.dot(D_hat.T, D_hat)
        H_hat = z * z.T * G_hat

        # J_D_hat = -1/2 * np.dot(np.dot(alpha.T, H_hat), alpha) + \
        #     np.dot(alpha.T, np.ones(N))

        # need to use multi_dot because it gives numerical problems otherwise
        J_D_hat = -1/2 * np.linalg.multi_dot([alpha.T, H_hat, alpha]) + \
            np.dot(alpha.T, np.ones(N))

        L_D_hat = - J_D_hat
        grad_L_D_hat = mcol(np.dot(H_hat, alpha) - np.ones(N))

        return L_D_hat, grad_L_D_hat
    return svm_dual


def svm_primal_from_dual(alpha, DTR, LTR, K):
    """
    """
    N = LTR.shape[0]
    z = mcol(np.array(2 * LTR - 1))
    D = np.vstack((DTR, np.ones(N) * K))
    w_s_hat = np.sum(alpha * z * D.T, axis=0)
    return w_s_hat


def svm_primal_objective(w_s_hat, DTR, LTR, K, C):
    """
    """
    N = LTR.shape[0]
    z = mrow(np.array(2 * LTR - 1))
    D_hat = np.vstack((DTR, np.ones(N) * K))
    f = 1 - z * np.dot(w_s_hat.T, D_hat)
    J_hat = 1/2 * (w_s_hat * w_s_hat).sum() + C * \
        np.sum(np.maximum(np.zeros(f.shape), f))
    return J_hat


def polynomial_kernel(X1, X2, c, d, gamma):
    """
    """
    return (np.dot(X1.T, X2) + c) ** d


def RBF_kernel(X1, X2, c, d, gamma):
    """
    """
    X1 = X1.T
    X2 = X2.T
    # return np.exp(-gamma*np.sum((X2-X1[:,np.newaxis])**2, axis=-1))

    # optimized version using property of norm:
    # ||X1-X2||^2 = ||X1||^2 + ||X2||^2 - 2 * X1.T * Y
    X1_norm = np.sum(X1 ** 2, axis=-1)
    X2_norm = np.sum(X2 ** 2, axis=-1)
    return np.exp(-gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))


def svm_dual_kernel_wrapper(DTR, LTR, kernel, K, c, d, gamma):
    """
    """
    def svm_dual_kernel(alpha):
        """
        """
        N = DTR.shape[1]

        z = mcol(np.array(2 * LTR - 1))

        # + K^2 (=Xi) for regularized bias in non-linear svm version
        H_hat = z * z.T * (kernel(DTR, DTR, c, d, gamma) + K**2)

        # J_D_hat = -1/2 * np.dot(np.dot(alpha.T, H_hat), alpha) + \
        #     np.dot(alpha.T, np.ones(N))

        # need to use multi_dot because it gives numerical problems otherwise
        J_D_hat = -1/2 * np.linalg.multi_dot([alpha.T, H_hat, alpha]) + \
            np.dot(alpha.T, np.ones(N))

        L_D_hat = - J_D_hat
        grad_L_D_hat = mcol(np.dot(H_hat, alpha) - np.ones(N))

        return L_D_hat, grad_L_D_hat
    return svm_dual_kernel


def svm_linear(DTR, LTR, DTE, K, C):
    """
    """

    N = DTR.shape[1]
    # starting point
    x0 = np.zeros(N)

    svm_dual = svm_dual_wrapper(DTR, LTR, K)

    bounds = []
    for i in range(N):
        bounds.append((0, C))

    x, f, d = op.fmin_l_bfgs_b(svm_dual, x0, factr=1.0, bounds=bounds)

    # Recover primal solution from dual solution
    w_s_hat = svm_primal_from_dual(mcol(x), DTR, LTR, K)

    # Compute scores S
    # DTE_ is the extended data matrix for the evaluation set
    DTE_ = np.vstack((DTE, np.ones(DTE.shape[1]) * K))
    S = np.dot(mcol(w_s_hat).T, DTE_)
    return S.reshape(S.size,)

def svm_kernel_polynomial(DTR, LTR, DTE, K, C, d=2, c=0):
    N = DTR.shape[1]
    # starting point
    x0 = np.zeros(N)
    bounds = []
    for i in range(N):
        bounds.append((0, C))

    svm_dual_kernel = svm_dual_kernel_wrapper(
                DTR, LTR, polynomial_kernel, K, c, d, 0)

    x, f, d_ = op.fmin_l_bfgs_b(svm_dual_kernel, x0, factr=1.0, bounds=bounds)

    # Compute scores of test samples
    S = np.empty(DTE.shape[1])
    z = mcol(np.array(2 * LTR - 1))

    for t in range(DTE.shape[1]):
        for i in range(N):
            if(mcol(x)[i] > 0):
                S[t] += mcol(x)[i] * z[i] * (polynomial_kernel(DTR.T[i], DTE.T[t], c, d, 0) + K**2)
    # + K^2 (=Xi) for regularized bias in non-linear svm version
    return S.reshape(S.size,) 


def svm_kernel_RBF(DTR, LTR, DTE, K, C, g):
    """
    """
    N = DTR.shape[1]
    # starting point
    x0 = np.zeros(N)
    bounds = []
    for i in range(N):
        bounds.append((0, C))


    svm_dual_kernel = svm_dual_kernel_wrapper(DTR, LTR, RBF_kernel, K, 0, 0, g)

    x, f, d_ = op.fmin_l_bfgs_b(svm_dual_kernel, x0, factr=1.0, bounds=bounds)

    # Compute scores of test samples
    S = np.empty(DTE.shape[1])
    z = mcol(np.array(2 * LTR - 1))

    for t in range(DTE.shape[1]):
        for i in range(N):
                if(mcol(x)[i] > 0):
                    S[t] += mcol(x)[i] * z[i] * (RBF_kernel(mcol(DTR.T[i]), mcol(DTE.T[t]), 0, 0, g) + K**2)
    # + K^2 (=Xi) for regularized bias in non-linear svm version
    return S.reshape(S.size,) 
