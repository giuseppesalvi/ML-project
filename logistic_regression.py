"""
    This module contains the implementation of the Logistic Regression model
"""
import numpy as np
import scipy.optimize as op


def logreg_obj_wrap(DTR, LTR, l):
    """ It's a wrapper for function logreg_obj that needs to access also to
        DTR, LTR, and lambda (l)
    """
    def logreg_obj(v):
        """ Computes the Logistic Regression objective J(w, b) using formula (2)
            v is a numpy array with shape (D+1,), where D is the dimensionality of 
            the feature space v = [w,b]
        """
        w, b = v[0:-1], v[-1]
        z = 2 * LTR - 1
        J = l / 2 * (w * w).sum() + \
            np.log1p(np.exp(-z * (w.T.dot(DTR) + b))).mean()

        # using formula 3
        # J = l / 2 * (w * w).sum() + (LTR * np.log1p(np.exp(-w.T.dot(DTR) - b)) + (1 - LTR) * np.log1p(np.exp(w.T.dot(DTR) + b))).mean()

        return J

    return logreg_obj


def logistic_regression(DTR, LTR, DTE, l):
    """ Implementation of Logistic Regression
        DTR, LTR are training data and labels
        DTE are evaluation data
        l is the hyperparameter lambda
        returns: logistic regression score, that can be interpreted as a class posterior log-likelihood ratio 
    """
    # starting point
    x0 = np.zeros(DTR.shape[0] + 1)

    # Use the wrapper to pass the parameters to the function
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)

    # Obtain the minimizer of J
    x, f, d = op.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)

    # Now that the model is trained we can compute posterior
    # log-likelihoods ratios by simply computing for each test sample xt
    # the score s(xt) = wT xt + b
    # compute the array of scores S
    S = np.dot(x[0:-1].T, DTE) + x[-1]
    return S
