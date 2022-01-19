"""
    This module contains the implementations of the Gaussian Models:
    multivariate gaussian model(MVG), naive bayes and tied
"""
import numpy as np
import sklearn.datasets as da
import scipy as sp

def vcol(x):
    """ reshape the vector x into a column vector """

    return x.reshape(x.shape[0], 1)
    
def covariance_matrix2(D):
    """ Computes and returns the covariance matrix given the dataset D
        this is a more efficient implementation
    """
    # compute the dataset mean mu
    mu = D.mean(1)

    # mu is a 1-D array, we need to reshape it to a column vector
    mu = vcol(mu)

    # remove the mean from all the points
    DC = D - mu

    # DC is the matrix of centered data
    C = np.dot(DC, DC.T)
    C = C / float(D.shape[1])

    return C


def logpdf_GAU_ND(x, mu, C):
    """ Computes the Multivariate Gaussian log density for the dataset x
        C represents the covariance matrix sigma
    """
    # M is the number of rows of x, n of attributes for each sample
    M = x.shape[0]
    first = -(M/2) * np.log(2*np.pi)
    second = -0.5 * np.linalg.slogdet(C)[1]
    third = -0.5 * np.dot(
        np.dot((x-mu).T, np.linalg.inv(C)), (x - mu))

    return np.diag(first+second+third)


def multivariate_gaussian_classifier2(DTR, LTR, DTE):
    """ implementation of the  Multivariate Gaussian Classifier
        using log_densities
        DTR and LTR are training data and labels
        DTE are evaluation data 
        returns: the log-likelihood ratio
    """
    # Compute the ML estimates for the classifier parameters
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    mu0 = DTR0.mean(1)
    C0 = covariance_matrix2(DTR0)
    mu1 = DTR1.mean(1)
    C1 = covariance_matrix2(DTR1)

    # Compute for each test sample the likelihoods
    # Initialize the score matrix, S[i,j] contains the
    # class conditional probability for sample j given class i
    S = np.zeros((2, DTE.shape[1]))
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[0, j] = logpdf_GAU_ND(sample, vcol(mu0), C0)
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[1, j] = logpdf_GAU_ND(sample, vcol(mu1), C1)

    # return llr
    return S[1] - S[0]
    

def naive_bayes_gaussian_classifier(DTR, LTR, DTE):
    """ implementation of the  Naive Bayes Gaussian Classifier
        based on MVG version with log_densities,
        covariance matrixes are diagonal
        DTR and LTR are training data and labels
        DTE evaluation data 
        returns: the log-likelihood ratio
    """
    # Compute the ML estimates for the classifier parameters
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    mu0 = DTR0.mean(1)
    C0 = covariance_matrix2(DTR0)
    mu1 = DTR1.mean(1)
    C1 = covariance_matrix2(DTR1)

    # We need to zeroing the out of diagonal elements of the MVG ML solution
    # This can be done multiplying element-wise the MVG ML solution
    # with the identity matrix
    C0 *= np.identity(C0.shape[0])
    C1 *= np.identity(C1.shape[0])

    # Compute for each test sample the likelihoods
    # Initialize the score matrix, S[i,j] contains the
    # class conditional probability for sample j given class i
    S = np.zeros((2, DTE.shape[1]))
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[0, j] = logpdf_GAU_ND(sample, vcol(mu0), C0)
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[1, j] = logpdf_GAU_ND(sample, vcol(mu1), C1)

    return S[1] - S[0]


def within_class_covariance_matrix(DTR, LTR):
    """ computes the within class covariance matrix SW for the dataset D"""

    # select the samples of the i-th class
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    # to compute the within class covariance matrix, we have to sum
    # the covariance matrices of each class
    C0 = covariance_matrix2(DTR0)
    C1 = covariance_matrix2(DTR1)
    SW = (DTR0.shape[1] * C0 +
          DTR1.shape[1] * C1 ) / DTR.shape[1]
    return SW



def tied_covariance_gaussian_classifier(DTR, LTR, DTE):
    """ implementation of the Tied Covariance Gaussian Classifier
        based on MVG version with log_densities
        DTR and LTR are training data and labels
        DTE are evaluation data
        returns: the log-likelihood ratio
    """
    # Compute the ML estimates for the classifier parameters
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]

    mu0 = DTR0.mean(1)
    mu1 = DTR1.mean(1)

    C_star = within_class_covariance_matrix(DTR, LTR)

    # Compute for each test sample the likelihoods
    # Initialize the score matrix, S[i,j] contains the
    # class conditional probability for sample j given class i
    S = np.zeros((2, DTE.shape[1]))
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[0, j] = logpdf_GAU_ND(sample, vcol(mu0), C_star)
    for j, sample in enumerate(DTE.T):
        sample = vcol(sample)
        S[1, j] = logpdf_GAU_ND(sample, vcol(mu1), C_star)

    return S[1] - S[0]