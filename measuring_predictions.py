"""
    In these module we have all the functions used to measure our predictions
"""

import numpy as np
from numpy.lib.scimath import log

def confusion_matrix(predicted_labels, real_labels, K):
    """ Computes the confusion matrix given the predicted labels and
        the real labels
        K is the size of the matrix (K x K)
    """
    # Initialize the matrix of size K x K with zeros
    conf_matrix = np.zeros((K, K))

    # The element of the matrix in position i,j represents the number
    # of samples belonging to class j that are predicted as class i
    for i in range(predicted_labels.size):
        conf_matrix[predicted_labels[i]][real_labels[i]] += 1

    return conf_matrix


def optimal_bayes_decisions(llr, pi1, Cfn, Cfp, threshold=None):
    """ Computes optimal Bayes decisions starting from the binary 
        log-likelihoods ratios
        llr is the array of log-likelihoods ratios
        pi1 is the prior class probability of class 1 (True)
        Cfp = C1,0 is the cost of false positive errors, that is the cost of 
        predicting class 1 (True) when the actual class is 0 (False)
        Cfn = C0,1 is the cost of false negative errors that is the cost of 
        predicting class 0 (False) when the actual class is 1 (True)
    """

    # initialize an empty array for predictions of samples
    predictions = np.empty(llr.shape, int)

    # compare the log-likelihood ratio with threshold to predict the class
    # if the threshold is not specified use the theoretical optimal threshold
    if (threshold == None):
        threshold = - log((pi1 * Cfn) / ((1 - pi1) * Cfp))
    for i in range(llr.size):
        if llr[i] > threshold:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return predictions


def empirical_bayes_risk(confusion_matrix, pi1, Cfn, Cfp):
    """ Computes the Bayes risk (or detection cost) from the consufion matrix 
        corresponding to the optimal decisions for an 
        application (pi1, Cfn, Cfp)
    """

    # FNR = false negative rate
    FNR = confusion_matrix[0][1] / \
        (confusion_matrix[0][1] + confusion_matrix[1][1])

    # FPR = false positive rate
    FPR = confusion_matrix[1][0] / \
        (confusion_matrix[0][0] + confusion_matrix[1][0])

    # We can compute the empirical bayes risk, or detection cost function DCF
    # using this formula
    DCF = pi1 * Cfn * FNR + (1-pi1) * Cfp * FPR

    return DCF
def normalized_detection_cost(DCF, pi1, Cfn, Cfp):
    """ Computes the normalized detection cost, given the detection cost DCF,
        and the parameters of the application, pi1, Cfn, Cfp
    """

    # We can compute the normalized detection cost (or bayes risk)
    # by dividing the bayes risk by the risk of an optimal system that doen not
    # use the test data at all

    # The cost of such system is given by this formula
    DCFdummy = pi1 * Cfn if (pi1 * Cfn < (1-pi1) * Cfp) else (1-pi1) * Cfp

    return DCF / DCFdummy


def minimum_detection_cost(llr, labels, pi1, Cfn, Cfp):
    """ Compute the minimum detection cost, given the binary
        log likelihood ratios llr
        labels is the array of labels
        pi1, Cfn, Cfp are the parameters for the application
    """

    # consider a set of thresholds corresponding to (-inf, s1, ... , sM, inf)
    # where s1 ... sM are the test scores, sorted in increasing order
    thresholds = np.append(llr, [np.inf, -np.inf])
    thresholds.sort()

    min_DCF = np.inf
    for t in thresholds:
        # compare the log-likelihood ratio with threshold to predict the class
        predictions = optimal_bayes_decisions(llr, pi1, Cfn, Cfp, t)

        # compute the confusion matrix
        conf = confusion_matrix(predictions, labels, 2)

        # compute DCF_norm
        DCF = empirical_bayes_risk(conf, pi1, Cfn, Cfp)
        DCF_norm = normalized_detection_cost(DCF, pi1, Cfn, Cfp)

        # set DCF as minimum if true
        if DCF_norm < min_DCF:
            min_DCF = DCF_norm

    return min_DCF


def actual_detection_cost(llr, labels, pi1, Cfn, Cfp):
    """ Compute the actual detection cost, given the binary
        log likelihood ratios llr
        labels is the array of labels
        pi1, Cfn, Cfp are the parameters for the application
    """

    min_DCF = np.inf
    # compare the log-likelihood ratio with  the optimal bayes decision threshold to predict the class
    predictions = optimal_bayes_decisions(llr, pi1, Cfn, Cfp)

    # compute the confusion matrix
    conf = confusion_matrix(predictions, labels, 2)

    # compute DCF_norm
    DCF = empirical_bayes_risk(conf, pi1, Cfn, Cfp)
    DCF_norm = normalized_detection_cost(DCF, pi1, Cfn, Cfp)

    return DCF_norm 

