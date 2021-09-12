"""
	 In this module we have functions that interact with the data, (load data, analyze data, ...)
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import corrcoef


def load_train():
    """ Loads the training dataset from file "Train.txt"
        Returns a matrix D with the attributes of the samples
        and an array L with the classes
    """
    with open("data/Train.txt") as f:
        samples = []
        labels = []
        for line in f:
            f_1, f_2, f_3, f_4, l = line.split(",")
            samples.append(
                np.array([float(f_1), float(f_2), float(f_3), float(f_4)]).reshape(4, 1))
            labels.append(int(l))
            D = np.hstack(samples)
            L = np.array(labels)
    return D, L


def load_trainH():
    """ Loads the training dataset from file "TrainH.txt", which contains the data 
        degraded with noise
        Returns a matrix D with the attributes of the samples 
        and an array L with the classes
    """
    with open("data/TrainH.txt") as f:
        samples = []
        labels = []
        for line in f:
            f_1, f_2, f_3, f_4, l = line.split(",")
            samples.append(
                np.array([float(f_1), float(f_2), float(f_3), float(f_4)]).reshape(4, 1))
            labels.append(int(l))
            D = np.hstack(samples)
            L = np.array(labels)
    return D, L


def load_test():
    """ Loads the testing dataset from file "Test.txt"
        Returns a matrix D with the attributes of the samples 
        and an array L with the classes
    """
    with open("data/Test.txt") as f:
        samples = []
        labels = []
        for line in f:
            f_1, f_2, f_3, f_4, l = line.split(",")
            samples.append(
                np.array([float(f_1), float(f_2), float(f_3), float(f_4)]).reshape(4, 1))
            labels.append(int(l))
            D = np.hstack(samples)
            L = np.array(labels)
    return D, L


def load_testH():
    """ Loads the testing dataset from file "TestH.txt", which contains the data 
        degraded with noise
        Returns a matrix D with the attributes of the samples 
        and an array L with the classes
    """
    with open("data/TestH.txt") as f:
        samples = []
        labels = []
        for line in f:
            f_1, f_2, f_3, f_4, l = line.split(",")
            samples.append(
                np.array([float(f_1), float(f_2), float(f_3), float(f_4)]).reshape(4, 1))
            labels.append(int(l))
            D = np.hstack(samples)
            L = np.array(labels)
    return D, L


def count_labels(L):
    """ Count how many labels are equal to 0 and how many are equal to 1
        0 corresponds to authentic banknote, 1 to fake banknote
        L is the array of labels of the dataset, returns the two values C0 and C1
    """
    C0 = len(L[L == 0])
    C1 = len(L[L == 1])
    return C0, C1


def plot_features(D, L, save_name):
    """ Plot histograms for the features of the dataset
        D is a matrix with the attributes of the samples
        L is an array with the classes of the samples
    """

    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    # let's filter only the data corrisponging to each class
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    attributes = {
        0: 'variance',
        1: 'skewness',
        2: 'kurtosis',
        3: 'entropy'
    }

    # plot an histogram with all the data and the data of the two classes with different colors
    for index in range(4):
        plt.figure()
        plt.xlabel(attributes[index])

        plt.hist(D0[index, :], bins=20, density=True, alpha=.9,
                 label='authentic banknotes', color="red")
        plt.hist(D1[index, :], bins=20, density=True, alpha=.9,
                 label='fake banknotes', color="blue")
        plt.hist(D[index, :], bins=20, density=True, alpha=.7,
                 label='all banknotes', color="purple")
        plt.legend()
        plt.tight_layout()
        plt.savefig('./images/%s_%d.pdf' % (save_name, index))

    plt.show()

    return


def plot_heatmap_pearson(D, save_name):
    pearson_matrix = corrcoef(D)
    plt.imshow(pearson_matrix, cmap='Purples')
    plt.savefig('./images/%s.pdf' % (save_name))
    plt.show()
    return pearson_matrix


if __name__ == "__main__":

    # Load Train dataset from Train.txt and print it
    DTR, LTR = load_train()
    print("-" * 74)
    print("Train Dataset\n")
    print("samples:\n")
    print(DTR)
    print("\n")
    print("labels:\n")
    print(LTR)
    print("-" * 74, "\n\n")

    # Load Test dataset from Test.txt and print it
    DTE, LTE = load_test()
    print("-" * 74)
    print("Test Dataset\n")
    print("samples:\n")
    print(DTE)
    print("\n")
    print("labels:\n")
    print(LTE)
    print("-" * 74, "\n\n")

    # Load TrainH dataset from TrainH.txt and print it
    DHTR, LHTR = load_trainH()
    print("-" * 74)
    print("TrainH Dataset\n")
    print("samples:\n")
    print(DHTR)
    print("\n")
    print("labels:\n")
    print(LHTR)
    print("-" * 74, "\n\n")

    # Load TestH dataset from TestH.txt and print it
    DHTE, LHTE = load_testH()
    print("-" * 74)
    print("TestH Dataset\n")
    print("samples:\n")
    print(DHTE)
    print("\n")
    print("labels:\n")
    print(LHTE)
    print("-" * 74, "\n\n")

    # Count how many samples of class 0 and 1 are present in the datasets
    C0_TR, C1_TR = count_labels(LTR)
    C0_TE, C1_TE = count_labels(LTE)
    C0_HTR, C1_HTR = count_labels(LHTR)
    C0_HTE, C1_HTE = count_labels(LHTE)
    print("-" * 74)
    print("Count samples for different classes\n")
    print("Train Dataset: total=%d, authentic(0)=%d, fake(1)=%d\n" % (
          len(LTR), C0_TR, C1_TR))
    print("Test Dataset: total=%d, authentic(0)=%d, fake(1)=%d\n" %
          (len(LTE), C0_TE, C1_TE))
    print("TrainH Dataset: total=%d, authentic(0)=%d, fake(1)=%d\n" %
          (len(LHTR), C0_HTR, C1_HTR))
    print("TestH Dataset: total=%d, authentic(0)=%d, fake(1)=%d\n" % (
          len(LHTE), C0_HTE, C1_HTE))
    print("-" * 74, "\n\n")

    # Plot the features
    plot_features(DTR, LTR, "hist_DTR")
    plot_features(DHTR, LHTR, "hist_DHTR")

    # Plot the correlation matrix using Pearson correlation coefficient and print the matrixes
    corr_matrix_DTR = plot_heatmap_pearson(DTR, "corr_matrix_DTR")
    corr_matrix_DHTR = plot_heatmap_pearson(DHTR, "corr_matrix_DHTR")
    print("-" * 74)
    print("Heat map correlation matrix Train dataset\n")
    print(corr_matrix_DTR)
    print("\nHeat map correlation matrix TrainH dataset\n\n")
    print(corr_matrix_DHTR)
    print("-" * 74, "\n\n")
