"""
    Main module, where all models are imported and executed
"""

from dataset import k_fold, load_train, load_test, load_trainH, load_testH, split_db_2to1
from gaussian_models import multivariate_gaussian_classifier2, naive_bayes_gaussian_classifier, tied_covariance_gaussian_classifier
from gmm import GMM_classifier
from logistic_regression import logistic_regression
from measuring_predictions import minimum_detection_cost, actual_detection_cost
from numpy.lib.scimath import log

from svm import svm_kernel_RBF, svm_kernel_polynomial, svm_linear


# Flags to execute only some algorithms
FLAG_TRAINING = True
FLAG_TESTING = True 

FLAG_SINGLEFOLD = True
FLAG_KFOLD = True

FLAG_GAUSSIANS = True 
FLAG_LOGREG = True  
FLAG_SVM = True 
FLAG_GMM = True

FLAG_ACTUALDCF = True

if __name__ == "__main__":

    # Load datasets
    DTR, LTR = load_train()
    DTE, LTE = load_test()
    DHTR, LHTR = load_trainH()
    DHTE, LHTE = load_testH()

    # 3 applications: main balanced one and two unbalanced
    applications = [[0.5, 1, 1], [0.1, 1, 1], [0.9, 1, 1]]

    # SINGLE FOLD
    # split training data in data for training and evaluation

    # version without noise
    (DTR_T, LTR_T), (DTR_E, LTR_E) = split_db_2to1(DTR, LTR)
    # version with noise
    (DHTR_T, LHTR_T), (DHTR_E, LHTR_E) = split_db_2to1(DHTR, LHTR)

    # TRAINING
    if FLAG_TRAINING:
        print("-" * 50)
        print("-" * 50)
        print("TRAINING...")
        print("-" * 50)
        print("-" * 50, "\n\n")

        print("COMPUTING minDCF for various models...\n\n")

        # SINGLE FOLD
        if FLAG_SINGLEFOLD:

            print("-" * 50)
            print("SINGLE FOLD")
            print("-" * 50, "\n\n")

            flag = 1

            for app in applications:
                pi1, Cfn, Cfp = app
                print("-" * 50)
                print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" %
                      (pi1, Cfn, Cfp))
                print("-" * 50, "\n")

                # Gaussian Models
                if FLAG_GAUSSIANS:

                    generative_models = [("MVG", multivariate_gaussian_classifier2), (
                        "Naive", naive_bayes_gaussian_classifier), ("Tied", tied_covariance_gaussian_classifier)]
                    for name, algo in generative_models:
                        DCF_min = minimum_detection_cost(
                            algo(DTR_T, LTR_T, DTR_E), LTR_E, pi1, Cfn, Cfp)
                        print("%s: minDCF = %f" % (name, DCF_min), "\n")
                        DCF_min = minimum_detection_cost(
                            algo(DHTR_T, LHTR_T, DHTR_E), LHTR_E, pi1, Cfn, Cfp)
                        print("%s: minDCF = %f" %
                              (name, DCF_min), " (noise version)\n")
                    print("")

                # Logistic Regression
                if FLAG_LOGREG:
                    lambda_list = [0., 1e-6, 1e-3, 1.]
                    for l in lambda_list:
                        S = logistic_regression(DTR_T, LTR_T, DTR_E, l)
                        # We can recover log-likelihood ratios by subtracting from the score s
                        # the empirical prior log-odds of the training set (slide 31)
                        llr = S - log(pi1 / (1-pi1))
                        DCF_min = minimum_detection_cost(
                            llr, LTR_E, pi1, Cfn, Cfp)
                        print("Logistic Regression: lambda = %f, minDCF = %f" %
                              (l, DCF_min), "\n")

                        S = logistic_regression(DHTR_T, LHTR_T, DHTR_E, l)
                        # We can recover log-likelihood ratios by subtracting from the score s
                        # the empirical prior log-odds of the training set (slide 31)
                        llr = S - log(pi1 / (1-pi1))
                        DCF_min = minimum_detection_cost(
                            llr, LHTR_E, pi1, Cfn, Cfp)
                        print("Logistic Regression: lambda = %f, minDCF = %f" %
                              (l, DCF_min), " (noise version)\n")

                # SVM
                if FLAG_SVM:
                    # Linear
                    K_list = [1, 10]
                    C_list = [0.1, 1.0, 10.0]
                    for K in K_list:
                        for C in C_list:
                            DCF_min = minimum_detection_cost(svm_linear(
                                DTR_T, LTR_T, DTR_E, K, C), LTR_E, pi1, Cfn, Cfp)
                            print("SVM Linear: K = %f, C = %f, minDCF = %f" %
                                  (K, C, DCF_min), "\n")
                            DCF_min = minimum_detection_cost(svm_linear(
                                DHTR_T, LHTR_T, DHTR_E, K, C), LHTR_E, pi1, Cfn, Cfp)
                            print("SVM Linear: K = %f, C = %f, minDCF = %f" %
                                  (K, C, DCF_min), " (noise version)\n")
                    # Polynomial Kernel
                    c_list = [0, 1]
                    for K in K_list:
                        for C in C_list:
                            for c in c_list:
                                DCF_min = minimum_detection_cost(svm_kernel_polynomial(
                                    DTR_T, LTR_T, DTR_E, K, C, d=2, c=c), LTR_E, pi1, Cfn, Cfp)
                                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" % (
                                    K, C, c, DCF_min), "\n")
                                DCF_min = minimum_detection_cost(svm_kernel_polynomial(
                                    DHTR_T, LHTR_T, DHTR_E, K, C, d=2, c=c), LHTR_E, pi1, Cfn, Cfp)
                                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" % (
                                    K, C, c, DCF_min), " (noise version)\n")
                    # RBF Kernel
                    g_list = [1, 10]
                    for K in K_list:
                        for C in C_list:
                            for g in g_list:
                                DCF_min = minimum_detection_cost(svm_kernel_RBF(
                                    DTR_T, LTR_T, DTR_E, K, C, g=g), LTR_E, pi1, Cfn, Cfp)
                                print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" % (
                                    K, C, g, DCF_min), "\n")
                                DCF_min = minimum_detection_cost(svm_kernel_RBF(
                                    DHTR_T, LHTR_T, DHTR_E, K, C, g=g), LHTR_E, pi1, Cfn, Cfp)
                                print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" % (
                                    K, C, g, DCF_min), " (noise version)\n")

                # GMM
                if FLAG_GMM:
                    psi = 0.01
                    M_list = [2, 4, 8, 16]
                    versions = ["full", "diagonal", "tied"]
                    for version in versions:
                        for M in M_list:
                            DCF_min = minimum_detection_cost(GMM_classifier(
                                DTR_T, LTR_T, DTR_E, M, psi, version=version), LTR_E, pi1, Cfn, Cfp)
                            print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (
                                version, M, psi, DCF_min), "\n")
                            DCF_min = minimum_detection_cost(GMM_classifier(
                                DHTR_T, LHTR_T, DHTR_E, M, psi, version=version), LHTR_E, pi1, Cfn, Cfp)
                            print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (
                                version, M, psi, DCF_min), "(noise version)\n")

        # K-FOLD
        K = 5
        if FLAG_KFOLD:

            print("-" * 50)
            print("K FOLD: K = %d" % (K))
            print("-" * 50, "\n\n")

            for app in applications:
                pi1, Cfn, Cfp = app
                print("-" * 50)
                print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" %
                      (pi1, Cfn, Cfp))
                print("-" * 50, "\n")

                # Gaussian Models
                if FLAG_GAUSSIANS:

                    generative_models = [("MVG", multivariate_gaussian_classifier2), (
                        "Naive", naive_bayes_gaussian_classifier), ("Tied", tied_covariance_gaussian_classifier)]
                    for name, algo in generative_models:
                        llr, labels = k_fold(DTR, LTR, K, algo)
                        DCF_min = minimum_detection_cost(
                            llr, labels, pi1, Cfn, Cfp)
                        print("%s: minDCF = %f" % (name, DCF_min), "\n")

                        llr, labels = k_fold(DHTR, LHTR, K, algo, None)
                        DCF_min = minimum_detection_cost(
                            llr, labels, pi1, Cfn, Cfp)
                        print("%s: minDCF = %f" %
                              (name, DCF_min), " (noise version)\n")
                    print("")

                # Logistic Regression
                if FLAG_LOGREG:

                    lambda_list = [0., 1e-6, 1e-3, 1.]
                    for l in lambda_list:
                        S, labels = k_fold(
                            DTR, LTR, K, logistic_regression, (l,))
                        # We can recover log-likelihood ratios by subtracting from the score s
                        # the empirical prior log-odds of the training set (slide 31)
                        llr = S - log(pi1 / (1-pi1))
                        DCF_min = minimum_detection_cost(
                            llr, labels, pi1, Cfn, Cfp)
                        print("Logistic Regression: lambda = %f, minDCF = %f" %
                              (l, DCF_min), "\n")

                        S, labels = k_fold(
                            DHTR, LHTR, K, logistic_regression, (l,))
                        # We can recover log-likelihood ratios by subtracting from the score s
                        # the empirical prior log-odds of the training set (slide 31)
                        llr = S - log(pi1 / (1-pi1))
                        DCF_min = minimum_detection_cost(
                            llr, labels, pi1, Cfn, Cfp)
                        print("Logistic Regression: lambda = %f, minDCF = %f" %
                              (l, DCF_min), " (noise version)\n")

                # SVM
                if FLAG_SVM:
                    # Linear
                    K_list = [1, 10]
                    C_list = [0.1, 1.0, 10.0]
                    for K_ in K_list:
                        for C in C_list:
                            llr, labels = k_fold(
                                DTR, LTR, K, svm_linear, (K_, C))
                            DCF_min = minimum_detection_cost(
                                llr, labels, pi1, Cfn, Cfp)
                            print("SVM Linear: K = %f, C = %f, minDCF = %f" %
                                  (K_, C, DCF_min), "\n")

                            llr, labels = k_fold(
                                DHTR, LHTR, K, svm_linear, (K_, C))
                            DCF_min = minimum_detection_cost(
                                llr, labels, pi1, Cfn, Cfp)
                            print("SVM Linear: K = %f, C = %f, minDCF = %f" %
                                  (K_, C, DCF_min), " (noise version)\n")
                    # Polynomial Kernel
                    c_list = [0, 1]
                    for K_ in K_list:
                        for C in C_list:
                            for c in c_list:
                                llr, labels = k_fold(
                                    DTR, LTR, K, svm_kernel_polynomial, (K_, C, 2, c))
                                DCF_min = minimum_detection_cost(
                                    llr, labels, pi1, Cfn, Cfp)
                                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" % (
                                    K_, C, c, DCF_min), "\n")

                                llr, labels = k_fold(
                                    DHTR, LHTR, K, svm_kernel_polynomial, (K_, C, 2, c))
                                DCF_min = minimum_detection_cost(
                                    llr, labels, pi1, Cfn, Cfp)
                                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" % (
                                    K_, C, c, DCF_min), " (noise version)\n")
                    # RBF Kernel
                    g_list = [1, 10]
                    for K_ in K_list:
                        for C in C_list:
                            for g in g_list:
                                llr, labels = k_fold(
                                    DTR, LTR, K, svm_kernel_RBF, (K_, C, g))
                                DCF_min = minimum_detection_cost(
                                    llr, labels, pi1, Cfn, Cfp)
                                print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" % (
                                    K_, C, g, DCF_min), "\n")

                                llr, labels = k_fold(
                                    DHTR, LHTR, K, svm_kernel_RBF, (K_, C, g))
                                DCF_min = minimum_detection_cost(
                                    llr, labels, pi1, Cfn, Cfp)
                                print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" % (
                                    K_, C, g, DCF_min), " (noise version)\n")

                # GMM
                if FLAG_GMM:
                    psi = 0.01
                    M_list = [2, 4, 8, 16]
                    versions = ["full", "diagonal", "tied"]
                    for version in versions:
                        for M in M_list:
                            llr, labels = k_fold(
                                DTR, LTR, K, GMM_classifier, (M, psi, version))
                            DCF_min = minimum_detection_cost(
                                llr, labels, pi1, Cfn, Cfp)
                            print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (
                                version, M, psi, DCF_min), "\n")

                            llr, labels = k_fold(
                                DHTR, LHTR, K, GMM_classifier, (M, psi, version))
                            DCF_min = minimum_detection_cost(
                                llr, labels, pi1, Cfn, Cfp)
                            print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (
                                version, M, psi, DCF_min), " (noise version)\n")

        if FLAG_ACTUALDCF:
            # Calculate Actual DCF for the chosen models
            print("\n\nCOMPUTING actDCF for various models...\n\n")

            print("-" * 50)
            print("SINGLE FOLD")
            print("-" * 50, "\n\n")

            for app in applications:
                pi1, Cfn, Cfp = app
                print("-" * 50)
                print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" %
                      (pi1, Cfn, Cfp))
                print("-" * 50, "\n")

                # normal dataset

                # MVG full
                DCF_act = actual_detection_cost(multivariate_gaussian_classifier2(
                    DTR_T, LTR_T, DTR_E), LTR_E, pi1, Cfn, Cfp)
                print("%s: actDCF = %f" % ("MVG", DCF_act), "\n")

                # GMM full M = 2
                DCF_act = actual_detection_cost(GMM_classifier(
                    DTR_T, LTR_T, DTR_E, 2, 0.001, version="full"), LTR_E, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("full", 2, 0.001, DCF_act), "\n")

                # SVM RBF K=1, C=10, g=1
                DCF_act = actual_detection_cost(svm_kernel_RBF(
                    DTR_T, LTR_T, DTR_E, 1, 10, g=1), LTR_E, pi1, Cfn, Cfp)
                print("SVM RBF Kernel: K = %f, C = %f, g = %f, actDCF = %f" %
                      (1, 10, 1, DCF_act), "\n")

                # degraded dataset

                # GMM full M = 4
                DCF_act = actual_detection_cost(GMM_classifier(
                    DHTR_T, LHTR_T, DHTR_E, 4, 0.001, version="full"), LHTR_E, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("full", 4, 0.001, DCF_act), " (noise version)\n")

                # GMM diagonal M = 16
                DCF_act = actual_detection_cost(GMM_classifier(
                    DHTR_T, LHTR_T, DHTR_E, 16, 0.001, version="diagonal"), LHTR_E, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("diagonal", 16, 0.001, DCF_act), " (noise version)\n")

                # SVM Quad. K=1, C=0.1, c=0, d=2
                DCF_act = actual_detection_cost(svm_kernel_polynomial(
                    DHTR_T, LHTR_T, DHTR_E, 1, 0.1, d=2, c=0), LHTR_E, pi1, Cfn, Cfp)
                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, actDCF = %f" % (
                    1, 0.1, 0, DCF_act), " (noise version)\n")

            K = 5
            print("-" * 50)
            print("K FOLD: K = %d" % (K))
            print("-" * 50, "\n\n")

            for app in applications:
                pi1, Cfn, Cfp = app
                print("-" * 50)
                print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" %
                      (pi1, Cfn, Cfp))
                print("-" * 50, "\n")

                # normal dataset

                # MVG full

                llr, labels = k_fold(
                    DTR, LTR, K, multivariate_gaussian_classifier2)
                DCF_min = actual_detection_cost(llr, labels, pi1, Cfn, Cfp)
                print("%s: actDCF = %f" % ("MVG", DCF_act), "\n")

                # GMM full M = 2
                llr, labels = k_fold(
                    DTR, LTR, K, GMM_classifier, (2, 0.001, "full"))
                DCF_act = actual_detection_cost(llr, labels, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("full", 2, 0.001, DCF_act), "\n")

                # SVM RBF K=1, C=10, g=1
                llr, labels = k_fold(DTR, LTR, K, svm_kernel_RBF, (1, 10, 1))
                DCF_act = actual_detection_cost(llr, labels, pi1, Cfn, Cfp)
                print("SVM RBF Kernel: K = %f, C = %f, g = %f, actDCF = %f" %
                      (1, 10, 1, DCF_act), "\n")

                # degraded dataset

                # GMM full M = 4
                llr, labels = k_fold(
                    DHTR, LHTR, K, GMM_classifier, (4, 0.001, "full"))
                DCF_act = actual_detection_cost(llr, labels, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("full", 4, 0.001, DCF_act), " (noise version)\n")

                # GMM diagonal M = 16
                llr, labels = k_fold(
                    DHTR, LHTR, K, GMM_classifier, (16, 0.001, "diagonal"))
                DCF_act = actual_detection_cost(llr, labels, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("diagonal", 16, 0.001, DCF_act), " (noise version)\n")

                # SVM Quad. K=1, C=0.1, c=0, d=2

                llr, labels = k_fold(
                    DHTR, LHTR, K, svm_kernel_polynomial, (1, 0.1, 2, 0))
                DCF_act = actual_detection_cost(llr, labels, pi1, Cfn, Cfp)
                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, actDCF = %f" % (
                    1, 0.1, 0, DCF_act), " (noise version)\n")

    # TESTING
    if FLAG_TESTING:

        print("-" * 50)
        print("-" * 50)
        print("TESTING...")
        print("-" * 50)
        print("-" * 50, "\n\n")

        # SINGLEFOLD using only the splitted data for model training (spit is 2/3 like in TRAINING)
        if FLAG_SINGLEFOLD:

            print("-" * 50)
            print("SINGLE FOLD")
            print("-" * 50, "\n\n")

            for app in applications:
                pi1, Cfn, Cfp = app
                print("-" * 50)
                print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" %
                      (pi1, Cfn, Cfp))
                print("-" * 50, "\n")

                # Gaussian Models
                if FLAG_GAUSSIANS:

                    generative_models = [("MVG", multivariate_gaussian_classifier2), (
                        "Naive", naive_bayes_gaussian_classifier), ("Tied", tied_covariance_gaussian_classifier)]
                    for name, algo in generative_models:
                        DCF_min = minimum_detection_cost(
                            algo(DTR_T, LTR_T, DTE), LTE, pi1, Cfn, Cfp)
                        print("%s: minDCF = %f" % (name, DCF_min), "\n")
                        DCF_min = minimum_detection_cost(
                            algo(DHTR_T, LHTR_T, DHTE), LHTE, pi1, Cfn, Cfp)
                        print("%s: minDCF = %f" %
                              (name, DCF_min), " (noise version)\n")
                    print("")

                # Logistic Regression
                if FLAG_LOGREG:
                    lambda_list = [0., 1e-6, 1e-3, 1.]
                    for l in lambda_list:
                        S = logistic_regression(DTR_T, LTR_T, DTE, l)
                        # We can recover log-likelihood ratios by subtracting from the score s
                        # the empirical prior log-odds of the training set (slide 31)
                        llr = S - log(pi1 / (1-pi1))
                        DCF_min = minimum_detection_cost(
                            llr, LTE, pi1, Cfn, Cfp)
                        print("Logistic Regression: lambda = %f, minDCF = %f" %
                              (l, DCF_min), "\n")

                        S = logistic_regression(DHTR_T, LHTR_T, DHTE, l)
                        # We can recover log-likelihood ratios by subtracting from the score s
                        # the empirical prior log-odds of the training set (slide 31)
                        llr = S - log(pi1 / (1-pi1))
                        DCF_min = minimum_detection_cost(
                            llr, LHTE, pi1, Cfn, Cfp)
                        print("Logistic Regression: lambda = %f, minDCF = %f" %
                              (l, DCF_min), " (noise version)\n")

                # SVM
                if FLAG_SVM:
                    # Linear
                    K_list = [1, 10]
                    C_list = [0.1, 1.0, 10.0]
                    for K in K_list:
                        for C in C_list:
                            DCF_min = minimum_detection_cost(svm_linear(
                                DTR_T, LTR_T, DTE, K, C), LTE, pi1, Cfn, Cfp)
                            print("SVM Linear: K = %f, C = %f, minDCF = %f" %
                                  (K, C, DCF_min), "\n")
                            DCF_min = minimum_detection_cost(svm_linear(
                                DHTR_T, LHTR_T, DHTE, K, C), LHTE, pi1, Cfn, Cfp)
                            print("SVM Linear: K = %f, C = %f, minDCF = %f" %
                                  (K, C, DCF_min), " (noise version)\n")
                    # Polynomial Kernel
                    c_list = [0, 1]
                    for K in K_list:
                        for C in C_list:
                            for c in c_list:
                                DCF_min = minimum_detection_cost(svm_kernel_polynomial(
                                    DTR_T, LTR_T, DTE, K, C, d=2, c=c), LTE, pi1, Cfn, Cfp)
                                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" % (
                                    K, C, c, DCF_min), "\n")
                                DCF_min = minimum_detection_cost(svm_kernel_polynomial(
                                    DHTR_T, LHTR_T, DHTE, K, C, d=2, c=c), LHTE, pi1, Cfn, Cfp)
                                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" % (
                                    K, C, c, DCF_min), " (noise version)\n")
                    # RBF Kernel
                    g_list = [1, 10]
                    for K in K_list:
                        for C in C_list:
                            for g in g_list:
                                DCF_min = minimum_detection_cost(svm_kernel_RBF(
                                    DTR_T, LTR_T, DTE, K, C, g=g), LTE, pi1, Cfn, Cfp)
                                print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" % (
                                    K, C, g, DCF_min), "\n")
                                DCF_min = minimum_detection_cost(svm_kernel_RBF(
                                    DHTR_T, LHTR_T, DHTE, K, C, g=g), LHTE, pi1, Cfn, Cfp)
                                print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" % (
                                    K, C, g, DCF_min), " (noise version)\n")

                # GMM
                if FLAG_GMM:
                    psi = 0.01
                    M_list = [2, 4, 8, 16]
                    versions = ["full", "diagonal", "tied"]
                    for version in versions:
                        for M in M_list:
                            DCF_min = minimum_detection_cost(GMM_classifier(
                                DTR_T, LTR_T, DTE, M, psi, version=version), LTE, pi1, Cfn, Cfp)
                            print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (
                                version, M, psi, DCF_min), "\n")
                            DCF_min = minimum_detection_cost(GMM_classifier(
                                DHTR_T, LHTR_T, DHTE, M, psi, version=version), LHTE, pi1, Cfn, Cfp)
                            print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (
                                version, M, psi, DCF_min), "(noise version)\n")

        # K-FOLD: using all training data for the models
        if FLAG_KFOLD:

            print("-" * 50)
            print("K FOLD")
            print("-" * 50, "\n\n")

            for app in applications:
                pi1, Cfn, Cfp = app
                print("-" * 50)
                print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" %
                      (pi1, Cfn, Cfp))
                print("-" * 50, "\n")

                # Gaussian Models
                if FLAG_GAUSSIANS:

                    generative_models = [("MVG", multivariate_gaussian_classifier2), (
                        "Naive", naive_bayes_gaussian_classifier), ("Tied", tied_covariance_gaussian_classifier)]
                    for name, algo in generative_models:
                        DCF_min = minimum_detection_cost(
                            algo(DTR, LTR, DTE), LTE, pi1, Cfn, Cfp)
                        print("%s: minDCF = %f" % (name, DCF_min), "\n")
                        DCF_min = minimum_detection_cost(
                            algo(DHTR, LHTR, DHTE), LHTE, pi1, Cfn, Cfp)
                        print("%s: minDCF = %f" %
                              (name, DCF_min), " (noise version)\n")
                    print("")

                # Logistic Regression
                if FLAG_LOGREG:
                    lambda_list = [0., 1e-6, 1e-3, 1.]
                    for l in lambda_list:
                        S = logistic_regression(DTR, LTR, DTE, l)
                        # We can recover log-likelihood ratios by subtracting from the score s
                        # the empirical prior log-odds of the training set (slide 31)
                        llr = S - log(pi1 / (1-pi1))
                        DCF_min = minimum_detection_cost(
                            llr, LTE, pi1, Cfn, Cfp)
                        print("Logistic Regression: lambda = %f, minDCF = %f" %
                              (l, DCF_min), "\n")

                        S = logistic_regression(DHTR, LHTR, DHTE, l)
                        # We can recover log-likelihood ratios by subtracting from the score s
                        # the empirical prior log-odds of the training set (slide 31)
                        llr = S - log(pi1 / (1-pi1))
                        DCF_min = minimum_detection_cost(
                            llr, LHTE, pi1, Cfn, Cfp)
                        print("Logistic Regression: lambda = %f, minDCF = %f" %
                              (l, DCF_min), " (noise version)\n")

                # SVM
                if FLAG_SVM:
                    # Linear
                    K_list = [1, 10]
                    C_list = [0.1, 1.0, 10.0]
                    for K in K_list:
                        for C in C_list:
                            DCF_min = minimum_detection_cost(svm_linear(
                                DTR, LTR, DTE, K, C), LTE, pi1, Cfn, Cfp)
                            print("SVM Linear: K = %f, C = %f, minDCF = %f" %
                                  (K, C, DCF_min), "\n")
                            DCF_min = minimum_detection_cost(svm_linear(
                                DHTR, LHTR, DHTE, K, C), LHTE, pi1, Cfn, Cfp)
                            print("SVM Linear: K = %f, C = %f, minDCF = %f" %
                                  (K, C, DCF_min), " (noise version)\n")
                    # Polynomial Kernel
                    c_list = [0, 1]
                    for K in K_list:
                        for C in C_list:
                            for c in c_list:
                                DCF_min = minimum_detection_cost(svm_kernel_polynomial(
                                    DTR, LTR, DTE, K, C, d=2, c=c), LTE, pi1, Cfn, Cfp)
                                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" % (
                                    K, C, c, DCF_min), "\n")
                                DCF_min = minimum_detection_cost(svm_kernel_polynomial(
                                    DHTR, LHTR, DHTE, K, C, d=2, c=c), LHTE, pi1, Cfn, Cfp)
                                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" % (
                                    K, C, c, DCF_min), " (noise version)\n")
                    # RBF Kernel
                    g_list = [1, 10]
                    for K in K_list:
                        for C in C_list:
                            for g in g_list:
                                DCF_min = minimum_detection_cost(svm_kernel_RBF(
                                    DTR, LTR, DTE, K, C, g=g), LTE, pi1, Cfn, Cfp)
                                print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" % (
                                    K, C, g, DCF_min), "\n")
                                DCF_min = minimum_detection_cost(svm_kernel_RBF(
                                    DHTR, LHTR, DHTE, K, C, g=g), LHTE, pi1, Cfn, Cfp)
                                print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" % (
                                    K, C, g, DCF_min), " (noise version)\n")

                # GMM
                if FLAG_GMM:
                    psi = 0.01
                    M_list = [2, 4, 8, 16]
                    versions = ["full", "diagonal", "tied"]
                    for version in versions:
                        for M in M_list:
                            DCF_min = minimum_detection_cost(GMM_classifier(
                                DTR, LTR, DTE, M, psi, version=version), LTE, pi1, Cfn, Cfp)
                            print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (
                                version, M, psi, DCF_min), "\n")
                            DCF_min = minimum_detection_cost(GMM_classifier(
                                DHTR, LHTR, DHTE, M, psi, version=version), LHTE, pi1, Cfn, Cfp)
                            print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (
                                version, M, psi, DCF_min), "(noise version)\n")

        if FLAG_ACTUALDCF:
            # Calculate Actual DCF for the chosen models
            print("\n\nCOMPUTING actDCF for various models...\n\n")

            print("-" * 50)
            print("SINGLE FOLD")
            print("-" * 50, "\n\n")

            for app in applications:
                pi1, Cfn, Cfp = app
                print("-" * 50)
                print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" %
                      (pi1, Cfn, Cfp))
                print("-" * 50, "\n")

                # normal dataset

                # MVG full
                DCF_act = actual_detection_cost(multivariate_gaussian_classifier2(
                    DTR_T, LTR_T, DTE), LTE, pi1, Cfn, Cfp)
                print("%s: actDCF = %f" % ("MVG", DCF_act), "\n")

                # GMM full M = 2
                DCF_act = actual_detection_cost(GMM_classifier(
                    DTR_T, LTR_T, DTE, 2, 0.001, version="full"), LTE, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("full", 2, 0.001, DCF_act), "\n")

                # SVM RBF K=1, C=10, g=1
                DCF_act = actual_detection_cost(svm_kernel_RBF(
                    DTR_T, LTR_T, DTE, 1, 10, g=1), LTE, pi1, Cfn, Cfp)
                print("SVM RBF Kernel: K = %f, C = %f, g = %f, actDCF = %f" %
                      (1, 10, 1, DCF_act), "\n")

                # degraded dataset

                # GMM full M = 4
                DCF_act = actual_detection_cost(GMM_classifier(
                    DHTR_T, LHTR_T, DHTE, 4, 0.001, version="full"), LHTE, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("full", 4, 0.001, DCF_act), " (noise version)\n")

                # GMM diagonal M = 16
                DCF_act = actual_detection_cost(GMM_classifier(
                    DHTR_T, LHTR_T, DHTE, 16, 0.001, version="diagonal"), LHTE, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("diagonal", 16, 0.001, DCF_act), " (noise version)\n")

                # SVM Quad. K=1, C=0.1, c=0, d=2
                DCF_act = actual_detection_cost(svm_kernel_polynomial(
                    DHTR_T, LHTR_T, DHTE, 1, 0.1, d=2, c=0), LHTE, pi1, Cfn, Cfp)
                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, actDCF = %f" % (
                    1, 0.1, 0, DCF_act), " (noise version)\n")

            K = 5
            print("-" * 50)
            print("K FOLD: K = %d" % (K))
            print("-" * 50, "\n\n")

            for app in applications:
                pi1, Cfn, Cfp = app
                print("-" * 50)
                print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" %
                      (pi1, Cfn, Cfp))
                print("-" * 50, "\n")

                # normal dataset

                # MVG full
                DCF_act = actual_detection_cost(
                    multivariate_gaussian_classifier2(DTR, LTR, DTE), LTE, pi1, Cfn, Cfp)
                print("%s: actDCF = %f" % ("MVG", DCF_act), "\n")

                # GMM full M = 2
                DCF_act = actual_detection_cost(GMM_classifier(
                    DTR, LTR, DTE, 2, 0.001, version="full"), LTE, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("full", 2, 0.001, DCF_act), "\n")

                # SVM RBF K=1, C=10, g=1
                DCF_act = actual_detection_cost(svm_kernel_RBF(
                    DTR, LTR, DTE, 1, 10, g=1), LTE, pi1, Cfn, Cfp)
                print("SVM RBF Kernel: K = %f, C = %f, g = %f, actDCF = %f" %
                      (1, 10, 1, DCF_act), "\n")

                # degraded dataset

                # GMM full M = 4
                DCF_act = actual_detection_cost(GMM_classifier(
                    DHTR, LHTR, DHTE, 4, 0.001, version="full"), LHTE, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("full", 4, 0.001, DCF_act), " (noise version)\n")

                # GMM diagonal M = 16
                DCF_act = actual_detection_cost(GMM_classifier(
                    DHTR, LHTR, DHTE, 16, 0.001, version="diagonal"), LHTE, pi1, Cfn, Cfp)
                print("GMM: version = %s, M = %d, psi = %f, actDCF = %f" %
                      ("diagonal", 16, 0.001, DCF_act), " (noise version)\n")

                # SVM Quad. K=1, C=0.1, c=0, d=2
                DCF_act = actual_detection_cost(svm_kernel_polynomial(
                    DHTR, LHTR, DHTE, 1, 0.1, d=2, c=0), LHTE, pi1, Cfn, Cfp)
                print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, actDCF = %f" % (
                    1, 0.1, 0, DCF_act), " (noise version)\n")
