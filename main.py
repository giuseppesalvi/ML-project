"""
    main module, where all models are imported and executed
"""

from dataset import k_fold, load_train, load_test, load_trainH, load_testH, split_db_2to1
from gaussian_models import multivariate_gaussian_classifier2, naive_bayes_gaussian_classifier, tied_covariance_gaussian_classifier
from gmm import GMM_classifier
from logistic_regression import logistic_regression
from measuring_predictions import minimum_detection_cost
from numpy.lib.scimath import log

from svm import svm_kernel_RBF, svm_kernel_polynomial, svm_linear


# Flags to execute only some algorithms
FLAG_SINGLEFOLD = False 
FLAG_KFOLD = False 

FLAG_GAUSSIANS = True 
FLAG_LOGREG = True 
FLAG_SVM = False 
FLAG_GMM = False 

if __name__ == "__main__":

    # Load datasets
    DTR, LTR = load_train()
    DTE, LTE = load_test()
    DHTR, LHTR = load_trainH()
    DHTE, LHTE = load_testH()


    print("COMPUTING minDCF for various models...\n\n")
   
    # 3 applications: main balanced one and two unbalanced
    #applications = [[0.5, 1, 1], [0.1, 1, 1], [0.9, 1, 1]]
    applications = [[0.5, 1, 1]]
    #applications = [[0.9, 1, 1]]


    # SINGLE FOLD
    if FLAG_SINGLEFOLD:
    
        # split training data in data for training and evaluation 

        # version without noise
        (DTR_T, LTR_T), (DTR_E, LTR_E) = split_db_2to1(DTR, LTR)
        # version with noise
        (DHTR_T, LHTR_T), (DHTR_E, LHTR_E) = split_db_2to1(DHTR, LHTR)


        print("-" * 50)
        print("SINGLE FOLD")
        print("-" * 50, "\n\n")

        flag = 1


        for app in applications:
            pi1, Cfn, Cfp = app
            print("-" * 50)
            print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" % (pi1, Cfn, Cfp))
            print("-" * 50, "\n")

            # Gaussian Models
            if FLAG_GAUSSIANS:

                generative_models = [("MVG", multivariate_gaussian_classifier2), ("Naive", naive_bayes_gaussian_classifier), ("Tied", tied_covariance_gaussian_classifier)]
                for name, algo in generative_models:
                    DCF_min = minimum_detection_cost(algo(DTR_T, LTR_T, DTR_E), LTR_E, pi1, Cfn, Cfp)
                    print("%s: minDCF = %f" %(name,DCF_min),"\n") # 0.0 for MVG and tied!!
                    if flag == 1:
                        if DCF_min != 0:
                            flag = 2
                        else: 
                            flag = 0

                    DCF_min = minimum_detection_cost(algo(DHTR_T, LHTR_T, DHTR_E), LHTR_E, pi1, Cfn, Cfp)
                    print("%s: minDCF = %f" %(name,DCF_min)," (noise version)\n") 
                print("")

            # Logistic Regression
            if FLAG_LOGREG:
                lambda_list = [0., 1e-6, 1e-3, 1.]
                for l in lambda_list:
                    S = logistic_regression(DTR_T, LTR_T, DTR_E, l)
                    # We can recover log-likelihood ratios by subtracting from the score s
                    # the empirical prior log-odds of the training set (slide 31)
                    llr = S - log(pi1/ (1-pi1))
                    DCF_min = minimum_detection_cost(llr, LTR_E, pi1, Cfn, Cfp)
                    print("Logistic Regression: lambda = %f, minDCF = %f" %(l,DCF_min),"\n") # 0.0 for MVG and tied!!

                    S = logistic_regression(DHTR_T, LHTR_T, DHTR_E, l)
                    # We can recover log-likelihood ratios by subtracting from the score s
                    # the empirical prior log-odds of the training set (slide 31)
                    llr = S - log(pi1/ (1-pi1))
                    DCF_min = minimum_detection_cost(llr, LHTR_E, pi1, Cfn, Cfp)
                    print("Logistic Regression: lambda = %f, minDCF = %f" %(l,DCF_min)," (noise version)\n") # 0.0 for MVG and tied!!

            # SVM
            if FLAG_SVM:
                # Linear
                K_list = [1, 10]
                C_list = [0.1, 1.0, 10.0]
                for K in K_list:
                    for C in C_list:
                        DCF_min = minimum_detection_cost(svm_linear(DTR_T, LTR_T, DTR_E, K, C), LTR_E, pi1, Cfn, Cfp)
                        print("SVM Linear: K = %f, C = %f, minDCF = %f" %(K, C, DCF_min),"\n")
                        DCF_min = minimum_detection_cost(svm_linear(DHTR_T, LHTR_T, DHTR_E, K, C), LHTR_E, pi1, Cfn, Cfp)
                        print("SVM Linear: K = %f, C = %f, minDCF = %f" %(K, C, DCF_min)," (noise version)\n")
                # Polynomial Kernel
                c_list = [0,1]
                for K in K_list:
                    for C in C_list:
                        for c in c_list:
                            DCF_min = minimum_detection_cost(svm_kernel_polynomial(DTR_T, LTR_T, DTR_E, K, C, d=2, c=c), LTR_E, pi1, Cfn, Cfp)
                            print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" %(K, C, c, DCF_min),"\n")
                            DCF_min = minimum_detection_cost(svm_kernel_polynomial(DHTR_T, LHTR_T, DHTR_E, K, C, d=2, c=c), LHTR_E, pi1, Cfn, Cfp)
                            print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" %(K, C, c, DCF_min)," (noise version)\n")
                # RBF Kernel
                g_list = [1, 10]
                for K in K_list:
                    for C in C_list:
                        for g in g_list:
                            DCF_min = minimum_detection_cost(svm_kernel_RBF(DTR_T, LTR_T, DTR_E, K, C, g=g), LTR_E, pi1, Cfn, Cfp)
                            print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" %(K, C, g, DCF_min),"\n")
                            DCF_min = minimum_detection_cost(svm_kernel_RBF(DHTR_T, LHTR_T, DHTR_E, K, C, g=g), LHTR_E, pi1, Cfn, Cfp)
                            print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" %(K, C, g, DCF_min)," (noise version)\n")


            # GMM
            if FLAG_GMM:
                psi = 0.01
                M_list = [2, 4, 8, 16]
                versions = ["full", "diagonal", "tied"]
                for version in versions:
                    for M in M_list:
                        DCF_min = minimum_detection_cost(GMM_classifier(DTR_T, LTR_T, DTR_E, M, psi, version=version), LTR_E, pi1, Cfn, Cfp)
                        print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (version, M, psi, DCF_min), "\n")
                        DCF_min = minimum_detection_cost(GMM_classifier(DHTR_T, LHTR_T, DHTR_E, M, psi, version=version), LHTR_E, pi1, Cfn, Cfp)
                        print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (version, M, psi, DCF_min), "(noise version)\n")

        print("\n\n\n", flag) # TODO: remove

        
    # K-FOLD
    K = 5 
    #K = LTR.size
    if FLAG_KFOLD:
    
        print("-" * 50)
        print("K FOLD: K = %d" %(K))
        print("-" * 50, "\n\n")


        for app in applications:
            pi1, Cfn, Cfp = app

            # Gaussian Models
            if FLAG_GAUSSIANS:

                generative_models = [("MVG", multivariate_gaussian_classifier2), ("Naive", naive_bayes_gaussian_classifier), ("Tied", tied_covariance_gaussian_classifier)]
                for name, algo in generative_models:
                    llr, labels = k_fold(DTR, LTR, K, algo)
                    DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                    print("%s: minDCF = %f" %(name,DCF_min),"\n") 

                    llr, labels = k_fold(DHTR, LHTR, K, algo, None)
                    DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                    print("%s: minDCF = %f" %(name,DCF_min)," (noise version)\n") 
                print("")


            # Logistic Regression
            if FLAG_LOGREG:

                lambda_list = [0., 1e-6, 1e-3, 1.]
                for l in lambda_list:
                    S, labels = k_fold(DTR, LTR, K, logistic_regression, (l,))
                    # We can recover log-likelihood ratios by subtracting from the score s
                    # the empirical prior log-odds of the training set (slide 31)
                    llr = S - log(pi1/ (1-pi1))
                    DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                    print("Logistic Regression: lambda = %f, minDCF = %f" %(l,DCF_min),"\n") 

                    S, labels = k_fold(DHTR, LHTR, K, logistic_regression, (l,))
                    # We can recover log-likelihood ratios by subtracting from the score s
                    # the empirical prior log-odds of the training set (slide 31)
                    llr = S - log(pi1/ (1-pi1))
                    DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                    print("Logistic Regression: lambda = %f, minDCF = %f" %(l,DCF_min)," (noise version)\n") 

            # SVM
            if FLAG_SVM:
                # Linear
                K_list = [1, 10]
                C_list = [0.1, 1.0, 10.0]
                for K_ in K_list:
                    for C in C_list:
                        llr, labels = k_fold(DTR, LTR, K, svm_linear, (K_, C))
                        DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                        print("SVM Linear: K = %f, C = %f, minDCF = %f" %(K_, C, DCF_min),"\n")

                        llr, labels = k_fold(DHTR, LHTR, K, svm_linear, (K_, C))
                        DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                        print("SVM Linear: K = %f, C = %f, minDCF = %f" %(K_, C, DCF_min)," (noise version)\n")
                # Polynomial Kernel
                c_list = [0,1]
                for K_ in K_list:
                    for C in C_list:
                        for c in c_list:
                            llr, labels = k_fold(DTR, LTR, K, svm_kernel_polynomial, (K_, C, 2, c))
                            DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                            print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" %(K_, C, c, DCF_min),"\n")

                            llr, labels = k_fold(DHTR, LHTR, K, svm_kernel_polynomial, (K_, C, 2, c))
                            DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                            print("SVM Polynomial Kernel: K = %f, C = %f, d = 2, c = %f, minDCF = %f" %(K_, C, c, DCF_min)," (noise version)\n")
                # RBF Kernel
                g_list = [1, 10]
                for K_ in K_list:
                    for C in C_list:
                        for g in g_list:
                            llr, labels = k_fold(DTR, LTR, K, svm_kernel_RBF, (K_, C, g))
                            DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                            print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" %(K_, C, g, DCF_min),"\n")

                            llr, labels = k_fold(DHTR, LHTR, K, svm_kernel_RBF, (K_, C, g))
                            DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                            print("SVM RBF Kernel: K = %f, C = %f, g = %f, minDCF = %f" %(K_, C, g, DCF_min)," (noise version)\n")

            # GMM
            if FLAG_GMM:
                psi = 0.01
                M_list = [2, 4, 8, 16]
                versions = ["full", "diagonal", "tied"]
                for version in versions:
                    for M in M_list:
                        llr, labels = k_fold(DTR, LTR, K, GMM_classifier, (M, psi, version))
                        DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                        print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (version, M, psi, DCF_min), "\n")

                        llr, labels = k_fold(DHTR, LHTR, K, GMM_classifier, (M, psi, version))
                        DCF_min = minimum_detection_cost(llr, labels, pi1, Cfn, Cfp)
                        print("GMM: version = %s, M = %d, psi = %f, minDCF = %f" % (version, M, psi, DCF_min), " (noise version)\n")





