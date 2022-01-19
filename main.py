"""
    main module, where all models are imported and executed
"""

from dataset import load_train, load_test, load_trainH, load_testH, split_db_2to1
from gaussian_models import multivariate_gaussian_classifier2, naive_bayes_gaussian_classifier, tied_covariance_gaussian_classifier
from logistic_regression import logistic_regression
from measuring_predictions import minimum_detection_cost
from numpy.lib.scimath import log


if __name__ == "__main__":

    # Load datasets
    DTR, LTR = load_train()
    DTE, LTE = load_test()
    DHTR, LHTR = load_trainH()
    DHTE, LHTE = load_testH()


    print("COMPUTING minDCF for various models...\n\n")
   
    # 3 applications: main balanced one and two unbalanced
    applications = [[0.5, 1, 1], [0.1, 1, 1], [0.9, 1, 1]]

    generative_models = [("MVG", multivariate_gaussian_classifier2), ("Naive", naive_bayes_gaussian_classifier), ("Tied", tied_covariance_gaussian_classifier)]


    # SINGLE FOLD
    
    # split training data in data for training and evaluation 

    # version without noise
    (DTR_T, LTR_T), (DTR_E, LTR_E) = split_db_2to1(DTR, LTR)
    # version with noise
    (DHTR_T, LHTR_T), (DHTR_E, LHTR_E) = split_db_2to1(DHTR, LHTR)


    print("-" * 50)
    print("SINGLE FOLD")
    print("-" * 50, "\n\n")


    for app in applications:
        pi1, Cfn, Cfp = app
        print("-" * 50)
        print("Application: pi1 = %.1f, Cfn = %d, Cfn = %d" % (pi1, Cfn, Cfp))
        print("-" * 50, "\n")

        # Gaussian Models
        #print("-" * 50)
        #print("GENERATIVE MODELS")
        #print("-" * 50, "\n\n")
        for name, algo in generative_models:
            DCF_min = minimum_detection_cost(algo(DTR_T, LTR_T, DTR_E), LTR_E, pi1, Cfn, Cfp)
            print("%s: minDCF = %f" %(name,DCF_min),"\n") # 0.0 for MVG and tied!!

            DCF_min = minimum_detection_cost(algo(DHTR_T, LHTR_T, DHTR_E), LHTR_E, pi1, Cfn, Cfp)
            print("%s: minDCF = %f" %(name,DCF_min)," (noise version)\n") 
        print("")

        # Logistic Regression

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
            print("Logistic Regression: lambda = %f, minDCF = %f" %(l,DCF_min),"(noise version)\n") # 0.0 for MVG and tied!!



        
    # K-FOLD