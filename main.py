"""
    main module, where all models are imported from other files and executed
"""

from posixpath import split
from dataset import load_train, load_test, load_trainH, load_testH, split_db_2to1
from gaussian_models import multivariate_gaussian_classifier2, naive_bayes_gaussian_classifier, tied_covariance_gaussian_classifier
from measuring_predictions import minimum_detection_cost

if __name__ == "__main__":

    # Load datasets
    DTR, LTR = load_train()
    DTE, LTE = load_test()
    DHTR, LHTR = load_trainH()
    DHTE, LHTE = load_testH()


    print("COMPUTING minDCF for various models...\n\n")

    # Single fold
    print("SINGLE FOLD\n")

    # split training data in data for training and evaluation 

    # version without noise
    (DTR_T, LTR_T), (DTR_E, LTR_E) = split_db_2to1(DTR, LTR)
    # version with noise
    (DHTR_T, LHTR_T), (DHTR_E, LHTR_E) = split_db_2to1(DHTR, LHTR)
    

    # Main Application
    pi1 = 0.5
    Cfn = 1
    Cfp = 1
    print("MAIN APPLICATION: pi1 = %f, Cfn = %d, Cfp = %d\n" %(pi1,Cfn,Cfp))

    # Gaussian Models

    # MVG

    DCF_min = minimum_detection_cost(multivariate_gaussian_classifier2(DTR_T, LTR_T, DTR_E), LTR_E, pi1, Cfn, Cfp)
    print("MVG: minDCF = %f" %(DCF_min),"\n") # 0.0 !!

    DCF_min = minimum_detection_cost(multivariate_gaussian_classifier2(DHTR_T, LHTR_T, DHTR_E), LHTR_E, pi1, Cfn, Cfp)
    print("MVG (noise version): minDCF = %f" %(DCF_min),"\n") # 0.0 !!


    # Naive 

    DCF_min = minimum_detection_cost(naive_bayes_gaussian_classifier(DTR_T, LTR_T, DTR_E), LTR_E, pi1, Cfn, Cfp)
    print("naive: minDCF = %f" %(DCF_min),"\n") # 0.0 !!

    DCF_min = minimum_detection_cost(naive_bayes_gaussian_classifier(DHTR_T, LHTR_T, DHTR_E), LHTR_E, pi1, Cfn, Cfp)
    print("naive (noise version): minDCF = %f" %(DCF_min),"\n") # 0.0 !!


    # Tied

    DCF_min = minimum_detection_cost(tied_covariance_gaussian_classifier(DTR_T, LTR_T, DTR_E), LTR_E, pi1, Cfn, Cfp)
    print("tied: minDCF = %f" %(DCF_min),"\n") # 0.0 !!

    DCF_min = minimum_detection_cost(tied_covariance_gaussian_classifier(DHTR_T, LHTR_T, DHTR_E), LHTR_E, pi1, Cfn, Cfp)
    print("tied (noise version): minDCF = %f" %(DCF_min),"\n") # 0.0 !!


    # Unbalanced Applications



    # K - fold


    # Main Application
    # Unbalanced Applications


    applications = [[0.5, 1, 1], [0.1, 1, 1], [0.9, 1, 1]]

    generative_models = [("MVG", multivariate_gaussian_classifier2), ("Naive", naive_bayes_gaussian_classifier), ("Tied", tied_covariance_gaussian_classifier)]


    for app in applications:
        pi1, Cfn, Cfp = app
        print("-" * 70)
        print("Application: pi1 = %f, Cfn = %d, Cfn = %d" % (pi1, Cfn, Cfp))
        print("-" * 70, "\n")

        for model in generative_models:
            name, algo = model
            DCF_min = minimum_detection_cost(algo(DTR_T, LTR_T, DTR_E), LTR_E, pi1, Cfn, Cfp)
            print("%s: minDCF = %f" %(name,DCF_min),"\n") # 0.0 !!

            DCF_min = minimum_detection_cost(algo(DHTR_T, LHTR_T, DHTR_E), LHTR_E, pi1, Cfn, Cfp)
            print("%s (noise version): minDCF = %f" %(name,DCF_min),"\n") # 0.0 !!
        
        print("")


        