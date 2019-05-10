Steps for using this code:
step 0: replace the file path of HSI dataset in HSI_Search.py and HSI_Classificaion.py
step 1: run HSI_Search.py, the main function receives a parameter(i.e., random seed) and returns the optimized neural architecture.
step 2: replace the architecture parameter (i.e., HSI) in genotypes.py with the one returned by step 1.
step 3: run HSI_Classificaion.py, the main function receives two parameters (i.e., architecture parameter and random seed) and
        returns the confusion matrix of classification result on test dataset.

The parameter random seed in step1 and step2 can control the split of training, validation, and test dataset. To ensure the
training and validation samples are the same in step1 and step2, the random seed parameter should be set to the same number
in step1 and step2.
