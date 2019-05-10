Steps for using this code:

step 0: replace the file path of HSI dataset in train_HSI.py and test_HSI.py

step 1: run train_HSI.py, the main function receives two parameters(i.e., random seed and cutout) and returns the optimized neural architecture.

step 2: replace the architecture parameter (i.e., HSI) in genotypes.py with the one returned by step 1.

step 3: run test_HSI.py, the main function receives three parameters (i.e., architecture parameter, random seed, and cutout) and
returns the confusion matrix of classification result on test dataset.

The parameter random seed in step1 and step3 can control the split of training, validation, and test dataset. To ensure the
training and validation samples are the same in step1 and step3, the random seed parameter should be set to the same number
in step1 and step3.
