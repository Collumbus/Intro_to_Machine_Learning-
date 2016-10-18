#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

def classify(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your SVM classifier """
    ### import the sklearn module for SVM

    from sklearn.svm import SVC


    #features_train = features_train[:len(features_train)/100]
    #labels_train = labels_train[:len(labels_train)/100]

    clf = SVC(kernel="rbf", C=10000)
    t0 = time()
    clf.fit(features_train, labels_train)
    fit_t = round(time()-t0, 3)

    t0 = time()
    pred = clf.predict(features_test)
    pred_t = round(time()-t0, 3)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return [fit_t, pred_t, acc, pred]

print ''
results = classify(features_train, labels_train, features_test, labels_test)
print ('The fit time is: %.2f s' % results[0])
print ('The training time is: %.2f s' % results[1])
print ('The accuracy is: %.2f' % results[2])

l = results[3]

print ('The class of index 10, 26, 50 is: %i, %i, %i' % (l[10], l[26], l[50]))

from collections import Counter
lll = Counter(l)
print ('The nunber of predicted Chris (1) class is %i ' % lll[1])
print ''


#########################################################
