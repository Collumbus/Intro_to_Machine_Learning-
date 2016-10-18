#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
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
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    t0 = time()
    clf.fit(features_train, labels_train)
    fit_t = round(time()-t0, 3)

    ### use the trained classifier to predict labels for the test features
    t0 = time()
    pred = clf.predict(features_test)
    pred_t = round(time()-t0, 3)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(labels_test, pred)
    return [fit_t, pred_t, acc]

results = classify(features_train, labels_train, features_test, labels_test)
print ('The fit time is: %.2f s' % results[0])
print ('The training time is: %.2f s' % results[1])
print ('The accuracy is: %.2f' % results[2])


#########################################################
