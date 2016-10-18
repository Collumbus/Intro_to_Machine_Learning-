#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary

from time import time

'''def classify(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your SVM classifier """

    from sklearn.ensemble import RandomForestClassifier


    rfc = RandomForestClassifier(n_estimators=100)
    t0 = time()
    rfc.fit(features_train, labels_train)
    fit_t = round(time()-t0, 3)

    t0 = time()
    pred = rfc.predict(features_test)
    pred_t = round(time()-t0, 3)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return [fit_t, pred_t, acc, pred]

print ''
results = classify(features_train, labels_train, features_test, labels_test)
print ('The fit time is: %.2f s' % results[0])
print ('The training time is: %.2f s' % results[1])
print ('The accuracy is: %.2f' % results[2])



try:
    prettyPicture(rfc, features_test, labels_test)
except NameError:
    pass'''

def classify(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your SVM classifier """
    ### import the sklearn module for SVM

    from sklearn.svm import SVC


    #features_train = features_train[:len(features_train)/100]
    #labels_train = labels_train[:len(labels_train)/100]

    clf = SVC(kernel="rbf", C=1e8)
    t0 = time()
    clf.fit(features_train, labels_train)
    fit_t = round(time()-t0, 3)

    t0 = time()
    pred = clf.predict(features_test)
    pred_t = round(time()-t0, 3)

    from sklearn.metrics import accuracy_score
    acc = (accuracy_score(pred, labels_test))*100
    return [fit_t, pred_t, acc, pred]

print ''
results = classify(features_train, labels_train, features_test, labels_test)
print ('The fit time is: %.2f s' % results[0])
print ('The training time is: %.2f s' % results[1])
print ('The accuracy is: %.2f' % results[2])



try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
