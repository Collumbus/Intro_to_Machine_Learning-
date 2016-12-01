#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)
sort_keys = '../tools/python2_lesson14_keys.pkl'

### your code goes here
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,
labels,test_size=0.3,random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print ''
print ('The accuracy is: %.4f' % (accuracy_score(labels_test,pred)))
print ''
print (confusion_matrix(labels_test, pred))
print ''
print ('There are %i POIs.' % len([e for e in labels_test if e == 1.0]))
print ''
print ('There are %i people in test set.' %len(features_test))
print ''
print ('The accuracy would be %.3f' % ((29.-4.)/(29.)))
print ''
print 'By confusion matrix we can see that there are any true positive.'
print ''
print ('The precision is: %.4f' % (precision_score(labels_test,pred)))
print ''
print ('The recall is: %.4f' % (recall_score(labels_test,pred)))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

tp = 0
for p,t in zip(predictions,true_labels):
    if (p == 1) and (t==1):
        tp += 1
print ''
print ('There are %i true positives.' % tp)
print ''
tp = 0
for p,t in zip(predictions,true_labels):
    if (p == 0) and (t==0):
        tp += 1

print ('There are %i true negatives.' % tp)
print ''
tp = 0
for p,t in zip(predictions,true_labels):
    if (p == 1) and (t==0):
        tp += 1

print ('There are %i false positives.' % tp)
print ''
tp = 0
for p,t in zip(predictions,true_labels):
    if (p == 0) and (t==1):
        tp += 1

print ('There are %i false negatives.' % tp)
print ''
print ('The precision is: %.4f' % (precision_score(true_labels, predictions)))

print ''
print ('The recall is: %.4f' % (recall_score(true_labels, predictions)))
print ''
