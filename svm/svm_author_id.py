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

from sklearn.metrics import accuracy_score
from sklearn import svm

svm_classifier = svm.SVC(kernel='rbf', C=10000)
svm_classifier.fit(features_train, labels_train)
pred = svm_classifier.predict(features_test)
print accuracy_score(pred, labels_test)


print 'Prediction for 10 ' + str(pred[10])
print 'Prediction for 26 ' + str(pred[26])
print 'Prediction for 50 ' + str(pred[50])
print 'Chris number of predictions ' + str(len(filter(lambda x: x == 1,pred)))