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
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import svm

gaussian_naive_clf = GaussianNB()
gaussian_naive_clf.fit(features_train, labels_train)
gaussian_naive_pred = gaussian_naive_clf.predict(features_test)
print '\n\n Gaussian Naive bayes \n ------------------------\n' + str(
    accuracy_score(labels_test, gaussian_naive_pred) * 100)

bernoulli_naive_clf = BernoulliNB()
bernoulli_naive_clf.fit(features_train, labels_train)
bernoulli_naive_pred = bernoulli_naive_clf.predict(features_test)
print '\n\n Bernoulli Naive bayes \n ------------------------\n' + str(
    accuracy_score(labels_test, bernoulli_naive_pred) * 100)

multinomial_naive_clf = MultinomialNB()
multinomial_naive_clf.fit(features_train, labels_train)
multinomial_naive_pred = multinomial_naive_clf.predict(features_test)
print '\n\n Multinomial Naive bayes \n ------------------------\n' + str(
    accuracy_score(labels_test, multinomial_naive_pred) * 100)

svm_linear = svm.LinearSVC()
svm_linear.fit(features_train, labels_train)
svm_linear_pred = svm_linear.predict(features_test)
print '\n\nLinear SVM \n ------------------------\n' + str(accuracy_score(labels_test, svm_linear_pred) * 100)

from sklearn.linear_model import SGDClassifier

sgdc = SGDClassifier(loss="hinge", penalty="l2", max_iter=3, tol=None)
sgdc.fit(features_train, labels_train)
sgdc_pred = sgdc.predict(features_test)
print '\n\nSGDC \n ------------------------\n' + str(accuracy_score(labels_test, sgdc_pred) * 100)


