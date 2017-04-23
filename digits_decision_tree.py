#!/usr/bin/python

###################################################
# module: iris_decision_tree.py
# description: A decision tree for DIGITS datasets
# Your Name
# Your A-Number
###################################################

from sklearn.datasets import load_digits
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np
from random import randint

digits_data = load_digits()
data_items = digits_data.data
target = digits_data.target

def run_train_test_split(classifier, n, test_size):
    for i in xrange(n):
        train_data, test_data, train_target, test_target = \
                    train_test_split(data_items, target,
                                     test_size=test_size, random_state=randint(0, 1000))
        dt = classifier.fit(train_data, train_target)
        print 'train/test run ',i,': accuracy = ',(sum(dt.predict(test_data) == test_target)/float(len(test_target)))
        print '------------------------------------------------------'
    return dt

def run_cross_validation(dtr, n):
    for i in xrange(n):
        print 'cross-validation run ',n
        for cv in xrange(5, 16):
            cross_val = cross_val_predict(dtr, data_items, target, cv=cv)
            acc = sum(cross_val==target)/float(len(target))
            print 'num_folders ',cv,', accuracy = ',acc
            print '------------------------------'

from sklearn.metrics import confusion_matrix
from matplotlib import pylab


def compute_train_test_confusion_matrix(classifier, test_size):
    train_data, test_data, train_target, test_target = \
                    train_test_split(data_items, target,
                                     test_size=test_size, random_state=randint(0, 1000))
    dt = classifier.fit(train_data, train_target)
    test_predict = dt.predict(test_data)
    cm = confusion_matrix(test_target, test_predict)
    plot_confusion_matrix(cm, ['0','1','2','3','4','5','6','7','8','9'], 'Digits Decision Tree')

def plot_confusion_matrix(cm, target_name_list, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(target_name_list)))
    ax.set_xticklabels(target_name_list)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(range(len(target_name_list)))
    ax.set_yticklabels(target_name_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.show()


## command line arguments
if __name__ == '__main__':
    clf = tree.DecisionTreeClassifier(random_state=0)
    #df = run_train_test_split(clf, 10, 0.3)
    #run_cross_validation(df, 2)
    compute_train_test_confusion_matrix(clf,200)

