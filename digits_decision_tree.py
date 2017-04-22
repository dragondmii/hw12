#!/usr/bin/python

###################################################
# module: iris_decision_tree.py
# description: A decision tree for DIGITS datasets
# Your Name
# Your A-Number
###################################################

from sklearn.datasets import load_digits
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
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
        #print('train data\'s size %d' % train_data.shape[0])
        #print('train target: %s' % str(train_target))
        #print('test data\'s size %d' % test_data.shape[0])
        #print('test target: %s' % str(test_target))
        dt = classifier.fit(train_data, train_target)
        print(sum(dt.predict(test_data) == test_target)/float(len(test_target)))

def run_cross_validation(dtr, n):
    for i in xrange(n):
        ## cv specifies the number of folds data is split
        for cv in xrange(5, 16):
            cross_val = cross_val_predict(dtr, data_items, target, cv=cv)
            acc = sum(cross_val==target)/float(len(target))
            print cv, acc

from sklearn.metrics import confusion_matrix
from matplotlib import pylab


def compute_train_test_confusion_matrix(classifier, test_size):
    train_data, test_data, train_target, test_target = \
                    train_test_split(data_items, target,
                                     test_size=test_size, random_state=randint(0, 1000))
    dt = classifier.fit(train_data, train_target)
    test_predict = dt.predict(test_data)
    cm = confusion_matrix(test_target, test_predict)
    plot_confusion_matrix(cm, ['setosa', 'versicolor', 'virginica'], 'IRIS DT CM')

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




