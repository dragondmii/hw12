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
    ## your code
    pass

def run_cross_validation(dtr, n):
    ## your code
    pass

from sklearn.metrics import confusion_matrix
from matplotlib import pylab


def compute_train_test_confusion_matrix(classifier, test_size):
    ## your code
    pass



