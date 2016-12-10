# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:58 2016

@author: cs390mb

Assignment 2 : Activity Recognition

This is the starter script used to train an activity recognition
classifier on accelerometer data.

See the assignment details for instructions. Basically you will train
a decision tree classifier and vary its parameters and evalute its
performance by computing the average accuracy, precision and recall
metrics over 10-fold cross-validation. You will then train another
classifier for comparison.

Once you get to part 4 of the assignment, where you will collect your
own data, change the filename to reference the file containing the
data you collected. Then retrain the classifier and choose the best
classifier to save to disk. This will be used in your final system.

Make sure to chek the assignment details, since the instructions here are
not complete.

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from features import extract_features # make sure features.py is in the same directory
from util import slidingWindow, reorient, reset_vars
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pickle



# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = os.path.join('my-activity-data.csv')
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

# print("Reorienting accelerometer data...")
# sys.stdout.flush()
# reset_vars()
# reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
# reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
# data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)


# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# you may want to play around with the window and step sizes
window_size = 100
step_size = 100

# sampling rate for the sample data should be about 25 Hz; take a brief window to confirm this
# n_samples = 1000
# time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
# sampling_rate = n_samples / time_elapsed_seconds

feature_names = ["mean X", "mean Y", "mean Z",
                 "Variance X","Variance Y", "Variance Z",
                 "minX","minY","minZ",
                 "maxX","maxY","maxZ",
                 "medianX","medianY","medianZ",
                 "zeroX","zeroY","zeroZ",
                 "meanCX","meanCY","meanCZ",
                 "FFTX","FFTY","FFTZ",
                 "magnitude","entropy","PeakX",
                 "PeakY","PeakZ","location_distance","mean_heartRate"]
class_names = ["Walking", "Running","Sitting","Driving"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

n_features = len(feature_names)

X = np.zeros((0,n_features))
y = np.zeros(0,)

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    # omit timestamp and label from accelerometer window for feature extraction:
    window = window_with_timestamp_and_label[:,1:-1]
    # extract features over window:
    x = extract_features(window)
    # append features:
    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    # append label:
    print(window_with_timestamp_and_label[10, -1])
    y = np.append(y, window_with_timestamp_and_label[10, -1])
    print("Getting label")
    print(np.shape(y))

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Plot data points
#
# -----------------------------------------------------------------------------

# We provided you with an example of plotting two features.
# We plotted the mean X acceleration against the mean Y acceleration.
# It should be clear from the plot that these two features are alone very uninformative.
# print("Plotting data points...")
# lenghtArray = range(len(feature_names))
# mapDic = {feature_names[i]:i for i in lenghtArray}
# sys.stdout.flush()
# plt.figure()
# formats = ['bo', 'go']
# for i in range(0,len(y),10): # only plot 1/10th of the points, it's a lot of data!
#     plt.plot(X[i,mapDic["zeroX"]], X[i,mapDic["PeakX"]], formats[int(y[i])])
#
# plt.show()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

# TODO: Train and evaluate your decision tree classifier over 10-fold CV.
# Report average accuracy, precision and recall metrics.
tree  =   DecisionTreeClassifier ( criterion = "entropy" ,  max_depth = 3)
cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
for i, (train_indexes, test_indexes) in enumerate(cv):
    print("Fold {} : The confusion matrix is :".format(i))
    X_train = X[train_indexes, :]
    y_train = y[train_indexes]
    X_test = X[test_indexes, :]
    y_test = y[test_indexes]
    tree.fit(X_train, y_train)
    print(X_test)
    y_pred = tree.predict(X_test)
    conf = confusion_matrix(y_test,y_pred)
    print(conf)

sum_accuracy = []
sum_precision = []
sum_recall = []
accuracy = sum(np.diagonal(conf))/(np.sum(conf)*1.0)
precision_stationary= (conf[0,0]/(sum(conf[:, 0]*1.0)))
print("Precision of Stationary is {}".format(precision_stationary))
precision_walking = (conf[1,1]/(sum(conf[:, 1])*1.0))
print("Precision of walking is {}".format(precision_walking))
recall_stationary = (conf[0,0]/(sum(conf[0])*1.0))
print("Recall of Stationary is {}".format(recall_stationary))
recall_walking = (conf[1,1]/(sum(conf[1])*1.0))
print("Recall of walking is {}".format(recall_walking))
sum_output = []

sum_output.append(accuracy)
sum_accuracy.append(accuracy)

sum_output.append(precision_walking)
sum_precision.append(precision_walking)
sum_output.append(precision_walking)
sum_precision.append(precision_walking)

sum_output.append(recall_stationary)
sum_recall.append(recall_stationary)
sum_output.append(recall_walking)
sum_recall.append(recall_walking)


for index in range(len(sum_output)):
        if np.isnan(sum_output[index]):
            sum_output[index] = 0

for index in range(len(sum_accuracy)):
            if np.isnan(sum_accuracy[index]):
                sum_accuracy[index] = 0

for index in range(len(sum_precision)):
            if np.isnan(sum_precision[index]):
                sum_precision[index] = 0

for index in range(len(sum_recall)):
            if np.isnan(sum_recall[index]):
                sum_recall[index] = 0

print("the average of accuracy is {}".format(np.mean(sum_accuracy)))
print("the average of precision is {}".format(np.mean(sum_precision)))
print("the average of recall is {}".format(np.mean(sum_recall)))


# TODO: Evaluate another classifier, i.e. SVM, Logistic Regression, k-NN, etc.
#SVM
svc = svm.LinearSVC()
for i, (train_indexes, test_indexes) in enumerate(cv):
        print("Fold {} : The confusion matrix is :".format(i))
        X_train = X[train_indexes, :]
        y_train = y[train_indexes]
        X_test = X[test_indexes, :]
        y_test = y[test_indexes]
        svc.fit(X_train,y_train)
        y_pred = svc.predict(X_test)
        conf = confusion_matrix(y_test,y_pred)
        print(conf)
sum_accuracy = []
sum_precision = []
sum_recall = []
accuracy = sum(np.diagonal(conf))/(np.sum(conf)*1.0)
precision_stationary= (conf[0,0]/(sum(conf[:, 0]*1.0)))
print("Precision of Stationary is {}".format(precision_stationary))
precision_walking = (conf[1,1]/(sum(conf[:, 1])*1.0))
print("Precision of walking is {}".format(precision_walking))
recall_stationary = (conf[0,0]/(sum(conf[0])*1.0))
print("Recall of Stationary is {}".format(recall_stationary))
recall_walking = (conf[1,1]/(sum(conf[1])*1.0))
print("Recall of walking is {}".format(recall_walking))

sum_output.append(accuracy)
sum_accuracy.append(accuracy)

sum_output.append(precision_walking)
sum_precision.append(precision_walking)
sum_output.append(precision_walking)
sum_precision.append(precision_walking)

sum_output.append(recall_stationary)
sum_recall.append(recall_stationary)
sum_output.append(recall_walking)
sum_recall.append(recall_walking)


for index in range(len(sum_output)):
        if np.isnan(sum_output[index]):
            sum_output[index] = 0

for index in range(len(sum_accuracy)):
            if np.isnan(sum_accuracy[index]):
                sum_accuracy[index] = 0

for index in range(len(sum_precision)):
            if np.isnan(sum_precision[index]):
                sum_precision[index] = 0

for index in range(len(sum_recall)):
            if np.isnan(sum_recall[index]):
                sum_recall[index] = 0

print("the average of accuracy is {}".format(np.mean(sum_accuracy)))
print("the average of precision is {}".format(np.mean(sum_precision)))
print("the average of recall is {}".format(np.mean(sum_recall)))

# TODO: Once you have collected data, train your best model on the entire
# dataset. Then save it to disk as follows:
tree.fit(X, y)
export_graphviz (tree ,out_file = 'tree.dot' ,  feature_names  =  feature_names)

# when ready, set this to the best model you found, trained on all the data:
best_classifier = svc
with open('classifier.pickle', 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)
