from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd

# built-in dataset from sklearn
from sklearn.datasets import load_breast_cancer

# load data
data = load_breast_cancer()

# check the type of data
type(data)

# note: it is a bunch object
# it acts as a dictionary where you can treat the keys like attributes
print(data.keys())

# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

# the 'data' attribute means the input data
print(data.data.shape)
# (569, 30) a two-dimensional arrary
# it has 569 samepls, 30 features

# 'targets'
print(data.target)
# note how the targets are just 0s and 1s
# normally, when you have K targets, they are labled 0..k-1

# but their meaning is not lost
# 0, 1 corresponds with ['malignant' 'benign']
print(data.target_names)

# determine the meaning of each feature
print(data.feature_names)

# normally we would put all of our imports
# at the top, but this lets us tell a story
from sklearn.model_selection import train_test_split

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

# instantiate a classifier and train it
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# evaluate the model's performance
# generally speaking, you do better on the train set rather than the test set
print(model.score(X_train, y_train))
print(model.score(X_test, y_test ))

# how you can make predictions
predictions = model.predict(X_test)

# what did we get?
print(predictions)

# manually check the performance
# nbr correct / ttl nbr of predictions 

N = len(y_test)

print(np.sum(predictions == y_test) / N)

# we can even use deep learning to solve the same problem!
from sklearn.neural_network import MLPClassifier

# you'll learn why scaling is needed later
from sklearn.preprocessing import StandardScaler