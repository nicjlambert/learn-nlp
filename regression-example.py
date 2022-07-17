from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import statsmodels.api as sm

# load data
df = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data

# check the data
print(df.head(5))
df.info()


# get the inputs
data = df[["cyl","disp","hp","drat", "wt","qsec","vs","am","gear","carb"]].values

# get the outputs
target = df[["mpg"]].values

# normally we would put all of our imports
# at the top, but this lets us tell a story
from sklearn.model_selection import train_test_split

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)

# instantiate a regression and train it
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# R^2
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# how can you make predictions
predictions = model.predict(X_test)
# what did we get?
print(predictions)

# we can use the same model (random forest) used in classification example
from sklearn.ensemble import RandomForestRegressor

model2 = RandomForestRegressor()
model2.fit(X_train, y_train)

print(model2.score(X_train, y_train))
print(model2.score(X_test, y_test))