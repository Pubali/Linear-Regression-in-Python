# -*- coding: utf-8 -*-
"""
Created on Tue May 08 18:40:42 2018

@author: pubali.bhaduri
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
#import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv("weight-height.csv",index_col=False)
df.head()
df.shape
df.plot(kind='scatter',
        x='Height',
        y='Weight',
        title='Weight and Height in adults')
x_train, x_test, y_train, y_test = train_test_split(df['Height'], df['Weight'], test_size=0.3)
x_train = np.reshape(x_train, (-1,1))
x_test = np.reshape(x_test, (-1,1))
y_train = np.reshape(y_train, (-1,1))
y_test = np.reshape(y_test, (-1,1))
print x_train
print('Train - Predictors shape', x_train.shape)
print('Test - Predictors shape', x_test.shape)
print('Train - Target shape', y_train.shape)
print('Test - Target shape', y_test.shape)

cls = linear_model.LinearRegression()
#Fit method is used for fitting your training data into the model
cls.fit(x_train,y_train)
prediction = cls.predict(x_test)
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
print('Model R^2 Square value', metrics.r2_score(y_test, prediction))
plt.scatter(x_test, y_test)
plt.plot(x_test, prediction, color='red', linewidth=3)
plt.xlabel('Hours')
plt.ylabel('Marks')
plt.title('Linear Regression')
#residual Plot

plt.scatter(cls.predict(x_test), cls.predict(x_test) - y_test, c='g', s = 40)
plt.hlines(y=0, xmin=0, xmax=100)
plt.title('Residual plot')
plt.ylabel('Residual')