# -*- coding: utf-8 -*-
"""Assignment__3.ipynb

"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

# 1. Download the dataset
# 2. Load the dataset
data = pd.read_csv('/content/abalone.csv')
print(data)

data.head()

data.dtypes

data['age'] = data['Rings']+1.5
data = data.drop('Rings', axis = 1)

# 3.1. Univariate analysis (Scatter plot)
plt.scatter(data.Height,data['Whole weight'])
plt.show()

# 3.1. Univariate Analysis(Histogram)
plt.hist(data['age'])

# 3.2 bivariate analysis (barplot)
sns.barplot(x='Sex',y='Length',data=data)

# 3.2 Bivariate analysis(countplot)
sns.countplot(x='Sex',data=data)

# 3.3 Multivariate analysis
pd.plotting.scatter_matrix(data.loc[:,"Sex":"age" ],diagonal="kde",figsize=(20,15))
plt.show()

# 4. Perform descriptive statistics on the dataset
d = {'Sex':pd.Series(['M','M','F','M','I','I','F','F','M']),'age':pd.Series([16.5,8.5,10.5,11.5,8.5,9.5,21.5,17.5,10.5])}
df = pd.DataFrame(d)
print (df)

print (df.sum())

print (df.mean())

print (df.mode())

print (df.median())

print (df.count())

# 5. Handle the missing values
data.isnull()

# 6. Find the outliers 
# Visualization using box plot
sns.boxplot(data['Shucked weight'])

data = pd.get_dummies(data)
dummy_data = data

var = 'Viscera weight'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)

data.drop(data[(data['Viscera weight'] > 0.5) &
          (data['age'] < 20)].index, inplace = True)
data.drop(data[(data['Viscera weight']<0.5) & (
data['age'] > 25)].index, inplace = True)

var = 'Shell weight'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)

data.drop(data[(data['Shell weight'] > 0.6) &
          (data['age'] < 25)].index, inplace = True)
data.drop(data[(data['Shell weight']<0.8) & (
data['age'] > 25)].index, inplace = True)

var = 'Shucked weight'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)

data.drop(data[(data['Shucked weight'] >= 1) &
          (data['age'] < 20)].index, inplace = True)
data.drop(data[(data['Viscera weight']<1) & (
data['age'] > 20)].index, inplace = True)

var = 'Whole weight'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)

data.drop(data[(data['Whole weight'] >= 2.5) &
          (data['age'] < 25)].index, inplace = True)
data.drop(data[(data['Whole weight']<2.5) & (
data['age'] > 25)].index, inplace = True)

var = 'Diameter'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)

data.drop(data[(data['Diameter'] <0.1) &
          (data['age'] < 5)].index, inplace = True)
data.drop(data[(data['Diameter']<0.6) & (
data['age'] > 25)].index, inplace = True)
data.drop(data[(data['Diameter']>=0.6) & (
data['age'] < 25)].index, inplace = True)

var = 'Height'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)

var = 'Length'
plt.scatter(x = data[var], y = data['age'])
plt.grid(True)

# 8.Split the dependent and independent variables
X = data.drop('age', axis = 1)
y = data['age']

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest

#9.Scale the independent variables
standardScale = StandardScaler()
standardScale.fit_transform(X)

selectkBest = SelectKBest()
X_new = selectkBest.fit_transform(X, y)

# 10.Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25)

from sklearn.linear_model import LinearRegression

# 11.Build the model using LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

#12.Train the model
y_train_pred = lm.predict(X_train)

#13.Test the model
y_test_pred = lm.predict(X_test)

#14.Measure the performance using Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
s = mean_squared_error(y_train, y_train_pred)
print('Mean Squared error of training set :%2f'%s)

p = mean_squared_error(y_test, y_test_pred)
print('Mean Squared error of testing set :%2f'%p)

from sklearn.metrics import r2_score
s = r2_score(y_train, y_train_pred)
print('R2 Score of training set:%.2f'%s)

p = r2_score(y_test, y_test_pred)
print('R2 Score of testing set:%.2f'%p)
