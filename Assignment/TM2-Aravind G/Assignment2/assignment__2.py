# -*- coding: utf-8 -*-
"""Assignment__2.ipynb



1. Download the dataset: Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""2. Load the dataset."""

df=pd.read_csv('/Churn_Modelling.csv')
df.head()

"""3. Perform Below Visualizations.
Univariate Analysis
"""

sns.displot(df.CreditScore)

sns.barplot(df.HasCrCard.value_counts().index,df.HasCrCard.value_counts())

"""Bi - Variate Analysis"""

sns.lineplot(df.Age,df.CreditScore)

sns.lineplot(df.Tenure,df.EstimatedSalary)

"""Multi - Variate Analysis"""

sns.pairplot(df)

"""4. Perform descriptive statistics on the dataset."""

df.describe()

df.Gender.value_counts()

df.Surname.value_counts()

df.Geography.value_counts()

"""5. Handle the Missing values."""

df.isnull().any()

"""6. Find the outliers and replace the outliers"""

sns.boxplot(df.RowNumber)

q1=df.RowNumber.quantile(0.25)
q3=df.Age.quantile(0.75)

IQR=q3-q1

upper_limit = q3 + 1.5 * IQR
lower_limit = q1- 1.5 * IQR

upper_limit

df.median()

"""7. Check for Categorical columns and perform encoding."""

df.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df.Surname=le.fit_transform(df.Surname)
df.Gender=le.fit_transform(df.Gender)

df.head()

df_main=pd.get_dummies(df,columns=['Tenure'])
df_main.head()

df_main.corr()

"""8. Split the data into dependent and independent variables."""

Y=df_main['Balance']
Y

X=df_main.drop(columns=['Balance'],axis=1)
X.head()

X=df_main.drop(columns=['Geography'],axis=1)
X.head()

"""9. Scale the independent variables"""

from sklearn.preprocessing import scale

X_scaled=pd.DataFrame(scale(X),columns=X.columns)
X_scaled.head()

"""10. Split the data into training and testing"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,Y,test_size=0.3,random_state=0)

X_train.shape

y_train.shape

X_test.shape

y_test.shape
