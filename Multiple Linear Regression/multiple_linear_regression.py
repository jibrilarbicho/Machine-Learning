# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
#Avoiding  The Dummy Variable trap
#X=X[:,1:] to make not select 1st column but we don't make this manually b/c there library that handle this
print(X)
from sklearn.model_selection import train_test_split

# Assuming X and Y are your feature and target variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
