# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X =np.array( ct.fit_transform(X))
#Avoiding  The Dummy Variable trap
#X=X[:,1:] to make not select 1st column but we don't make this manually b/c there library that handle this
#print(X)
from sklearn.model_selection import train_test_split

# Assuming X and Y are your feature and target variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_predict=regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1 )),1))
#import statsmodels.formula.api as sm
#X=np.append(arr=X,values=np.ones((50,1)).astype(int))
from sklearn.metrics import r2_score
print(r2_score(y_test,y_predict))