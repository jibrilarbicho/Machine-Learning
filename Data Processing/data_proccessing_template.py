#Data processing
#Importing Libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
datasets=pd.read_csv("Data.csv")
X=datasets.iloc[:,:-1].values
Y=datasets.iloc[:,3].values #to display the 4th  column
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

columnTransformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [0])
    ],
    remainder='passthrough'  
)
X= columnTransformer.fit_transform(X)
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)
#Spliting the Datset into Training set and test set
from sklearn.model_selection import train_test_split

# Assuming X and Y are your feature and target variables
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)