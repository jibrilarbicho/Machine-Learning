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