#Data processing
#Importing Libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
datasets=pd.read_csv("Data.csv")
X=datasets.iloc[:,:-1].values
Y=datasets.iloc[:,3].values #to display the 4th  column