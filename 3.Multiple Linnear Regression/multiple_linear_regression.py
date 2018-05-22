#Import libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#Import the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding the State feature
# 0 0 1 : New York
# 1 0 0 : Cali
# 0 1 0 : Florida
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap


#Splitting the dataset into Trainning set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2 , random_state = 0)

#Fitting multiple linear regression to the trainning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict the test set result
y_pred = regressor.predict(X_test)

w = regressor.coef_
x1 = 1
x2 = 0
x3 = 120000
x4 = 100000
x5 = 310000

y0 = w[0] + w[1]*x1 + w[2]*x2 + w[3]*x3 + w[4]*x4 + w[5]*x5

print(dataset)
print('\n\tX')
print(X)
print("\n\ty")
print(y)
print("\n\tX_train")
print(X_train)
print("\n\tX_test")
print(X_test)
print("\n\ty_train")
print(y_train)
print("\n\ty_test")
print(y_test)
print("\n\ty_pred")
print(y_pred)
print("\n\tResult")
print(regressor.coef_)
print(y0)
