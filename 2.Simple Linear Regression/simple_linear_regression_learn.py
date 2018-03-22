#Import libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

#Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Data preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fit linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Visualize training set
plt.figure(1)
plt.scatter(X_train, y_train, color = 'red')
plt.title('Training set')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.xlabel('Year of Ex')
plt.ylabel('Salary')
plt.show()

#Visualize test set
plt.figure(2)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Test set')
plt.xlabel('Year of Ex')
plt.ylabel('Salary')
plt.show()


print('\n\tX')
print(X)
print("\n\t y")
print(y)
print('\n\tX_train')
print(X_train)
print('\n\tX_test')
print(X_test)
print('\n\ty_train')
print(y_train)
print('\n\ty_test')
print(y_test)