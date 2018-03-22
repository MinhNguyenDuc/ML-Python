#Data preprocessing
#Import libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to the traing set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict the test result
y_pred = regressor.predict(X_test)

#Visualize the Training set result
plt.figure(1)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Year of Ex')
plt.ylabel('Salary')
plt.show()

#Visualize the Test set
plt.figure(2)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



print(dataset)
print("\nX")
print(X)
print("\n\ty")
print(y)

print("\nX_train")
print(X_train)
print("\nX_test")
print(X_test)
print("\ny_train")
print(y_train)
print("\ny_test")
print(y_test)
print("y_pred")
print(y_pred)
