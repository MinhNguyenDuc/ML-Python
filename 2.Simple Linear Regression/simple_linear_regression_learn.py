import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fit linear regerssion
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Visualize the trainning set
plt.figure(1)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Trainning set')
plt.xlabel("Year of Ex")
plt.ylabel("Salary")

#Visualize the test set
plt.figure(2)
plt.scatter(X_test , y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Test set")
plt.xlabel("Year of Ex")
plt.ylabel("Salary")
plt.show()

print(dataset)
print(X)
print(y)
print(X_train)
print(X_test)
print(y_train)
print(y_test)