import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Get dataset
dataset = pd.read_csv('ex1data1.txt')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1:].values
y0 = dataset.iloc[:, 1].values

#First Solution
ones = np.ones((X.shape[0], 1))
Xbar = np.concatenate((ones,X), 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

w0 = w[0][0]
w1 = w[1][0]
y1 = w0 + w1*X
#End First Solution


#Second Solution
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept=False)
regressor.fit(Xbar, y)
#End Second Solution

#Third Solution
regressor0 = LinearRegression()
regressor0.fit(X, y)
#End Third Solution


#3 ways of Visualize data
plt.figure(1)
plt.scatter(X, y, color='red')
plt.plot(X, y1, color='blue')

plt.figure(2)
plt.scatter(X, y, color ='red')
plt.plot(X, regressor.predict(Xbar), color = 'green')

plt.figure(3)
plt.scatter(X, y, color ='red')
plt.plot(X, regressor0.predict(X), color ='yellow')

plt.show()

print(w)
print(regressor.coef_)
print(regressor0.coef_)