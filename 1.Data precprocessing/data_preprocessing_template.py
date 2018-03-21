#Data preprocessing

#Importing libraries

#Math library numpy
import numpy as np
#Plot data library use matplotlib
import matplotlib.pyplot as plt
#pandas use to import the dataset
import pandas as pd

#Importing the dataset
dataset= pd.read_csv('Data.csv')
#Tạo 1 ma trận X, hàm iloc cho phép lấy dữ liệu vào ma trận
#Ở đây t lấy toàn bộ 3 cột đầu tiên
#iloc[ , ] nhận 2 tham số truyền vào, số dòng của dữ liệu và số cột của dữ liệu
#iloc[:, :-1] : lấy toàn bộ các dòng và lấy toàn bộ số cột trừ cột cuối cùng
X = dataset.iloc[:, :-1].values
#Tạo 1 vector y, lấy toàn bộ số hàng ở cột số 3
y = dataset.iloc[:, 3].values

#Spliting Dataset into TrainingSet and Test set
#Chia dữ liệu thành traning set and test set với test_size chiếm 20% dữ liệu và train_size chiếm 80% dữ liệu
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, train_size = 0.8, random_state = 0)


# #Feature scaling
# #training set thì dùng fit_transform , test set thì chỉ dùng transform
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)


print(X)
print(y)

print("Train and Test set ")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)



