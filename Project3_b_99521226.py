import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 

dataset = ['child' , 'tree', 'car']
# dataset = ['car' , 'tree2']
# dataset = ['child' , 'tree2']
X = []
y = []
for d in dataset:
    MyPath = "images" + r"\\" + d
    for pic in os.listdir(MyPath):
        picAddr = MyPath + r"\\" + pic
        image = cv2.imread(picAddr , 0)
        try:
            image1 = cv2.resize(image , (150,150))
            image2 = np.array(image1).flatten()
            X.append(image2)
            y.append(dataset.index(d))
        except Exception as e:
            num=2

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1234)
clf = SVC(kernel='rbf')
# rbf, sigmoid, poly, linear
clf.fit(x_train, y_train)
ys_predict = clf.predict(x_test) 

sum = 0
for i in range(len(y_test)):
    if y_test[i] == ys_predict[i]:
        sum += 1
    res = sum / len(y_test)

print(res) 


