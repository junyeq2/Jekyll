# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 09:11:42 2020

@author: Jekyll
"""

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit( X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    scores.append(accuracy_score(y_test,y_pred))
    
import matplotlib.pyplot as plt 
plt.plot(k_range,scores)
plt.title('KNN')
plt.xlabel('k_neighbors')
plt.ylabel('scores')
plt.show()

from sklearn.tree import DecisionTreeClassifier
tree_scores = []
d_range = range(1,21)
for  k in d_range:
    tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=k, 
                              random_state=1)
    tree.fit(X_train_std, y_train)
    tree_y_pred = tree.predict(X_test_std)
    tree_scores.append(accuracy_score(y_test, tree_y_pred))
plt.figure(2)
plt.plot(d_range,tree_scores)
plt.title('DecisionTreeClassifier')
plt.xlabel('depth')
plt.ylabel('scores')
plt.show()

print("My name is Junye Qiu")
print("My NetID is: junyeq2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")