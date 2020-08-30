# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
df = pd.read_csv('HW1data.csv', header = None)
f = np.array(df)
X = np.array([f[1:,2],f[1:,3]])
X = np.transpose(X)
df.shape[1]-1
y = f[1:,df.shape[1]-1]
for i in range(len(y)):
    if y[i] == 'TRUE':y[i] = 1 
    if y[i] == 'FALSE':y[i] = 0 
y = np.array(y, dtype=int)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=33)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'blue']
tf = ['FALSE','TRUE']
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, s=32*(len(colors)-i), c=colors[i], label = tf[i])
plt.title('squeeze')
plt.legend( loc = 'center', fontsize= 20)
plt.xlabel('price_crossing')
plt.ylabel('price_ditortion')

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)
print( clf.coef_)
print( clf.intercept_)

x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
#error in case Xs or xs
Xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1,1)
fig.set_size_inches(10, 6)
axes.set_aspect('equal')
# axes.set_title('Class '+ str(1) + ' versus the rest')
axes.set_xlabel('price_crossing')
axes.set_ylabel('price_ditortion')
axes.set_xlim(x_min, x_max)
axes.set_ylim(y_min, y_max)
#error here need plt.
plt.sca(axes)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
ys = (-clf.intercept_- Xs * clf.coef_[0,0]) / clf.coef_[0,1]
plt.plot(Xs, ys)

print( clf.predict(scaler.transform([[1,1]])))
print( clf.decision_function(scaler.transform([[1,1]])))

from sklearn import metrics
y_train_pred = clf.predict(X_train)
print( metrics.accuracy_score(y_train, y_train_pred) )

y_pred = clf.predict(X_test)
print( metrics.accuracy_score(y_test, y_pred) )
print( metrics.classification_report(y_test, y_pred, target_names= ['0','1']))
print( metrics.confusion_matrix(y_test, y_pred) )

print("My name is Junye Qiu")
print("My NetID is: junyeq2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")