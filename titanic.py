import pandas as pd
import numpy as np

dt = pd.read_csv("train.csv")

nv = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']
dt['Age'] = dt['Age'].fillna(dt['Age']).median()
y = dt['Survived']
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(dt[nv],y)
test = pd.read_csv('test.csv')
test['Age'] = test['Age'].fillna(test['Age']).median()
test = test[nv].fillna(test.mean())
yp = model.predict(test[nv])
print(yp)
