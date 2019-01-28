import pandas as pd
import numpy as np

dt = pd.read_csv("train.csv")

nv = ['PassengerId','Pclass','Age','SibSp','Parch','Fare']
dt['Age'] = dt['Age'].fillna(dt['Age']).median()
y = dt['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dt[nv],dt['Survived'],test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)
#test = pd.read_csv('test.csv')
#test['Age'] = test['Age'].fillna(test['Age']).median()
#test = test[nv].fillna(test.mean())
yp = model.predict(X_test)
print(yp)
print(model.score(X_test,Y_test))