import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

df=pd.read_csv("kyphosis.csv")

print(df.head())
print(df.describe())
print(df.info())

sns.histplot(df['Age'],bins=20,kde=True)

plt.figure()

sns.heatmap(df.corr(),cmap='Blues',annot=True)

X=df.drop('Kyphosis',axis=1,inplace=False)
y=df['Kyphosis']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#single descision tree
dtree=DecisionTreeClassifier()

dtree.fit(X_train,y_train)

pred=dtree.predict(X_test)

print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))   

#random forest classifier

rfc=RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

rfc_pred=rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))

print(confusion_matrix(y_test,rfc_pred))   

