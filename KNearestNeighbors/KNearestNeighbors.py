import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#importing data
df=pd.read_csv('KNN_Project_Data.csv')

#getting basic details
print(df.head())
print(df.info())
print(df.describe())


sns.pairplot(data=df,hue="TARGET CLASS")
plt.figure()
# Standardizing data
scalar=StandardScaler()
scalar.fit(df.drop('TARGET CLASS',1))
scaled_features=scalar.transform(df.drop('TARGET CLASS',1))
df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])

print(df_feat.head())

#Predicting

#Independednt variables
X=scaled_features

#dependent variables
y=df['TARGET CLASS']

#splitting into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)


#Selecting k value using elbow method
error_rate=[]
for i in range( 1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))

plt.figure

plt.plot(range(1,40),error_rate,color='blue',ls='--',marker='o',markerfacecolor='red',markersize=10)

#fitting the data  into model
#k=30 as the error rate is min
knn=KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)

# #predicting
pred=knn.predict(X_test)

#checking model performance
print(classification_report(y_test,pred))

