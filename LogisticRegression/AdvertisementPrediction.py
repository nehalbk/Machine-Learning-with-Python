import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#importing data
df=pd.read_csv('advertising.csv')

#getting basic details
print(df.head())
print(df.info())
print(df.describe())

#checking the distribution of input
sns.histplot(df['Age'],kde=True)

#Exploratory analysis
plt.figure()
sns.jointplot(data=df,x='Daily Time Spent on Site',y='Daily Internet Usage',kind='kde')

#Predicting

#Independednt variables
X=df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]

#dependent variables
y=df['Clicked on Ad']

#splitting into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

#instanciating the model
logMod=LogisticRegression()

#fitting the data  into model
logMod.fit(X_train,y_train)

#predicting
pred=logMod.predict(X_test)

#checking model performance
print(classification_report(y_test,pred))

