import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#import the dataset
df=pd.read_csv('USA_Housing.csv')

#check data content
print("Data:")
print(df.head())

#print some info
print("Data info:")
print(df.info())

#get some basic statistics
print("Data Description:")
print(df.describe())

#get list of column names
cols=df.columns
print("Columns:")
print(str(cols))

#plot a pairplot to visualize data relation
sns.pairplot(df)

plt.figure()

#plot histogram to visualize how data is distributed
sns.histplot(data=df['Price'],kde=True,bins=45) 

plt.figure()

#check correlation of columns
sns.heatmap(data=df.corr(),annot=True,cmap='rocket_r')

#Divide data into train and test datasets

#Independent Variables
X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]

#depemdedent variables
y=df['Price']

#splitting the datasets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)

#instanciate the model
lm=LinearRegression()

#fitting the model
lm.fit(X_train,y_train)

#get basic info on correlation
print("Intercept:")
print(lm.intercept_)

#get coefficiants for the columns
cdf= pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
print("Coefficents:")
print(cdf)

plt.figure()

#get Predictions and plot again expected values
pred=lm.predict(X_test)
sns.scatterplot(x=pred,y=y_test)

#get basic errors
print("Errors:")
print("mean_absolute_error: ",metrics.mean_absolute_error(y_test,pred))
print("mean_squared_error: ",metrics.mean_squared_error(y_test,pred))
print("root_mean_squared_error: ",np.sqrt(metrics.mean_squared_error(y_test,pred)))
