import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df=pd.read_csv('MedicalData.csv')

print(df.head())

df['sex']=df['sex'].apply(lambda x: 0 if x=='male' else 1)
df['smoker']=df['smoker'].apply(lambda x: 0 if x=='no' else 1)
df['region'].replace(['southwest', 'southeast', 'northwest', 'northeast'],
                        [1,2,3,4], inplace=True)


sns.heatmap(df.corr(),cmap='gray_r',annot=True)
print(df.head())

plt.figure()

X= df[['age', 'bmi', 'smoker']]
y=df['charges']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=33)

print(X_train.head())

sns.histplot(y,bins=100)

lm=LinearRegression()

lm.fit(X_train,y_train)
pred=lm.predict(X_test)


print("Errors:")
print("mean_absolute_error: ",metrics.mean_absolute_error(y_test,pred))
print("mean_squared_error: ",metrics.mean_squared_error(y_test,pred))
print("root_mean_squared_error: ",np.sqrt(metrics.mean_squared_error(y_test,pred)))

plt.figure()
res_df=pd.DataFrame(y_test,pred)
sns.jointplot(data=res_df,x=y_test,y=pred,kind='reg')

print("R2 score: ",round(metrics.r2_score(y_test, pred),2))

    