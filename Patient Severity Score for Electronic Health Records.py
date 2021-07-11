import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/Patient Severity Score for Electronic Health Records.csv')

df

df.describe()

df.info()

df['SCORE']=df['SCORE ']
df=df.drop(columns='SCORE ')

hassan=df['SCORE'].value_counts()
hassan

hassan.plot.pie(autopct='%1.1f%%');

hassna=df.corr()

hassna

f,ax=plt.subplots(figsize=(10,10))
mask = np.triu(np.ones_like(hassna, dtype=bool))
sns.heatmap(hassna,annot=True,ax=ax,linewidths=.5,cmap="YlGnBu",mask=mask);

colors = ['red', 'orange', 'blue','green']
for i in range(4):
    x = df[df['SCORE'] == i]
    plt.scatter(x['BPSYS'], x['BPDIAS'],c = colors[i],label=i)
plt.xlabel("BPSYS")
plt.ylabel("BPDIAS")
plt.legend()

for i in range(4):
    x = df[df['SCORE'] == i]
    plt.scatter(x['BPDIAS'], x['PULSE'],c = colors[i],label=i)
plt.xlabel("BPDIAS")
plt.ylabel("PULSE")
plt.legend()

X=df.iloc[:,:-1]
y=df['SCORE']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
(scaler.fit_transform(df))

from sklearn.neighbors import KNeighborsClassifier

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier( n_neighbors = i)
    knn.fit(X_train , y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure( figsize=(10,8))
plt.plot( range(1,40) , error_rate , color='purple', linestyle='--', marker='o',
         markerfacecolor='yellow', markersize=10);

nnn=error_rate.index(min(error_rate))+1
print('the best number of neighbors is:',nnn)

knn=KNeighborsClassifier(n_neighbors=nnn)
knn.fit(X_train,y_train)
print('knn score is:',knn.score(X_test,y_test)*100,'%')

