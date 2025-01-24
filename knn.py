import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Classified Data',index_col=0)
print(df.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df.drop("TARGET CLASS",axis=1))

scaled_features = scaler.transform(df.drop("TARGET CLASS",axis=1))

df_features = pd.DataFrame(scaled_features,columns=df.columns[:-1])

print(df_features)



from sklearn.model_selection import train_test_split

X=df_features
y=df["TARGET CLASS"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)

predict=knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predict))
print(classification_report(y_test,predict))


error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predict_i= knn.predict(X_test)
    error_rate.append(np.mean(predict!=predict_i))  #here np.mean() expects no. of boolean as input treat true as 1 false as 0 and divide the sum of inputs by the no. of inputs
#print(error_rate)
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color="brown",linestyle="solid",marker="o",
         markerfacecolor="yellow",markersize=8)
plt.title('Error_Rate vs K values')
plt.ylabel('Error_Rate')
plt.xlabel('K values')
plt.show()