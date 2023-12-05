import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

Treino = pd.read_csv('nba_treino.csv')
Teste = pd.read_csv('nba_teste.csv')
real_labels = []

X_train = Treino.drop('TARGET_5Yrs',axis=1)
y_train = Treino['TARGET_5Yrs']

X_test = Teste.drop('TARGET_5Yrs',axis=1)
y_test = Teste['TARGET_5Yrs']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn_2 = KNeighborsClassifier(n_neighbors=2)
knn_2.fit(X_train, y_train)

knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_10.fit(X_train, y_train)

knn_25 = KNeighborsClassifier(n_neighbors=25)
knn_25.fit(X_train, y_train)

knn_50 = KNeighborsClassifier(n_neighbors=50)
knn_50.fit(X_train, y_train)

y_pred_2 = knn_2.predict(X_test)
y_pred_10 = knn_10.predict(X_test)
y_pred_25 = knn_25.predict(X_test)
y_pred_50 = knn_50.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred_2)
accuracy10 = accuracy_score(y_test, y_pred_10)
accuracy25 = accuracy_score(y_test, y_pred_25)
accuracy50 = accuracy_score(y_test, y_pred_50)

print("Accuracy for k = 2:", accuracy2)
print("Accuracy for k = 10:", accuracy10)
print("Accuracy for k = 25:", accuracy25)
print("Accuracy for k = 50:", accuracy50)

precision2 = precision_score(y_test, y_pred_2)
precision10 = precision_score(y_test, y_pred_10)
precision25 = precision_score(y_test, y_pred_25)
precision50 = precision_score(y_test, y_pred_50)

print("precision for k = 2:", precision2)
print("precision for k = 10:", precision10)
print("precision for k = 25:", precision25)
print("precision for k = 50:", precision50)

recall2 = recall_score(y_test, y_pred_2)
recall10 = recall_score(y_test, y_pred_10)
recall25 = recall_score(y_test, y_pred_25)
recall50 = recall_score(y_test, y_pred_50)

print("recall for k = 2:", recall2)
print("recall for k = 10:", recall10)
print("recall for k = 25:", recall25)
print("recall for k = 50:", recall50)

f1_score2 = f1_score(y_test, y_pred_2)
f1_score10 = f1_score(y_test, y_pred_10)
f1_score25 = f1_score(y_test, y_pred_25)
f1_score50 = f1_score(y_test, y_pred_50)

print("f1_score for k = 2:", f1_score2)
print("f1_score for k = 10:", f1_score10)
print("f1_score for k = 25:", f1_score25)
print("f1_score for k = 50:", recall50)
