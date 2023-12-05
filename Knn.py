import pandas as pd
from Algorithms import model,Knn_predict
from Metrics import Precision,Accuracy,F1,Recall,Confusion_Matrix

Treino = pd.read_csv('nba_treino.csv')
Teste = pd.read_csv('nba_teste.csv')

real_labels = []

for idx,row in Teste.iterrows():
    real_labels.append(row['TARGET_5Yrs'])


k = 2

Knn_Model = model(Treino)

labels_2 = Knn_predict(Teste,Knn_Model,2)
labels_10 = Knn_predict(Teste,Knn_Model,10)
labels_25 = Knn_predict(Teste,Knn_Model,25)
labels_50 = Knn_predict(Teste,Knn_Model,50)

print("Accuracy dos modelos\n")
print(f"Accuracy for k = 2: {Accuracy(labels_2,real_labels)}")
print(f"Accuracy for k = 10: {Accuracy(labels_10,real_labels)}")
print(f"Accuracy for k = 25: {Accuracy(labels_25,real_labels)}")
print(f"Accuracy for k = 50: {Accuracy(labels_50,real_labels)}")
print("\n")

Precision_2 = Precision(labels_2,real_labels)
Precision_10 = Precision(labels_10,real_labels)
Precision_25 = Precision(labels_25,real_labels)
Precision_50 = Precision(labels_50,real_labels)

print("Precisão dos modelos\n")
print(f"Precision for k = 2: {Precision_2}")
print(f"Precision for k = 10: {Precision_10}")
print(f"Precision for k = 25: {Precision_25}")
print(f"Precision for k = 50: {Precision_50}")
print("\n")

Recall_2 = Recall(labels_2,real_labels)
Recall_10 = Recall(labels_10,real_labels)
Recall_25 = Recall(labels_25,real_labels)
Recall_50 = Recall(labels_50,real_labels)

print("Recall dos modelos \n")
print(f"Recall for k = 2: {Recall_2}")
print(f"Recall for k = 10: {Recall_10}")
print(f"Recall for k = 25: {Recall_25}")
print(f"Recall for k = 50: {Recall_50}")
print("\n")

print("F1-score dos modelos \n")
print(f"F1 for k = 2: {F1(Precision_2,Recall_2)}")
print(f"F1 for k = 10: {F1(Precision_10,Recall_10)}")
print(f"F1 for k = 25: {F1(Precision_25,Recall_25)}")
print(f"F1 for k = 50: {F1(Precision_50,Recall_50)}")

print("\n")

Confusion_Matrix(labels_2,real_labels,2)

print("\n")

Confusion_Matrix(labels_10,real_labels,10)

print("\n")

Confusion_Matrix(labels_25,real_labels,25)

print("\n")

Confusion_Matrix(labels_50,real_labels,50)




