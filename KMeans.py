import pandas as pd
from Algorithms import model,K_means
from Metrics import KMeansCheck

Treino = pd.read_csv('nba_treino.csv')
Teste = pd.read_csv('nba_teste.csv')

data = pd.concat([Treino, Teste])

Model = model(data)

centroids2,clusters2 = K_means(2,Model,1000)
centroids3,clusters3 = K_means(3,Model,1000)

print("Centroides de Kmenans with k = 2\n")
print(centroids2[0])
print(centroids2[1])
print("\n")

print("Centroides de Kmenans with k = 3\n")
print(centroids2[0])
print(centroids2[1])
print(centroids3[2])
print("\n")


print("Labels para Clusters com k = 2")
KMeansCheck(clusters2)
print("\n")

print("Labels para Clusterss com k = 3")
KMeansCheck(clusters3)
print("\n")