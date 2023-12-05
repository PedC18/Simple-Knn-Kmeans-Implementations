import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

Treino = pd.read_csv('Nba_treino.csv')
Teste = pd.read_csv('Nba_teste.csv')

data = pd.concat([Treino, Teste])

X = data.drop('TARGET_5Yrs', axis=1) 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_2 = KMeans(n_clusters=2, random_state=0) 
kmeans_3 = KMeans(n_clusters=3, random_state=0) 

kmeans_2.fit(X_scaled)
kmeans_3.fit(X_scaled)

cluster_labels_2 = kmeans_2.labels_
cluster_labels_3 = kmeans_3.labels_

silhouette_avg_2 = silhouette_score(X_scaled, cluster_labels_2)
silhouette_avg_3 = silhouette_score(X_scaled, cluster_labels_3)

print(f"Silhouette Score para k = 2: {silhouette_avg_2}")
print(f"Silhouette Score para k = 3: {silhouette_avg_3}")
