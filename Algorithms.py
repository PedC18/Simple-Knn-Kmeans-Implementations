import numpy as np
import random
import math

def DistanciaEuclidiana(a,b):
    dist = 0
    for i in range(len(a)):
        value = pow((a[i] - b[i]),2)
        dist += value
    
    return math.sqrt(dist)


def model(data):
    model = []
    for _,row in data.iterrows():
        Row_values = []
        for f in data.columns:            
            Row_values.append(row[f])

        model.append(Row_values)
    
    
    return model


def Knn_predict(data,model,k):
    labels = []

    for idx,row in data.iterrows():
        Row_values = []
        # Pega os dados de teste
        for f in data.columns:
            Row_values.append(row[f])
        
        distances = []
        # Calcula a distancia entre o ponto atual e os pontos do modelo
        for m in model:
            dist = DistanciaEuclidiana(Row_values,m)
            distances.append(dist)
        
        # Retorna o index dos k-vizinhos mais próximos
        idx = np.argsort(distances)
        idx = idx[:k]

        count0 = 0
        count1 = 0

        # Conta quantos os k-vizinhos mais próximos possuem a label 0 ou 1
        for i in range(k):
            if(model[idx[i]][19] == 1):
                count1 += 1
            else:
                count0 += 1
        
        # A label que rotula mais vizinhos será a escolhido pra rotular esse ponto
        if count0 > count1:
            labels.append(0)
        else:
            labels.append(1)
    
    return labels

def K_means(k,model,max_iters):
    #Inicializa centroides randomicamente
    centroids = random.sample(model, k)
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
    
        # Distancia do ponto para os centroides, escolhe a menor disntacia e coloca no cluster
        # A razão do [:19] é para desrotular os dados
        for m in model:
            distances = [DistanciaEuclidiana(m[:19], centroid[:19]) for centroid in centroids]
            centroid_idx = distances.index(min(distances))
            clusters[centroid_idx].append(m)
        

        # Escolhe um novo centroide a partir da média dos pontos do cluster
        Novo_Centroides = []
        for cluster in clusters:
            Novo_Centroide = [sum(ponto[:19]) / len(cluster[:19]) for ponto in zip(*cluster)]
            Novo_Centroides.append(Novo_Centroide)
        
        #Para se não houve alteração nos centroides
        if centroids == Novo_Centroide:
            break
        
        centroids = Novo_Centroides
            
    
    return centroids, clusters
    