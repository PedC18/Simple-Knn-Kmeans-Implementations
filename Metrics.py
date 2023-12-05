def Precision(labels, real_labels):

    TP = 0
    FP = 0
    for i in range(len(labels)):
        if labels[i] == 1 and real_labels[i] == 1:
            TP += 1
        elif labels[i] == 1 and real_labels[i] == 0:
            FP += 1

    return TP/(TP+FP)

def Recall(labels, real_labels):
    
    TP = 0
    FN = 0
    for i in range(len(labels)):
        if labels[i] == 1 and real_labels[i] == 1:
            TP += 1
        elif labels[i] == 0 and real_labels[i] == 1:
            FN += 1

    return TP/(TP+FN)

def F1(precision,recall):

    return (2*precision*recall)/(precision + recall)

def Accuracy(labels,real_labels):
    TP = 0
    TN = 0
    for i in range(len(labels)):
        if labels[i] == 1 and real_labels[i] == 1:
            TP += 1
        elif labels[i] == 0 and real_labels[i] == 0:
            TN += 1

    return (TP+TN)/len(labels)


def Confusion_Matrix(labels,real_labels,k):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(labels)):
        if labels[i] == 1 and real_labels[i] == 1:
            TP += 1
        elif labels[i] == 0 and real_labels[i] == 0:
            TN += 1
        elif labels[i] == 0 and real_labels[i] == 1:
            FN += 1
        else:
            FP += 1
    
    print(f"----------------------Matriz de confusão para k = {k} --------------------------")
    print("                            Classe Resposta")
    print("                        Label 0             Label 1        ")
    print("                  ---------------------------------------------")
    print(f"Classe  | Label 0 |     {TP}       ||         {FN}        |")
    print(f" Real   | Label 1 |     {FP}       ||         {TN}        |")


def KMeansCheck(clusters):

    for i in range(len(clusters)):
        label0 = 0
        label1 = 0
        for data in clusters[i]:
            if data[19] == 0:
                label0 +=1
            if data[19] == 1:
                label1 +=1

        print(f"Cluster {i}: Labels 0 = {label0} / Labels 1 = {label1}")
    



        
