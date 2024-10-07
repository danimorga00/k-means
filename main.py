from kmeans import generateData
from kmeans import KMeans
from kmeans import train_test_split
import time
import numpy as np
from joblib import Parallel, delayed
import os


if __name__ == "__main__":
    # Genera 1000 punti in un dataset con 2 caratteristiche e 3 cluster
    X, true_labels = generateData(n_samples=2000000, n_features=2, n_clusters=5, cluster_std=2, random_seed=42)
    
    # Suddivide i dati in train e test (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, true_labels, train_size=1, random_seed=42)
    
    # Crea l'oggetto KMeans con 3 cluster
    kmeansSeq = KMeans(n_clusters=5, n_jobs=1)
    kmeansPar = KMeans(n_clusters=5, n_jobs=10)
    
    start_time = time.time()

    # Addestra il modello
    kmeansSeq.fit(X_train)
    
    duration1 = time.time() - start_time

    print("durata kmeans sequenziale: "+str(duration1))

    start_time = time.time()

    # Addestra il modello
    kmeansPar.fit(X_train)
    
    duration2 = time.time() - start_time

    print("durata kmeans parallelo: "+str(duration2))

    print("Speedup: "+str(duration1/duration2))

    # Stampa i centroidi
    #print("Centroidi:", kmeans.centroids)
    
    # Assegna i cluster ai punti di dati
    #labels = kmeans.predict(X_test)
    #print("Etichette:", labels)