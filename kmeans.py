import time
import numpy as np
from joblib import Parallel, delayed
import os

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, n_jobs=1):
        self.n_clusters = n_clusters      # Numero di cluster
        self.max_iter = max_iter          # Numero massimo di iterazioni
        self.tol = tol                    # Tolleranza per la convergenza
        self.n_jobs = n_jobs

    def fit(self, X):
        # Inizializza i centroidi selezionando casualmente k punti dai dati
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            print(f"iterazione: {i}")
            # Assegna ogni punto dati al cluster più vicino
            self.labels = self._assign_clusters_parallel(X)
            
            # Calcola i nuovi centroidi come la media dei punti assegnati a ciascun cluster
            new_centroids = self._compute_centroids(X)
            
            # Se i centroidi non cambiano significativamente, termina l'algoritmo
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break
                
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        # Calcola le distanze tra ogni punto dati e i centroidi
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Restituisce l'indice del centroide più vicino per ogni punto dati
        return np.argmin(distances, axis=1)
    
    def _assign_clusters_parallel(self, X):
        """
        Parallelizza il calcolo delle distanze tra i punti dati e i centroidi e l'assegnazione ai cluster.
        """
        def compute_chunk(chunk):
            """
            Calcola le distanze tra un chunk di punti dati e i centroidi, restituendo l'indice del centroide più vicino.
            """
            processId = os.getpid()
            print(f"Process ID: {processId}, chunk size: {len(chunk)}")
            distances_chunk = np.linalg.norm(chunk[:, np.newaxis] - self.centroids, axis=2)
            return np.argmin(distances_chunk, axis=1)
        
        # Definisce la dimensione dei chunk (blocchi di punti da processare)
        n_samples = X.shape[0]
        chunk_size = n_samples // self.n_jobs
        
        # Suddivide i dati in chunk e parallelizza il calcolo della distanza
        chunks = [X[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
        
        # Parallelizza il calcolo per ciascun chunk
        labels = Parallel(n_jobs=self.n_jobs)(delayed(compute_chunk)(chunk) for chunk in chunks)
        
        # Concatena i risultati dei chunk
        return np.concatenate(labels)
    
    def _compute_centroids(self, X):
        # Funzione per calcolare la media dei punti assegnati a ciascun cluster
        def compute_single_centroid(i):
            return X[self.labels == i].mean(axis=0)

        # Parallelizza il calcolo dei centroidi per ogni cluster
        centroids = Parallel(n_jobs=self.n_jobs)(delayed(compute_single_centroid)(i) for i in range(self.n_clusters))

        return np.array(centroids)

    def predict(self, X):
        # Restituisce l'indice del cluster più vicino per i nuovi dati
        return self._assign_clusters(X)

def generateData(n_samples=1000, n_features=2, n_clusters=3, cluster_std=1.0, random_seed=None):
    """
    Genera un set di dati sintetici distribuiti attorno a diversi centroidi.
    
    Parametri:
    - n_samples: Numero totale di punti dati da generare.
    - n_features: Numero di caratteristiche (dimensioni dei dati).
    - n_clusters: Numero di cluster o gruppi.
    - cluster_std: Deviazione standard di ciascun cluster (quanto sono dispersi i punti intorno al centroide).
    - random_seed: Semina casuale per ottenere risultati ripetibili.
    
    Restituisce:
    - X: Array di dati generato (n_samples, n_features).
    - true_labels: Etichette dei cluster a cui i punti appartengono (utili per la valutazione).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Inizializza un array vuoto per memorizzare i dati
    X = np.empty((0, n_features))
    true_labels = np.empty(0, dtype=int)
    
    # Genera i centroidi casuali
    centroids = np.random.uniform(-10, 10, size=(n_clusters, n_features))
    
    for i in range(n_clusters):
        # Per ogni centroide, genera punti intorno ad esso con una deviazione standard definita
        points = np.random.normal(loc=centroids[i], scale=cluster_std, size=(n_samples // n_clusters, n_features))
        X = np.vstack((X, points))
        true_labels = np.hstack((true_labels, np.full(points.shape[0], i)))
    
    return X, true_labels

def train_test_split(X, y=None, train_size=0.8, random_seed=None):
    """
    Suddivide i dati X (e opzionalmente le etichette y) in insiemi di train e test.
    
    Parametri:
    - X: Array dei dati (n_samples, n_features).
    - y: Etichette associate ai dati, se presenti.
    - train_size: Percentuale di dati da usare per il train (default: 0.8 per l'80%).
    - random_seed: Semina casuale per ottenere una suddivisione ripetibile.
    
    Restituisce:
    - X_train: Dati di addestramento.
    - X_test: Dati di test.
    - y_train: Etichette di addestramento (se y è fornito).
    - y_test: Etichette di test (se y è fornito).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Numero totale di campioni
    n_samples = X.shape[0]
    
    # Genera un array di indici casuali
    indices = np.random.permutation(n_samples)
    
    # Determina il numero di campioni di train
    train_size = int(train_size * n_samples)
    
    # Separa gli indici per train e test
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Suddividi i dati
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    if y is not None:
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test
    
