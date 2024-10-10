import threading
import time
import numpy as np
from joblib import Parallel, delayed
import os

class KMeans:
    def __init__(self, X, n_clusters=3, max_iter=300, tol=1e-4, assign_jobs=1, compute_jobs=1):
        self.n_clusters = n_clusters      # Numero di cluster
        self.max_iter = max_iter          # Numero massimo di iterazioni
        self.tol = tol                    # Tolleranza per la convergenza
        self.assign_jobs = assign_jobs
        self.compute_jobs = compute_jobs
        self.X = X
        self.initialCentroids = self.X[np.random.choice(self.X.shape[0], self.n_clusters, replace=False)]
        self.centroids = self.initialCentroids

    def fit(self):

        for i in range(self.max_iter):
            self.labels = self._assign_clusters_parallel()
            
            new_centroids = self._compute_centroids()
            
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break
                
            self.centroids = new_centroids

    def _assign_clusters(self):
        distances = np.linalg.norm(self.X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def _assign_clusters_parallel(self):
        """
        Parallelizza il calcolo delle distanze tra i punti dati e i centroidi e l'assegnazione ai cluster.
        """
        def compute_chunk(chunk):
            """
            Calcola le distanze tra un chunk di punti dati e i centroidi, restituendo l'indice del centroide più vicino.
            """
            processId = threading.get_ident()
            #print(f"Process ID: {processId}, chunk size: {len(chunk)}")
            distances_chunk = np.linalg.norm(chunk[:, np.newaxis] - self.centroids, axis=2)
            return np.argmin(distances_chunk, axis=1)
        
        n_samples = self.X.shape[0]
        chunk_size = n_samples // self.assign_jobs
        
        chunks = [self.X[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
        
        labels = Parallel(n_jobs=self.assign_jobs, backend="threading")(delayed(compute_chunk)(chunk) for chunk in chunks)
        
        return np.concatenate(labels)
    
    def _compute_centroids(self):
        
        def compute_single_centroid(i):
            return self.X[self.labels == i].mean(axis=0)

        centroids = Parallel(n_jobs=self.compute_jobs, backend="threading")(delayed(compute_single_centroid)(i) for i in range(self.n_clusters))

        return np.array(centroids)

    def predict(self):
        return self._assign_clusters(self.X)
    
    def setJobs(self, assign_jobs=1, compute_jobs=1):
        self.assign_jobs = assign_jobs
        self.compute_jobs = compute_jobs

    def setData(self, X):
        self.X = X

    def resetCentroids(self):
        self.centroids = self.initialCentroids


def generateData(n_samples=1000, n_features=2, n_clusters=3, cluster_std=0.1, random_seed=None):
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
    
    X = np.empty((0, n_features))
    true_labels = np.empty(0, dtype=int)
    
    centroids = np.random.uniform(-20, 20, size=(n_clusters, n_features))
    
    for i in range(n_clusters):
        points = np.random.normal(loc=centroids[i], scale=cluster_std, size=(n_samples // n_clusters, n_features))
        X = np.vstack((X, points))
        true_labels = np.hstack((true_labels, np.full(points.shape[0], i)))
    
    return X, true_labels, centroids

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
    
    n_samples = X.shape[0]
    
    indices = np.random.permutation(n_samples)
    
    train_size = int(train_size * n_samples)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    if y is not None:
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test
    
