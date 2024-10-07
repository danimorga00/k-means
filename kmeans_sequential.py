import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters      # Numero di cluster
        self.max_iter = max_iter          # Numero massimo di iterazioni
        self.tol = tol                    # Tolleranza per la convergenza

    def fit(self, X):
        # Inizializza i centroidi selezionando casualmente k punti dai dati
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for i in range(self.max_iter):
            # Assegna ogni punto dati al cluster più vicino
            self.labels = self._assign_clusters(X)
            
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
    
    def _compute_centroids(self, X):
        # Calcola i nuovi centroidi come media dei punti assegnati a ciascun cluster
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def predict(self, X):
        # Restituisce l'indice del cluster più vicino per i nuovi dati
        return self._assign_clusters(X)
import numpy as np

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

if __name__ == "__main__":
    # Genera 1000 punti in un dataset con 2 caratteristiche e 3 cluster
    X, true_labels = generateData(n_samples=1000, n_features=2, n_clusters=3, cluster_std=2, random_seed=42)
    
    # Crea l'oggetto KMeans con 3 cluster
    kmeans = KMeans(n_clusters=3)
    
    # Addestra il modello
    kmeans.fit(X)
    
    # Stampa i centroidi
    print("Centroidi:", kmeans.centroids)
    
    # Assegna i cluster ai punti di dati
    labels = kmeans.predict(X)
    print("Etichette:", labels)
