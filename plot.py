import numpy as np
import matplotlib.pyplot as plt

# Supponiamo di avere già:
# - X: i punti dati (2D)
# - labels: l'array delle etichette che assegna ogni punto a un cluster
# - centroids: la posizione dei centroidi

def plot_clusters(X, labels, centroids, title):
    # Numero di cluster
    n_clusters = centroids.shape[0]
    
    # Creiamo una mappa di colori
    colors = plt.cm.get_cmap('tab10', n_clusters)  # Usa una palette di 10 colori diversi
    
    # Plot dei punti dati con colori diversi per ogni cluster
    for i in range(n_clusters):
        points_in_cluster = X[labels == i]
        plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], 
                    s=50, c=[colors(i)], label=f'Cluster {i+1}')
    
    # Plot dei centroidi
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                s=200, c='red', marker='X', edgecolor='black', label='Centroidi')
    
    # Titolo e legenda
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()