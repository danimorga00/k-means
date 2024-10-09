from kmeans import generateData
from kmeans import KMeans
from kmeans import train_test_split
import time
import numpy as np
from joblib import Parallel, delayed
import os

from plot import plot_clusters


if __name__ == "__main__":
    X, true_labels, centroids = generateData(n_samples=2000000, n_features=2, n_clusters=5, cluster_std=1)
    
    # Suddivide i dati in train e test
    X_train, X_test, y_train, y_test = train_test_split(X, true_labels, train_size=0.5, random_seed=42)
    
    kmeansSeq = KMeans(n_clusters=5, assign_jobs=1, compute_jobs=1)
    kmeansPar = KMeans(n_clusters=5, assign_jobs=20, compute_jobs=20)
    
    start_time = time.time()

    kmeansSeq.fit(X_train)
    
    duration1 = time.time() - start_time

    print("durata kmeans sequenziale: "+str(duration1))

    start_time = time.time()

    kmeansPar.fit(X_train)
    
    duration2 = time.time() - start_time

    print("durata kmeans parallelo: "+str(duration2))

    print("Speedup: "+str(duration1/duration2))

    """
    y_preds = kmeansSeq.predict(X_test)
    y_predp = kmeansPar.predict(X_test)
    plot_clusters(X_test, y_preds, kmeansSeq.centroids, "sequenziale")
    plot_clusters(X_test, y_predp, kmeansPar.centroids, "parallela")
    plot_clusters(X_test, y_predp, centroids, "parallela")
    """