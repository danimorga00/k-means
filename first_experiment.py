import csv
from kmeans import generateData
from kmeans import KMeans
from kmeans import train_test_split
import time
import numpy as np
from joblib import Parallel, delayed
import os
from tqdm import tqdm
from datetime import datetime

from plot import plot_clusters, plotColumn

report = []
header = ["n_samples", "n_jobs", "time", "speedup"]

FEATURES = 2
CLUSTERS = 10
ITERATION_INCREMENT = 100000
ITERATIONS = 1
MAX_JOBS = 16

if __name__ == "__main__":

    for i in tqdm(range(1,ITERATIONS+1), desc=" running", unit="iteration"):
        X, true_labels, centroids = generateData(n_samples=i*ITERATION_INCREMENT, n_features=FEATURES, n_clusters=CLUSTERS, cluster_std=1)
        X_train, X_test, y_train, y_test = train_test_split(X, true_labels, train_size=0.5, random_seed=42)

        kmeans = KMeans(X_train, n_clusters=CLUSTERS, assign_jobs=i, compute_jobs=i)
        
        log = []
        log.append(i*ITERATION_INCREMENT)
        log.append(1)
        start_time = time.time()

        kmeans.fit()
        
        baseline = time.time() - start_time

        log.append(baseline)
        log.append(1)
        report.append(log)

        for jobs in tqdm(range(2,MAX_JOBS+1), desc=" running", unit="jobs"):
        #for jobs in range(2,MAX_JOBS+1):
            
            log = []
            log.append(i*ITERATION_INCREMENT)
            log.append(jobs)

            kmeans.setJobs(jobs, jobs)
            kmeans.resetCentroids()

            start_time = time.time()

            kmeans.fit()
            
            duration = time.time() - start_time
            log.append(duration)
            log.append(baseline/duration)
            report.append(log)

    with open("report2_"+str(datetime.now().date()), mode='w', newline='', encoding='utf-8') as file_csv:
        scrittore = csv.writer(file_csv)
        scrittore.writerow(header)
        scrittore.writerows(report)

    plotColumn("report2_2024-10-09")