import cv2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2422)

def euclidean_distance(x, y):
    sum = np.sum((x - y)**2)
    square_root = np.sqrt(sum)
    return square_root

class KMeansClustering():

    def __init__(self, K = 3, max_iterations = 100):
        self.K = K
        self.max_iterations = max_iterations
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.samples_n, self.features_n = X.shape

        random_sample_idxs = np.random.choice(self.samples_n, self.K, replace = False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        for _ in range(self.max_iterations):
            self.clusters = self.create_clusters(self.centroids)
            centroids_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)
            if self.is_converged(centroids_old, self.centroids):
                break
        
        return self.get_cluster_labels(self.clusters)

    def get_cluster_labels(self, clusters):
        labels = np.empty(self.samples_n)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        
        return labels

    def create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        
        for idx, sample in enumerate(self.X):
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        
        return clusters
    
    def closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.features_n))
        
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0)
            centroids[cluster_idx] = cluster_mean
        
        return centroids

    def is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    def cent(self):
        return self.centroids


