from sklearn.cluster import KMeans
import numpy as np

class GaussianMixture(object):
    def __init__(self, points, n_components = 5, noise = 0.01):
        self.n_components = n_components
        self.dimension = points.shape[-1]
        self.mu = np.zeros((n_components, self.dimension))
        self.cov = np.zeros((n_components, self.dimension, self.dimension))
        self.pi = np.zeros(n_components)
        self.noise = noise

        self.Kmeans_initialisation(points)

    def Kmeans_initialisation(self, X):
        label = KMeans(n_clusters = self.n_components, n_init = 1).fit(X).labels_
        self.fit(X, label)

    def calculate_probability(self, X):
        prob = np.zeros((X.shape[0], self.n_components))
        for c in range(self.n_components):
            data = X - self.mu[c]
            exponential = np.einsum('ij,ij->i', data, np.dot(np.linalg.inv(self.cov[c]), data.T).T)
            prob[:, c] = np.exp(-0.5 * exponential) / np.sqrt(2 * np.pi) / np.sqrt(np.linalg.det(self.cov[c]))
        return np.dot(prob, self.pi)

    def components_registration(self, X):
        prob = np.zeros((X.shape[0], self.n_components))
        for c in range(self.n_components):
            data = X - self.mu[c]
            exponential = np.einsum('ij,ij->i', data, np.dot(np.linalg.inv(self.cov[c]), data.T).T)
            prob[:, c] = np.exp(-0.5 * exponential) / np.sqrt(2 * np.pi) / np.sqrt(np.linalg.det(self.cov[c]))
        return np.argmax(prob * self.pi, axis = 1)

    def fit(self, X, label):
        n_elements = np.zeros(self.n_components)

        for c in range(self.n_components):
            index = [i for i in range(len(label)) if label[i] == c]
            n_elements[c] = len(index)
            elements = X[index]

            if len(index) == 0:
                self.mu[c] = np.zeros(self.dimension)
            else:
                self.mu[c] = np.mean(elements, axis = 0)

            if len(index) <= 1:
                self.cov[c] = np.zeros((self.dimension, self.dimension))
            else:
                self.cov[c] = np.cov(elements.T)
            if np.linalg.det(self.cov[c] == 0):
                self.cov[c] = self.cov[c] + self.noise * np.eyes(self.dimension)

        self.pi = n_elements / len(X)
        assert abs(np.sum(self.pi) - 1.0) < 1e-10
