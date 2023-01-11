import sklearn.neighbors
import sklearn.preprocessing
from src.base_index import BaseANN


class BruteForce(BaseANN):
    def __init__(self, method_param):
        if method_param["metric"] not in ("angular", "euclidean", "hamming"):
            raise NotImplementedError(
                "BruteForce doesn't support metric %s" % method_param["metric"]
            )
        self.metric = {"angular": "cosine", "euclidean": "l2", "hamming": "hamming"}[
            method_param["metric"]
        ]
        self.name = "BruteForce()"

    def fit(self, X):
        self.nbrs = sklearn.neighbors.NearestNeighbors(
            algorithm="brute", metric=self.metric
        )
        self.nbrs.fit(X)

    def query(self, x, k):
        indexes = self.nbrs.kneighbors([x], return_distance=False, n_neighbors=k)
        return indexes

    def batch_query(self, X, k):
        indexes = self.nbrs.kneighbors(X, return_distance=False, n_neighbors=k)
        return indexes

    def freeIndex(self):
        del self.nbrs
