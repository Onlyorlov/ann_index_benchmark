import numpy as np
import scann
from src.base_index import BaseANN


class Scann(BaseANN):
    def __init__(self, method_param):
        self.name = "Scann (%s)" % (method_param)
        if method_param["metric"] not in ("angular", "euclidean"):
            raise NotImplementedError(
                "Scann doesn't support metric %s" % method_param["metric"]
            )

        self.dist = {"angular": "dot_product", "euclidean": "squared_l2"}[
            method_param["metric"]
        ]
        self.n_leaves = method_param["n_leaves"]
        self.avq_threshold = method_param["avq_threshold"]
        self.dims_per_block = method_param["dims_per_block"]

        self.leaves_to_search = method_param["leaves_to_search"]
        self.reorder = method_param["reorder"]

    def fit(self, X):
        if self.dist == "dot_product":
            spherical = True
            X[np.linalg.norm(X, axis=1) == 0] = 1.0 / np.sqrt(X.shape[1])
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]
        else:
            spherical = False

        self.searcher = (
            scann.scann_ops_pybind.builder(X, 10, self.dist)
            .tree(
                self.n_leaves,
                self.leaves_to_search,
                training_sample_size=len(X),
                spherical=spherical,
                quantize_centroids=True,
            )
            .score_ah(
                self.dims_per_block,
                anisotropic_quantization_threshold=self.avq_threshold,
            )
            .reorder(self.reorder)
            .build()
        )

    def query(self, x, k):
        indexes, _ = self.searcher.search(x, k)
        return indexes

    def batch_query(self, X, k):
        indexes, _ = self.searcher.search_batched(X, k)
        return indexes
