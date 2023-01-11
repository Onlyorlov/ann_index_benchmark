import hnswlib
import numpy as np
from src.base_index import BaseANN


class HnswLib(BaseANN):
    def __init__(self, method_param):
        self.name = "hnswlib (%s)" % (method_param)
        if method_param["metric"] not in ("angular", "euclidean"):
            raise NotImplementedError(
                "HnswLib doesn't support metric %s" % method_param["metric"]
            )

        self.metric = {"angular": "cosine", "euclidean": "l2"}[method_param["metric"]]
        self.M = method_param["M"]
        self.efConstruction = method_param["efConstruction"]
        self.ef = method_param["ef"]

    def fit(self, X):
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(
            max_elements=len(X), ef_construction=self.efConstruction, M=self.M
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)
        self.p.set_ef(self.ef)

    def query(self, x, k):
        # print(np.expand_dims(x,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        indexes, _ = self.p.knn_query(np.expand_dims(x, axis=0), k=k)
        return indexes

    def batch_query(self, X, k):
        indexes, _ = self.p.knn_query(X, k=k)
        return indexes

    def freeIndex(self):
        del self.p
