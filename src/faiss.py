import faiss
import numpy as np
from src.base_index import BaseANN


class Faiss(BaseANN):
    def query(self, x, k):
        _, I = self.index.search(np.expand_dims(x, axis=0).astype(np.float32), k)
        return I

    def batch_query(self, X, k):
        _, I = self.index.search(X.astype(np.float32), k)
        return I

    def freeIndex(self):
        del self.index


class FaissLSH(Faiss):
    def __init__(self, method_param):
        self.name = "FaissLSH (%s)" % (method_param)
        if method_param["metric"] not in ("hamming"):
            raise NotImplementedError(
                "FaissLSH doesn't support metric %s" % method_param["metric"]
            )
        self.n_bits = method_param[
            "n_bits"
        ]  # the number of bits to use per stored vector

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        f = X.shape[1]
        self.index = faiss.IndexLSH(f, self.n_bits)
        self.index.train(X)
        self.index.add(X)


class FaissIVF(Faiss):
    def __init__(self, method_param):
        self.name = "FaissIVF (%s)" % (method_param)
        if method_param["metric"] not in ("angular", "euclidean"):
            raise NotImplementedError(
                "FaissIVF doesn't support metric %s" % method_param["metric"]
            )
        self.metric = (
            faiss.METRIC_INNER_PRODUCT
            if method_param["metric"] == "angular"
            else faiss.METRIC_L2
        )
        self.n_list = method_param["n_list"]  # the number of cells/centroids
        self.n_probe = method_param["n_probe"]  # Search for top-n centroids

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        self.index = faiss.IndexIVFFlat(
            self.quantizer, X.shape[1], self.n_list, self.metric
        )
        self.index.train(X)
        self.index.add(X)
        self.index.nprobe = self.n_probe


class FaissIVFPQfs(Faiss):
    def __init__(self, method_param):
        self.name = "FaissIVF (%s)" % (method_param)
        if method_param["metric"] not in ("angular", "euclidean"):
            raise NotImplementedError(
                "FaissIVF doesn't support metric %s" % method_param["metric"]
            )
        self.metric = (
            faiss.METRIC_INNER_PRODUCT
            if method_param["metric"] == "angular"
            else faiss.METRIC_L2
        )
        self.n_list = method_param["n_list"]  # the number of cells/centroids
        self.n_probe = method_param["n_probe"]  # Search for top-n centroids
        self.k_reorder = method_param[
            "k_reorder"
        ]  # i k!=0: re-ranks k search results with real distance computations

    def fit(self, X):
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        d = X.shape[1]
        factory_string = f"IVF{self.n_list},PQ{d//2}x4fs"
        index = faiss.index_factory(d, factory_string, self.metric)
        index.train(X)
        index.add(X)
        index.nprobe = self.n_probe

        if self.k_reorder == 0:
            self.index = index
        else:  # re-rank the search results with real distance computations
            index_refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(X))
            index_refine.k_factor = self.k_reorder
            self.index = index_refine
