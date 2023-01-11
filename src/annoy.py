import annoy
from src.base_index import BaseANN
from multiprocessing.pool import ThreadPool


class Annoy(BaseANN):
    def __init__(self, method_param):
        self.name = "annoy (%s)" % (method_param)

        self.n_trees = method_param["n_trees"]
        self.search_k = method_param["search_k"]
        self.metric = method_param["metric"]

    def fit(self, X):
        self.annoy = annoy.AnnoyIndex(X.shape[1], metric=self.metric)
        for i, x in enumerate(X):
            self.annoy.add_item(i, x.tolist())
        self.annoy.build(self.n_trees)

    def query(self, x, k):
        return self.annoy.get_nns_by_vector(x.tolist(), k, self.search_k)

    def batch_query(self, X, k):
        pool = ThreadPool()
        return pool.map(lambda q: self.query(q, k), X)

    def freeIndex(self):
        del self.annoy
