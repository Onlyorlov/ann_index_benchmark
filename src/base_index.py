import psutil


class BaseANN(object):
    def fit(self, dataset):
        """
        Build the index for the data points given in dataset name.
        Assumes that after fitting index is loaded in memory.
        Args:
            dataset:np.array
        """
        raise NotImplementedError()

    def load_index(self, dataset):
        """
        Load the index for dataset. Returns False if index
        is not available, True otherwise.
        Checking the index usually involves the dataset name
        and the index build paramters passed during construction.
        """
        raise NotImplementedError()

    def save_index(self, dataset):
        """
        Saves the index for dataset. Returns False if index
        is not available, True otherwise.
        """
        raise NotImplementedError()

    def query(self, x, k):
        """Carry out a query for k-NN of query x."""
        raise NotImplementedError()

    def batch_query(self, X, k):
        """
        Carry out a batch query for k-NN of query set X.
        """
        raise NotImplementedError()

    def freeIndex(self):
        """
        This is called after results have been processed.
        Use it for cleaning up if necessary.
        """
        pass

    def __str__(self):
        return self.name

    def get_memory_usage(self):
        """
        Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available.
        """
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024
