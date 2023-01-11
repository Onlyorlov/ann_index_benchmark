import time
import importlib
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Any, Dict, Type
from itertools import product
from src.base_index import BaseANN


def measure_recall(
    index: Type[BaseANN],
    bf_index: Type[BaseANN],
    run_parameters: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    silent: bool = True,
) -> Dict[str, Any]:
    k = run_parameters["k"]
    run_count = run_parameters["run_count"]
    nun_queries = run_parameters["nun_queries"]
    bf_index.fit(X_train)

    recall = MyMeter()
    search_time = MyMeter()
    build_time = MyMeter()
    for i in range(run_count):
        if not silent:
            print("Run %d/%d..." % (i + 1, run_count))
        t0 = time.time()
        if not silent:
            print("Adding batch of %d elements" % (X_train.shape[0]))
        index.fit(X_train)
        t1 = time.time()
        if not silent:
            print("Indices built")

        # Generating query data
        query_data = X_test[
            np.random.choice(X_test.shape[0], size=nun_queries, replace=False)
        ]

        # Query the elements and measure recall:
        t2 = time.time()
        labels = index.batch_query(query_data, k)
        t3 = time.time()
        labels_bf = bf_index.batch_query(query_data, k)

        # Measure recall <- rewrite it please!
        correct = 0
        for i in range(nun_queries):
            for label in labels[i]:
                for correct_label in labels_bf[i]:
                    if label == correct_label:
                        correct += 1
                        break
        curr_recall = float(correct) / (k * nun_queries)
        build_time.update(t1 - t0)
        search_time.update(t3 - t2)
        recall.update(curr_recall)
        if not silent:
            print("Recall is: ", recall.val)
            print("Time to build index: ", build_time.val)
            print("Time to search index: ", search_time.val)
        index.freeIndex()

    result = {
        "recall": recall,
        "search_time": search_time,
        "build_time": build_time,
    }
    return result


@dataclass
class Algorithm_definition:
    constructor: str
    module: str
    arguments: dict


def instantiate_algorithm(
    definition: Algorithm_definition, silent: bool = True
) -> Type[BaseANN]:
    if not silent:
        print(
            "Trying to instantiate %s.%s(%s)"
            % (definition.module, definition.constructor, definition.arguments)
        )
    module = importlib.import_module(definition.module)
    constructor = getattr(module, definition.constructor)
    return constructor(definition.arguments)


def grid_parameters(parameters: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))


class MyMeter(object):
    """
    Meter class, use the update to add the current value
    self.avg to get the avg value
    self.max to get the max value
    self.min to get the min value
    """

    def __init__(self):
        self.val = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.max = max(self.max, val)
        self.min = min(self.min, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
