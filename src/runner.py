from pathlib import Path
import argparse
import yaml
import numpy as np
from attrdict import AttrDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
from tqdm import tqdm
import math

from src.utils import (
    Algorithm_definition,
    instantiate_algorithm,
    measure_recall,
    grid_parameters,
)


def run(experiment_name, opt):
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(experiment_name)
    parameters = (
        opt.search_parameters
    )  # parameters ={"learning_rate":[0.1, 1, 2], "penalty":[1, 2, 3]}

    # Load dataset
    data = np.load(opt.dataset_pth)
    if opt.split:
        X_train, X_test = train_test_split(
            data, random_state=104, test_size=0.25, shuffle=True
        )
    else:  # train and test index on the same dataset
        X_train = X_test = data
    num_comb = math.prod(
        [
            len(search_space)
            for search_space in parameters.values()
            if isinstance(search_space, list)
        ]
    )
    for settings in (pbar := tqdm(grid_parameters(parameters), total=num_comb)):
        pbar.write("Params: (%s)" % (settings))
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("Train/Test split", opt.split)
            mlflow.log_param("index_name", opt.constructor)
            for param_name, val in settings.items():
                mlflow.log_param(f"index_param_{param_name}", val)
            mlflow.log_param("gt_index_name", "BruteForce")

            definition = Algorithm_definition(
                module=opt.module, constructor=opt.constructor, arguments=settings
            )
            bf_definition = Algorithm_definition(
                module="src.brute_force",
                constructor="BruteForce",
                arguments=settings,  # may differ, that's why initialisation inside the loop
            )
            # Init idex from parameters
            index = instantiate_algorithm(definition)
            # Init gt_index
            bf_index = instantiate_algorithm(bf_definition)

            settings["k"] = opt.k
            settings["run_count"] = opt.run_count
            settings["nun_queries"] = opt.nun_queries

            mlflow.log_param("Number of runs", settings["run_count"])
            mlflow.log_param("KNN", settings["k"])
            mlflow.log_param("Nun_queries", settings["nun_queries"])
            mlflow.log_param("Metric", settings["metric"])
            if settings["metric"] == "angular":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            result = measure_recall(index, bf_index, settings, X_train, X_test, True)
            mlflow.log_metric("best_recall", result["recall"].min)
            mlflow.log_metric("avg_recall", result["recall"].avg)
            mlflow.log_metric("best_search_time", result["search_time"].min)
            mlflow.log_metric("avg_search_time", result["search_time"].avg)
            mlflow.log_metric("best_build_time", result["build_time"].min)
            mlflow.log_metric("avg_build_time", result["build_time"].avg)


def main(opt):
    with open(opt.config_pth) as f:
        config = yaml.safe_load(f)
        cfg = AttrDict(config)
    run(opt.experiment_name, cfg)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Experiment name for mlflow(create new or continue)",
        required=True,
    )
    parser.add_argument(
        "--config_pth", type=str, help="Config file with run parameters", required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
