import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.metrics import accuracy_score

from ray.air import RunConfig
from ray import tune
from ray import train
from config import Config
from custom_log import LocalFileCallback

import time

def train_fn(config):
    X, y = make_classification(
        n_samples=11000,
        n_features=1000,
        n_informative=50,
        n_redundant=0,
        n_classes=10,
        class_sep=2.5,
    )
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1000)

    # Hyperparameters
    alpha, epsilon = config["alpha"], config["epsilon"]

    # Train
    clf = SGDClassifier(alpha=alpha, epsilon=epsilon)
    clf = clf.fit(x_train, y_train)

    # Eval
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    metrics = {"score": score, "test_score": score, "dummy_metric": 100}

    # Simulate epoch training with improving score per epoch
    for _ in range(config["mlflow"]["max_epochs"]):
       score *= 1.02
       metrics["score"] = score
       train.report(metrics)
    # return metrics


def start_parallel(config: Config):
    mlflow_tracking_uri = config.mlflow_filepath
    ray_tracking_results = config.ray_filepath
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=config.experiment_name)

    param_space = {
        "alpha": tune.grid_search([1e-4, 1e-1, 1]),
        "epsilon": tune.grid_search([0.01, 0.1]),
        "mlflow": {
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_name": config.experiment_name,
            "max_epochs": config.max_epochs
        }
    }

    tuner = tune.Tuner(
        train_epochs,
        tune_config=tune.TuneConfig(
            num_samples=1,
        ),
        param_space=param_space,
        run_config=RunConfig(
            name="sgd-test-classifier",
            storage_path=ray_tracking_results,
            callbacks=[
                LocalFileCallback(dir="./result_directory") # get path from config
            ],
        )
    )
    results = tuner.fit()
    print("BEST RESULT  ",results.get_best_result(metric="score", mode="max"))

def train_epochs(config):
    metrics = train_fn(config)
    return metrics




def main():
    config = Config(
        mlflow_filepath="/tmp/mlflow",
        ray_filepath="/tmp/ray_results",
        experiment_name="test_experiment_epoch",
        max_epochs=5
    )
    start_parallel(config)


if __name__ == '__main__':
    main()
