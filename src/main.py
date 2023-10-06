import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn import tree

from ray.train import RunConfig
from ray.tune.sklearn import TuneGridSearchCV
from ray.tune.sklearn import TuneSearchCV
from ray.tune.search.bayesopt import BayesOptSearch
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
from ray import tune
from config import Config


import time

def train(config):
    tracking_uri = config.pop("tracking_uri", None)
    #setup_mlflow(
    #    config,
    #    experiment_name="setup_mlflow_example",
    #    tracking_uri=tracking_uri,
    #)

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
    metrics = {"score": score}

    #mlflow.log_metric()
    return metrics


def start_parallel(config: Config):
    mlflow_tracking_uri = config.mlflow_filepath
    ray_tracking_results = config.ray_filepath
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name="setup_mlflow_example")
    # algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})

    param_space = {
        "alpha": tune.grid_search([1e-4, 1e-1, 1]),
        "epsilon": tune.grid_search([0.01, 0.1]),
        "tracking_uri": mlflow.get_tracking_uri(),
    }

    tuner = tune.Tuner(
        train,
        tune_config=tune.TuneConfig(
            mode="min",
            #search_alg=algo,
            num_samples=1,
        ),
        param_space=param_space,
        run_config=RunConfig(
            name="sgd-test-classifier",
            storage_path=ray_tracking_results,
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=mlflow_tracking_uri,
                    experiment_name="mlflow_callback_example",
                    save_artifact=True,
                )
            ],
        )
    )
    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="min"))



def main():
    config = Config(mlflow_filepath="/tmp/mlflow", ray_filepath="/tmp/ray_results")
    start_parallel(config)


if __name__ == '__main__':
    main()
