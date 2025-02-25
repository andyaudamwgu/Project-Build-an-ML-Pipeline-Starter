#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error

from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    print("Starting go function")
    run = wandb.init(job_type="test_model")
    print("W&B initialized")
    run.config.update(args)

    logger.info("Downloading artifacts")
    model_local_path = run.use_artifact(args.mlflow_model).download()
    print(f"Model downloaded: {model_local_path}")
    test_dataset_path = run.use_artifact(args.test_dataset).file()
    print(f"Test dataset: {test_dataset_path}")

    logger.info("Loading model and performing inference on test set")
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("price")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Scoring")
    r_squared = sk_pipe.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    run.summary['r2'] = r_squared
    run.summary['mae'] = mae
    print("Metrics logged to W&B")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str, 
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
