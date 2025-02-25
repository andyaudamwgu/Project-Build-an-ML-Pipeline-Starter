import json
import mlflow
import tempfile
import os
import wandb
import hydra
import subprocess
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # "test_regression_model"
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _run_basic_cleaning(input_artifact, output_artifact, output_type, output_description, min_price, max_price):
    run_id = mlflow.run(
        os.path.join(ROOT_DIR, "src", "basic_cleaning"),
        "main",
        env_manager="conda",
        parameters={
            "input_artifact": input_artifact,
            "output_artifact": output_artifact,
            "output_type": output_type,
            "output_description": output_description,
            "min_price": min_price,
            "max_price": max_price
        }
    )
    return run_id

def _run_data_check(csv, ref, kl_threshold, min_price, max_price):
    run_id = mlflow.run(
        os.path.join(ROOT_DIR, "src", "data_check"),
        "main",
        env_manager="conda",
        parameters={
            "csv": csv,
            "ref": ref,
            "kl_threshold": kl_threshold,
            "min_price": min_price,
            "max_price": max_price
        }
    )
    return run_id

def _run_train_random_forest(trainval_artifact, val_size, random_seed, stratify_by, output_artifact, rf_config, max_tfidf_features):
    run_id = mlflow.run(
        os.path.join(ROOT_DIR, "src", "train_random_forest"),
        "main",
        env_manager="conda",
        parameters={
            "trainval_artifact": trainval_artifact,
            "val_size": val_size,
            "random_seed": random_seed,
            "stratify_by": stratify_by,
            "output_artifact": output_artifact,
            "rf_config": rf_config,
            "max_tfidf_features": max_tfidf_features
        }
    )
    return run_id

@hydra.main(config_name='config')
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:
        if "download" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            basic_cleaning_run_id = _run_basic_cleaning(
                input_artifact="adam6-western-governors-university/nyc_airbnb/sample.csv:latest",
                output_artifact="clean_sample.csv",
                output_type="clean_data",
                output_description="Cleaned NYC Airbnb data",
                min_price=config["etl"]["min_price"],
                max_price=config["etl"]["max_price"]
            )

        if "data_check" in active_steps:
            _run_data_check(
                csv="adam6-western-governors-university/Project-Build-an-ML-Pipeline-Starter-src_basic_cleaning/clean_sample.csv:latest",
                ref="adam6-western-governors-university/Project-Build-an-ML-Pipeline-Starter-src_basic_cleaning/clean_sample.csv:reference",
                kl_threshold=config["data_check"]["kl_threshold"],
                min_price=config["etl"]["min_price"],
                max_price=config["etl"]["max_price"]
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                env_manager="conda",
                parameters={
                    "input": "adam6-western-governors-university/Project-Build-an-ML-Pipeline-Starter-src_basic_cleaning/clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                }
            )

        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)
            _run_train_random_forest(
                trainval_artifact="adam6-western-governors-university/nyc_airbnb/trainval_data.csv:latest",
                val_size=config["modeling"]["val_size"],
                random_seed=config["modeling"]["random_seed"],
                stratify_by=config["modeling"]["stratify_by"],
                output_artifact="random_forest_export",
                rf_config=rf_config,
                max_tfidf_features=config["modeling"]["max_tfidf_features"]
            )

        if "test_regression_model" in active_steps:
            script_path = os.path.join(ROOT_DIR, "test_run.py")
            cmd = [
                "python", script_path,
                "--mlflow_model", "adam6-western-governors-university/nyc_airbnb/random_forest_export:prod",
                "--test_dataset", "adam6-western-governors-university/nyc_airbnb/test_data.csv:latest"
            ]
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    go()