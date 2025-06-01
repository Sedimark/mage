import os
import mlflow
import numpy as np
from mage_ai.settings.repo import get_repo_path
from os import path
import yaml
import pandas as pd
import datetime
from mlflow import MlflowClient
from mlflow.models import infer_signature

#
from default_repo.utils.crossformer_wrap.wrap import inference
import torch  # it should be installed when installing crossformer


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

"""
This part has no difference with the train script in data_exporters.
TODO: It would be better to check with the Stefan's work
"""
# config MLFlow env
config_path = path.join(get_repo_path(), "io_config.yaml")
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# Set MLflow environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
os.environ['AWS_ACCESS_KEY_ID'] = 'super'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://minio.sedimark.work'
os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://mlflow.sedimark.work/") 

# Create experiment
experiment_name = "CrossFormer"
current_experiment = mlflow.get_experiment_by_name(experiment_name)

if current_experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    current_experiment = mlflow.get_experiment(experiment_id)

# End any active run
if mlflow.active_run():
    mlflow.end_run()

@custom
def transform_custom(data, *args, **kwargs):
    """transform_inference_crossformer

    Transform the input data (values-only) into the predictions by
    the trained crossformer model to MLFlow.

    Args:
        data (pd.DataFrame): values-only DataFrame from the upstream block.
    """

    data = data.iloc[:24]
    print(data)
    # load the model
    model = mlflow.pytorch.load_model("models:/model/latest")  # TODO: /model/latest is /registered_name/version and should be replaced
    predictions = inference(
        model=model,
        df=data,
    )
    return predictions


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
