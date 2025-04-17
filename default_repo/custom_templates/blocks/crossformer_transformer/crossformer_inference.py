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
from default_repo.utils.crossformer_wrap import inference
import torch  # it should be installed when installing crossformer

if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


"""
This part has no difference with the train script in data_exporters.
TODO: It would be better to check with the Stefan's work
"""
# config MLFlow env
config_path = path.join(get_repo_path(), "io_config.yaml")
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


MLFLOW_TRACKING_USERNAME = config["default"]["MLFLOW_TRACKING_USERNAME"]
MLFLOW_TRACKING_PASSWORD = config["default"]["MLFLOW_TRACKING_PASSWORD"]
AWS_ACCESS_KEY_ID = config["default"]["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = config["default"]["AWS_SECRET_ACCESS_KEY"]
MLFLOW_S3_ENDPOINT_URL = config["default"]["MLFLOW_S3_ENDPOINT_URL"]
MLFLOW_TRACKING_INSECURE_TLS = config["default"]["MLFLOW_TRACKING_INSECURE_TLS"]


os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = MLFLOW_TRACKING_INSECURE_TLS
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "1000"

mlflow.set_tracking_uri("http://62.72.21.79:5000")  # change tracking uri later

mlflow.pytorch.autolog()  # pay attention here, monitored metrics may be essential


@transformer
def transform_inference_crossformer(data, *args, **kwargs):
    """transform_inference_crossformer

    Transform the input data (values-only) into the predictions by
    the trained crossformer model to MLFlow.

    Args:
        data (pd.DataFrame): values-only DataFrame from the upstream block.
    """

    # load the model
    model = mlflow.pytorch.load_model(logged_model)  # TODO: check the loading
    predictions = inference(
        model=model,
        data=data,
    )
    return predictions


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
