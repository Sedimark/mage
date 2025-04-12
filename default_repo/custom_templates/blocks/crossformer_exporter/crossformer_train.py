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
from default_repo.utils.crossformer_wrap import setup_fit
import torch  # it should be installed when installing crossformer

if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

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
def train_crossformer(data, *args, **kwargs):
    """
    Train crossformer block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """

    model, dm, trainer = setup_fit(
        cfg=cfg,
        df=data,
        callbacks=None,
    )

    client = MlflowClient()
    # signature
    input_example = torch.randn(1, cfg["in_len"], cfg["data_dim"])
    output_example = model(input_example)
    signature = infer_signature(
        input_example.numpy(), output_example.detach().numpy()
    )
    mlflow.pytorch.autolog(checkpoint_monitor="val_SCORE", silent=True)
    with mlflow.start_run() as run:
        trainer.fit(model, data)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
        )
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, f"{cfg['experiment_name']}_best_model")
    test_result = trainer.test(model, data, ckpt_path="best")

    return data


@data_exporter
def export_train_crossformer(data, *args, **kwargs) -> None:
    """export_train_crossformer

    Export the trained crossformer model to MLFlow.

    Args:
        data (pd.DataFrame): values-only DataFrame from the upstream block.
    """

    # Setup the training configuration
    model, dm, trainer = setup_fit(
        cfg=cfg,
        df=data,
        callbacks=None,
    )  # TODO: where to load the cfg?

    # Create the signature for the model
    input_example = torch.randn(1, cfg["in_len"], cfg["data_dim"])
    output_example = model(input_example)
    signature = infer_signature(
        input_example.numpy(), output_example.detach().numpy()
    )

    # Start the MLflow run
    client = MlflowClient()
    mlflow.pytorch.autolog(checkpoint_monitor="val_SCORE", silent=True)
    with mlflow.start_run() as run:
        # Train the model
        trainer.fit(model, dm)

        # Log the model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
        )
