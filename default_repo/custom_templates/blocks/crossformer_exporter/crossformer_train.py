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

    # MLFlow setup
    client = MlflowClient()

    model_name = "crossformer_inference"
    # TODO: add the case information to the model name

    try:
        reigsted_model = client.get_registered_model(model_name)
    except Exception as e:
        print(f"Model {model_name} not found, creating a new one.")
        client.create_registered_model(
            model_name,
            tags={
                "model_type": "crossformer",
                "mage_model": "true",
            },  # TODO: double-check
        )

    mlflow.pytorch.autolog(checkpoint_monitor="val_SCORE", silent=True)
    with mlflow.start_run(
        experiment_id=mlflow.get_experiment_by_name(
            "status_model"
        ).experiment_id
    ) as run:
        # TODO: check the "status_model"

        # Train the model
        trainer.fit(model, dm)

        # Log the model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",  # case_model_inference
            signature=signature,
            reigsted_model_name="model",  # TODO: check the name
        )

        # Set tags
        mlflow.set_tag("model_type", "crossformer")
        mlflow.set_tag("dataset", "CASE")  # TODO: pass the case name
        # mlflow.set_tag("model_name", model_name)  # TODO: add the hyper-parameters

        run_id = run.info.run_id

    src_uri = f"runs://{run_id}/temperature_model_test"
    result = client.create_model_version(
        name=model_name,
        source=src_uri,
        run_id=run_id,
    )

    return model_name
