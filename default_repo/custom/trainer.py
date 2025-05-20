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


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def train_crossformer(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here


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

    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
