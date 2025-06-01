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

from default_repo.utils.crossformer_wrap.wrap import setup_fit, cfg_base
import torch 

if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


# Set MLflow environment variables
# os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
# os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
# os.environ['AWS_ACCESS_KEY_ID'] = 'super'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://minio.sedimark.work'
# os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://172.20.132.105:5000") # it is local ip and should be replaced by https://mlflow.sedimark.work/ 

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
def train_crossformer(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here

    # load cfg & consider apply new changes to cfg
    # TODO: We should connect with UI to adjust the cfg
    cfg = cfg_base
    
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
    mlflow.pytorch.autolog(checkpoint_monitor="val_SCORE", silent=True)
    with mlflow.start_run(
        experiment_id=current_experiment.experiment_id
    ) as run:
        # TODO: check the "status_model"

        # Train the model
        trainer.fit(model, dm)

        # Log the model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="Trained_Model",  # TODO: check the name
        )
    
    test_result = trainer.test(model, dm, ckpt_path="best")

    return test_result, registered_model_name


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
