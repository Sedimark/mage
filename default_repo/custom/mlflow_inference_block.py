import os
import pandas as pd
from mlflow.tracking import MlflowClient

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def ensure_envs() -> None:
    """
    Iterate over a list of environment variables and raise ValueError if not exist.
    """

    required_envs =[
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_FLASK_SERVER_SECRET_KEY",
        "MLFLOW_EXPERIMENT_NAME",
    ]
    optional_env = "MLFLOW_TRACKING_INSECURE_TLS"

    for env in required_envs:
        if os.getenv(env) is None:
            raise ValueError(f"Environment variable {env} not present. Please make sure Mage is configured to work with a MLFlow instance.")

    if os.getenv(optional_env) is None:
        os.environ[optional_env] = "true"


def mlflow_client() -> MlflowClient:
    """
    Ensure that envs are present.

    Create and return an instance of the MlflowClient.

    Returns:
        MlflowClient
    """
    ensure_envs()

    client = MlflowClient(os.getenv("MLFLOW_TRACKING_URI"))

    return client


@custom
def mlflow_inference(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)
    kwargs: Key value pairs containing the dataset and variables

    Returns:
        The SEDIMARK DataFrame.
    """
    model_name = kwargs.get("model_name")
    model_version = kwargs.get("model_version")

    if not model_name or not model_version:
        raise ValueError("Kwargs not present for this block.")

    client = mlflow_client()

    data = kwargs.get("data_flatten", None)

    if data is None:
        raise ValueError("No input data provided.")
    
    data = pd.DataFrame(data)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")

    try:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

        model_version = client.get_model_version(name=model_name, version=model_version)
    
        model_uri = f"s3://models/{experiment_id}/{model_version.run_id}/artifacts"

        model = mlflow.pyfunc.load_model(model_uri)

        predictions = model.predict(data)
    except Exception as ex:
        print(f"Failed to execute the inference process: {ex}")
        raise RuntimeError("Failed to execute the inference process.")
    finally:
        if predictions:
            return predictions

    return None