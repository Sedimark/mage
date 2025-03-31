from mage_ai.settings.repo import get_repo_path
import mlflow

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def load_env():
    config_path = f"{get_repo_path()}/io_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ['MLFLOW_TRACKING_USERNAME'] = config["MLFLOW"]["MLFLOW_TRACKING_USERNAME"].strip().replace("\n", "")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config["MLFLOW"]["MLFLOW_TRACKING_PASSWORD"].strip().replace("\n", "")
    os.environ['AWS_ACCESS_KEY_ID'] = config["MLFLOW"]["AWS_ACCESS_KEY_ID"].strip().replace("\n", "")
    os.environ['AWS_SECRET_ACCESS_KEY'] = config["MLFLOW"]["AWS_SECRET_ACCESS_KEY"].strip().replace("\n", "")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = config["MLFLOW"]["MLFLOW_S3_ENDPOINT_URL"].strip().replace("\n", "")
    os.environ['MLFLOW_EXPERIMENT'] = config["MLFLOW"]["MFLOW_EXPERIMENT_NAME"]
    os.environ['MLFLOW_TRACKING_URI'] = config["MLFLOW"]["MLFLOW_TRACKING_URI"]
    os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = "true"
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = "1000"


@transformer
def mlflow_loader(data, *args, **kwargs):
    """
    Transformer block for loading mlflow models and doing a prediction.

    Returns:
        pandas DataFrame
    """
    load_env()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    model_name = kwargs.get("model_name")
    model_version = kwargs.get("model_version")

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    
    return model.predict(data)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
