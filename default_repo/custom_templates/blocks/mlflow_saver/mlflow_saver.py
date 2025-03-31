from mage_ai.settings.repo import get_repo_path
from mlflow import MlflowClient
from PIL import Image
import base64
import yaml
import os
import io

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


def checker(item: dict, item_type: str, key_type: str, value_type: str) -> None:
    if not isinstance(item, eval(item_type)):
        raise TypeError(f"Item is not of type {item_type}")

    key_types = [type(k).__name__ for k in item.keys()]
    if any(kt != key_type for kt in key_types):
        raise TypeError(f"Not all keys in the dictionary are of type {key_type}")

    value_types = [type(k).__name__ for v in item.values()]
    if any((vt != value_type) or value_type =="any" for vt in value_types):
        raise TypeError(f"Not all values in the dictionary are of type {value_type}")


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


def find_and_import_class(class_name: str):
    for path in sys.path:
        if not os.path.isdir(path):
            continue
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    module_path = os.path.join(root, file)
                    module_name = os.path.relpath(module_path, path).replace(os.sep, ".").removesuffix(".py")
                    
                    try:
                        module = importlib.import_module(module_name)
                        if hasattr(module, class_name):
                            cls = getattr(module, class_name)
                            if inspect.isclass(cls):
                                print(f"Class '{class_name}' found in module '{module_name}'.")
                                return cls
                    except Exception as e:
                        pass
    
    raise ImportError(f"Class '{class_name}' not found in the Python path.")


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    if not isinstance(data, dict):
        raise TypeError("Data is not of type dict!")

    required_keys = ["model", "model_name", "model_type", "images", "parameters", "metrics"]
    optional_keys = ["pytorch_model_class", "pytorch_model_parameters"]
    for key in required_keys:
        if key not in data.keys():
            raise KeyError(f"Required key {key} not present in data!") 

    checker(data["model"], "dict", "str", "any")
    checker(data["model_name"], "dict", "str", "str")
    checker(data["model_type"], "dict", "str", "str")
    checker(data["images"], "dict", "str", "str")
    checker(data["parameters"], "dict", "str", "any")
    checker(data["metrics"], "dict", "str", "any")

    if data["model_type"] not in ["sklearn", "pytorch", "onnx"]:
        raise ValueError("model_type must be: sklearn | pytroch | onnx")
    
    if data["model_type"] == "pytorch":
        for key in optional_keys:
            if key not in data.keys():
                raise KeyError(f"Optional key {key} not present in data!") 

    model_class = find_and_import_class(data["pytorch_model_class"])

    load_env()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    client = MlflowClient()
    with mlfow.start_run(experiment_id=mlflow.get_experiment_by_name(os.getenv("MFLOW_EXPERIMENT_NAME"))) as run:
        mlflow.log_metrics(data["metrics"])
        mlflow.log_params(data["parameters"])

        for image_name, image_data in data["images"].items():
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            mlflow.log_image(buffer, f"images/{image_name}")
        
        if data["model_type"] == "sklearn":
            mlflow.sklearn.log_model(data["model"], "model")
        elif data["model_type"] == "onnx":
            import onnx
            onnx_model = onnx.load_model_from_string(data["model"])
            mlflow.onnx.log_model(onnx_model, "model")
        elif data["model_type"] == "pytorch":
            import torch
            model = model_class(*data["pytorch_model_parameters"])
            state_dict = {k: torch.tensor(v) for k, v in data["model"].items()}
            model.load_state_dict(state_dict)
            mlflow.pytorch.log_model(model, "model")

    artifact_uri = mlflow.get_artifact_uri(artifact_path)

    run_id = run.info.run_id

    src_uri = f"runs:/{run_id}/model"
    try:
        registered_model = client.get_registered_model(data["model_name"])
    except mlflow.exceptions.RestException:
        client.create_registered_model(data["model_name"])

    model_version = client.create_model_version(
        name=data["model_name"],
        source=artifact_uri,
        run_id=run_id,
    )
