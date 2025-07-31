from datetime import datetime
from minio import Minio
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


def ensure_envs() -> None:
    required_envs = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_S3_ENDPOINT_URL",
    ]

    for env in required_envs:
        if os.getenv(env) is None:
            raise ValueError(f"Environment variable {env} not present. Please make sure Mage is configured to work with a Minio instance.")


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports predictions to a MinIO bucket.

    Args:
        data: The predictions from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    ensure_envs()

    minio_uri = os.getenv("MLFLOW_S3_ENDPOINT_URL", "").split("/")[-1]
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    prediction_name = kwargs.get("prediction_name")

    if not prediction_name:
        raise ValueError("Prediction name not present in kwargs.")

    if data is None:
        print("No predictions to upload, skipping...")
        return

    client = Minio(minio_uri, access_key=access_key, secret_key=secret_key)

    if not client.bucket_exists("predictions"):
        client.make_bucket("predictions")

    buffer = BytesIO()

    pickle.dump(data, buffer)

    buffer.seek(0)

    now = int(datetime.utcnow().timestamp())

    client.put_object("predictions", f"{prediction_name}/{now}.pkl", buffer, length=-1, part_size=10*1024*1024)