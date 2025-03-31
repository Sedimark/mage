# Variables {"username":{"type":"str","description":"The username for the user to login inside the database.","regex":"^.*$"},"password":{"type":"secret","description":"The password for the user to login inside the database."},"host":{"type":"str","description":"The host address where the database resides.","regex":"^((25[0-5]|(2[0-4]|1\\d|[1-9]|)\\d)\\.?\\b){4}$"},"port":{"type":"int","description":"The port on which the database runs.","range":[0,65535]},"bucket":{"type":"str","description":"The name of the bucket.","regex":"^.*$"},"object":{"type":"str","description":"The name of the object to load datafrom.","regex":"^.*$"}}

import pandas as pd
from minio import Minio
from io import BytesIO
from mage_ai.data_preparation.shared.secrets import get_secret_value

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def exporter(df: pd.DataFrame, *args, **kwargs):
    """
    Template code for exporting data to a Minio bucket.

    Args:
    - kwargs should include 'username', 'password', 'host', 'port', 'bucket', and 'object'.

    """

    username = kwargs.get('username')
    host = kwargs.get('host')
    port = kwargs.get('port')
    bucket = kwargs.get('bucket')
    obj = kwargs.get('object')

    secret_name = "password-" + kwargs.get("PIPELINE_NAME")

    password = get_secret_value(secret_name)

    if None in [username, password, host, port, bucket, obj]:
        raise ValueError("All connection parameters (username, password, host, port, bucket, obj) must be provided.")

    client = Minio(
        f"{host}:{port}",
        access_key=username,
        secret_key=password,
        secure=False
    )

    # Convert DataFrame to CSV in memory
    csv_data = df.to_csv(index=False)
    csv_bytes = BytesIO(csv_data.encode('utf-8'))

    # Upload CSV to Minio
    client.put_object(
        bucket,
        obj,
        data=csv_bytes,
        length=csv_bytes.getbuffer().nbytes,
        content_type='application/csv'
    )