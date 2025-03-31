# Variables {"username":{"type":"str","description":"The username for the user to login inside the database.","regex":"^.*$"},"password":{"type":"secret","description":"The password for the user to login inside the database."},"host":{"type":"str","description":"The host address where the database resides.","regex":"^((25[0-5]|(2[0-4]|1\\d|[1-9]|)\\d)\\.?\\b){4}$"},"port":{"type":"int","description":"The port on which the database runs.","range":[0,65535]},"bucket":{"type":"str","description":"The name of the bucket.","regex":"^.*$"},"object":{"type":"str","description":"The name of the object to load datafrom.","regex":"^.*$"}}

import pandas as pd
from minio import Minio
from io import StringIO
from mage_ai.data_preparation.shared.secrets import get_secret_value

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data_minio(*args, **kwargs):
    """
    Template code for loading data from a Minio bucket.

    Args:
    - kwargs should include 'username', 'password', 'host', 'port', 'database', 'table', and 'file_key'.

    Returns:
        pandas.DataFrame - Data loaded from the specified Minio file.
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

    data = client.get_object(bucket, obj).read().decode('utf-8')

    df = pd.read_csv(StringIO(data))

    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'Output is not a DataFrame'
