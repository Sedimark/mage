# Variables {"username":{"type":"str","description":"The username for the user to login inside the database.","regex":"^.*$"},"password":{"type":"secret","description":"The password for the user to login inside the database."},"host":{"type":"str","description":"The host address where the database resides.","regex":"^((25[0-5]|(2[0-4]|1\\d|[1-9]|)\\d)\\.?\\b){4}$"},"port":{"type":"int","description":"The port on which the database runs.","range":[0,65535]},"database":{"type":"str","description":"The name of the database.","regex":"^.*$"},"collection":{"type":"str","description":"The name of the collection to load data from.","regex":"^.*$"}}

import pandas as pd
from pymongo import MongoClient
from mage_ai.data_preparation.shared.secrets import get_secret_value

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data_mongodb(*args, **kwargs):
    """
    Template code for loading data from a MongoDB database.

    Args:
    - kwargs should include 'username', 'password', 'host', 'port', 'database', and 'table'.

    Returns:
        pandas.DataFrame - Data loaded from the specified MongoDB collection.
    """

    username = kwargs.get('username')
    host = kwargs.get('host')
    port = kwargs.get('port')
    database = kwargs.get('database')
    collection = kwargs.get('tabcollectionle')

    secret_name = "password-" + kwargs.get("PIPELINE_NAME")

    password = get_secret_value(secret_name)

    if None in [username, password, host, port, database, collection]:
        raise ValueError("All connection parameters (username, password, host, port, database, collection) must be provided.")

    connection_string = f"mongodb://{username}:{password}@{host}:{port}/{database}"

    client = MongoClient(connection_string)
    db = client[database]
    collection = db[collection]

    df = pd.DataFrame(list(collection.find()))

    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'Output is not a DataFrame'
