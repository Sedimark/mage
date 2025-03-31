# Variables {"schema":{"type":"str","description":"The schema for the PostgreSQL deployment.","regex":"^.*$"},"user_id":{"type":"str","description":"The ID of the user who wants to run this pipeline.","regex":"^.*$"},"username":{"type":"str","description":"The username for the user to login inside the database.","regex":"^.*$"},"password":{"type":"secret","description":"The password for the user to login inside the database."},"host":{"type":"str","description":"The host address where the database resides.","regex":"^((25[0-5]|(2[0-4]|1\\d|[1-9]|)\\d)\\.?\\b){4}$"},"port":{"type":"int","description":"The port on which the database runs.","range":[0,65535]},"database":{"type":"str","description":"The name of the database.","regex":"^.*$"},"table":{"type":"str","description":"The name of the table to load datafrom.","regex":"^.*$"}}

from mage_ai.data_preparation.shared.secrets import get_secret_value
from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.mssql import MSSQL
from pandas import DataFrame
from os import path
import yaml

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_mssql(df: DataFrame, **kwargs) -> None:
    """
    Template for exporting data to a MSSQL database.
    Specify your configuration settings in 'io_config.yaml'.
    Set the following in your io_config:

    Docs: https://docs.mage.ai/integrations/databases/MicrosoftSQLServer
    """
    username = kwargs.get('username')
    host = kwargs.get('host')
    port = kwargs.get('port')
    database = kwargs.get('database')
    table = kwargs.get('table')
    schema = kwargs.get('schema')
    user_id = kwargs.get('user_id')

    secret_name = "password-" + kwargs.get("PIPELINE_NAME")

    password = get_secret_value(secret_name)

    if None in [user_id, username, password, host, port, database, schema, table]:
        raise ValueError(
            "All connection parameters (user_id, username, password, host, port, database, schema, table) must be provided.")

    new_entry = {
        user_id: {
            "MSSQL_DATABASE": database,
            "MSSQL_DRIVER": "ODBC Driver 18 for SQL Server",
            "MSSQL_HOST": host,
            "MSSQL_PASSWORD": password,
            "MSSQL_PORT": int(port),
            "MSSQL_SCHEMA": schema,
            "MSSQL_USER": username
        }
    }

    schema_name = schema  # Specify the name of the schema to export data to
    table_name = table  # Specify the name of the table to export data to
    config_path = path.join(get_repo_path(), 'io_config.yaml')

    try:
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
            if data is None:
                data = {}
    except FileNotFoundError:
        data = {}

    data.update(new_entry)

    with open(config_path, 'w') as file:
        yaml.dump(data, file)

    config_profile = user_id

    with MSSQL.with_config(ConfigFileLoader(config_path, config_profile)) as loader:
        loader.export(
            df,
            schema_name,
            table_name,
            index=False,  # Specifies whether to include index in exported table
            if_exists='replace',  # Specify resolution policy if table name already exists
        )
