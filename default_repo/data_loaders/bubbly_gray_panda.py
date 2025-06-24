import io
import pandas as pd
import requests
import subprocess
from mage_ai.settings.repo import get_repo_path
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """

    # here I try to run my python process
    try:
        # Run the script with Python 3.11
        config_file = get_repo_path() + "/configs/<pipeline_name>/config.yaml"
        process_server = subprocess.run(
            [ 'python3.11', "/home/src/default_repo/utils/fleviden/scripts/server.py",  "--config",
            config_file
            ],
            capture_output=True,
            text=True,
            check=True
        )

        process_client = subprocess.run(
            [ 'python3.11', "/home/src/default_repo/utils/fleviden/scripts/client.py",  "--config",
            config_file
            ],
            capture_output=True,
            text=True,
            check=True
        )
        print("Output:", process.stdout)
        return process.stdout
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

    return None

