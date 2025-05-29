import os
import json
import pandas as pd
import requests
from mage_ai.data_preparation.shared.secrets import get_secret_value
from urllib.parse import urlparse, unquote


if 'data_loader' not in globals():
     from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
     from mage_ai.data_preparation.decorators import test


@data_loader
def data_loader_ngsi_ld_file(*args, **kwargs):
    """
        Template for loading data from a github repository
    """
    url = 'https://raw.githubusercontent.com/Sedimark/SEDIMARK_DPP/refs/heads/version_2/datasets/urban%20bike%20use%20case/temporal_station.jsonld'

    # Retrieve the token from environment variable
    token = get_secret_value('github_token')

    # Add token to headers for authentication
    headers = {'Authorization': f'token {token}'}

    # Make request
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Parse response as JSON
    data = response.json()

    return data

# @test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'