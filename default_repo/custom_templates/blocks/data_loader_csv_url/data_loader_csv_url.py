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
def data_loader_csv_url(*args, **kwargs):
    """
        Template for loading data from a github repository
    """
    url = 'https://raw.githubusercontent.com/Sedimark/SEDIMARK_DPP/refs/heads/version_2/datasets/water%20use%20case/surrey_AI_data/broker.csv'

    # Retrieve the token from environment variable
    token = get_secret_value('github_token')

    # Add token to headers for authentication
    headers = {'Authorization': f'token {token}'}

    # Make request
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.text

    # Return CSV content as string
    return data


# @test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'