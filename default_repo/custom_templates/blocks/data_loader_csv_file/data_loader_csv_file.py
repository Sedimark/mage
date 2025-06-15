import os
import json
import pandas as pd
import requests
from urllib.parse import urlparse, unquote


if 'data_loader' not in globals():
     from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
     from mage_ai.data_preparation.decorators import test


@data_loader
def data_loader_csv_file(*args, **kwargs):
    """
        Template for loading csv data
    """
    file_path = "default_repo/broker_sample.csv"

    # Read the file contents
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # Return the data (you could also process it further)
    return data

# @test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'