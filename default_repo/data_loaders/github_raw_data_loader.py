import io
import pandas as pd
import requests
import os
from mage_ai.data_preparation.shared.secrets import get_secret_value

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """
    url = 'https://raw.githubusercontent.com/Sedimark/wings_energy_consumption_prediction_model/main/Data/raw/ENERGY_DATA_MYT.csv'

    # Retrieve the token from environment variable 
    token = get_secret_value('github_token')

    # Add token to headers for authentication
    headers = {'Authorization': f'token {token}'}

    # Make the GET request with headers
    response = requests.get(url, headers=headers)

    # Raise an error if the request failed
    response.raise_for_status()

    data = pd.read_csv(io.StringIO(response.text), header=None, sep=',')

    data = data_transformer(data=data)

    return data

def data_transformer(data, *args, **kwargs):

    # Expand the single column into multiple columns based on commas
    data = data[0].str.split(',', expand=True)

    # Set columns the first row 
    data.columns = data.iloc[0]

    # Remove the ID from columns the first column
    data = data.drop(columns=[data.columns[0]])

    # Remove the first row
    df = data.drop(index=0)

    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'