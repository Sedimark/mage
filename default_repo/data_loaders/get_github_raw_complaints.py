if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd
import requests
from mage_ai.data_preparation.shared.secrets import get_secret_value


@data_loader
def get_raw_complaints_data(*args, **kwargs):
    """
    Template for loading data from API
    """
    url_complaints_data = 'https://raw.githubusercontent.com/Sedimark/wings_customer_churn_prediction_model/main/data/raw/parapona_data.xlsx'

    # If the data is private and you need authentication
    token = get_secret_value('github_token')
    headers = {'Authorization': f'token {token}'}
    
    # You can use requests if you want to download the file manually, otherwise skip it
    response = requests.get(url_complaints_data, headers=headers)
    response.raise_for_status()

    # Read Excel file directly from the response content
    from io import BytesIO
    complaints_df = pd.read_excel(BytesIO(response.content))

    return complaints_df



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
