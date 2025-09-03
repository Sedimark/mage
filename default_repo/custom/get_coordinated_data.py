import io
import pandas as pd
import requests
import os
from mage_ai.data_preparation.shared.secrets import get_secret_value

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

# THIS BLOCK IS USED INSTEAD OF THE PARENT BLOCK TO AVOID RUNNING IT and use it as a shortcut

# @custom
# def fetch_coordinates_data(*args, **kwargs):
#     """
#     Template for loading data from API
#     """
#     url = 'https://raw.githubusercontent.com/Sedimark/wings_energy_consumption_prediction_model/main/Data/Dataset_mytilinaios_with_lat_long.csv'

#     # Retrieve the token from environment variable 
#     token = get_secret_value('github_token')

#     # Add token to headers for authentication
#     headers = {'Authorization': f'token {token}'}

#     # Make the GET request with headers
#     response = requests.get(url, headers=headers)

#     # Raise an error if the request failed
#     response.raise_for_status()

#     df = pd.read_csv(io.StringIO(response.text), sep=',')

#     df = df.dropna(axis=0)

#     df = df_groupby(df=df)

#     return df


# # Groupby supply_id in df
# @custom
# def df_groupby(df):

#     grouped_df = df.groupby('SUPPLY_ID')
#     groups = grouped_df.groups

#     # Convert keys to strings
#     data_dict = {}
    
#     for supply_id, indices in list(groups.items()):
#         data_dict[str(supply_id)] = df.loc[indices]  # Convert `supply_id` to string
    
#     single_row_keys = [key for key, value in data_dict.items() if len(value) == 1]
#     keys_removed = [key for key in single_row_keys]

#     for key in keys_removed:
#         data_dict.pop(key, None)

#     zero_cons_keys = [key for key, value in data_dict.items() if value["ENERGY_CONSUMPTION"].iloc[0] == 0.0]
#     keys_removed = [key for key in zero_cons_keys]

#     for key in keys_removed:
#         data_dict.pop(key, None)

#     return data_dict

#     # # Read and return the CSV data
#     # return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
