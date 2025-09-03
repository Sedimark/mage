import requests
import pandas as pd
from mage_ai.data_preparation.shared.secrets import get_secret_value

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

# Get access token using client credentials
def get_access_token():

    token_url = 'https://sso.eglobalmark.com/auth/realms/sedimark/protocol/openid-connect/token'
    client_id = 'wings'
    client_secret = '0hxl53prwJw28H04gI9pThXEYT8GvsfB'

    # token_url = get_secret_value('stelio_token_url')
    # client_id = get_secret_value('client_id')
    # client_secret = get_secret_value('client_secret')

    response = requests.post(token_url, data={
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    })

    if response.status_code != 200:
        raise Exception(f"Token error: {response.status_code} {response.text}")

    return response.json()['access_token']

def transform_stellio_data(df):
    # Make sure every entry is a list of two floats, else replace with [None, None]
    def ensure_coords(x):
        if isinstance(x, list) and len(x) == 2:
            try:
                return [float(x[0]), float(x[1])]
            except:
                return [None, None]
        return [None, None]

    df['location.value.coordinates'] = df['location.value.coordinates'].apply(ensure_coords)

    # Now split the list into two separate columns
    df[['LONGITUDE', 'LATITUDE']] = pd.DataFrame(df['location.value.coordinates'].tolist(), index=df.index)

    df = df.drop(columns=[
            'id', 
            'type', 
            '@context', 
            'location.type',
            'location.value.type',
            'location.value.coordinates',
            'taxId.type', 
            'zipCode.type', 
            'supplyId.type', 
            'invoiceDate.type', 
            'squareMeters.type', 
            'energyConsumption.type'
            ])

    df = df.rename(columns={
            'taxId.value': 'TAX_ID',
            'zipCode.value': 'ZIP_CODE',
            'supplyId.value': 'SUPPLY_ID',
            'invoiceDate.value': 'INVOICE_DATE',
            'squareMeters.value': 'SQUARE_METERS',
            'energyConsumption.value': 'ENERGY_CONSUMPTION'
            })

    # Convert SUPPLY_ID to string properly
    def clean_supply_id(x):
        # If x is dict or list, extract the string value properly
        if isinstance(x, dict):
            # This depends on actual structure, adjust 'value' key as needed
            return str(x.get('value', 'unknown'))
        elif isinstance(x, list):
            # If list, join or take first element as string
            return str(x[0]) if len(x) > 0 else 'unknown'
        else:
            return str(x)

    df['SUPPLY_ID'] = df['SUPPLY_ID'].apply(clean_supply_id)

    # Convert ENERGY_CONSUMPTION to numeric as before
    df['ENERGY_CONSUMPTION'] = pd.to_numeric(df['ENERGY_CONSUMPTION'], errors='coerce')

    return df

@data_loader
def load_data_from_stelio(*args, **kwargs):

    stelio_broker_url = 'https://stellio-dev.eglobalmark.com'
    tenant = 'urn:ngsi-ld:tenant:sedimark'

    # # Retrieve secrets
    # stelio_broker_url = get_secret_value('stelio_broker_url')
    # tenant = get_secret_value('tenant')

    # Get access token
    token = get_access_token()

    # Prepare request headers
    headers = {
        'Accept': 'application/ld+json',
        'Authorization': f'Bearer {token}',
        'NGSILD-Tenant': tenant
    }

    # Query all EnergyConsumptionRecord entities
    entity_type = 'EnergyConsumptionRecord'
    url = f"{stelio_broker_url}/ngsi-ld/v1/entities?type={entity_type}&limit=100"

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Data fetch error: {response.status_code} {response.text}")

    # Convert response JSON to DataFrame
    data = response.json()
    df = pd.json_normalize(data)
    # df = response.json
    df = transform_stellio_data(df)

    data_dict = df_groupby(df=df)

    return data_dict


def df_groupby(df):
    
    grouped_df = df.groupby('SUPPLY_ID')
    groups = grouped_df.groups

    data_dict = {
        str(supply_id): df.loc[indices].copy()
        for supply_id, indices in groups.items()
    }

    print(f"Initial group count: {len(data_dict)}")

    # Remove groups where ALL energy consumption is 0 or NaN
    zero_cons_keys = [
        key for key, value in data_dict.items()
        if value["ENERGY_CONSUMPTION"].fillna(0).sum() == 0.0
    ]
    print(f"Removing {len(zero_cons_keys)} groups with 0 energy consumption")

    for key in zero_cons_keys:
        data_dict.pop(key, None)

    print(f"Remaining groups: {len(data_dict)}")
    return data_dict



# @data_loader
# def load_data_from_api(*args, **kwargs):

#     import io
#     import pandas as pd
#     import requests
#     import os
#     from mage_ai.data_preparation.shared.secrets import get_secret_value
#     """
#     Template for loading data from API
#     """
#     url = 'https://raw.githubusercontent.com/Sedimark/wings_energy_consumption_prediction_model/main/Data/raw/ENERGY_DATA_MYT.csv'

#     # Retrieve the token from environment variable 
#     token = get_secret_value('github_token')

#     print(token)

#     # Add token to headers for authentication
#     headers = {'Authorization': f'token {token}'}

#     # Make the GET request with headers
#     response = requests.get(url, headers=headers)

#     # Raise an error if the request failed
#     response.raise_for_status()

#     df = pd.read_csv(io.StringIO(response.text), header=None, sep=',')

#     df = data_transformer(df=df)

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
    
#     # print(f"data_dict type: {type(data_dict)}")

#     return data_dict

    

# def data_transformer(df, *args, **kwargs):

#     # Expand the single column into multiple columns based on commas
#     df = df[0].str.split(',', expand=True)

#     # Set columns the first row 
#     df.columns = df.iloc[0]

#     # Remove the ID from columns the first column
#     df = df.drop(columns=[df.columns[0]])

#     # Remove the first row
#     df = df.drop(index=0)

#     return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'