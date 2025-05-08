if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@custom
# #Use Google API to find the latitude and the longitude for the different zip codes
# def extract_lat_long_via_address(address_or_zipcode, country='Greece'):
#     """Fetch latitude and longitude for a given address/zipcode from Google Maps API."""
#     with open("config.json") as config_file:
#         config = json.load(config_file)
#         GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
#     try:
#         endpoint = f"https://maps.googleapis.com/maps/api/geocode/json?address={address_or_zipcode}&components=country:{country}&key={GOOGLE_API_KEY}"
#         response = requests.get(endpoint)
#         if response.status_code == 200:
#             result = response.json()['results'][0]
#             return result['geometry']['location']['lat'], result['geometry']['location']['lng']
#     except (IndexError, KeyError):
#         return None, None

# df_split['ZIP_CODE'] = df_split['ZIP_CODE'].astype(str)
# df_split[['LATITUDE', 'LONGITUDE']] = df_split['ZIP_CODE'].apply(extract_lat_long_via_address).apply(pd.Series)

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'