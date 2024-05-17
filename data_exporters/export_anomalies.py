from mage_ai.data_preparation.variable_manager import get_variable
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from default_repo.sedimark_demo import secret
from default_repo.sedimark_demo import connector
import copy
import json
import requests

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

def convert_numbers(data):
    if isinstance(data, dict):
        return {k: convert_numbers(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numbers(item) for item in data]
    elif isinstance(data, (int, float)):
        return int(data) if data == int(data) else float(data)
    return data
     
def export_to_broker(data, load_date: str, save_name: str = None):
    if save_name is None:
        entity_id = 'urn:ngsi-ld:WeatherInformation:Forecasted:Hourly:France:Les_Orres-Annotated-Anomaly-UCD'
    else:
        entity_id = f'urn:ngsi-ld:WeatherInformation:Forecasted:Hourly:France:{save_name}'
    
    bucket = {'host': 'https://stellio-dev.eglobalmark.com',
              'url_keycloak': 'https://sso.eglobalmark.com/auth/realms/sedimark/protocol/openid-connect/token',
              'client_id': secret.client_id,
              'client_secret': secret.client_secret,
              'username': secret.username,
              'password': secret.password,
              'entity_to_load_from': "urn:ngsi-ld:WeatherInformation:Forecasted:Hourly:France:Les_Orres",
              'entity_to_save_in': entity_id,
              'link_context': 'https://easy-global-market.github.io/ngsild-api-data-models/projects/jsonld-contexts/sedimark.jsonld',
              'tenant': 'urn:ngsi-ld:tenant:sedimark',
              'time_query': f'timerel=after&timeAt={load_date}T00:00:00Z',
              'content_type': 'application/json'
              }

    stellio_dev = connector.DataStore_NGSILD(bucket['host'], bucket['url_keycloak'])
    stellio_dev.getToken(bucket['client_id'], bucket['client_secret'], bucket['username'], bucket['password'])

    
    load_data = connector.LoadData_NGSILD(data_store=stellio_dev, entity_id=bucket['entity_to_load_from'], context=bucket['link_context'], tenant="urn:ngsi-ld:tenant:sedimark")
    load_data.run(bucket)

    bucket['processed_contextual_data'] = copy.deepcopy(bucket['contextual_data'])
    bucket['processed_contextual_data']["is_anomaly"] = [{'type': 'Property', 'value': 0, 'datasetId': 'urn:ngsi-ld:Dataset:Is-Anomaly', 'observedAt': '2024-04-23T23:00:00Z'}]
    bucket['processed_contextual_data']["anomaly_scores"] = [{'type': 'Property', 'value': 0.0, 'datasetId': 'urn:ngsi-ld:Dataset:Anomaly-Score', 'observedAt': '2024-04-23T23:00:00Z'}]

    bucket['processed_temporal_data'] = data[[column for column in data.columns if column not in ["_is_anomaly", "_anomaly_score"]]].copy()

    bucket['processed_temporal_data']['is_anomaly'] = data["_is_anomaly"]
    bucket['processed_temporal_data']['anomaly_scores'] = data["_anomaly_score"]
    bucket['entity_id'] = bucket['entity_to_save_in']

    for col in bucket['processed_temporal_data'].columns:
        if pd.api.types.is_numeric_dtype(bucket['processed_temporal_data'][col]):
            bucket['processed_temporal_data'][col] =  bucket['processed_temporal_data'][col].apply(lambda x: float(x))

    save_data = connector.SaveData_NGSILD(data_store=stellio_dev, entity_id=bucket['entity_to_save_in'], context=bucket['link_context'], tenant=bucket['tenant'])
    save_data.run(bucket)

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """

    load_date = kwargs.get("load_date")

    load_date = '2024-04-17'

    export_to_broker(data, load_date)
