import os
import requests
from typing import Any
from datetime import datetime, timezone

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def load_temporal_data(entity_id: str, start_time: str) -> Any:
    context = os.getenv("NGSI_LD_LINK_CONTEXT", "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.8.jsonld")
    host = os.getenv("NGSI_LD_HOST", "")

    if start_time == "":
        start_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    end_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = {
        "Link": f'<{context}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    }

    params = {
        "timerel": "between",
        "timeAt": start_time,
        "endTimeAt": end_time
    }

    url = f"{host}/ngsi-ld/v1/temporal/entities/{entity_id}"

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()

    return response.json()


def load_contextual_data(entity_id) -> Any:
    context = os.getenv("NGSI_LD_LINK_CONTEXT", "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.8.jsonld")
    host = os.getenv("NGSI_LD_HOST", "")

    headers = {
        "Link": f'<{context}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    }

    url = f"{host}/ngsi-ld/v1/entities/{entity_id}"

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return response.json()


@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    start_time = kwargs.get("date", "")
    entity_id = kwargs.get("entity_id", "")

    if entity_id == "":
        raise ValueError("entity_id must be provided")

    temporal_data = load_temporal_data(entity_id, start_time)
    contextual_data = load_contextual_data(entity_id)

    return temporal_data, contextual_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
