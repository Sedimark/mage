import io
import os
import pandas as pd
import numpy as np
import requests
import datetime
from datetime import timezone

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def load_temporal_data(entity_id: str, start_time: str | None, end_time: str | None, attrs: str | None):
    context = os.getenv("NGSI_LD_LINK_CONTEXT", "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.8.jsonld")
    host = os.getenv("NGSI_LD_HOST", "")

    url = f"{host}/ngsi-ld/v1/temporal/entities/{entity_id}"

    headers = {
        "Link": f'<{context}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    }

    datetime_format = "%Y-%m-%dT%H:%M:%SZ"

    # Validate datetime format
    if start_time is not None:
        try:
            datetime.strptime(start_time, datetime_format)
        except ValueError:
            raise ValueError(f"start_time must be in format {datetime_format}")
    
    if end_time is not None:
        try:
            datetime.strptime(end_time, datetime_format)
        except ValueError:
            raise ValueError(f"end_time must be in format {datetime_format}")

    params = None
    timerel = None
    if start_time is not None:
        if end_time is None:
            end_time = datetime.now(timezone.utc).strftime(datetime_format)
        timerel = "between" 

    params = {
        "timerel": timerel,
        "timeAt": start_time,
        "endTimeAt": end_time,
        "attrs": attrs
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    return response.json()


@data_loader
def load_data_from_api(*args, **kwargs):
    start_time = kwargs.get("start_time", None)
    end_time = kwargs.get("end_time", None)
    attrs = kwargs.get("attrs", None)
    entity_id = kwargs.get("entity_id", "urn:ngsi-ld:Sedimark:CrowdFlowObserved:100016667")

    # start_time = "2022-11-16T07:00:00Z"
    # start_time = None

    # end_time = None
    # attrs = "https://vocab.egm.io/flow"
    # entity_id = "urn:ngsi-ld:Sedimark:CrowdFlowObserved:100016667"

    if entity_id is None:
        raise ValueError("entity_id must be provided")

    temporal_data = load_temporal_data(entity_id, start_time, end_time, attrs)
    return temporal_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'