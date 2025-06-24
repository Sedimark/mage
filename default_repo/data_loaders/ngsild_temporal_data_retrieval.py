import os
import requests
import pandas as pd
from typing import Any
from datetime import datetime, timezone

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def load_temporal_data(entity_id: str, start_time: str | None, end_time: str | None) -> Any:
    context = os.getenv("NGSI_LD_LINK_CONTEXT", "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.8.jsonld")
    host = os.getenv("NGSI_LD_HOST", "")
    url = f"{host}/ngsi-ld/v1/temporal/entities/{entity_id}"

    headers = {
        "Link": f'<{context}>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    }

    params = dict()
    if start_time is not None:
        if end_time is None:
            end_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "timerel": "between",
            "timeAt": start_time,
            "endTimeAt": end_time
        }

    if not params:
        params = None

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

    start_time = kwargs.get("start_time", None)
    end_time = kwargs.get("end_time", None)
    entity_id = kwargs.get("entity_id", "")

    if entity_id is None:
        raise ValueError("entity_id must be provided")

    temporal_data = load_temporal_data(entity_id, start_time, end_time)

    timeseries = {k: v for k, v in temporal_data.items() if isinstance(v, list)}
    other_properties = {k: v for k, v in temporal_data.items() if not isinstance(v, list)}

    print(other_properties)
    df = pd.DataFrame()
    for name, ts in timeseries.items():
        temp_df = pd.DataFrame(ts)
        temp_df['property'] = name

        if df.empty:
            df = temp_df
            continue

        df = pd.concat([df, temp_df], axis=0, ignore_index=True)

    for prop, value in other_properties.items():
        df[prop] = value

    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
