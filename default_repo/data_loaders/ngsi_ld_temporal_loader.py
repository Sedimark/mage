import os
import pandas as pd
from default_repo.utils.generic_pipeline_enabler.default import load_temporal_data

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data_from_api(*args, **kwargs):
    query_broker_flag = kwargs.get('get_data_from_broker', True)

    if not query_broker_flag:
        return {}

    context = os.getenv("NGSI_LD_LINK_CONTEXT", "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.8.jsonld")
    host = os.getenv("NGSI_LD_HOST", None)
    start_time = kwargs.get("start_time", None)
    end_time = kwargs.get("end_time", None)
    attrs = kwargs.get("attrs", None)
    entity_id = kwargs.get("entity_id", "urn:ngsi-ld:Sedimark:CrowdFlowObserved:100016667")

    # start_time = "2022-11-16T07:00:00Z"
    # start_time = None
    # end_time = None
    # attrs = "https://vocab.egm.io/flow"
    # entity_id = "urn:ngsi-ld:Sedimark:CrowdFlowObserved:100016667"

    temporal_data = load_temporal_data(host, context, entity_id, start_time, end_time, attrs)
    return temporal_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'