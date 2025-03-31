from default_repo.utils.ngsi_ld import connector
import os

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    """
    Loader block for NGSI-LD entities.

    Returns:
        pandas.DataFrame
    """
    host = os.getenv("NGSI_LD_HOST")
    link_context = os.getenv("NGSI_LD_LINK_CONTEXT")
    entity_id = kwargs.get("entity_id")
    load_date = kwargs.get("load_date")

    if not host or not link_context or not entity_id or not load_date:
        raise Exception("Needed information to run the block is not provided!")

    bucket = {
        'host': host,
        'entity_id': entity_id,
        'link_context': link_context,
        'time_query': f'timerel=after&timeAt={load_date}T00:00:00Z',
    }

    stellio_broker = connector.DataStore_NGSILD(bucket['host'])

    load_data = connector.LoadData_NGSILD(
        data_store=stellio_broker, 
        entity_id=bucket['entity_id'],
        context=bucket['link_context'],
    )

    load_data.run(bucket)

    data = bucket['temporal_data']
    data.reset_index(inplace=True)

    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
