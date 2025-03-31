from default_repo.utils.ngsi_ld import connector
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


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
    host = os.getenv("NGSI_LD_HOST")
    link_context = os.getenv("NGSI_LD_LINK_CONTEXT")
    entity_id = kwargs.get("entity_id")
    load_date = kwargs.get("load_date")

    if not host or not link_context or not entity_id:
        raise Exception("Needed information to run the block is not provided!")

    bucket = {
        'host': host,
        'entity_id': entity_id,
        'link_context': link_context,
        'time_query': f'timerel=after&timeAt={load_date}T00:00:00Z',
    }

    stellio_broker = connector.DataStore_NGSILD(bucket['host'])

    save_data = connector.SaveData_NGSILD(
        data_store=stellio_broker, 
        entity_id=bucket['entity_id'],
        context=bucket['link_context'],
    )

    save_data.run(bucket)
