import os
from default_repo.utils.generic_pipeline_enabler.default import export_temporal_data

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    # context = os.getenv("NGSI_LD_LINK_CONTEXT", "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context-v1.8.jsonld")
    context = "https://sedimark.github.io/broker/jsonld-contexts/sedimark-helsinki-compound.jsonld"

    host = os.getenv("NGSI_LD_HOST", None)

    # print(data)
    response = export_temporal_data(data, host, context)

    print(response)

    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here

