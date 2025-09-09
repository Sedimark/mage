import os
from default_repo.utils.generic_pipeline_enabler.default import interoperability_enabler_to_ngsild

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Transform DataFrame back to JSON-LD format
    """
    context_df, temporal_df = data
    data = interoperability_enabler_to_ngsild(context_df, temporal_df)
    return data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'