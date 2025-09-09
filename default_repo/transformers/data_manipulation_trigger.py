import os
import pandas as pd
from default_repo.utils.generic_pipeline_enabler.default import execute_pipeline_and_get_final_result

# Restored decorator imports to ensure compatibility with your environment
if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Main transformation block that triggers a child pipeline and processes its results.
    """
    context_df, temporal_df = data
    child_pipeline_uuid = kwargs.get("data_manipulation_pipeline_trigger", "data_manipulation_test")

    df = execute_pipeline_and_get_final_result(
        pipeline_uuid=child_pipeline_uuid,
        data=temporal_df,
        kwargs=kwargs,
        max_output_chars=300,
        poll_interval=10
    )
    
    return context_df, df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert isinstance(output, pd.DataFrame), 'The output should be a pandas DataFrame.'
    assert not output.empty, 'The output DataFrame should not be empty if the child pipeline produced data.'