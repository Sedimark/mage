import os
import pandas as pd
from default_repo.utils.generic_pipeline_enabler.default import interoperability_enabler_to_df

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    if not data:
        return pd.DataFrame()
    
    df = interoperability_enabler_to_df(data)
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'