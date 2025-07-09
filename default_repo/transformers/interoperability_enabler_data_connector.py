import re
import pandas as pd
from InteroperabilityEnabler.utils.data_formatter import data_to_dataframe

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def data_interoperability_enabler(temporal_data):
    metadata_columns = ['id', 'type']
    metadata_values = {f'entity_{k}': v for k, v in temporal_data.items() if k in metadata_columns}


    def is_uri(s):
        # Simple URI check: starts with http:// or https://
        return isinstance(s, str) and re.match(r'^https?://', s) is not None

    value_columns = {k: v for k, v in temporal_data.items() if k not in metadata_columns and isinstance(v, list)}

    result_df = pd.DataFrame()
    for k, v in value_columns.items():
        temp_df = data_to_dataframe(v)
        if is_uri(k):
            temp_df['schema_uri'] = k
            temp_df['schema'] = k.split('/')[-1].split('#')[-1]
        else:
            temp_df['schema'] = k
        for prop, value in metadata_values.items():
            temp_df[prop] = value

        result_df = pd.concat([result_df, temp_df], ignore_index=True)

    return result_df


@transformer
def transform(data, *args, **kwargs):
    df = data_interoperability_enabler(data)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'