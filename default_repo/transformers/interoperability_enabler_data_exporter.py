import re
import pandas as pd
from InteroperabilityEnabler.utils.data_mapper import data_conversion, restore_ngsi_ld_structure

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def flatten_dataframe_by_schema_uri(data):
    """
    Flatten dataframe based on schema_uri with column names in format: schema_uri[index].column_name
    
    Args:
        data: DataFrame with schema_uri column
        
    Returns:
        DataFrame: Flattened dataframe with new column naming convention
    """
    entity_id = data['entity_id'][0]
    entity_type = data['entity_type'][0]

    schema_col = 'schema_uri' if 'schema_uri' in data.columns else 'schema'

    if schema_col not in data.columns:
        raise ValueError(f"DataFrame must contain {schema_col} column")
    
    schema = data[schema_col][0]
    cols_to_remove = ['entity_id', 'entity_type']

    if schema_col == 'schema_uri':
        cols_to_remove.append('schema')

    data = data.drop(cols_to_remove, axis=1)
    grouped = data.groupby(schema_col)
    
    flattened_data = {}
    
    for schema, group in grouped:
        # Reset index to get sequential indexing within each group
        group_reset = group.reset_index(drop=True)
        
        # For each row in the group, create columns with the new naming convention
        for idx, (_, row) in enumerate(group_reset.iterrows()):
            for col_name, value in row.items():
                if col_name == schema_col:
                    continue
                new_col_name = f"{schema}[{idx}].{col_name}"
                flattened_data[new_col_name] = value
    
    flattened_df = pd.DataFrame([flattened_data])
    flattened_df['id'] = entity_id
    flattened_df['type'] = entity_type

    return flattened_df

 
@transformer
def transform(data, *args, **kwargs):
    """
    Transform DataFrame back to JSON-LD format
    """
    df = flatten_dataframe_by_schema_uri(data)
    converted_data = data_conversion(df)
    restored_data = restore_ngsi_ld_structure(converted_data)
    print(restored_data)
    
    return restored_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'