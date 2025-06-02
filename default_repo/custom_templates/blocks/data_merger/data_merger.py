from InteroperabilityEnabler.utils.merge_data import merge_predicted_data

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df_initial, predicted_df_with_metadata, *args, **kwargs):
    """
    Merge predicted data into the initial DataFrame by matching column names.
    Add 'null' for missing columns in the predicted data.

    Args:
        df_initial: DataFrame containing the original selected columns.
        predicted_df_with_metadata: DataFrame containing the predicted data with metadata (column names).

    Returns:
        A merged Pandas DataFrame with 'null' for missing columns.
    """
    merged_df = merge_predicted_data(df_initial, predicted_df_with_metadata)

    return merged_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'