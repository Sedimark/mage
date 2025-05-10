from InteroperabilityEnabler.utils.add_metadata import add_metadata_to_predictions_from_dataframe

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(predicted_df, selected_column_names, *args, **kwargs):
    """
    Add metadata (column names) back to the predicted DataFrame (from an AI model).

    Args:
        predicted_df: DataFrame containing the predictions without metadata.
        column_names: List of column names corresponding to the predictions.

    Returns:
        Pandas DataFrame with metadata (column names).
    """
    predicted_df = add_metadata_to_predictions_from_dataframe(
        predicted_df, selected_column_names
    )

    return predicted_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'