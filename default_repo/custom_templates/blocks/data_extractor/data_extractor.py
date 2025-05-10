from InteroperabilityEnabler.utils.extract_data import extract_columns

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df, column_indices, *args, **kwargs):
    """
        Process specific columns and return them as a pandas DataFrame.

    Args:
        df: pandas DataFrame.
        column_indices: List of column indices to be selected (0-based index).

    Returns:
        The column names corresponding to the indices as a separate output
        with the DataFrame (with the selected columns).
    """
    selected_df, selected_column_names = extract_columns(df, column_indices)

    return selected_df, selected_column_names


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'