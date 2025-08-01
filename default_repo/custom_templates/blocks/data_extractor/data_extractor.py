from InteroperabilityEnabler.utils.extract_data import extract_columns

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def data_extractor(df, *args, **kwargs):
    """
        Process specific columns and return them as a pandas DataFrame.

    Args:
        df: input DataFrame

    Returns:
        The column names corresponding to the indices as a separate output
        with the DataFrame (with the selected columns).
    """
    # Select columns by index
    column_indices = [2, 4]

    selected_df, selected_column_names = extract_columns(df, column_indices)

    return selected_df, selected_column_names