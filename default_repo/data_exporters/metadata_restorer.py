from InteroperabilityEnabler.utils.add_metadata import add_metadata_to_predictions_from_dataframe

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def metadata_restorer(data, *args, **kwargs):
    """
    Add metadata (column names) back to the predicted DataFrame (from an AI model).

    Args:
        predicted_df: DataFrame containing the predictions without metadata.
        column_names: List of column names corresponding to the predictions.

    Returns:
        Pandas DataFrame with metadata (column names).
    """
    predictions, df, selected_df, selected_column_names = data


    predicted_df = add_metadata_to_predictions_from_dataframe(
        predictions, selected_column_names
    )

    return df, predicted_df

