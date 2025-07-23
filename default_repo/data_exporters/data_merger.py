from InteroperabilityEnabler.utils.merge_data import merge_predicted_data

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def data_merger(data, *args, **kwargs):
    """
    Merge predicted data into the initial DataFrame by matching column names.
    Add 'null' for missing columns in the predicted data.

    Args:
        df_initial: DataFrame containing the original selected columns.
        predicted_df_with_metadata: DataFrame containing the predicted data with metadata (column names).

    Returns:
        A merged Pandas DataFrame with 'null' for missing columns.
    """
    df, predicted_df = data

    merged_df = merge_predicted_data(df, predicted_df)

    return merged_df
