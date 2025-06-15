from InteroperabilityEnabler.utils.merge_data import merge_predicted_data

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


def transform(df_initial, predicted_df_with_metadata, *args, **kwargs):
    """
    Merge predicted data into the initial DataFrame by matching column names.
    Add 'null' for missing columns in the predicted data.

    Args:
        data: Containing the initial data (input data) and the predicted data.

    Returns:
        A merged Pandas DataFrame with 'null' for missing columns.
    """
    df, predicted_df = data

    merged_df = merge_predicted_data(df, predicted_df)

    return merged_df
