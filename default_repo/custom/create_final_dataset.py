if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def merge_datasets(invoice_grouped_df, complaints_grouped_df, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Ensure SUPPLY_ID exists in both DataFrames
    complaints_grouped_df = complaints_grouped_df.reset_index()
    invoice_grouped_df = invoice_grouped_df.reset_index()

    # Merge using SUPPLY_ID as the key
    final_df = invoice_grouped_df.merge(complaints_grouped_df, on='SUPPLY_ID', how='left')

    # Fill NaN values (if needed, because some supplies may not have complaints)
    final_df = final_df.fillna(0)

    final_df.drop(columns=[ 'index_x', 'index_y', 'SUPPLY_ID', 'level_0'], inplace=True)

    return final_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
