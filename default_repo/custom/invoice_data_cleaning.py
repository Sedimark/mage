if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def invoice_data_cleaning(invoice_df, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Replace missing values with the mean of the group (per SUPPLY_ID)
    invoice_df['ENERGY_AMOUNT'] = invoice_df.groupby('supply_id')['ENERGY_AMOUNT'].transform(lambda x: x.fillna(x.mean()))

    invoice_df ['TAX_NUMBER'] = invoice_df['tax_number']

    invoice_df['SUPPLY_ID'] = invoice_df['supply_id']

    invoice_df = invoice_df.drop(columns=['tax_number', 'supply_id'])   

    # Count occurrences of each unique REPAYMENT_STATUS per SUPPLY_ID
    repayment_status_counts = invoice_df.pivot_table(
        index='SUPPLY_ID',
        columns='REPAYMENT_STATUS',
        aggfunc='size',
        fill_value=0
    )

    # Count occurrences of each unique PAYMENT_STATUS per SUPPLY_ID
    payment_status_counts = invoice_df.pivot_table(
        index='SUPPLY_ID',
        columns='PAYMENT_STATUS',
        aggfunc='size',
        fill_value=0
    )

    # Aggregate other numerical columns
    invoice_grouped_df = invoice_df.groupby('SUPPLY_ID').agg(
        TOTAL_ENERGY_AMOUNT=('ENERGY_AMOUNT', 'sum'),
        TOTAL_PAYMENT=('TOTAL_PAYMENT_AMOUNT', 'sum')
    )

    # Merge and add suffixes to avoid column name conflicts
    invoice_grouped_df = invoice_grouped_df.join(
        repayment_status_counts, rsuffix='_repay'
    ).join(
        payment_status_counts, rsuffix='_pay'
    )

    # Reset column names
    invoice_grouped_df = invoice_grouped_df.reset_index()
    invoice_grouped_df.columns.name = None  # Remove column group name

    # Reset column names
    invoice_grouped_df = invoice_grouped_df.reset_index(drop=True)
    invoice_grouped_df = invoice_grouped_df.drop(columns=['level_0', 'index'], errors='ignore')

    return invoice_grouped_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
