if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def complaints_data_preprocessing(complaints_df, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """

    complaints_df = complaints_df.drop(columns=['tax_number', 'Unnamed: 0'], axis=1)

    # Calculate postal code frequency and add it in columns
    postal_code_counts = complaints_df['CMPOSTALCODE'].value_counts() / complaints_df['CMPOSTALCODE'].nunique()
    complaints_df['postal_code_freq'] = complaints_df['CMPOSTALCODE'].map(postal_code_counts)
    complaints_df.drop('CMPOSTALCODE', axis=1, inplace=True)

    complaints_df.rename(columns={'Τυπος': 'COMPLAINTS_TYPE', 
                              'ΤΥΠΟΣ ΠΑΡΟΧΗΣ': 'SUPPLY_TYPE', 
                              'Κατηγορία Θέματος': 'ISSUE_CATEGORY', 
                              'Θέμα': 'ISSUE',
                              'Churn': 'label',
                              'supply_id': 'SUPPLY_ID',
                              'postal_code_freq': 'POSTAL_CODE_FREQ'},inplace=True
                               )

    # Group by SUPPLY_ID, taking the max label (churn) and the mean postal code frequency
    churn_postal_grouped_df = complaints_df.groupby('SUPPLY_ID').agg({
        'label': 'max',  # If any instance of this supply_id has churned, set it to 1
        'POSTAL_CODE_FREQ': 'mean'  # Average postal code frequency per supply_id
    }).reset_index()

    # Count occurrences of each unique COMPLAINT TYPE per SUPPLY_ID
    complaints_type_counts = complaints_df.pivot_table(
        index='SUPPLY_ID',
        columns='COMPLAINTS_TYPE',
        aggfunc='size',
        fill_value=0
    )
    # Count occurrences of each unique SUPPLY TYPE per SUPPLY_ID
    supply_type_counts = complaints_df.pivot_table(
        index='SUPPLY_ID',
        columns='SUPPLY_TYPE',
        aggfunc='size',
        fill_value=0
    )
    # Count occurrences of each unique ISSUE_CATEGORY per SUPPLY_ID
    issue_category_counts = complaints_df.pivot_table(
        index='SUPPLY_ID',
        columns='ISSUE_CATEGORY',
        aggfunc='size',
        fill_value=0
    )
    # Count occurrences of each unique ISSUE per SUPPLY_ID
    issue_counts = complaints_df.pivot_table(
        index='SUPPLY_ID',
        columns='ISSUE',
        aggfunc='size',
        fill_value=0
    )

    # Merge all pivot tables into one DataFrame
    complaints_grouped_df = (
        churn_postal_grouped_df
        .join(complaints_type_counts, rsuffix='_type')
        .join(supply_type_counts, rsuffix='_supply')
        .join(issue_category_counts, rsuffix='_category')
        .join(issue_counts, rsuffix='_issue')
    )

    # Reset index if needed
    complaints_grouped_df = complaints_grouped_df.reset_index()

    return complaints_grouped_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
