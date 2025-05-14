if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def nan_outliers_handling(final_df):

    nan_features = ['SKEW', 'KURT', 'CREST', 'ENTROPY_DETAIL']

    for feat in nan_features:
        if final_df[feat].isnull().any():
            print("Warning: NaN values in ENERGY_CONSUMPTION! Filling with 0.")
            final_df[feat].fillna(-1, inplace=True)

    final_df = outliers_handling(df=final_df)

    contains_strings = final_df.select_dtypes(include=['object']).any().any()

    print("DataFrame contains strings:", contains_strings)  

    return final_df

def outliers_handling(df):

    # Calculate the Q1, Q3, and IQR for the 'label' column
    Q1 = df['label'].quantile(0.25)
    Q3 = df['label'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outliers as values greater than Q3 + 3*IQR or less than Q1 - 3*IQR
    outliers = (df['label'] < (Q1 - 3.0 * IQR)) | (df['label'] > (Q3 + 3.0 * IQR))

    # Print how many extreme outliers exist
    print("Number of extreme outliers:", outliers.sum())

    # Drop the rows with outliers in the 'label' column
    df= df[~outliers]

    df = df.drop(columns='ID', axis=1)  

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'

