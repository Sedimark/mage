from sklearn.model_selection import train_test_split

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def train_val_test_split(final_df, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    print(type(final_df.columns[0]))  # Should be <class 'str'>


    X = final_df.drop(columns='label', axis=1)
    y = final_df['label']

    # First split to get 'train + validation' set and 'test' set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Then split to create training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

    data_dict = {
        "X_train": X_train.to_dict(orient='records'),
        "X_test": X_test.to_dict(orient='records'),
        "X_val": X_val.to_dict(orient='records'),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
        "y_val": y_val.tolist()
    }

    return data_dict


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
