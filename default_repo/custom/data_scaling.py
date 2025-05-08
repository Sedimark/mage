from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def split_normalization(final_df):

    X = final_df.drop(columns='label', axis=1)
    y = final_df['label']

    # Split the dataset into training (80%) and testing (20%) sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split 80% for training and 10% of the original dataset for validation
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)


    # Perform scaling so we can have scaled & preprocessed datasets to be used in shamrock framework!!!!
    # Compute 2nd (q5) and 98th (q95) quantiles for feature scaling
    q5_X = np.quantile(X_train, 0.02, axis=0)
    q95_X = np.quantile(X_train, 0.98, axis=0)

    # Apply scaling to features
    X_train_norm = (X_train - q5_X) / (q95_X - q5_X)
    X_test_norm = (X_test - q5_X) / (q95_X - q5_X)
    X_val_norm = (X_val - q5_X) / (q95_X - q5_X)

    # Compute 2nd (q5) and 98th (q95) quantiles for target scaling
    q5_y = np.quantile(y_train, 0.02, axis=0)
    q95_y = np.quantile(y_train, 0.98, axis=0)

    # Apply scaling to target variable
    y_train_norm = (y_train - q5_y) / (q95_y - q5_y)
    y_test_norm = (y_test - q5_y) / (q95_y - q5_y)
    y_val_norm = (y_val - q5_y) / (q95_y - q5_y)

    print(X_train_norm.shape)
    print(y_train_norm.shape)

    print(f"y_train dtype: {y_train_norm.dtype}")
    print(f"y_val dtype: {y_val_norm.dtype}")

    data_dict = {
        "X_train": X_train_norm.to_dict(orient='records'),
        "X_test": X_test_norm.to_dict(orient='records'),
        "X_val": X_val_norm.to_dict(orient='records'),
        "y_train": y_train_norm.tolist(),
        "y_test": y_test_norm.tolist(),
        "y_val": y_val_norm.tolist()
    }
    # # Convert back to DataFrame
    # train_df = pd.DataFrame(np.column_stack((X_train_norm, y_train_norm)), columns=final_df.columns)
    # test_df = pd.DataFrame(np.column_stack((X_test_norm, y_test_norm)), columns=final_df.columns)

    # # Original training data
    # X_train_full = train_df.drop(columns='label', axis=1)
    # y_train_full = train_df['label']

    # # Test set remains the same
    # X_test = test_df.drop(columns='label')
    # y_test = test_df['label']

    return data_dict

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'