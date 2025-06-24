import pandas as pd
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def profile_value_column(data, *args, **kwargs):
    """
    Generates a profile report specifically for the 'value' column of the input df.

    Args:
        data: The input df to be converted to DataFrame. Expected to have a column named 'value'.
        *args: Additional arguments (not used in this block).
        **kwargs: Additional keyword arguments (not used in this block).
                logger: The logger object from MageAI for logging.

    Returns:
        A pandas DataFrame containing the profile report for the 'value' column.
        The output DataFrame will have a single row.
        Columns in the output DataFrame include:
        - 'column_name': Will be 'value'.
        - 'data_type': Data type of the 'value' column.
        - 'missing_values_count': Number of missing values in 'value'.
        - 'missing_values_percentage': Percentage of missing values in 'value'.
        - 'unique_values_count': Number of unique values in 'value'.
        - 'unique_values_percentage': Percentage of unique values in 'value'.
        - 'mean': Mean of the 'value' column (if numeric).
        - 'median': Median of the 'value' column (if numeric).
        - 'std_dev': Standard deviation of the 'value' column (if numeric).
        - 'min': Minimum value of the 'value' column (if numeric).
        - '25%': 25th percentile (if numeric).
        - '50%': 50th percentile (median, if numeric).
        - '75%': 75th percentile (if numeric).
        - 'max': Maximum value of the 'value' column (if numeric).
        - 'mode': Most frequent value in 'value'.
        - 'mode_count': Count of the most frequent value in 'value'.
        - 'mode_percentage': Percentage of the most frequent value in 'value'.
    """
    logger = kwargs.get('logger')

    if data.empty:
        if logger:
            logger.warning("Input DataFrame is empty. Returning an empty profile.")
        return pd.DataFrame()

    if 'value' not in data.columns:
        if logger:
            logger.error("Column 'value' not found in the input DataFrame.")
        # Return an empty DataFrame or raise an error, depending on desired behavior
        # For now, let's return an empty DataFrame with expected columns to avoid downstream errors if possible
        # Or, more strictly:
        raise KeyError("Column 'value' not found in the input DataFrame. Cannot perform profiling.")


    if logger:
        logger.info(f"Starting data profiling for the 'value' column from DataFrame with {data.shape[0]} rows.")

    total_rows = len(data)
    column_data = data['value']
    col_name = 'value'

    # Basic stats
    record = {
        'column_name': col_name,
        'data_type': str(column_data.dtype),
        'missing_values_count': int(column_data.isnull().sum()),
        'missing_values_percentage': round((column_data.isnull().sum() / total_rows) * 100, 2) if total_rows > 0 else 0,
        'unique_values_count': int(column_data.nunique()),
        'unique_values_percentage': round((column_data.nunique() / total_rows) * 100, 2) if total_rows > 0 else 0,
        'mean': np.nan,
        'median': np.nan,
        'std_dev': np.nan,
        'min': np.nan,
        '25%': np.nan,
        '50%': np.nan,
        '75%': np.nan,
        'max': np.nan,
        'mode': np.nan,
        'mode_count': np.nan,
        'mode_percentage': np.nan,
    }

    # Numeric stats
    if pd.api.types.is_numeric_dtype(column_data):
        desc = column_data.describe()
        record['mean'] = round(desc.get('mean', np.nan), 4)
        record['median'] = round(desc.get('50%', np.nan), 4) # Median is the 50th percentile
        record['std_dev'] = round(desc.get('std', np.nan), 4)
        record['min'] = desc.get('min', np.nan)
        record['25%'] = desc.get('25%', np.nan)
        record['50%'] = desc.get('50%', np.nan)
        record['75%'] = desc.get('75%', np.nan)
        record['max'] = desc.get('max', np.nan)
    
    # Object/Categorical stats (Mode)
    if column_data.count() > 0: # Check if there are any non-NA values to calculate mode
        mode_series = column_data.mode()
        if not mode_series.empty:
            mode_val = mode_series[0] # Take the first mode if multiple exist
            record['mode'] = mode_val
            record['mode_count'] = int(column_data[column_data == mode_val].count())
            record['mode_percentage'] = round((record['mode_count'] / total_rows) * 100, 2) if total_rows > 0 else 0
    
    profile = pd.Series(record)
    
    # Reorder columns for better readability
    ordered_columns = [
        'column_name', 'data_type', 
        'missing_values_count', 'missing_values_percentage',
        'unique_values_count', 'unique_values_percentage',
        'mean', 'median', 'std_dev', 'min', '25%', '50%', '75%', 'max',
        'mode', 'mode_count', 'mode_percentage'
    ]
    profile = profile[ordered_columns]
    metadata_cols = data.nunique() == 1
    metadata_cols = list(metadata_cols[metadata_cols == True].index)
    metadatas = data[metadata_cols].iloc[0]

    processed_data = pd.concat([metadatas, profile])

    if logger:
        logger.info(f"Data profiling complete for the 'value' column. Profile:\n{processed_data.to_string()}")


    return processed_data