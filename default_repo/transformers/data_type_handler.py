import pandas as pd
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def correct_data_types(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Corrects data types for specified columns.

    Configuration through kwargs:
        type_conversions (dict): A dictionary where keys are column names and 
                                 values are the target pandas/numpy dtype string.
                                 e.g., {'col_A': 'int64', 'col_B': 'float', 'date_col': 'datetime64[ns]'}
        datetime_format (str or dict, optional): If converting to datetime, specify the format string.
                                                 If a dict, keys are column names, values are format strings.
                                                 e.g., '%Y-%m-%d %H:%M:%S'
        errors (str, optional): For pd.to_numeric and pd.to_datetime, specifies error handling.
                                'raise', 'coerce' (invalid parsing set as NaT/NaN), 'ignore'. Default 'raise'.
        logger: MageAI logger.
    """
    logger = kwargs.get('logger')
    df = data[[column for column in data.columns if column.endswith("__value")]]

    type_conversions = kwargs.get('type_conversions', {})
    datetime_format_config = kwargs.get('datetime_format') # Can be a string or a dict
    errors_handling = kwargs.get('errors', 'raise')

    if not type_conversions:
        if logger:
            logger.info("No type conversions specified. DataFrame types remain unchanged.")
        return df

    if logger:
        logger.info(f"Applying data type corrections: {type_conversions}")

    for column, target_type_str in type_conversions.items():
        if column not in df.columns:
            if logger:
                logger.warning(f"Column '{column}' for type conversion not found in DataFrame. Skipping.")
            continue
        
        try:
            current_format = None
            if isinstance(datetime_format_config, dict):
                current_format = datetime_format_config.get(column)
            elif isinstance(datetime_format_config, str) and target_type_str.startswith('datetime'):
                current_format = datetime_format_config

            if target_type_str.startswith('datetime'):
                df[column] = pd.to_datetime(df[column], format=current_format, errors=errors_handling)
                if logger: logger.info(f"Converted column '{column}' to {df[column].dtype} (format: {current_format if current_format else 'inferred'}).")
            elif target_type_str in ['int', 'integer', 'int64', 'int32', 'float', 'float64', 'float32', 'numeric']:
                df[column] = pd.to_numeric(df[column], errors=errors_handling)
                # Further cast if a specific int/float type is requested (e.g. int32) and pd.to_numeric result allows it
                if target_type_str not in ['numeric']: # 'numeric' is just a general conversion
                    if 'int' in target_type_str and df[column].isnull().any() and not target_type_str.lower().startswith('Int'): # Check for nullable integer type
                         if logger: logger.warning(f"Column '{column}' has NaNs. Cannot convert to standard int type '{target_type_str}'. Using float or Pandas nullable Int.")
                         # Pandas >= 0.24 allows nullable integers like 'Int64'
                         if target_type_str.lower() in ['int64', 'int32', 'int16', 'int8']:
                            df[column] = df[column].astype(target_type_str.capitalize()) # e.g. Int64
                         # else it will likely remain float or object if errors='coerce'
                    else:
                        df[column] = df[column].astype(target_type_str)
                if logger: logger.info(f"Converted column '{column}' to {df[column].dtype}.")
            elif target_type_str == 'category':
                df[column] = df[column].astype('category')
                if logger: logger.info(f"Converted column '{column}' to category.")
            elif target_type_str == 'bool':
                # Handle common string representations of bool before astype
                if df[column].dtype == 'object':
                    true_values = ['true', 'True', 'TRUE', 'yes', 'Yes', 'YES', '1', 1]
                    false_values = ['false', 'False', 'FALSE', 'no', 'No', 'NO', '0', 0]
                    df[column] = df[column].apply(lambda x: True if x in true_values else (False if x in false_values else np.nan))
                df[column] = df[column].astype(bool) # Note: This will convert NaN to True. Be cautious.
                                                    # For nullable boolean, use df[column].astype('boolean') (Pandas >= 1.0)
                if logger: logger.info(f"Converted column '{column}' to {df[column].dtype}.")
            else: # General string, object, etc.
                df[column] = df[column].astype(target_type_str)
                if logger: logger.info(f"Converted column '{column}' to {target_type_str}.")
        except Exception as e:
            if logger:
                logger.error(f"Error converting column '{column}' to '{target_type_str}': {e}")
            if errors_handling == 'raise':
                raise
            # If errors='coerce' or 'ignore', the error is handled by pandas, NaT/NaN inserted or original kept.
    return df