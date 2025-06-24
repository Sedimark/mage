import pandas as pd
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def handle_missing_values(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Handles missing values in a DataFrame.

    Configuration through kwargs:
        strategy (str): 'drop_rows', 'drop_cols', 'impute_mean', 'impute_median', 
                        'impute_mode', 'impute_constant'. Default: 'impute_mean' for numeric, 'impute_mode' for object.
        columns (list, optional): List of columns to apply the strategy. If None, applies to all columns 
                                  based on their dtype (numeric for mean/median, object/category for mode).
                                  For 'drop_rows' or 'drop_cols', 'columns' refers to a subset to consider for dropping.
        impute_value (any, optional): Constant value to use if strategy is 'impute_constant'. Default: 0.
        drop_threshold_col (float, optional): For 'drop_cols', percentage of missing values (0.0 to 1.0) 
                                            above which a column is dropped. Default: 0.5 (50%).
        logger: MageAI logger.
    """
    logger = kwargs.get('logger')
    df = data.copy() # Work on a copy

    strategy = kwargs.get('strategy', 'default_imputation') # 'default_imputation' will infer based on dtype
    columns_to_process = kwargs.get('columns', [])
    columns_to_process = list(columns_to_process) if isinstance(columns_to_process, str) else columns_to_process
    impute_value = kwargs.get('impute_value', 0)
    drop_threshold_col = kwargs.get('drop_threshold_col', 0.5)

    if logger:
        logger.info(f"Handling missing values with strategy: {strategy}")
        if columns_to_process:
            logger.info(f"Applying to columns: {columns_to_process}")
        if strategy == 'impute_constant':
            logger.info(f"Imputation constant value: {impute_value}")
        if strategy == 'drop_cols':
            logger.info(f"Column drop threshold (missing %): {drop_threshold_col*100}%")


    if strategy == 'drop_rows':
        subset_to_check = columns_to_process if columns_to_process else None
        original_rows = len(df)
        df.dropna(subset=subset_to_check, inplace=True)
        if logger:
            logger.info(f"Dropped {original_rows - len(df)} rows with missing values.")
    
    elif strategy == 'drop_cols':
        original_cols = df.shape[1]
        if columns_to_process: # Consider only specified columns for potential dropping
            cols_to_evaluate = [col for col in columns_to_process if col in df.columns]
        else: # Evaluate all columns
            cols_to_evaluate = df.columns

        cols_to_drop = []
        for col in cols_to_evaluate:
            missing_percentage = df[col].isnull().sum() / len(df)
            if missing_percentage > drop_threshold_col:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            if logger:
                logger.info(f"Dropped columns due to high missing values ({drop_threshold_col*100}%+): {cols_to_drop}")
        else:
             if logger:
                logger.info(f"No columns met the drop threshold of {drop_threshold_col*100}% missing values.")


    elif strategy in ['impute_mean', 'impute_median', 'impute_mode', 'impute_constant', 'default_imputation']:
        cols_for_imputation = columns_to_process if columns_to_process else df.columns
        
        for col in cols_for_imputation:
            if col not in df.columns:
                if logger: logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            if df[col].isnull().any(): # Only process if there are NaNs
                current_strategy = strategy
                
                if strategy == 'default_imputation':
                    if pd.api.types.is_numeric_dtype(df[col]):
                        current_strategy = 'impute_mean' # Default for numeric
                    else:
                        current_strategy = 'impute_mode'  # Default for object/categorical

                fill_value = None
                if current_strategy == 'impute_mean':
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].mean()
                    elif logger:
                        logger.warning(f"Cannot impute mean for non-numeric column '{col}'. Skipping imputation for this column.")
                        continue
                elif current_strategy == 'impute_median':
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_value = df[col].median()
                    elif logger:
                        logger.warning(f"Cannot impute median for non-numeric column '{col}'. Skipping imputation for this column.")
                        continue
                elif current_strategy == 'impute_mode':
                    modes = df[col].mode()
                    if not modes.empty:
                        fill_value = modes[0] # Take the first mode if multiple exist
                    else: # Column might be all NaNs or empty
                        if logger: logger.warning(f"No mode found for column '{col}' (it might be all NaNs or empty). Skipping imputation.")
                        continue
                elif current_strategy == 'impute_constant':
                    fill_value = impute_value
                
                if fill_value is not None:
                    df[col].fillna(fill_value, inplace=True)
                    if logger:
                        logger.info(f"Imputed missing values in column '{col}' with {current_strategy.split('_')[-1]}: {fill_value if current_strategy != 'impute_mode' else modes[0]}")
                elif current_strategy != 'default_imputation' and logger: # 'default_imputation' handles its own skipping messages
                     logger.warning(f"Could not determine fill value for column '{col}' with strategy '{current_strategy}'. Skipping.")
            elif logger:
                 logger.debug(f"No missing values in column '{col}'. Skipping imputation.")


    else:
        if logger:
            logger.error(f"Unknown missing value handling strategy: {strategy}")
        raise ValueError(f"Unknown missing value handling strategy: {strategy}")

    return df