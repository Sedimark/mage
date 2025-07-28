import pandas as pd
import numpy as np
from default_repo.utils.generic_pipeline_enabler.default import export_parent_data

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def manage_columns(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Renames specified columns and/or drops specified columns.

    Configuration through kwargs:
        rename_map (dict, optional): Dictionary for renaming columns. 
                                   e.g., {'old_name1': 'new_name1', 'old_name2': 'new_name2'}
        drop_columns (list, optional): List of column names to drop.
        logger: MageAI logger.
    """
    logger = kwargs.get('logger')
    df = data.copy()

    rename_map = kwargs.get('rename_map', {})
    drop_columns_list = kwargs.get('drop_columns', [])

    # Rename columns
    if rename_map:
        # Check if all old names in rename_map exist in df columns
        valid_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        if len(valid_rename_map) < len(rename_map) and logger:
            missing_keys = set(rename_map.keys()) - set(df.columns)
            logger.warning(f"Columns to rename not found in DataFrame: {missing_keys}. Skipping them.")
        
        if valid_rename_map:
            df.rename(columns=valid_rename_map, inplace=True)
            if logger:
                logger.info(f"Renamed columns: {valid_rename_map}")
        elif logger:
             logger.info("No columns renamed (either rename_map was empty or no specified old names found).")


    # Drop columns
    if drop_columns_list:
        actual_cols_to_drop = [col for col in drop_columns_list if col in df.columns]
        if len(actual_cols_to_drop) < len(drop_columns_list) and logger:
            missing_to_drop = set(drop_columns_list) - set(df.columns)
            logger.warning(f"Columns to drop not found in DataFrame: {missing_to_drop}. Skipping them.")

        if actual_cols_to_drop:
            df.drop(columns=actual_cols_to_drop, inplace=True)
            if logger:
                logger.info(f"Dropped columns: {actual_cols_to_drop}")
        elif logger:
            logger.info("No columns dropped (either drop_columns list was empty or no specified columns found).")
            
    if not rename_map and not drop_columns_list and logger:
        logger.info("No column renaming or dropping specified.")

    return export_parent_data(df)