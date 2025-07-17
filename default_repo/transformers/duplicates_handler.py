import pandas as pd
import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def remove_duplicates(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame.

    Configuration through kwargs:
        subset (list, optional): List of column names to consider for identifying duplicates. 
                                 If None, all columns are used.
        keep (str, optional): Which duplicate to keep ('first', 'last', False for dropping all). 
                              Default: 'first'.
        logger: MageAI logger.
    """
    logger = kwargs.get('logger')
    df = data.copy()

    subset = kwargs.get('subset')
    keep = kwargs.get('keep', 'first')

    if logger:
        logger.info(f"Removing duplicate rows. Keep: '{keep}'. Subset: {subset if subset else 'all columns'}.")

    original_rows = len(df)
    df.drop_duplicates(subset=subset, keep=keep, inplace=True)
    rows_dropped = original_rows - len(df)

    if logger:
        if rows_dropped > 0:
            logger.info(f"Removed {rows_dropped} duplicate rows.")
        else:
            logger.info("No duplicate rows found based on the criteria.")
            
    return df