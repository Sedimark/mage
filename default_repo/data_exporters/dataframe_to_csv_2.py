import pandas as pd
import os

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_to_csv(data, *args, **kwargs):
    """
    Exports a pandas DataFrame or Series to a CSV file.

    Args:
        data: The pandas DataFrame or Series to export.
        *args: Additional arguments (not used in this block).
        **kwargs: Keyword arguments passed from the MageAI UI or YAML.
            Expected keyword arguments:
            - output_path (str): The full path including filename for the output CSV file.
                                 e.g., '/path/to/your/output_directory/exported_data.csv'
                                 or just 'exported_data.csv' if you want it in the project's root 
                                 or a predefined data directory.
            - index (bool, optional): Write row names (index). Defaults to True if data is Series,
                                      False if data is DataFrame, unless 'header' is False for Series.
            - header (bool or list of str, optional): Write out the column names. If a list of strings
                                                    is given it is assumed to be aliases for the column names.
                                                    Defaults to True. If data is Series and header is False,
                                                    index defaults to False.
            - sep (str, optional): Delimiter to use. Defaults to ','.
            - logger: The logger object from MageAI.

    Output:
        Writes a CSV file to the specified `output_path`.
    """
    logger = kwargs.get('logger')

    # --- Configuration ---
    # Default output path if not provided - consider your project structure
    # For robust usage, output_path should ideally be explicitly provided.
    default_filename = 'default_repo/exported_data.csv'
    output_path = kwargs.get('output_path', default_filename)
    
    # pandas to_csv specific arguments
    # For Series: if index is not specified and header is False, index defaults to False.
    # For DataFrame: index defaults to False usually, but let's be explicit.
    if isinstance(data, pd.Series):
        default_index = True
        if kwargs.get('header') is False: # if header is explicitly False for Series
            default_index = False 
        index_arg = kwargs.get('index', default_index)
    else: # DataFrame or other
        index_arg = kwargs.get('index', False) # Common default for DataFrames is not to write index

    header_arg = kwargs.get('header', True)
    sep_arg = kwargs.get('sep', ',')
    # Add more to_csv arguments as needed (e.g., encoding, mode, quoting)

    if data is None:
        if logger:
            logger.warning("Input data is None. Nothing to export.")
        return

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        if logger:
            logger.error(f"Input data is not a pandas DataFrame or Series. Got {type(data)}. Cannot export to CSV.")
        raise TypeError("Input data must be a pandas DataFrame or Series to export to CSV.")

    if isinstance(data, pd.DataFrame) and data.empty:
        if logger:
            logger.info(f"Input DataFrame is empty. Exporting an empty CSV file to {output_path} (with headers if enabled).")
    elif isinstance(data, pd.Series) and data.empty:
        if logger:
            logger.info(f"Input Series is empty. Exporting an empty CSV file to {output_path} (with header/index if enabled).")
    elif logger:
        if isinstance(data, pd.DataFrame):
            logger.info(f"Exporting DataFrame with shape {data.shape} to CSV: {output_path}")
        else: # pd.Series
            logger.info(f"Exporting Series with length {len(data)} to CSV: {output_path}")

    try:
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            if logger:
                logger.info(f"Created output directory: {output_dir}")

        # Export to CSV
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=index_arg, header=header_arg, sep=sep_arg)
        elif isinstance(data, pd.Series):
            # For a Series, to_csv writes it as a single column.
            # If you want it as a single row (like your profiler output), you might convert to DataFrame first.
            # Assuming standard Series to CSV (single column or single row if header=False, index=True)
            data.to_csv(output_path, index=index_arg, header=header_arg, sep=sep_arg)
        
        if logger:
            logger.info(f"Successfully exported data to {output_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to export data to CSV at {output_path}. Error: {e}")
        raise
