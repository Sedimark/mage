import pandas as pd
import numpy as np
from faker import Faker # For generating fake data (generalization/perturbation)

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def anonymize_data(data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Applies various anonymization techniques to specified columns.

    Configuration via kwargs:
        anonymization_config (list of dicts): A list where each dict defines an anonymization task.
            Each dict should have:
            - 'columns' (list): Column(s) to apply this technique to.
            - 'technique' (str): 'generalization', 'suppression', 'perturbation', 
                                 'bucketing', 'pseudonymization'.
            - 'params' (dict, optional): Technique-specific parameters.
                - For 'generalization': {'type': 'age_ranges'/'zip_truncate', 'ranges': [[0,18,'0-18'], ...], 'truncate_digits': 2}
                - For 'suppression': No specific params, just suppresses to NaN or a placeholder.
                                     'placeholder' (str, optional): Value to use for suppression, e.g., '*'. Defaults to np.nan.
                - For 'perturbation': {'method': 'add_noise', 'noise_scale': 0.1} (for numeric)
                                    {'method': 'fake_names'} (for string names, uses Faker)
                - For 'bucketing': {'bins': 5 or list_of_bin_edges, 'labels': False or list_of_labels} (for numeric)
                - For 'pseudonymization': {'prefix': 'ID_'} (generates new IDs, simple counter for now)
                                        {'mapping_dict': {}} (optional, provide existing map, otherwise new map is built)
        logger: MageAI logger
    """
    logger = kwargs.get('logger')
    df = data.copy()
    fake = Faker() # Initialize Faker

    anonymization_config = kwargs.get('anonymization_config', [])
    anonymization_config = [{'columns': ['unitCode'], 'technique': 'pseudonymization'}]
    if not anonymization_config:
        if logger: logger.info("No anonymization configuration provided. Returning original data.")
        return df

    if logger: logger.info(f"Starting data anonymization with config: {anonymization_config}")

    pseudonym_maps = {} # To store mappings for pseudonymization if generated

    for config_item in anonymization_config:
        columns = config_item.get('columns', [])
        technique = config_item.get('technique')
        params = config_item.get('params', {})

        if not columns or not technique:
            if logger: logger.warning(f"Skipping invalid config item (missing columns or technique): {config_item}")
            continue

        for col in columns:
            if col not in df.columns:
                if logger: logger.warning(f"Column '{col}' for anonymization not found. Skipping.")
                continue
            
            if logger: logger.info(f"Applying '{technique}' to column '{col}' with params: {params}")

            try:
                if technique == 'suppression':
                    placeholder = params.get('placeholder', np.nan)
                    df[col] = placeholder
                    if logger: logger.debug(f"Suppressed column '{col}' with '{placeholder}'.")

                elif technique == 'generalization':
                    gen_type = params.get('type')
                    if gen_type == 'age_ranges' and 'ranges' in params:
                        # Example: params['ranges'] = [[0,18,'0-18'], [19,30,'19-30'], [31, np.inf, '31+']]
                        def generalize_age(age):
                            for r in params['ranges']:
                                if r[0] <= age <= r[1]:
                                    return r[2]
                            return np.nan # Or some default
                        df[col] = df[col].apply(generalize_age)
                    elif gen_type == 'zip_truncate' and 'truncate_digits' in params:
                        # Example: truncate_digits = 2 -> 90210 becomes 902**
                        digits = params['truncate_digits']
                        df[col] = df[col].astype(str).str[:digits] + '*' * (df[col].astype(str).str.len().max() - digits) # simplistic
                    else:
                        if logger: logger.warning(f"Unsupported generalization type or missing params for column '{col}'.")
                
                elif technique == 'perturbation':
                    method = params.get('method')
                    if method == 'add_noise' and pd.api.types.is_numeric_dtype(df[col]):
                        noise_scale = params.get('noise_scale', 0.1) # Percentage of std deviation
                        noise = np.random.normal(0, df[col].std() * noise_scale, size=len(df))
                        df[col] = df[col] + noise
                    elif method == 'fake_names' and df[col].dtype == 'object':
                        # This replaces all names, losing link if multiple people have same name
                        df[col] = df[col].apply(lambda x: fake.name() if pd.notnull(x) else np.nan)
                    else:
                        if logger: logger.warning(f"Unsupported perturbation method or non-numeric type for '{col}'.")

                elif technique == 'bucketing': # Primarily for numeric data
                    if pd.api.types.is_numeric_dtype(df[col]):
                        bins = params.get('bins', 5)
                        labels = params.get('labels', False) # If False, uses integer labels
                        df[col] = pd.cut(df[col], bins=bins, labels=labels, right=False, include_lowest=True)
                    else:
                        if logger: logger.warning(f"Bucketing is typically for numeric columns. Column '{col}' is not numeric.")
                
                elif technique == 'pseudonymization':
                    # Simple counter-based pseudonymization for this example.
                    # A more robust solution uses hashing or a secure lookup table.
                    prefix = params.get('prefix', f"{col}_PID_")
                    
                    # Use provided mapping or build a new one
                    if 'mapping_dict' in params and col in params['mapping_dict']:
                        current_map = params['mapping_dict'][col]
                        df[col] = df[col].map(current_map)
                        # Fill NaNs for values not in map, or create new pseudonyms
                        unmapped_mask = df[col].isnull() & data[col].notnull() # Original had value, but now it's NaN after map
                        if unmapped_mask.any():
                             if logger: logger.warning(f"Some values in '{col}' not found in provided map. They are now NaN or will get new pseudonyms if map is extended.")
                             # Option to extend map for unmapped values (not shown here for simplicity)
                    else:
                        if col not in pseudonym_maps:
                            unique_values = df[col].dropna().unique()
                            pseudonym_maps[col] = {val: f"{prefix}{i+1}" for i, val in enumerate(unique_values)}
                        
                        df[col] = df[col].map(pseudonym_maps[col])
                    if logger: logger.debug(f"Pseudonymized column '{col}'.")


                else:
                    if logger: logger.warning(f"Unknown anonymization technique '{technique}' for column '{col}'.")
            
            except Exception as e:
                if logger: logger.error(f"Error applying {technique} to {col}: {e}")
                # Decide if to raise or continue
                # raise e 

    if pseudonym_maps and logger:
        # For real scenarios, this map needs secure storage and management
        logger.info(f"Generated pseudonym maps (for re-identification if needed, handle securely!): {pseudonym_maps}")
        # Potentially output pseudonym_maps as a separate artifact of this block if needed downstream.
        # For now, it's just logged. You can store it in kwargs['block_output_metadata'] = {'pseudonym_maps': pseudonym_maps}

    return df