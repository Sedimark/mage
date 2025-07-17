import os
import pandas as pd
from default_repo.utils.generic_pipeline_enabler.default import load_parent_data

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """

    df = load_parent_data(kwargs)

    load_csv = kwargs.get('load_csv', False)
    csv_file = kwargs.get('csv_file', None)

    if csv_file is None or not load_csv:
        return df
    
    csv_dir = os.getenv('CSV_DIR', None)

    # strategies ['concat', 'merge', 'join', 'replace']
    strategy = kwargs.get('join_csv_strategy', 'concat')

    if csv_dir is None:
        print("CSV file reference found but no CSV_DIR environment variable was configured.")

    csv_path = f'{csv_dir}/{csv_file}'

    if os.path.exists(csv_path):
        csv_df = pd.read_csv(csv_path)

        if df is None:
            df = csv_df
            return df

        if strategy == 'concat':
            df = pd.concat([df, csv_df], ignore_index=True)
        elif strategy == 'merge':
            # Merge requires common columns - using outer join on all common columns
            common_cols = list(set(df.columns) & set(csv_df.columns))
            if common_cols:
                df = pd.merge(df, csv_df, on=common_cols, how='outer')
            else:
                # If no common columns, fall back to concat
                df = pd.concat([df, csv_df], ignore_index=True)
        elif strategy == 'join':
            # Join on index
            df = df.join(csv_df, how='outer', rsuffix='_csv')
        elif strategy == 'replace':
            df = csv_df
        else:
            df = pd.concat([df, csv_df], ignore_index=True)

    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
