import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Make some operation on the data argument

    column_name=kwargs.get('column_name')
    print(f"column name is {column_name}")
    time_column=kwargs.get('time_column')

    if column_name is None:
        column_name = ['temperature', 'windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:10MTR',
        'windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:80MTR',
        'windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:120MTR',
        'windSpeed # urn:ngsi-ld:Dataset:Open-Meteo:180MTR']
    else:
        column_name = list(column_name) # convert into list containing multiple columns
    print(f"column name is {column_name}")
        
    if time_column is None:
        time_column="observedAt"
        print(f"time column is {time_column}")
    
    data['Time'] = pd.to_datetime(data[time_column])
    data['UnixTime'] = data['Time'].astype(int) // 10**9

    selected_columns = ['UnixTime'] + column_name
    print(selected_columns)  

    return data[selected_columns]
