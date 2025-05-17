from InteroperabilityEnabler.utils.data_formatter import data_to_dataframe

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def data_formatter(FILE_PATH, *args, **kwargs):
    """
    Read data from different file types (xls, xlsx, csv, json, jsonld) and
    convert them into a pandas DataFrame.

    Args:
        FILE_PATH (str): The path to the data file.

    Return:
        Pandas DataFrame.
    """
    df = data_to_dataframe(FILE_PATH)

    return df