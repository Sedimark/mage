from InteroperabilityEnabler.utils.data_formatter import data_to_dataframe

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform(FILE_PATH, *args, **kwargs):
    """
    Read data from different file types (xls, xlsx, csv, json, jsonld) and
    convert them into a pandas DataFrame.

    Args:
        file_path (str): The path to the data file.

    Return:
        Pandas DataFrame.
    """
    df = data_to_dataframe(FILE_PATH)

    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
