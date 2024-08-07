import pandas as pd


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
    from mage_ai.data_preparation.decorators import test

@transformer
def normalize_data(data, *args, **kwargs):
    return (data - data.mean()) / data.std()

@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
