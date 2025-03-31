# Variables {}

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transformer(data, *args, **kwargs):
    data.fillna(data.mean(), inplace=True)  # Filling missing values with mean
    return data

@test
def test(output, *args) -> None:
    assert output is not None, 'The output is undefined'
