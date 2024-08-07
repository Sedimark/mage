# Variables {}

from sklearn.preprocessing import PowerTransformer

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transformer(data, *args, **kwargs):
    pt = PowerTransformer()
    data.iloc[:, :] = pt.fit_transform(data)
    return data

@test
def test(output, *args) -> None:
    assert output is not None, 'The output is undefined'
