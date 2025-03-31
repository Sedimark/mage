# Variables {}

import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transformer(data, *args, **kwargs):
    data['outlier'] = (np.abs(data - data.mean()) > 3 * data.std()).any(axis=1)
    return data

@test
def test(output, *args) -> None:
    assert output is not None, 'The output is undefined'
