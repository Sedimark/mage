import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from mage_ai.data_preparation.decorators import transformer
from mage_ai.data_preparation.decorators import test


if 'transformer' not in globals():
@transformer
def transformer(data, *args, **kwargs):
    n_components = 3 if kwargs.get('n_components') is None else kwargs.get('n_components')
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data.select_dtypes(include=[np.number]))
    data = pd.DataFrame(data_pca, columns=[f'PCA{i+1}' for i in range(data_pca.shape[1])])
    return data

@test
def test(output, *args) -> None:
    assert output is not None, 'The output is undefined'
