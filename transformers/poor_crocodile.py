import pandas as pd

import pandas as pd
from sklearn.decomposition import PCA

from mage_ai.data_preparation.decorators import transformer


@transformer
def apply_pca(df, *args, **kwargs):
    pca = PCA(n_components=2)  # You can adjust the number of components as needed
    return pd.DataFrame(pca.fit_transform(df))
