import pandas as pd

from sklearn.decomposition import PCA
from mage_ai.data_preparation.decorators import transformer


@transformer
def apply_pca(df, *args, **kwargs):
    """
    Applies Principal Component Analysis (PCA) on a pandas DataFrame.
    """
    pca = PCA(n_components=2)  # Change the number of components as needed
    df_pca = pd.DataFrame(data=pca.fit_transform(df), columns=[f'PC{i+1}' for i in range(2)])  # Change column names as needed
    return df_pca
