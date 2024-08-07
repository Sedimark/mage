import pandas as pd

from sklearn.decomposition import PCA
from pandas import DataFrame

from mage_ai.data_preparation.decorators import transformer


@transformer
def apply_pca(df: DataFrame, *args, **kwargs) -> DataFrame:
    """
    Apply Principal Component Analysis (PCA) to a pandas DataFrame.
    """
    pca = PCA(n_components=2)  # Change the number of components as needed
    df_pca = pca.fit_transform(df)
    return pd.DataFrame(data=df_pca, columns=['PC1', 'PC2'])  # Change column names as needed
