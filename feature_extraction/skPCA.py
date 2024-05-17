

from sklearn.decomposition import PCA

class skpca(PCA):
    """ Principal component analysis (PCA).
    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space. The input data is centered
    but not scaled for each feature before applying the SVD.
    This is a simple implementation version of PCA.
    """
    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X):
        return super().fit_transform(X)

    
