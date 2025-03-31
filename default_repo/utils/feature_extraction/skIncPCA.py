

from sklearn.decomposition import IncrementalPCA

class skincpca(IncrementalPCA):
    """ Incremental Principal component analysis (IPCA) builds a low-rank approximation 
    for the input data using an amount of memory which is independent of the number of 
    input data samples. It is still dependent on the input data features, but changing 
    the batch size allows for control of memory usage.
    """
    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X):
        return super().fit_transform(X)

    
