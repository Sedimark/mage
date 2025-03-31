

from sklearn.feature_extraction import FeatureHasher
import pandas as pd


class skfh(FeatureHasher):
    """Implements feature hashing, aka the hashing trick.

    This class turns sequences of symbolic feature names into matrices, using
    a hash function to compute the matrix column corresponding to a name. The 
    hash function employed is the signed 32-bit version of Murmurhash3.
    """
    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X):
        return super().fit_transform(X)

    
