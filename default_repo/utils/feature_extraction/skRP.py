

from sklearn.random_projection import GaussianRandomProjection



class skrp(GaussianRandomProjection):
    """  Random Projection is a dimensionality reduction technique that simplifies high-dimensional data by projecting it onto a
    lower-dimensional space using a random transformation. It retains essential information while reducing computational 
    complexity, making it useful for processing large datasets efficiently.
    """
    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X):
        return super().fit_transform(X)
 

    
