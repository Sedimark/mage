
from river.feature_extraction import PolynomialExtender


class skincpoly(PolynomialExtender):
    """ Polynomial feature extender generates features consisting of all polynomial 
    combinations of the features with degree less than or equal to the specified degree.
    """
    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X):
        return super().transform_one(X)
 