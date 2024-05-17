from river.preprocessing.random_projection import GaussianRandomProjector

class skincrp(GaussianRandomProjector):

    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X):
        return super().transform_one(X)
    


    

