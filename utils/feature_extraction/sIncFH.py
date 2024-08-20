from river.preprocessing import FeatureHasher

class skincfh(FeatureHasher):

    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X):
        return super().transform_one(X)
    


    

