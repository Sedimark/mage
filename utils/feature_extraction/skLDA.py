

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class sklda(LinearDiscriminantAnalysis):
    """ Linear Discriminant Analysis (LDA) is a dimensionality reduction technique that 
    maximizes class separability in data.  It does so by projecting the data 
    into a lower-dimensional space while preserving as much information as 
    possible about the class labels. 
    """
    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X, y):
        return super().fit_transform(X,y)
 

    
