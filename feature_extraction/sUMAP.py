

import umap

class sumap(umap.UMAP):
    """Uniform Manifold Approximation and Projection (UMAP) is a dimensionality reduction technique used
    in machine learning and data analysis. It is particularly well-suited for visualizing high-dimensional 
    data in a lower-dimensional space while preserving the underlying structure of the data. UMAP is 
    similar in purpose to t-SNE (t-Distributed Stochastic Neighbor Embedding) but offers several advantages,
    including faster computation and better preservation of global structure.
    """
    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X):
        return super().fit_transform(X)
 

    
