

from sklearn.manifold import TSNE

class sktsne(TSNE):
    """t-SNE, or t-distributed Stochastic Neighbor Embedding, is a popular and powerful dimensionality reduction technique 
    commonly used in machine learning and data visualization. It was introduced by Laurens van der Maaten and Geoffrey 
    Hinton in 2008. t-SNE is particularly well-suited for visualizing high-dimensional data in a lower-dimensional space 
    (of 3 or 2 dimensions) while preserving the structure and relationships between data points. 
    """
    def __init__(self, **params):
        super().__init__(**params)


    def fit_transform(self, X):
        return super().fit_transform(X)

    
