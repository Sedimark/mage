
"""Feature extraction or transformation.

The purpose of this module is to modify an existing set of features in order to be processed
by a machine learning algorithm. This includes transforming the input set of features to a new 
one by extracting new information from the input data.
This module could be used to data preprocessing or for the Energy efficiency module where we 
reduce the dimensionality/size of the input data.
"""

 
from .skFH import skfh
from .skPCA import skpca
from .skRP import skrp
# from .sUMAP import sumap
from .skTSNE import sktsne
from .skLDA import sklda 

__all__ = [
    "skfh",
    "skpca",
    "skrp",
    # "sumap",
    "sktsne",
    "sklda"
]