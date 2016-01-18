# Practical Machine learning
# Bayesian learning - Naive Bayes example 
# Chapter 9

from datatypes import Dataset

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA

def univariate_feature_selection(ds, n):
    """
    Selects 'n' features in the dataset. Returns the Reduced Dataset
    n (int), ds (Dataset) -> Dataset
    """

    selector = SelectKBest(f_classif, n)
    selector.fit(ds.data, ds.target)
    features = selector.get_support(indices=True)
    return Dataset(selector.transform(ds.data), ds.target)

def lda(ds, n):
    '''
        Outputs the projection of the data in the best
        discriminant dimension.
        Maximum of 2 dimensions for our binary case (values of n greater than this will be ignored by sklearn)
    '''
    selector = LDA(n_components=n)
    selector.fit(ds.data, ds.target)
    new_data = selector.transform(ds.data)
    return Dataset(new_data, ds.target)
    
def pca(ds,n):
    '''
        Uses the PCA classifier to reduces the dimensionality by choosing the n lastest elements
        of the transform.
    '''
    selector = PCA()
    selector.fit(ds.data, ds.target)    
    new_data = selector.transform(ds.data)[:, :-n]
    return Dataset(new_data, ds.target)
