import numpy as np
from adeft_indra.anomaly_detection._tree_kernel import tree_kernel


def make_random_forest_kernel(trained_forest):
    trees = [est.tree_ for est in trained_forest.estimators_]
    def K(X, Y):
        n1, p1 = X.shape
        n2, p2 = Y.shape
        if p1 != p2:
            raise ValueError('Input matrices do not have compatible shape.')
        X_32, Y_32 = X.astype(np.float32), Y.astype(np.float32)
        return tree_kernel(X_32, Y_32, trees)
    return K
