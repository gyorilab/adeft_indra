import cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset
from scipy.sparse import csr_matrix, issparse
from sklearn.tree._utils cimport safe_realloc
from sklearn.tree._tree cimport INT32_t, Node, SIZE_t, Tree


DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

TREE_LEAF = -1
cdef SIZE_t _TREE_LEAF = TREE_LEAF


cdef class KernelTree:
    cdef:
        Node *nodes
    def __cinit__(self, Tree tree):
        node_ndarray = tree._get_node_ndarray()
        capacity = node_ndarray.shape[0]
        self.nodes = <Node *> malloc(capacity * sizeof(Node))
        memcpy(self.nodes, <Node*> tree.nodes, capacity * sizeof(Node))

    def __dealloc__(self):
        free(self.nodes)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray _apply_dense(object X, KernelTree tree):
    """Finds the terminal region (=leaf node) for each sample in X."""
    # Check input
    if not isinstance(X, np.ndarray):
        raise ValueError("X should be in np.ndarray format, got %s"
                         % type(X))

    if X.dtype != DTYPE:
        raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

    # Extract input
    cdef const DTYPE_t[:, :] X_ndarray = X
    cdef SIZE_t n_samples = X.shape[0]

    # Initialize output
    cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
    cdef SIZE_t* out_ptr = <SIZE_t*> out.data

    # Initialize auxiliary data-structure
    cdef Node* node = NULL
    cdef SIZE_t i = 0

    with nogil:
        for i in range(n_samples):
            node = tree.nodes
            # While node not a leaf
            while node.left_child != _TREE_LEAF:
                # ... and node.right_child != _TREE_LEAF:
                if X_ndarray[i, node.feature] <= node.threshold:
                    node = &tree.nodes[node.left_child]
                else:
                    node = &tree.nodes[node.right_child]

            out_ptr[i] = <SIZE_t>(node - tree.nodes)  # node offset

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.ndarray _apply_sparse_csr(DTYPE_t *X_data,
                                         INT32_t *X_indices,
                                         INT32_t *X_indptr,
                                         SIZE_t n_samples,
                                         SIZE_t n_features,
                                         KernelTree tree):
    """Finds the terminal region (=leaf node) for each sample in sparse X.
    """
    # Initialize output
    cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                   dtype=np.intp)
    cdef SIZE_t* out_ptr = <SIZE_t*> out.data

    # Initialize auxiliary data-structure
    cdef DTYPE_t feature_value = 0.
    cdef Node* node = NULL
    cdef DTYPE_t* X_sample = NULL
    cdef SIZE_t i = 0
    cdef INT32_t k = 0

    # feature_to_sample as a data structure records the last seen sample
    # for each feature; functionally, it is an efficient way to identify
    # which features are nonzero in the present sample.
    cdef SIZE_t* feature_to_sample = NULL

    safe_realloc(&X_sample, n_features)
    safe_realloc(&feature_to_sample, n_features)
    with nogil:
        memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

        for i in range(n_samples):
            node = tree.nodes
            for k in range(X_indptr[i], X_indptr[i + 1]):
                feature_to_sample[X_indices[k]] = i
                X_sample[X_indices[k]] = X_data[k]

            # While node not a leaf
            while node.left_child != _TREE_LEAF:
                # ... and node.right_child != _TREE_LEAF:
                if feature_to_sample[node.feature] == i:
                    feature_value = X_sample[node.feature]

                else:
                    feature_value = 0.

                if feature_value <= node.threshold:
                    node = &tree.nodes[node.left_child]
                else:
                    node = &tree.nodes[node.right_child]

            out_ptr[i] = <SIZE_t>(node - tree.nodes)  # node offset

        # Free auxiliary arrays
        free(X_sample)
        free(feature_to_sample)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def tree_kernel(X, Y, trees):
    cdef int i, j
    cdef SIZE_t n_samples_X = X.shape[0]
    cdef SIZE_t n_features_X = X.shape[1]
    cdef SIZE_t n_samples_Y = Y.shape[0]
    cdef SIZE_t n_features_Y = Y.shape[1]
    cdef SIZE_t n_trees = len(trees)

    cdef np.ndarray[DTYPE_t, ndim=2] out = np.zeros((n_samples_X,
                                                     n_samples_Y),
                                                    dtype=DTYPE)

    cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
    cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
    cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr
    cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
    cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
    cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

    cdef np.ndarray[ndim=1, dtype=DTYPE_t] Y_data_ndarray = Y.data
    cdef np.ndarray[ndim=1, dtype=INT32_t] Y_indices_ndarray  = Y.indices
    cdef np.ndarray[ndim=1, dtype=INT32_t] Y_indptr_ndarray  = Y.indptr
    cdef DTYPE_t* Y_data = <DTYPE_t*>Y_data_ndarray.data
    cdef INT32_t* Y_indices = <INT32_t*>Y_indices_ndarray.data
    cdef INT32_t* Y_indptr = <INT32_t*>Y_indptr_ndarray.data

    cdef np.ndarray[ndim=1, dtype=SIZE_t] X_leaf_array
    cdef np.ndarray[ndim=1, dtype=SIZE_t] Y_leaf_array

    X_is_sparse = issparse(X)
    Y_is_sparse = issparse(Y)
    trees = [KernelTree(tree) for tree in trees]
    for tree in trees:
        if X_is_sparse:
            X_leaf_array = _apply_sparse_csr(X_data, X_indices, X_indptr,
                                             n_samples_X, n_features_X, tree)
        else:

            X_leaf_array = _apply_dense(X, tree)
        if Y_is_sparse:
            Y_leaf_array = _apply_sparse_csr(Y_data, Y_indices, Y_indptr,
                                             n_samples_Y, n_features_Y, tree)
        else:
            Y_leaf_array = _apply_dense(Y, tree)
        i = 0
        j = 0
        for i in range(len(X_leaf_array)):
            for j in range(len(Y_leaf_array)):
                out[i, j] += (X_leaf_array[i] == X_leaf_array[j])
    return out/len(trees)
