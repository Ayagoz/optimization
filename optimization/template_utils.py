import numpy as np
import itertools
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed
import gc

joblib_path='~/JOBLIB_TMP_FOLDER/'

def matrix_to_vec_indices(indices, shape):
    dot = np.array([np.prod(shape[::-1][:-i]) for i in range(1, len(shape))] + [1])
    indices = np.array(indices)
    return np.sum(indices * dot)


def vec_to_matrix_indices(I, shape):
    n = len(shape)

    dot = np.array([np.prod(shape[::-1][:-i]) for i in range(1, n)] + [1])
    indices = np.array([dot[0] + 1] + list(shape[1:]))

    return np.array([I / dot[i] % indices[i] for i in range(n)])


def create_arange(i, l, w=2):
    if i == 0:
        i_ = np.arange(i, i + w + 1)

    elif i == l:
        i_ = np.arange(i - w, l)

    else:
        if i + w >= l:
            i_ = np.arange(i - w, l)

        else:
            i_ = np.arange(i - w, i + w + 1)

    return i_


def neighbours_indices(shape, I, mode='vec', window=3):
    n = len(shape)
    w = window / 2

    indices = vec_to_matrix_indices(I, shape)

    bounds = [create_arange(i=indices[i], l=shape[i], w=w) for i in range(n)]

    idx = list(itertools.product(*bounds))

    if mode == 'mat':
        return idx
    elif mode == 'vec':
        idx = np.array(idx)
        return np.array([matrix_to_vec_indices(idx[i], shape) for i in range(len(idx))])

    else:
        raise TypeError('Not correct mode, support just `vec` and `mat`')


def one_line_sparse(vector, I, shape, window):
    rows = neighbours_indices(shape, I, 'vec', window)
    cols = np.repeat(I, len(rows))

    data = vector[I] * vector[rows, 0]

    return data, rows, cols


def sparse_dot_product_forward(vector, mat_shape, window):
    data = []
    rows, cols = [], []
    for I in range(len(vector)):
        d, r, c = one_line_sparse(vector, I, mat_shape, window)

        data = np.concatenate([data, d])
        rows = np.concatenate([rows, r])
        cols = np.concatenate([cols, c])

    rows = rows.astype(int)
    cols = cols.astype(int)

    gc.collect()

    return coo_matrix((data, (cols, rows)), shape=(len(vector), len(vector)))


def sparse_dot_product_parallel(vector, mat_shape, window, n_jobs=5, path_joblib='~/JOBLIB_TMP_FOLDER/'):
    result = Parallel(n_jobs=n_jobs, temp_folder=path_joblib)(delayed(one_line_sparse)(vector, I,
                                                                                       mat_shape,
                                                                                       window)
                                                                                       for I in range(len(vector)))

    data, rows, cols = map(np.concatenate, zip(*result))

    rows = rows.astype(int)
    cols = cols.astype(int)

    gc.collect()

    return coo_matrix((data, (cols, rows)), shape=(len(vector), len(vector)))


def sparse_dot_product(vector, mat_shape, window=2, mode='parallel', n_jobs=5, path=joblib_path):
    if mode == 'forward':
        return sparse_dot_product_forward(vector, mat_shape, window)
    elif mode == 'parallel':
        return sparse_dot_product_parallel(vector, mat_shape, window, n_jobs, path)
    else:
        raise TypeError('Do not support such type of calculating')

def double_dev_J_v(vec):
    shape = int(np.prod(vec.shape[1:]))

    data = []
    rows, cols = [], []

    for i in range(vec.shape[0]):
        data += list(vec[i].reshape(-1))
        rows += list(np.arange(shape))
        cols += list(np.arange(shape) + i * shape)

    return coo_matrix((data, (cols, rows)), shape=(np.prod(vec.shape), shape))