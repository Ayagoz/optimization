import numpy as np
import itertools
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed
import gc


def matrix_to_vec_indices(i, j, k, shape):
    l, n, m = shape
    return i * n * m + j * m + k


def vec_to_matrix_indices(I, shape):
    l, n, m = shape
    return I / (n * m), (I / m) % n, I % m


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
    l, n, m = shape
    i, j, k = vec_to_matrix_indices(I, shape)

    w = window / 2
    i_ = create_arange(i, l, w)
    j_ = create_arange(j, n, w)
    k_ = create_arange(k, m, w)

    idx = list(itertools.product(*list([i_, j_, k_])))
    if mode == 'mat':
        gc.collect()
        return idx
    elif mode == 'vec':
        idx = np.array(idx)

        gc.collect()

        return matrix_to_vec_indices(idx[:, 0], idx[:, 1], idx[:, 2], shape)

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
    result = Parallel(n_jobs=n_jobs, temp_folder=path_joblib)(
                    delayed(one_line_sparse)(vector, I,mat_shape, window)
                            for I in range(len(vector)))

    data, rows, cols = map(np.concatenate, zip(*result))

    rows = rows.astype(int)
    cols = cols.astype(int)

    gc.collect()

    return coo_matrix((data, (cols, rows)), shape=(len(vector), len(vector)))


def sparse_dot_product(vector, mat_shape, window=2, mode='parallel', n_jobs=5, path='~/JOBLIB_TMP_FOLDER/'):
    if mode == 'forward':
        return sparse_dot_product_forward(vector, mat_shape, window)

    elif mode == 'parallel':
        return sparse_dot_product_parallel(vector, mat_shape, window, n_jobs, path)

    else:
        raise TypeError('Do not support such type of calculating')