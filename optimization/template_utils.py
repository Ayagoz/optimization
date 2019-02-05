import numpy as np
import itertools
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed
import gc

import rtk
from rtk.registration.LDDMM import derivative

joblib_path = '~/JOBLIB_TMP_FOLDER/'


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


def deformation_grad(vf, n_steps, shape):
    deformation = rtk.DiffeomorphicDeformation(n_steps)
    deformation.set_shape(shape)

    v = 0.5 * (vf[:-1] + vf[1:])
    deformation.update_mappings(v)

    return deformation


def deformation_applied(moving, template, n_steps, deformation, inverse):
    if inverse:
        template_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=template), n_steps)
        moving_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=moving), n_steps)
    else:
        template_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=moving), n_steps)
        moving_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=template), n_steps)

    moving_imgs.apply_transforms(deformation.forward_mappings)
    template_imgs.apply_transforms(deformation.backward_mappings)

    return moving_imgs, template_imgs


def full_derivative_by_v(moving, template, n_steps, vf, similarity, regularizer, inverse):
    deformation = deformation_grad(vf, n_steps, template.shape)

    moving_imgs, template_imgs = deformation_applied(moving, template, n_steps, deformation, inverse)

    if inverse:
        T = -1
    else:
        T = 0

    grad_v = np.array([derivative(similarity=similarity, fixed=template_imgs[- T - 1],
                                  moving=moving_imgs[T], Dphi=deformation.backward_dets[- T - 1],
                                  vector_field=vf[T],
                                  regularizer=regularizer, learning_rate=1.)])

    return grad_v, deformation.backward_dets[-T - 1], moving_imgs[T]


def grad_of_derivative(moving, template, n_steps, vf, similarity, regularizer, inverse):
    pass


def intergal_of_action(vf, shape, a, b, n_steps):
    regularizer = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b)
    regularizer.set_shape(shape)
    A = regularizer.A
    K = np.array([path_length(A, vf[i], a, b) for i in range(n_steps + 1)])
    return 0.5 * (K[1:] + K[:-1])


def loss_func(vf, a, b, moving, template, sigma, n_steps, shape, inverse):
    deformation = deformation_grad(vf, n_steps, shape)
    deformed_moving, _ = deformation_applied(moving, template, n_steps, deformation, inverse)

    loss = np.sum((deformed_moving[-1] - template) ** 2) / sigma + intergal_of_action(vf, shape, a, b, n_steps)

    return loss


def second_derivative_ii(vf, i, epsilon, a, b, moving, template, sigma, n_steps, shape, inverse):
    loss = loss_func(vf, a, b, moving, template, sigma, n_steps, shape, inverse)
    copy_vf = vf.copy()
    copy_vf[i] += epsilon
    loss_forward = loss_func(copy_vf, a, b, moving, template, sigma, n_steps, shape, inverse)
    copy_vf = vf.copy()
    copy_vf[i] -= epsilon
    loss_backward = loss_func(copy_vf, a, b, moving, template, sigma, n_steps, shape, inverse)
    return (loss_forward - 2 * loss + loss_backward) / epsilon ** 2


def second_derivative_ij(vf, i, j, epsilon, a, b, moving, template, sigma, n_steps, shape, inverse):
    # d^2f/dx/dy ~ (f(x-e, y -e) + f(x+e, y+e) - f(x+e, y-e) - f(x-e, y+e))/ (4*e^2) with precision O(h^2)

    vf_forward_ij = vf.copy()
    vf_forward_ij[i] += epsilon
    vf_forward_ij[j] += epsilon
    loss_forward_ij = loss_func(vf_forward_ij, a, b, moving, template, sigma, n_steps, shape, inverse)

    vf_forward_i = vf.copy()
    vf_forward_i[i] += epsilon
    vf_forward_i[j] -= epsilon
    loss_forward_i = loss_func(vf_forward_i, a, b, moving, template, sigma, n_steps, shape, inverse)

    vf_forward_j = vf.copy()
    vf_forward_j[i] -= epsilon
    vf_forward_j[j] += epsilon
    loss_forward_j = loss_func(vf_forward_j, a, b, moving, template, sigma, n_steps, shape, inverse)

    vf_backward_ij = vf.copy()
    vf_backward_ij[i] -= epsilon
    vf_backward_ij[j] -= epsilon
    loss_backward_ij = loss_func(vf_backward_ij, a, b, moving, template, sigma, n_steps, shape, inverse)

    return (loss_backward_ij + loss_forward_ij - loss_forward_i - loss_forward_j) / (4 * epsilon ** 2)


def second_derivative_by_loss(vf, i, j, epsilon, a, b, moving, template, sigma, n_steps, shape, inverse):
    assert len(vf.shape) == len(i) == len(j), "Not correct indices"

    if i == j:
        return second_derivative_ii(vf, i, epsilon, a, b, moving, template, sigma, n_steps, shape, inverse)

    elif i != j:
        return second_derivative_ij(vf, i, j, epsilon, a, b, moving, template, sigma, n_steps, shape, inverse)
    else:
        raise TypeError('you should give correct indices')


def mixed_derivatives(vf, epsilon, similarity, regularizer, n_steps, inverse):
    # moving_imgs, template_imgs =

    pass


def one_line_sparse(vector, ndim, I, shape, window, ax):
    cols = neighbours_indices(shape, I, 'vec', window)
    rows = np.repeat(I, len(cols))
    data = vector[I] * vector[rows, 0]

    mat_shape = (ndim * len(vector), ndim * len(vector))
    return coo_matrix((data, (rows + ax * np.prod(shape), cols + ax * np.prod(shape))), shape=mat_shape)


def sparse_dot_product_forward(vector, ndim, mat_shape, window):
    mat_len = int(np.prod(mat_shape))

    assert ndim * mat_len == np.prod(vector.shape), "not correct shape of vector"

    result = coo_matrix((len(vector), len(vector)))

    for ax in range(ndim):
        for I in range(mat_len):
            loc_res = one_line_sparse(vector[ax * mat_len:(ax + 1) * mat_len], ndim, I, mat_shape,
                                      window, ax)
            result += loc_res

    gc.collect()

    return result


def sparse_dot_product_parallel(vector, ndim, mat_shape, window, n_jobs=5, path_joblib='~/JOBLIB_TMP_FOLDER/'):
    mat_len = int(np.prod(mat_shape))

    loc_res = Parallel(n_jobs=n_jobs, temp_folder=path_joblib)(
        delayed(one_line_sparse)(
            vector[ax * mat_len: (ax + 1) * mat_len],
            ndim, I, mat_shape, window, ax
        )
        for I in range(mat_len) for ax in range(ndim)
    )

    gc.collect()

    result = coo_matrix((len(vector), len(vector)))
    for one in loc_res:
        result += one

    return result


def sparse_dot_product(vector, ndim, mat_shape, window=2, mode='parallel', n_jobs=5, path=joblib_path):
    if mode == 'forward':
        return sparse_dot_product_forward(vector, ndim, mat_shape, window)
    elif mode == 'parallel':
        return sparse_dot_product_parallel(vector, ndim, mat_shape, window, n_jobs, path)
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
