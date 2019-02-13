import gc
import itertools

import numpy as np
from joblib import Parallel, delayed
from scipy.fftpack import fftn, ifftn
from scipy.sparse import coo_matrix

import rtk
from RegOptim.utils import import_func
from rtk.registration.LDDMM import derivative

joblib_path = '~/JOBLIB_TMP_FOLDER/'


def get_delta(A, a, b):
    delta = A - b * np.ones(A.shape)
    return delta / float(a)


def Lv(A, v):
    G = np.zeros(v.shape, dtype=np.complex128)
    for i in range(len(v)):
        G[i] = fftn(v[i])
    Lv = A * G
    Dv = np.zeros_like(G)

    for i in range(len(v)):
        Dv[i] = np.real(ifftn(Lv[i]))

    return np.real(Dv)


def get_der_dLv(A, v, a, b):
    # A = (a * delta + bE)
    # v = Km, m = Lv
    # L = A^(-2)
    G = np.zeros(v.shape, dtype=np.complex128)
    # delta = laplacian
    delta = get_delta(A, a, b)
    # turn v into furier space
    for i in range(len(v)):
        G[i] = fftn(v[i])
    # in fourier space we get a simple multiplication and get delta*v, delta^2*v(like Lv)
    delta_v = G * delta
    delta2_v = G * delta ** 2

    # get back from fourier space
    Dv = np.zeros_like(G)
    D2v = np.zeros_like(G)
    for i in range(len(v)):
        Dv[i] = np.real(ifftn(delta_v[i]))
        D2v[i] = np.real(ifftn(delta2_v[i]))

    # return derivative of Lv
    # dLv/da = ((a*delta + bE)^2 v)/da = 2(a*delta +bE)*delta*v = 2(a*delta^2 + b*delta)*v
    # dLv/db = 2(a*delta + bE) * E * v = 2(a*delta + bE)v
    del delta2_v, delta_v, G
    return np.real(2 * a * D2v + 2 * b * Dv), np.real(2 * a * Dv + 2 * b * v)


def derivative_path_length(A, vf, a, b):
    # count
    # dLv/da = 2(a*delta^2 + b*delta)*v - shape (ndim, image_shape)
    # dLv/db = 2(a*delta + bE) * E * v = 2(a*delta + bE)v - shape (ndim, image_shape)
    # shape of this dLv_da - (n_steps, ndim, image_shape)

    dLv_da, dLv_db = np.array([get_der_dLv(A=A, v=vf[i], a=a, b=b) for i in range(len(vf))]).T
    # axis (ndim, image_shape)
    axis = tuple(np.arange(vf.shape)[1:])
    # sum by space dimensions
    da, db = np.sum(dLv_da * vf, axis=axis), np.sum(dLv_db * vf, axis=axis)
    # by time dimensions (approx integral)
    da = 0.5 * (da[:-1] + da[1:])
    db = 0.5 * (db[:-1] + db[1:])

    return da, db


def path_length(regularizer, vf):
    momentum = regularizer(vf)
    return np.sum(momentum * vf)


def matrix_to_vec_indices(indices, shape):
    dot = np.array([np.prod(shape[::-1][:-i]) for i in range(1, len(shape))] + [1])
    indices = np.array(indices)
    return np.sum(indices * dot)


def vec_to_matrix_indices(I, shape):
    n = len(shape)

    dot = np.array([np.prod(shape[::-1][:-i]) for i in range(1, n)] + [1])
    indices = np.array([dot[0] + 1] + list(shape[1:]))

    return np.array([I / dot[i] % indices[i] for i in range(n)]).astype(int)


def create_arange(i, l, w=2):
    return np.arange(max(i - w, 0), min(i + w + 1, l), 1)


def neighbours_indices(shape, I, mode='vec', window=3):
    n = len(shape)
    w = int(window // 2)
    indices = vec_to_matrix_indices(I, shape)

    bounds = [create_arange(i=indices[i], l=shape[i], w=w) for i in range(n)]

    idx = list(itertools.product(*bounds))

    if mode == 'mat':
        return np.array(idx).astype(int)
    elif mode == 'vec':
        idx = np.array(idx)
        return np.array([matrix_to_vec_indices(idx[i], shape) for i in range(len(idx))]).astype(int)

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

    return moving_imgs, template_imgs


def full_derivative_by_v(moving, template, n_steps, vf, similarity, regularizer, inverse):
    deformation = deformation_grad(vf, n_steps, template.shape)

    moving_imgs, template_imgs = deformation_applied(moving, template, n_steps, deformation, inverse)

    if inverse:
        T = -1
    else:
        T = 0

    grad_v = np.squeeze([derivative(similarity=similarity, fixed=template_imgs[- T - 1],
                                  moving=moving_imgs[T], Dphi=deformation.backward_dets[- T - 1],
                                  vector_field=vf[T], regularizer=regularizer, learning_rate=1.) + \
                       Lv(regularizer.A, vf)
                       ])

    return grad_v, deformation.backward_dets[-T - 1], moving_imgs[T]


def intergal_of_action(vf, regularizer, n_steps):
    K = np.array([path_length(regularizer, vf[i]) for i in range(n_steps)])
    return np.sum(0.5 * (K[1:] + K[:-1]))


def loss_func(vf, moving, template, sigma, regularizer, n_steps, shape, inverse):
    deformation = deformation_grad(vf, n_steps, shape)
    deformed_moving, _ = deformation_applied(moving, template, n_steps, deformation, inverse)

    loss = np.sum((deformed_moving[-1] - template) ** 2) / float(sigma) + intergal_of_action(vf, regularizer, n_steps)

    return loss


def second_derivative_ii(vf, i, loss,  epsilon, moving, template, sigma, regularizer, n_steps, inverse):
    copy_vf = vf.copy()
    copy_vf[i] += epsilon
    loss_forward = loss_func(copy_vf, moving, template, sigma, regularizer, n_steps, template.shape, inverse)
    copy_vf = vf.copy()
    copy_vf[i] -= epsilon
    loss_backward = loss_func(copy_vf, moving, template, sigma, regularizer, n_steps, template.shape, inverse)

    return (loss_forward - 2 * loss + loss_backward) / epsilon ** 2


def second_derivative_ij(vf, i, j, epsilon, moving, template, sigma, regularizer, n_steps, inverse):
    # d^2f/dx/dy ~ (f(x-e, y -e) + f(x+e, y+e) - f(x+e, y-e) - f(x-e, y+e))/ (4*e^2) with precision O(h^2)

    vf_forward_ij = vf.copy()
    vf_forward_ij[i] += epsilon
    vf_forward_ij[j] += epsilon
    loss_forward_ij = loss_func(vf_forward_ij, moving, template, sigma, regularizer, n_steps,
                                template.shape, inverse)

    vf_forward_i = vf.copy()
    vf_forward_i[i] += epsilon
    vf_forward_i[j] -= epsilon
    loss_forward_i = loss_func(vf_forward_i, moving, template, sigma, regularizer, n_steps,
                               template.shape, inverse)

    vf_forward_j = vf.copy()
    vf_forward_j[i] -= epsilon
    vf_forward_j[j] += epsilon
    loss_forward_j = loss_func(vf_forward_j, moving, template, sigma, regularizer, n_steps,
                               template.shape, inverse)

    vf_backward_ij = vf.copy()
    vf_backward_ij[i] -= epsilon
    vf_backward_ij[j] -= epsilon
    loss_backward_ij = loss_func(vf_backward_ij, moving, template, sigma, regularizer, n_steps,
                                 template.shape, inverse)

    return ((loss_backward_ij + loss_forward_ij - loss_forward_i - loss_forward_j) / (4 * epsilon ** 2))


def second_derivative_by_loss(vf, i, j, loss, epsilon, moving, template, similarity, regularizer, n_steps, inverse):
    assert len(vf.shape) == len(i) == len(j), "Not correct indices"

    if i == j:
        return second_derivative_ii(vf=vf, i=i, loss=loss, epsilon=epsilon, moving=moving, template=template,
                                    sigma=similarity.variance, regularizer=regularizer,
                                    n_steps=n_steps, inverse=inverse)

    elif i != j:
        return second_derivative_ij(vf=vf, i=i, j=j, epsilon=epsilon, moving=moving, template=template,
                                    sigma=similarity.variance, regularizer=regularizer,
                                    n_steps=n_steps, inverse=inverse)
    else:
        raise TypeError('you should give correct indices')


def grad_of_derivative(vf, i, j, epsilon, moving, template, similarity, regularizer, n_steps, inverse, loss=None):
    vf_forward = vf.copy()
    vf_forward[j] += epsilon
    vf_backward = vf.copy()
    vf_backward[j] += epsilon

    grad_forward, _, _ = full_derivative_by_v(moving=moving, template=template, n_steps=n_steps, vf=vf_forward,
                                              similarity=similarity, regularizer=regularizer, inverse=inverse)
    grad_backward, _, _ = full_derivative_by_v(moving=moving, template=template, n_steps=n_steps, vf=vf_backward,
                                               similarity=similarity, regularizer=regularizer, inverse=inverse)



    return ((grad_forward - grad_backward) / (2 * epsilon))[i]


def one_line_sparse(vector, ndim, I, shape, window, loss, ax, params_grad, param_der):
    if params_grad['inverse']:
        T = -1
    else:
        T = 0
    cols = neighbours_indices(shape, I, 'vec', window)
    rows = np.repeat(I, len(cols))

    derivative_func = import_func(**param_der)

    data = np.array([
        derivative_func(i=(T, ax,) + tuple(vec_to_matrix_indices(I, shape)),
                        j=(T, ax,) + tuple(vec_to_matrix_indices(j, shape)),
                        vf=vector, loss=loss, **params_grad
                        )
        for j in cols
    ])

    mat_shape = (ndim * np.prod(shape), ndim * np.prod(shape))

    return coo_matrix((data, (rows + ax * int(np.prod(shape)), cols + ax * int(np.prod(shape)))), shape=mat_shape)


def sparse_dot_product_forward(vector, ndim, mat_shape, loss, window, params_grad, param_der):
    mat_len = int(np.prod(mat_shape))
    #
    # assert ndim * mat_len == len(vector), "not correct shape of vector"

    result = coo_matrix((ndim * mat_len, ndim * mat_len))

    for ax in range(ndim):
        for I in range(mat_len):
            loc_res = one_line_sparse(vector=vector, ndim=ndim, I=I, shape=mat_shape,
                                      window=window, ax=ax, loss=loss,
                                      params_grad=params_grad, param_der=param_der)
            result += loc_res

    gc.collect()

    return result


def sparse_dot_product_parallel(vector, ndim, mat_shape, loss, window, params_grad, param_der, n_jobs=5,
                                path_joblib='~/JOBLIB_TMP_FOLDER/'):
    mat_len = int(np.prod(mat_shape))

    # assert ndim * mat_len == len(vector), "not correct shape of vector"

    loc_res = Parallel(n_jobs=n_jobs, temp_folder=path_joblib)(
        delayed(one_line_sparse)(
            vector=vector, ndim=ndim, I=I, shape=mat_shape,
            loss=loss, window=window, ax=ax, params_grad=params_grad,
            param_der=param_der
        )
        for ax in range(ndim) for I in range(mat_len)
    )

    gc.collect()

    result = coo_matrix((ndim * mat_len,
                         ndim * mat_len))
    for one in loc_res:
        result += one

    return result


def sparse_dot_product(vector, ndim, mat_shape, loss, params_grad, param_der, window=2, mode='parallel',
                       n_jobs=5, path=joblib_path):
    if mode == 'forward' or n_jobs == 1:
        return sparse_dot_product_forward(vector=vector, ndim=ndim,
                                          mat_shape=mat_shape, loss=loss,
                                          window=window, params_grad=params_grad,
                                          param_der=param_der)
    elif mode == 'parallel':
        return sparse_dot_product_parallel(vector=vector, ndim=ndim,
                                           mat_shape=mat_shape, loss=loss,
                                           window=window, params_grad=params_grad,
                                           param_der=param_der,
                                           n_jobs=n_jobs, path_joblib=path)
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
