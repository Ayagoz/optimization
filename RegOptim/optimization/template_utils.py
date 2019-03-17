import gc
import itertools
import copy

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse.linalg import inv
from scipy.fftpack import fftn, ifftn
from scipy.sparse import coo_matrix

import rtk
from RegOptim.utils import import_func
from rtk import Deformation
import matplotlib.pyplot as plt

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
    momentum = Lv(regularizer.A ** 2, vf)
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


def intergal_of_action(vf, regularizer, n_steps):
    K = np.array([path_length(regularizer, vf[i]) for i in range(n_steps + 1)])
    return (K[0] / 2. + np.sum(K[1:-1]) + K[-1] / 2.) / float(n_steps)


def deformation_grad(vf, n_steps, shape):
    deformation = rtk.DiffeomorphicDeformation(n_steps)
    deformation.set_shape(shape)

    v = 0.5 * (vf[:-1] + vf[1:])
    deformation.update_mappings(v)

    return deformation


def gradient(reg, deformation, vf=None):
    if vf is not None:
        deformation = deformation_grad(vf=vf, n_steps=reg.n_steps, shape=reg.moving.shape)
    moving = copy.deepcopy(reg.moving)
    warp_moving = np.copy(moving.apply_transform(Deformation(grid=deformation.forward_mappings[-1]), order=3).data)

    grad = reg.regularizer(reg.similarity.derivative(reg.fixed.data, warp_moving) * deformation.forward_dets[-1])
    return grad


def grad_of_derivative_ii(vf, i, epsilon, reg, deformation):
    vf1 = np.copy(vf)
    vf2 = np.copy(vf)
    vf3 = np.copy(vf)
    vf4 = np.copy(vf)

    vf1[i] += epsilon
    vf2[i] -= epsilon
    vf3[i] += 2 * epsilon
    vf4[i] -= 2 * epsilon

    def1 = copy.deepcopy(deformation)
    def2 = copy.deepcopy(deformation)
    def3 = copy.deepcopy(deformation)
    def4 = copy.deepcopy(deformation)

    def1.update_mappings(0.5 * (vf1[1:] + vf1[:-1]))
    def2.update_mappings(0.5 * (vf2[1:] + vf2[:-1]))
    def3.update_mappings(0.5 * (vf3[1:] + vf3[:-1]))
    def4.update_mappings(0.5 * (vf4[1:] + vf4[:-1]))

    grad_f1 = gradient(reg=copy.deepcopy(reg), deformation=def1)
    grad_b1 = gradient(reg=copy.deepcopy(reg), deformation=def2)
    grad_f2 = gradient(reg=copy.deepcopy(reg), deformation=def3)
    grad_b2 = gradient(reg=copy.deepcopy(reg), deformation=def4)
    ind = i[1:]
    res = -grad_f2[ind] + 8 * grad_f1[ind] - 8 * grad_b1[ind] + grad_b2[ind]
    print('ii in grad der ', res)
    return res / (12 * epsilon)


def grad_of_derivative_ij(vf, i, j, epsilon, reg, deformation):
    vf1 = np.copy(vf)
    vf2 = np.copy(vf)
    vf3 = np.copy(vf)
    vf4 = np.copy(vf)

    vf1[j] += epsilon
    vf2[j] -= epsilon
    vf3[j] += 2 * epsilon
    vf4[j] -= 2 * epsilon

    def1 = copy.deepcopy(deformation)
    def2 = copy.deepcopy(deformation)
    def3 = copy.deepcopy(deformation)
    def4 = copy.deepcopy(deformation)

    def1.update_mappings(0.5 * (vf1[1:] + vf1[:-1]))
    def2.update_mappings(0.5 * (vf2[1:] + vf2[:-1]))
    def3.update_mappings(0.5 * (vf3[1:] + vf3[:-1]))
    def4.update_mappings(0.5 * (vf4[1:] + vf4[:-1]))

    grad_f1 = gradient(reg=copy.deepcopy(reg), deformation=def1)
    grad_b1 = gradient(reg=copy.deepcopy(reg), deformation=def2)
    grad_f2 = gradient(reg=copy.deepcopy(reg), deformation=def3)
    grad_b2 = gradient(reg=copy.deepcopy(reg), deformation=def4)
    ind = i[1:]
    res = -grad_f2[ind] + 8 * grad_f1[ind] - 8 * grad_b1[ind] + grad_b2[ind]
    print('ii in grad der ', res)

    return res / (12 * epsilon)


def grad_of_derivative(vf, i, j, epsilon, reg, deformation, loss=None):
    assert len(vf.shape) == len(i) == len(j), "Not correct indices"

    if i == j:
        return grad_of_derivative_ii(vf=np.copy(vf), i=i, epsilon=epsilon, reg=copy.deepcopy(reg),
                                     deformation=copy.deepcopy(deformation))

    elif i != j:
        return grad_of_derivative_ij(vf=np.copy(vf), i=i, j=j, epsilon=epsilon,
                                     reg=copy.deepcopy(reg), deformation=copy.deepcopy(deformation))
    else:
        raise TypeError('you should give correct indices')


def loss_func(reg, deformation, vf=None, show=False):
    if vf is not None:
        deformation = deformation_grad(vf=vf, n_steps=reg.n_steps, shape=reg.moving.shape)

    moving = copy.deepcopy(reg.moving)
    warp_moving = np.copy(moving.apply_transform(Deformation(grid=deformation.forward_mappings[-1]), order=3).data)

    if show:
        f, ax = plt.subplots(2, figsize=(10, 10))
        ax[0].imshow(moving)
        ax[0].set_title('moving')
        ax[1].imshow(warp_moving.data)
        ax[1].set_title('in loss warped moving')
        plt.axis('off')
        plt.show()

    loss = np.sum(np.square(warp_moving - reg.fixed.data)) / float(reg.similarity.variance)
    # + intergal_of_action(vf, regularizer, n_steps)
    gc.collect()
    return loss


def second_derivative_ii(vf, i, loss, epsilon, reg, deformation):
    vf1 = np.copy(vf)
    vf2 = np.copy(vf)
    vf3 = np.copy(vf)
    vf4 = np.copy(vf)

    vf1[i] += epsilon
    vf2[i] -= epsilon
    vf3[i] += 2 * epsilon
    vf4[i] -= 2 * epsilon

    def1 = copy.deepcopy(deformation)
    def2 = copy.deepcopy(deformation)
    def3 = copy.deepcopy(deformation)
    def4 = copy.deepcopy(deformation)

    def1.update_mappings(0.5 * (vf1[1:] + vf1[:-1]))
    def2.update_mappings(0.5 * (vf2[1:] + vf2[:-1]))
    def3.update_mappings(0.5 * (vf3[1:] + vf3[:-1]))
    def4.update_mappings(0.5 * (vf4[1:] + vf4[:-1]))

    # print(np.allclose(def1.backward_dets, def2.backward_dets))
    # print(np.allclose(def1.backward_mappings, def2.backward_mappings))
    # print(np.allclose(def1.forward_dets, def2.forward_dets))
    # print(np.allclose(def1.forward_mappings, def2.forward_mappings))
    # print(np.abs(def1.backward_dets - def2.backward_dets).sum())
    # print(np.abs(def1.backward_mappings - def2.backward_mappings).sum())
    # print(np.abs(def1.forward_dets - def2.forward_dets).sum())
    # print(np.abs(def1.forward_mappings - def2.forward_mappings).sum())

    loss_forward = loss_func(reg=copy.deepcopy(reg), deformation=def1)
    loss_backward = loss_func(reg=copy.deepcopy(reg), deformation=def2)
    loss_forward2 = loss_func(reg=copy.deepcopy(reg), deformation=def3)
    loss_backward2 = loss_func(reg=copy.deepcopy(reg), deformation=def4)

    res = -loss_forward2 + 16 * loss_forward - 30 * loss + 16 * loss_backward - loss_backward2

    # print('ii forward2 {}, forward {} \n loss {} backward {} backward2 {}, \n res {}'.format(loss_forward2,
    #                                                                                          loss_forward, loss,
    #                                                                                          loss_backward,
    #                                                                                          loss_backward2,
    #                                                                                          res))
    gc.collect()
    return res / float(12 * epsilon ** 2)


def second_derivative_ij(vf, i, j, loss, epsilon, reg, deformation):
    # d^2f/dx/dy ~ (f(x-e, y -e) + f(x+e, y+e) - f(x+e, y-e) - f(x-e, y+e))/ (4*e^2) with precision O(h^2)
    # f(x+1, y + 1)
    vf1 = np.copy(vf)
    vf2 = np.copy(vf)
    vf3 = np.copy(vf)
    vf4 = np.copy(vf)
    # f(x+1, y+1)
    vf1[i] += epsilon
    vf1[j] += epsilon
    # f(x+1, y - 1)
    vf2[i] += epsilon
    vf2[j] -= epsilon
    # f(x-1, y + 1)
    vf3[i] -= epsilon
    vf3[j] += epsilon
    # f(x-1, y -1)
    vf4[i] -= epsilon
    vf4[j] -= epsilon

    def1 = copy.deepcopy(deformation)
    def2 = copy.deepcopy(deformation)
    def3 = copy.deepcopy(deformation)
    def4 = copy.deepcopy(deformation)

    def1.update_mappings(0.5 * (vf1[1:] + vf1[:-1]))
    def2.update_mappings(0.5 * (vf2[1:] + vf2[:-1]))
    def3.update_mappings(0.5 * (vf3[1:] + vf3[:-1]))
    def4.update_mappings(0.5 * (vf4[1:] + vf4[:-1]))

    loss_f11 = loss_func(reg=copy.deepcopy(reg), deformation=def1)
    loss_f1_b1 = loss_func(reg=copy.deepcopy(reg), deformation=def2)
    loss_b1_f1 = loss_func(reg=copy.deepcopy(reg), deformation=def3)
    loss_b11 = loss_func(reg=copy.deepcopy(reg), deformation=def4)

    a = (loss_b11 + loss_f11 - loss_f1_b1 - loss_b1_f1)
    del vf1, vf2, vf3, vf4, def1, def2, def3, def4, loss_f11, loss_f1_b1, loss_b1_f1, loss_b11

    vf1 = np.copy(vf)
    vf2 = np.copy(vf)
    vf3 = np.copy(vf)
    vf4 = np.copy(vf)
    # f(x+1, y-2)
    vf1[i] += epsilon
    vf1[j] -= 2 * epsilon
    # f(x+2, y -1)
    vf2[i] += 2 * epsilon
    vf2[j] -= epsilon
    # f(x-2, y+1)
    vf3[i] -= 2 * epsilon
    vf3[j] += epsilon
    # f(x-1, y+2)
    vf4[i] -= epsilon
    vf4[j] += 2 * epsilon

    def1 = copy.deepcopy(deformation)
    def2 = copy.deepcopy(deformation)
    def3 = copy.deepcopy(deformation)
    def4 = copy.deepcopy(deformation)

    def1.update_mappings(0.5 * (vf1[1:] + vf1[:-1]))
    def2.update_mappings(0.5 * (vf2[1:] + vf2[:-1]))
    def3.update_mappings(0.5 * (vf3[1:] + vf3[:-1]))
    def4.update_mappings(0.5 * (vf4[1:] + vf4[:-1]))

    loss_f1_b2 = loss_func(reg=copy.deepcopy(reg), deformation=def1)
    loss_f2_b1 = loss_func(reg=copy.deepcopy(reg), deformation=def2)
    loss_b2_f1 = loss_func(reg=copy.deepcopy(reg), deformation=def3)
    loss_b1_f2 = loss_func(reg=copy.deepcopy(reg), deformation=def4)

    b = (loss_f1_b2 + loss_f2_b1 + loss_b2_f1 + loss_b1_f2)
    del vf1, vf2, vf3, vf4, def1, def2, def3, def4, loss_f1_b2, loss_f2_b1, loss_b2_f1, loss_b1_f2

    vf1 = np.copy(vf)
    vf2 = np.copy(vf)
    vf3 = np.copy(vf)
    vf4 = np.copy(vf)

    # f(x-1, y-2)
    vf1[i] -= epsilon
    vf1[j] -= 2 * epsilon
    # f(x-2, y-1)
    vf2[i] -= 2 * epsilon
    vf2[j] -= epsilon
    # f(x+1, y+2)
    vf3[i] += epsilon
    vf3[j] += 2 * epsilon
    # f(x+2, y+1)
    vf4[i] += 2 * epsilon
    vf4[j] += epsilon

    def1 = copy.deepcopy(deformation)
    def2 = copy.deepcopy(deformation)
    def3 = copy.deepcopy(deformation)
    def4 = copy.deepcopy(deformation)

    def1.update_mappings(0.5 * (vf1[1:] + vf1[:-1]))
    def2.update_mappings(0.5 * (vf2[1:] + vf2[:-1]))
    def3.update_mappings(0.5 * (vf3[1:] + vf3[:-1]))
    def4.update_mappings(0.5 * (vf4[1:] + vf4[:-1]))

    loss_b1_b2 = loss_func(reg=copy.deepcopy(reg), deformation=def1)
    loss_b2_b1 = loss_func(reg=copy.deepcopy(reg), deformation=def2)
    loss_f1_f2 = loss_func(reg=copy.deepcopy(reg), deformation=def3)
    loss_f2_f1 = loss_func(reg=copy.deepcopy(reg), deformation=def4)

    c = loss_b1_b2 + loss_b2_b1 + loss_f1_f2 + loss_f2_f1
    del vf1, vf2, vf3, vf4, def1, def2, def3, def4, loss_b2_b1, loss_b1_b2, loss_f2_f1, loss_f1_f2

    vf1 = np.copy(vf)
    vf2 = np.copy(vf)
    vf3 = np.copy(vf)
    vf4 = np.copy(vf)

    # f(x+2, y-2)
    vf1[i] += 2 * epsilon
    vf1[j] -= 2 * epsilon
    # f(x-2, y+2)
    vf2[i] -= 2 * epsilon
    vf2[j] += 2 * epsilon
    # f(x-2, y-2)
    vf3[i] -= 2 * epsilon
    vf3[j] -= 2 * epsilon
    # f(x+2, y+2)
    vf4[i] += 2 * epsilon
    vf4[j] += 2 * epsilon

    def1 = copy.deepcopy(deformation)
    def2 = copy.deepcopy(deformation)
    def3 = copy.deepcopy(deformation)
    def4 = copy.deepcopy(deformation)

    def1.update_mappings(0.5 * (vf1[1:] + vf1[:-1]))
    def2.update_mappings(0.5 * (vf2[1:] + vf2[:-1]))
    def3.update_mappings(0.5 * (vf3[1:] + vf3[:-1]))
    def4.update_mappings(0.5 * (vf4[1:] + vf4[:-1]))

    loss_f2_b2 = loss_func(reg=copy.deepcopy(reg), deformation=def1)
    loss_b2_f2 = loss_func(reg=copy.deepcopy(reg), deformation=def2)
    loss_b2_b2 = loss_func(reg=copy.deepcopy(reg), deformation=def3)
    loss_f2_f2 = loss_func(reg=copy.deepcopy(reg), deformation=def4)

    d = loss_f2_b2 + loss_b2_f2 - loss_b2_b2 - loss_f2_f2
    del vf1, vf2, vf3, vf4, def1, def2, def3, def4, loss_b2_f2, loss_b2_b2, loss_f2_b2, loss_f2_f2

    res = 64 * a + 8 * b - 8 * c - d
    # print('a {}, b{}, c{}, d{} \n res {}'.format(a, b, c, d, res))
    gc.collect()
    return res / float(144 * epsilon ** 2)


def second_derivative_by_loss(vf, i, j, loss, epsilon, reg, deformation):
    assert len(vf.shape) == len(i) == len(j), "Not correct indices"

    if i == j:
        return second_derivative_ii(vf=np.copy(vf), i=i, loss=loss, epsilon=epsilon, reg=copy.deepcopy(reg),
                                    deformation=copy.deepcopy(deformation))

    elif i != j:
        return second_derivative_ij(vf=np.copy(vf), i=i, j=j, loss=loss, epsilon=epsilon,
                                    reg=copy.deepcopy(reg), deformation=copy.deepcopy(deformation))
    else:
        raise TypeError('you should give correct indices')


def sparse_dot_product_forward(vector, ndim, mat_shape, T, loss, window, params_grad, param_der):
    mat_len = int(np.prod(mat_shape))
    # assert ndim * mat_len == len(vector), "not correct shape of vector"

    derivative_func = import_func(**param_der)

    deltas = list(itertools.product(range(-window, window + 1), repeat=vector.ndim - 2))
    mn, mx = (0,) * ndim, vector.shape[2:]

    data, rows, cols = [], [], []

    for ax in range(ndim):
        for i in np.ndindex(*vector.shape[2:]):
            I = matrix_to_vec_indices(i, mat_shape)

            for delta in deltas:
                j = np.array(i) + delta

                if not ((j < mn).any() or (j >= mx).any()) and i <= tuple(j):
                    der = derivative_func(
                        i=(T, ax,) + i,
                        j=(T, ax,) + tuple(j),
                        vf=np.copy(vector), loss=loss, **params_grad
                    )

                    i_loc = I + ax * mat_len
                    j_loc = matrix_to_vec_indices(j, mat_shape) + ax * mat_len
                    # if np.abs(der) > 1e-8:
                    data.extend([der, der])
                    rows.extend([i_loc, j_loc])
                    cols.extend([j_loc, i_loc])

    gc.collect()
    shape = (ndim * mat_len, ndim * mat_len)
    r = np.arange(shape[0])

    reg = coo_matrix((np.repeat(1e-14, shape[0]), (r, r)), shape = shape)
    return inv(coo_matrix((data, (rows, cols)), shape=shape) + reg)


def double_dev_J_v(vec):
    shape = int(np.prod(vec.shape[1:]))

    data = []
    rows, cols = [], []

    for i in range(vec.shape[0]):
        data += list(vec[i].reshape(-1))
        rows += list(np.arange(shape))
        cols += list(np.arange(shape) + i * shape)

    return coo_matrix((data, (cols, rows)), shape=(np.prod(vec.shape), shape))

# def sparse_dot_product_parallel(vector, ndim, mat_shape, loss, window, params_grad, param_der, n_jobs=5,
#                                 path_joblib='~/JOBLIB_TMP_FOLDER/'):
#     mat_len = int(np.prod(mat_shape))
#
#     # assert ndim * mat_len == len(vector), "not correct shape of vector"
#
#     loc_res = Parallel(n_jobs=n_jobs, temp_folder=path_joblib)(
#         delayed(one_line_sparse)(
#             vector=vector, ndim=ndim, I=I, shape=mat_shape,
#             loss=loss, window=window, ax=ax, params_grad=params_grad,
#             param_der=param_der, ind=[]
#         )
#         for ax in range(ndim) for I in range(mat_len)
#     )
#
#     gc.collect()
#
#     result = coo_matrix((ndim * mat_len,
#                          ndim * mat_len))
#     for one in loc_res:
#         result += one
#
#     return result


# def sparse_dot_product(vector, ndim, mat_shape, loss, params_grad, param_der, window=2, mode='parallel',
#                        n_jobs=5, path=joblib_path):
#     if mode == 'forward' or n_jobs == 1:
#         return sparse_dot_product_forward(vector=vector, ndim=ndim,
#                                           mat_shape=mat_shape, loss=loss,
#                                           window=window, params_grad=params_grad,
#                                           param_der=param_der)
#     elif mode == 'parallel':
#         return sparse_dot_product_parallel(vector=vector, ndim=ndim,
#                                            mat_shape=mat_shape, loss=loss,
#                                            window=window, params_grad=params_grad,
#                                            param_der=param_der,
#                                            n_jobs=n_jobs, path_joblib=path)
#     else:
#         raise TypeError('Do not support such type of calculating')
# def one_line_sparse(vector, ndim, I, shape, window, loss, ax, params_grad, param_der):
#     if params_grad['inverse']:
#         T = -1
#     else:
#         T = 0
#     cols = neighbours_indices(shape, I, 'vec', window)
#
#     rows = np.repeat(I, len(cols))
#
#     derivative_func = import_func(**param_der)
#
#     data = np.array([
#         derivative_func(i=(T, ax,) + tuple(vec_to_matrix_indices(I, shape)),
#                         j=(T, ax,) + tuple(vec_to_matrix_indices(j, shape)),
#                         vf=vector, loss=loss, **params_grad
#                         )
#         for j in cols
#     ])
#
#     mat_shape = (ndim * np.prod(shape), ndim * np.prod(shape))
#
#     return coo_matrix((data, (rows + ax * int(np.prod(shape)), cols + ax * int(np.prod(shape)))), shape=mat_shape)
#
