import rtk
from rtk.registration.LDDMM import derivative
from rtk import gradient

import numpy as np
from scipy.fftpack import fftn, ifftn

from RegOptim.preprocessing import to_one_resolution
from RegOptim.optimization.template_utils import sparse_dot_product, double_dev_J_v
from RegOptim.optimization.pipeline_utils import count_da_db_to_template, count_K_to_template, count_dJ

from tqdm import tqdm
import gc

joblib_path = '~/JOBLIB_TMP_FOLDER/'


def get_delta(A, a, b):
    delta = A - b * np.ones(A.shape)
    return delta / float(a)


def get_derivative_Lv(A, v, a, b):
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


def get_resulting_dv(reg, regularizer, inverse):
    reg.set_params(**{'regularizer': regularizer})
    reg.execute()
    return to_one_resolution(resulting_vector_fields=reg.resulting_vector_fields,
                             resolutions=reg.resolutions,
                             n_steps=reg.n_step, zoom_grid=reg.zoom_grid,
                             vf0=True, inverse=inverse)


def dv(reg, a, b, epsilon, name_of_param, inverse):
    # set first regularizer with a + epsilon
    if name_of_param == 'convexity':
        regularizer1 = rtk.BiharmonicRegularizer(convexity_penalty=a + epsilon, norm_penalty=b)
        regularizer2 = rtk.BiharmonicRegularizer(convexity_penalty=a - epsilon, norm_penalty=b)

    elif name_of_param == 'normalization':
        regularizer1 = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b + epsilon)
        regularizer2 = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b - epsilon)

    else:
        raise TypeError('No such metric parameter')

    # compute resulting vector fields
    vf1 = get_resulting_dv(reg, regularizer1, inverse)

    # count resulting vector fields
    vf2 = get_resulting_dv(reg, regularizer2, inverse)
    del regularizer1, regularizer2
    # f'(x) = (f(x+e) - f(x-e))/(2*e)
    return (vf1 - vf2) / (2 * epsilon)


def get_derivative_v(a, b, reg, epsilon=0.1, inverse=True, data=None, template=None):
    # check if exist fixed and moving images
    if not hasattr(reg, 'fixed'):
        # inverse means that we would like to find path from X to template
        # it means that we would like to find vf[-1] ~ vf^(-1)[0]
        if data is None or template is None:
            print 'Error, not optimized LDDMM registration is passed'
            raise TypeError
        if inverse:
            # first image fixed, second moving
            reg.set_images(rtk.ScalarImage(data=template), rtk.ScalarImage(data=data))
        else:
            reg.set_images(rtk.ScalarImage(data=data), rtk.ScalarImage(data=template))

    dv_da = dv(reg=reg, a=a, b=b, epsilon=epsilon, name_of_param='convexity', inverse=inverse)
    dv_db = dv(reg=reg, a=a, b=b, epsilon=epsilon, name_of_param='normalization', inverse=inverse)

    return dv_da, dv_db


def derivatives_of_pipeline_with_template(result, train_idx, n_total, img_shape):
    n_train = len(train_idx)
    Lvfs, vfs, dv_da, dv_db, dL_da, dL_db, dv_dJ = map(np.concatenate, zip(*result))
    shape = np.array(Lvfs).shape[2:]
    ndim = len(shape)

    # (t=1, ndim, img_shape)-for v and (img_shape,)- for template img J
    if isinstance(dv_dJ[0], (str, np.str, np.string_, np.unicode_)):
        shape_res = (ndim,) + img_shape + img_shape
    else:
        shape_res = (1, ndim) + img_shape + img_shape

    metric = np.array([count_dJ(Lvfs[idx1], Lvfs[idx2], dv_dJ[idx1].copy(), dv_dJ[idx2].copy(), ndim, shape=shape_res)
                       for i, idx1 in tqdm(enumerate(train_idx), desc='dJ_train')
                       for idx2 in train_idx[i:]])
    dJ = np.zeros((n_train, n_train) + shape)

    i, j = np.triu_indices(n_train, 0)
    k, l = np.tril_indices(n_train, 0)
    dJ[i, j] = metric
    dJ[k, l] = metric

    K = count_K_to_template(Lvfs, vfs, n_total)
    da, db = count_da_db_to_template(Lvfs, vfs, dv_da, dv_db, dL_da, dL_db, n_total)
    gc.collect()
    return K, da, db, dJ


# @profile
def derivatives_of_pipeline_without_template(result, n_total):
    Lvfs, vfs, dv_da, dv_db, dL_da, dL_db = map(np.concatenate, zip(*result))
    K = count_K_to_template(Lvfs, vfs, n_total)
    da, db = count_da_db_to_template(Lvfs, vfs, dv_da, dv_db, dL_da, dL_db, n_total)
    gc.collect()
    return K, da, db


def get_derivative_template(data, template, n_steps, vf_all_in_one_resolution,
                            similarity, regularizer, inverse, path, n_jobs=5, window=3):
    if inverse:
        template_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=template), n_steps)
        moving_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=data), n_steps)
    else:
        template_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=template), n_steps)
        moving_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=template), n_steps)

    deformation = rtk.DiffeomorphicDeformation(n_steps)
    deformation.set_shape(template.shape)

    v = 0.5 * (vf_all_in_one_resolution[:-1] + vf_all_in_one_resolution[1:])
    deformation.update_mappings(v)

    moving_imgs.apply_transforms(deformation.forward_mappings)
    template_imgs.apply_transforms(deformation.backward_mappings)

    if inverse:
        T = -1
    else:
        T = 0

    inv_grad_v = np.array([1 / derivative(similarity=similarity, fixed=template_imgs[- T - 1],
                                          moving=moving_imgs[T], Dphi=deformation.backward_dets[- T - 1],
                                          vector_field=vf_all_in_one_resolution[T],
                                          regularizer=regularizer, learning_rate=1.)]).reshape(-1, 1)

    # get I composed with phi
    # print moving_imgs.data[-1].shape, template_img.shape

    dl_dv = - 2. / similarity.variance * gradient(moving_imgs[T]) * deformation.backward_dets[- T - 1]
    #     dl_dv = - similarity.derivative(moving_imgs.data[-1], template_img.data) * deformation.backward_dets[-1] #/ \
    # np.array(moving_imgs[-1] - template_img.data).astype(np.float)
    # if you want to use sparsity
    # dv_dJ = sparse_dot_product(vector=inv_grad_v, mat_shape=shape_res[:-template.ndim], window=window,
    #                            mode='parallel', n_jobs=n_jobs, path=joblib_path).dot(dl_dJ)

    # (t=1, ndim, img_shape)-for v and (img_shape,)- for template img J
    if path is not None:
        shape_res = (template.ndim,) + template.shape + template.shape
    else:
        shape_res = (1, template.ndim) + template.shape + template.shape

    dl_dJ_dv = double_dev_J_v(dl_dv)

    dv_dJ = sparse_dot_product(vector=inv_grad_v, mat_shape=shape_res[:-template.ndim], window=window,
                               mode='parallel', n_jobs=n_jobs, path=joblib_path).dot(dl_dJ_dv)

    del moving_imgs, template_imgs
    del dl_dv, inv_grad_v, dl_dJ_dv

    gc.collect()

    return dv_dJ
