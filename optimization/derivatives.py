import rtk
from rtk import gradient

import numpy as np
import copy
# import gc
from scipy.fftpack import fftn, ifftn

from RegOptim.preprocessing import to_one_resolution
from RegOptim.optimization.template_utils import sparse_dot_product, double_dev_J_v, full_derivative_by_v

joblib_path = '~/JOBLIB_TMP_FOLDER/'


def path_length(A, vf, a, b):
    # count
    # dLv/da = 2(a*delta^2 + b*delta)*v - shape (ndim, image_shape)
    # dLv/db = 2(a*delta + bE) * E * v = 2(a*delta + bE)v - shape (ndim, image_shape)
    # shape of this dLv_da - (n_steps, ndim, image_shape)
    dLv_da, dLv_db = np.array([get_derivative_Lv(A=A, v=vf[i], a=a, b=b) for i in range(len(vf))]).T
    # axis (ndim, image_shape)
    axis = tuple(np.arange(vf.shape)[1:])
    # sum by space dimensions
    da, db = np.sum(dLv_da * vf, axis=axis), np.sum(dLv_db * vf, axis=axis)
    # by time dimensions (approx integral)
    da = 0.5 * (da[:-1] + da[1:])
    db = 0.5 * (db[:-1] + db[1:])

    return da, db


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


def pairwise_pipeline_derivatives(reg, inverse):
    in_one_res = to_one_resolution(reg.resulting_vector_fields, reg.resolutions, reg.n_step,
                                   reg.zoom_grid, False, inverse)

    return np.sum(reg.resulting_metric) / len(reg.resolutions), in_one_res


# @profile
def template_pipeline_derivatives(reg, similarity, regularizer, data, template, a, b,
                                  epsilon, shape, inverse, optim_template, path,
                                  n_jobs, window):
    in_one_res = to_one_resolution(resulting_vector_fields=reg.resulting_vector_fields,
                                   resolutions=reg.resolutions,
                                   n_steps=reg.n_step, zoom_grid=reg.zoom_grid, vf0=True, inverse=inverse)

    vf_all_in_one_res = to_one_resolution(resulting_vector_fields=reg.resulting_vector_fields,
                                          resolutions=reg.resolutions,
                                          n_steps=reg.n_step, zoom_grid=reg.zoom_grid,
                                          vf0=False, inverse=inverse)

    regularizer.set_operator(shape)

    # find Lv(t=1,ndim, img_shape)
    Lvf = regularizer(in_one_res[0])[None]
    # get derivatives of Lv
    Deltas_v = get_derivative_Lv(A=reg.As[-1], v=in_one_res, a=a, b=b)
    dLv_da, dLv_db = Deltas_v[0], Deltas_v[1]
    del Deltas_v

    # after that will be change reg so, save everything what you need
    dv_da, dv_db = get_derivative_v(a=a, b=b, reg=copy.deepcopy(reg),
                                    epsilon=epsilon, inverse=inverse, data=data,
                                    template=template)

    if optim_template:
        dv_dJ = get_derivative_template(data=data, template=template, n_steps=reg.n_step,
                                        vf_all_in_one_resolution=vf_all_in_one_res,
                                        similarity=similarity, regularizer=regularizer,
                                        inverse=inverse, path=path, n_jobs=n_jobs, window=window)
        # gc.collect()
        return Lvf, in_one_res, dv_da, dv_db, dLv_da, dLv_db, dv_dJ

    # gc.collect()

    return Lvf, in_one_res, dv_da, dv_db, dLv_da, dLv_db


def get_derivative_template(data, template, n_steps, vf_all_in_one_resolution,
                            similarity, regularizer, inverse, path, n_jobs=5, window=3):
    grad_v, det, moving_img = full_derivative_by_v(data, template, n_steps, vf_all_in_one_resolution,
                                                   similarity, regularizer, inverse)

    # get I composed with phi
    # print moving_imgs.data[-1].shape, template_img.shape

    dl_dv = - 2. / similarity.variance * gradient(moving_img) * det

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

    dv_dJ = sparse_dot_product(vector=grad_v.reshape(-1, 1).copy(), ndim=template.ndim,
                               mat_shape=shape_res[:-template.ndim], window=window,
                               mode='forward', n_jobs=n_jobs, path=joblib_path).dot(dl_dJ_dv)
    dv_dJ = dl_dJ_dv
    del dl_dv, dl_dJ_dv

    # gc.collect()

    return [dv_dJ]
