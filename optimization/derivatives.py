import os
import sys

module_path = '~/src/rtk/'
if module_path not in sys.path:
    sys.path.append(module_path)
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(module_path)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import rtk
from rtk.registration.LDDMM import derivative

import numpy as np
from scipy.fftpack import fftn, ifftn

from RegOptim.preprocessing import to_one_resolution
from .template_utils import sparse_dot_product
import pickle
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


def get_resulting_dv(reg, regularizer, vf0, inverse):
    reg.set_params(**{'regularizer': regularizer})
    reg.execute()
    return to_one_resolution(resulting_vector_fields=reg.resulting_vector_fields,
                             resolutions=reg.resolutions,
                             n_steps=reg.n_step, zoom_grid=reg.zoom_grid,
                             vf0=vf0, inverse=inverse)


def dv(reg, a, b, vf, epsilon, name_of_param, vf0, inverse):
    # set first regularizer with a + epsilon
    if name_of_param == 'convexity':
        regularizer1 = rtk.BiharmonicRegularizer(convexity_penalty=a + epsilon, norm_penalty=b)
        regularizer2 = rtk.BiharmonicRegularizer(convexity_penalty=a - epsilon, norm_penalty=b)

    elif name_of_param == 'normalization':
        regularizer1 = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b + epsilon)
        regularizer2 = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b - epsilon)

    else:
        raise TypeError

    # compute resulting vector fields
    vf1 = get_resulting_dv(reg, regularizer1, vf0, inverse)

    # count resulting vector fields
    vf2 = get_resulting_dv(reg, regularizer2, vf0, inverse)
    del regularizer1, regularizer2
    # f''(x) = (f(x+e) - 2f(x)+f(x-e))/e^2
    return (vf1 - 2 * vf + vf2) / epsilon ** 2


def get_derivative_v(a, b, reg, vf, epsilon=0.1, vf0=True, inverse=True, data=None, template=None):
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

    dv_da = dv(reg=reg, a=a, b=b, vf=vf, epsilon=epsilon, name_of_param='convexity', vf0=vf0, inverse=inverse)
    dv_db = dv(reg=reg, a=a, b=b, vf=vf, epsilon=epsilon, name_of_param='normalization',
               vf0=vf0, inverse=inverse)

    return dv_da, dv_db


def get_derivative_template(data, template, n_steps, vf_all_in_one_resolution,
                            similarity, regularizer, vf0, inverse, path, n_jobs=5, window=3):
    if inverse:
        template_img = rtk.ScalarImage(data=template)
        moving_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=data), n_steps)
    else:
        template_img = rtk.ScalarImage(data=data)
        moving_imgs = rtk.SequentialScalarImages(rtk.ScalarImage(data=template), n_steps)

    deformation = rtk.DiffeomorphicDeformation(n_steps)
    deformation.set_shape(template.shape)

    v = 0.5 * (vf_all_in_one_resolution[:-1] + vf_all_in_one_resolution[1:])
    deformation.update_mappings(v)

    moving_imgs.apply_transforms(deformation.forward_mappings)

    if vf0:
        if inverse:
            T = -1
        else:
            T = 0
        inv_grad_v = np.array([1 / derivative(similarity=similarity, fixed=template_img.data,
                                              moving=moving_imgs[T], Dphi=deformation.backward_dets[-T - 1],
                                              vector_field=vf_all_in_one_resolution[T],
                                              regularizer=regularizer, learning_rate=1.)]).reshape(-1, 1)
    else:
        inv_grad_v = np.stack([np.array([1 / derivative(similarity=similarity, fixed=template_img.data,
                                                        moving=moving_imgs[i], Dphi=deformation.backward_dets[-i - 1],
                                                        vector_field=vf_all_in_one_resolution[i],
                                                        regularizer=regularizer, learning_rate=1.)])
                               for i in range(n_steps + 1)], 0).reshape(-1, 1)

    if vf0:
        # (t=1, ndim, img_shape)-for v and (img_shape,)- for template img J
        shape_res = (1, template.ndim) + template.shape + template.shape
    else:
        shape_res = (n_steps + 1, template.ndim) + template.shape + template.shape
    # get I composed with phi
    del template_img
    #
    dl_dJ = (-2 * (moving_imgs[-1] - template)).reshape(1,-1)
    #if you want to use sparsity
    # dv_dJ = sparse_dot_product(vector=inv_grad_v, mat_shape=shape_res[:-template.ndim], window=window,
    #                            mode='parallel', n_jobs=n_jobs, path=joblib_path).dot(dl_dJ)
    dv_dJ = inv_grad_v.dot(dl_dJ)
    del moving_imgs
    del dl_dJ, inv_grad_v

    gc.collect()

    dv_dJ = dv_dJ.reshape(shape_res)

    if path is not None:
        with open(path, 'wb') as f:
            pickle.dump(dv_dJ, f, protocol=2)

        del dv_dJ
        return [path]
    else:
        return  dv_dJ