try:
    import rtk
except:
    import os
    import sys

    module_path = '~/workspace/rtk/'
    if module_path not in sys.path:
        sys.path.append(module_path)
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(module_path)
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    import rtk

import numpy as np
import nibabel as nib
from tqdm import tqdm
from joblib import Parallel, delayed

from models.RegPred.ml.ml_utils import expand_dims
from models.RegPred.utils import load_images, load_nii
from models.RegPred.preprocessing import change_resolution
from .derivatives import get_delta, get_derivative_Lv


def create_template(path_to_data, train_idx, exp_path, resolution, sigma=0.01, inverse=False):
    images = []
    path_to_template = os.path.join(exp_path, 'templates/')

    os.makedirs(path_to_template, exist_ok=True)

    if isinstance(path_to_data[0], (str, np.string_)):
        images = load_images(path_to_data[np.ix_(train_idx)])

    if isinstance(path_to_data[0], np.ndarray):
        images = path_to_data[np.ix_(train_idx)]

    if resolution != 1:
        images = change_resolution(images, resolution, sigma)

    images = np.array(images)
    template = np.mean(images, axis=0)
    if inverse:
        template = change_resolution(template, 1 / float(resolution), sigma, multiple=False)

    template_nifty = nib.Nifti1Image(template, np.eye(4))
    nib.save(template_nifty, path_to_template)
    return images


###change!!! something wrong
def update_template(template_path, delta, it, learning_rate=0.1, resolution=1., inverse=False, data_type=np.ndarray):
    if isinstance(data_type, (str, np.string_, np.str)):
        image = load_nii(template_path)
    if isinstance(data_type, np.ndarray):
        image = template_path

    if image.shape != delta.shape:
        print 'Error not correct shape or resolution'
        raise TypeError

    image -= learning_rate * delta

    if isinstance(data_type, np.ndarray):
        return image

    image = nib.Nifti1Image(image, np.eye(4))
    new_path = template_path.split('.nii')[0] + '_' + str(it) + '.nii'
    nib.save(image, new_path)

    if isinstance(data_type, (str, np.string_, np.str)):
        return new_path


def count_K_with_template(Lvf, vf, n):
    K = np.zeros((n, n))
    # vf  shape (n_samples, t, ndim, img_shape)
    axis = tuple(np.arange(vf.ndim)[1:])
    # K = (<Lv_i, v_j>), is sum by space(ndim, img_shape,) and if t!=0 to get length of path sum of t
    for i, one in enumerate(Lvf):
        # vf ~ (n_sample, t, ndim, img_shape) * Lv[i]~ (i_th sample=1, t, ndim, img_shape)
        K[i, :] = np.sum(vf * one, axis=axis)

    return K


def count_K_without_template(metrics, n):
    K = np.zeros((n, n))
    i, j = np.triu_indices(n=n, k=0)
    K[i, j] = metrics

    return K + K.T - K.diagonal()


def path_length(A, vf, a, b):
    # count
    # dLv/da = 2(a*delta^2 + b*delta)*v - shape (ndim, image_shape)
    # dLv/db = 2(a*delta + bE) * E * v = 2(a*delta + bE)v - shape (ndim, image_shape)
    # shape of this dLv_da - (n_steps, ndim, image_shape)
    dLv_da, dLv_db = np.array([get_derivative_Lv(A, vf[i], a, b) for i in range(len(vf))]).T
    # axis (ndim, image_shape)
    axis = tuple(np.arange(vf.shape)[1:])
    # sum by space dimentions
    da, db = np.sum(dLv_da * vf, axis=axis), np.sum(dLv_db * vf, axis=axis)
    # by time dimention (approx integral)
    da = 0.5 * (da[:-1] + da[1:])
    db = 0.5 * (db[:-1] + db[1:])

    return [da], [db]


def count_da_db_without_template(vf, a, b, shape, n_job, n):
    regularizer = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b)

    regularizer.set_operator(shape=shape)
    train_devs = Parallel(n_jobs=n_job)(delayed(path_length)(regularizer.A, vf[i], a, b)
                                        for i in tqdm(range(len(vf)), desc='da_db_train'))

    train_da, train_db = map(np.concatenate, zip(*train_devs))

    da = np.zeros((n, n))
    db = np.zeros((n, n))

    i, j = np.triu_indices(n=n, k=0)
    da[i, j] = train_da
    db[i, j] = train_db

    return da + da.T - da.diagonal(), db + db.T - db.diagonal()


def count_da_db_with_template(Lvf, vf, dv_da, dv_db, dLv_da, dLv_db, n):
    da = np.zeros((n, n))
    # dv_da = (v(a + e) - 2*v(a) + v(a-e))/e^2, shape(n_sample, t, ndim, img_shape)
    # dLv = dL/da * v, shape (n_samples, t, ndim, img_shape)
    db = np.zeros((n, n))
    # vf  shape  (n_samples, t, ndim, img_shape)
    axis = tuple(np.arange(vf.ndim)[1:])

    # count derivatives
    for i in tqdm(range(n), desc="da_db"):
        # correct derivative of K=(<Lv_i,v_j>)
        # dK/da = <dL v_i, v_j> + <L dv_i/da, v_j> + <Lv_i, dv_j>
        # because <Lv_i, v_j> = v_i^T.dot(L.dot(v_j)) = v_j^T.dot(L.dot(v_i)), it is scalar value
        # dK/da = vf(n_sample,t,ndim,img_shape) * dLv[i](t,ndim, img_shape) +
        # + Lv[j] * dv_i
        da[i, :] = np.sum(vf * dLv_da[i], axis=axis) + \
                   np.sum(dv_da[i] * Lvf, axis=axis) + \
                   np.sum(dv_da * Lvf[i], axis=axis)
        db[i, :] = np.sum(vf * dLv_db[i], axis=axis) + \
                   np.sum(dv_db[i] * Lvf, axis=axis) + \
                   np.sum(dv_db * Lvf[i], axis=axis)

    return da, db


def count_dJ(Lvfs_i, Lvfs_j, dv_dJ_i, dv_dJ_j, ndim):
    # we want to differenciate K(kernel) by J (template)
    # K_ij = <Lv_i, v_j>
    # dK/dJ = <Ldv_i/dJ, v_j> + <Lv_i, dv_j/dJ>
    # <Ldv_i/dJ, v_j> = <dv_i/dJ, Lv_j>, because L - is self-adjoint
    # dK/dJ = <dv_i/dJ, Lv_j> + <Lv_i, dv_j/dJ>
    axis = tuple(np.arange(Lvfs_i.ndim))
    # np.sum(b[0] * a[..., None, None], axis=(1,2,3,4))
    dK_dJ = np.sum(dv_dJ_i * expand_dims(Lvfs_j, ndim), axis=axis) + \
            np.sum(expand_dims(Lvfs_i, ndim) * dv_dJ_j, axis=axis)

    return dK_dJ
