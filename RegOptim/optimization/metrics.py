import numpy as np
import gc
from joblib import Parallel, delayed
from tqdm import tqdm

import rtk
from RegOptim.ml.ml_utils import expand_dims
from RegOptim.optimization.template_utils import path_length
from RegOptim.optimization.derivatives import Lv



def count_K_to_template(Lvf, vf, n):
    K = np.zeros((n, n))
    # vf  shape (n_samples, t, ndim, img_shape)
    axis = tuple(np.arange(vf.ndim)[1:])
    # K = (<Lv_i, v_j>), is sum by space(ndim, img_shape,) and if t!=0 to get length of path sum of t
    for i, one in enumerate(Lvf):
        # vf ~ (n_sample, t, ndim, img_shape) * Lv[i]~ (i_th sample=1, t, ndim, img_shape)
        K[i, :] = np.sum(vf * one, axis=axis)

    return K


def count_K_pairwise(metrics, n):
    K = np.zeros((n, n))
    i, j = np.triu_indices(n=n, k=0)
    K[i, j] = metrics

    return K + K.T - K.diagonal()


def count_da_db_pairwise(vf, a, b, shape, n_job, n):
    regularizer = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b)
    regularizer.set_operator(shape=shape)

    train_devs = Parallel(n_jobs=n_job)(delayed(path_length)(A=regularizer.A, vf=vf[i], a=a, b=b)
                                        for i in tqdm(range(len(vf)), desc='da_db_train'))

    train_da, train_db = map(np.concatenate, zip(*train_devs))

    da = np.zeros((n, n))
    db = np.zeros((n, n))

    i, j = np.triu_indices(n=n, k=0)
    da[i, j] = train_da
    db[i, j] = train_db

    return da + da.T - da.diagonal(), db + db.T - db.diagonal()


def count_da_db_to_template(Lvf, vf, dv_da, dv_db, dLv_da, dLv_db, n):
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


def count_dJ(Lvfs_i, vfs_j, dv_dJ_i, dv_dJ_j, ndim, shape, A):
    # we want to differentiate K(kernel) by J (template)
    # K_ij = <Lv_i, v_j>
    # dK/dJ = <Ldv_i/dJ, v_j> + <Lv_i, dv_j/dJ>
    # <Ldv_i/dJ, v_j> = <dv_i/dJ, Lv_j>, because L - is self-adjoint
    # dK/dJ = <dv_i/dJ, Lv_j> + <Lv_i, dv_j/dJ>

    dv_dJ_i = dv_dJ_i.toarray().reshape(shape)
    dv_dJ_j = dv_dJ_j.toarray().reshape(shape)
    # with open(dv_dJ_i, 'rb') as f:
    #     dv_dJ_i = pickle.load(f).astype(np.float32)
    # with open(dv_dJ_j, 'rb') as f:
    #     dv_dJ_j = pickle.load(f).astype(np.float32)
    axis = tuple(np.arange(Lvfs_i.ndim))
    Lvfs_j = Lv(A**2, vfs_j)
    # np.sum(b[0] * a[..., None, None], axis=(1,2,3,4))
    dK_dJ = np.sum(dv_dJ_i * expand_dims(Lvfs_j, ndim), axis=axis) + \
            + np.sum(expand_dims(Lvfs_i, ndim) * dv_dJ_j, axis=axis)

    gc.collect()
    return dK_dJ
