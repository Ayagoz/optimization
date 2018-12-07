try:
    import rtk
except:
    import os
    import sys

    module_path = '~/src/rtk/'
    if module_path not in sys.path:
        sys.path.append(module_path)
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(module_path)
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    import rtk

import numpy as np
from tqdm import tqdm
import copy

from joblib import Parallel, delayed

joblib_folder = '~/JOBLIB_TMP_FOLDER/'

from RegOptim.optimization.derivatives import to_one_resolution, get_derivative_Lv, get_derivative_v, \
    get_derivative_template
from RegOptim.optimization.pipeline_utils import count_dJ, count_K_with_template, count_da_db_with_template
from RegOptim.optimization.pipeline_utils import count_K_without_template, count_da_db_without_template
from RegOptim.utils import load_nii
from RegOptim.preprocessing import change_resolution


def pairwise_pipeline_derivatives(reg, inverse):
    in_one_res = to_one_resolution(reg.resulting_vector_fields, reg.resolutions, reg.n_step,
                                   reg.zoom_grid, False, inverse)
    return [np.sum(reg.resulting_metric) / len(reg.resolutions)], [in_one_res]


def template_pipeline_derivatives(reg, similarity, regularizer, data, template, a, b,
                                  epsilon, shape, vf0, inverse, optim_template):
    in_one_res = to_one_resolution(reg.resulting_vector_fields, reg.resolutions,
                                   reg.n_step, reg.zoom_grid, vf0, inverse)

    vf_all_in_one_res = to_one_resolution(reg.resulting_vector_fields, reg.resolutions,
                                          reg.n_step, reg.zoom_grid, False, inverse)

    regularizer.set_operator(shape)

    if vf0:
        # find Lv(t=1,ndim, img_shape)
        Lvf = regularizer(in_one_res[0])[None]
        # get derivatives of Lv
        Deltas_v = get_derivative_Lv(reg.As[-1], in_one_res, a, b)
        dLv_da, dLv_db = Deltas_v[0], Deltas_v[1]
    else:
        Lvf = np.stack([regularizer(in_one_res[i]) for i in range(reg.n_step + 1)], 0)
        Deltas_v = np.array([get_derivative_Lv(reg.As[-1], in_one_res[i], a, b) for i in range(reg.n_step + 1)])
        dLv_da, dLv_db = Deltas_v[:, 0], Deltas_v[:, 1]
    # after that will be change reg so, save everything what you need
    dv_da, dv_db = get_derivative_v(a, b, copy.deepcopy(reg), in_one_res, epsilon, vf0, inverse, data, template)

    if optim_template:
        dv_dJ = get_derivative_template(data, template, reg.n_step, vf_all_in_one_res,
                                        similarity, regularizer, vf0, inverse)

        return Lvf, in_one_res, dv_da, dv_db, dLv_da, dLv_db, dv_dJ

    return Lvf, in_one_res, dv_da, dv_db, dLv_da, dLv_db


def one_to_one(data1, data2, a, b, epsilon, ssd_var, n_steps,
               n_iters, resolutions, smoothing_sigmas,
               delta_phi_threshold, unit_threshold, learning_rate, n_jobs,
               vf0, inverse, optim_template, data_type='path',
               change_res=True, init_resolution=4., pipe_template=True):
    # registration
    similarity = rtk.SSD(variance=ssd_var)
    regularizer = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b)

    reg = rtk.LDDMM(
        n_step=n_steps,
        similarity=similarity,
        regularizer=regularizer,
        n_iters=n_iters,
        resolutions=resolutions,
        smoothing_sigmas=smoothing_sigmas,
        delta_phi_threshold=delta_phi_threshold,
        unit_threshold=unit_threshold,
        learning_rate=learning_rate, n_jobs=n_jobs)

    if data_type == 'path':
        data1 = load_nii(data1)
    if isinstance(data2, (str, np.str, np.string_)):
        data2 = load_nii(data2)

    if change_res:
        data1 = np.squeeze(change_resolution(data1, init_resolution, multiple=False))
        data2 = np.squeeze(change_resolution(data2, init_resolution, multiple=False))

    # inverse means that we would like to find path from X to template
    # it means that we would like to find vf[-1] ~ vf^(-1)[0]

    if inverse:
        # first image = fixed, second = moving
        reg.set_images(rtk.ScalarImage(data=data2), rtk.ScalarImage(data=data1))
    else:
        reg.set_images(rtk.ScalarImage(data=data1), rtk.ScalarImage(data=data2))

    reg.execute()

    # get resulting vector field in one resolution
    # if vf0=True, get (1, ndim, img_shape), else get (t, ndim, img_shape)

    if pipe_template:
        return template_pipeline_derivatives(reg, similarity, regularizer, data1, data2, a, b,
                                             epsilon, data1.shape, vf0, inverse, optim_template)

    else:
        return pairwise_pipeline_derivatives(reg, inverse)


def derivatives_of_pipeline_with_template(result, train_idx, n_total, n_job=1):
    n_train = len(train_idx)

    Lvfs, vfs, dv_da, dv_db, dL_da, dL_db, dv_dJ = map(np.concatenate, zip(*result))
    shape = np.array(Lvfs).shape[2:]
    ndim = len(shape)

    dJ = np.zeros((n_train, n_train) + shape)
    i, j = np.triu_indices(n_train, 0)
    dJ[i, j] = np.array([count_dJ(Lvfs[idx1], Lvfs[idx2], dv_dJ[idx1], dv_dJ[idx2], ndim)
                for i, idx1 in tqdm(enumerate(train_idx), desc='dJ_train')
                for idx2 in train_idx[i:]])

    K = count_K_with_template(Lvfs, vfs, n_total)
    da, db = count_da_db_with_template(Lvfs, vfs, dv_da, dv_db, dL_da, dL_db, n_total)
    return K, da, db, dJ


def derivatives_of_pipeline(result, n_total):
    Lvfs, vfs, dv_da, dv_db, dL_da, dL_db = map(np.concatenate, zip(*result))
    K = count_K_with_template(Lvfs, vfs, n_total)
    da, db = count_da_db_with_template(Lvfs, vfs, dv_da, dv_db, dL_da, dL_db, n_total)

    return K, da, db


def count_dist_matrix_to_template(data, template, a, b, train_idx, epsilon=0.1, n_job=5, ssd_var=1000.,
                                  n_steps=30, n_iters=(50, 20, 10), resolutions=(4, 2, 1), smoothing_sigmas=(2., 1., 0.),
                                  delta_phi_threshold=0.1, unit_threshold=0.01, learning_rate=0.01,
                                  change_res=True, init_resolution=4, data_type='path', vf0=True,
                                  inverse=False, optim_template=True):
    n = len(data)

    result = Parallel(n_jobs=n_job, temp_folder=joblib_folder)(delayed(one_to_one)(data1=data[i],
                                                                                   data2=template,
                                                                       a=a, b=b,
                                                                       epsilon=epsilon,
                                                                       ssd_var=ssd_var,
                                                                       n_steps=n_steps,
                                                                       n_iters=n_iters,
                                                                       resolutions=resolutions,
                                                                       smoothing_sigmas=smoothing_sigmas,
                                                                       delta_phi_threshold=delta_phi_threshold,
                                                                       unit_threshold=unit_threshold,
                                                                       learning_rate=learning_rate,
                                                                       n_jobs=1, vf0=vf0, inverse=inverse,
                                                                       optim_template=optim_template,
                                                                       data_type=data_type,
                                                                       change_res=change_res,
                                                                       init_resolution=init_resolution,
                                                                       pipe_template=True)
                                                               for i in tqdm(range(n), desc="registration"))

    if optim_template:
        return derivatives_of_pipeline_with_template(result, train_idx, n, n_job)

    else:
        return derivatives_of_pipeline(result, n)


def count_dist_matrix(data, y, a, b, idx_train=None, idx_test=None, n_job=5, ssd_var=1000., n_steps=20,
                      n_iters=(50, 20, 10), resolutions=(4, 2, 1), smoothing_sigmas=(2., 1., 0.),
                      delta_phi_threshold=0.1, unit_threshold=0.01, learning_rate=0.01,
                      inverse=True, change_res=True, init_resolution=4, data_type='path'):
    if idx_train is None:
        idx_train = []
    if idx_test is None:
        idx_test = []

    n_train = len(idx_train)
    n_test = len(idx_test)
    if n_train == 0:
        n_train = len(y)
        idx_train = np.arange(n_train)

    train_result = Parallel(n_jobs=n_job, temp_folder=joblib_folder)(
        delayed(one_to_one)(data1=copy.deepcopy(data[idx1]), data2=copy.deepcopy(data[idx2]), a=a, b=b,
                            ssd_var=ssd_var, n_steps=n_steps, n_iters=n_iters,
                            resolutions=resolutions, smoothing_sigmas=smoothing_sigmas,
                            delta_phi_threshold=delta_phi_threshold,
                            unit_threshold=unit_threshold, learning_rate=learning_rate,
                            n_jobs=1, vf0=False, inverse=inverse,
                            optim_template=False, data_type=data_type, change_res=change_res,
                            init_resolution=init_resolution, pipe_template=False)
        for i, idx1 in tqdm(enumerate(idx_train), desc='train') for idx2 in idx_train[i:])

    train_metric, train_vector_fields = map(np.concatenate, zip(*train_result))

    train_K = count_K_without_template(train_metric, n_train)
    shape = train_vector_fields.shape[2:]
    train_da, train_db = count_da_db_without_template(train_vector_fields, a, b, shape, n_job, n_train)

    if n_test != 0:
        test_result = Parallel(n_jobs=n_job, temp_folder=joblib_folder)(
            delayed(one_to_one)(data1=copy.deepcopy(data[idx1]), data2=copy.deepcopy(data[idx2]), a=a, b=b,
                                ssd_var=ssd_var, n_steps=n_steps, n_iters=n_iters,
                                resolutions=resolutions, smoothing_sigmas=smoothing_sigmas,
                                delta_phi_threshold=delta_phi_threshold,
                                unit_threshold=unit_threshold, learning_rate=learning_rate,
                                n_jobs=1, vf0=False, inverse=inverse,
                                optim_template=False, data_type=data_type, change_res=change_res,
                                init_resolution=init_resolution, pipe_template=False)
            for idx1 in tqdm(idx_test, desc='test') for idx2 in idx_train)

        test_metric, test_vector_fields = map(np.concatenate, zip(*test_result))
        test_K = np.array(test_metric).reshape(n_test, n_train)

        K = np.zeros((n_train + n_test, n_train + n_test))

        K[np.ix_(idx_train, idx_train)] = train_K
        K[np.ix_(idx_test, idx_train)] = test_K
        K[np.ix_(idx_train, idx_test)] = test_K.T

        return K, train_da, train_db
    else:
        return train_K, train_da, train_db
