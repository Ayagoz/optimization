import os
import rtk

import numpy as np
import copy
import gc
from tqdm import tqdm
# from memory_profiler import profile
from joblib import Parallel, delayed


from RegOptim.optimization.derivatives import to_one_resolution, get_derivative_Lv, get_derivative_v, \
    get_derivative_template, derivatives_of_pipeline_with_template, \
    derivatives_of_pipeline_without_template
from RegOptim.utils import load_nii
from RegOptim.preprocessing import change_resolution
from RegOptim.image_utils import padding
from RegOptim.optimization.pipeline_utils import create_exp_folders, create_template, count_K_pairwise,\
    count_da_db_pairwise

joblib_folder = '~/JOBLIB_TMP_FOLDER/'

# @profile
def pairwise_pipeline_derivatives(reg, inverse):
    in_one_res = to_one_resolution(reg.resulting_vector_fields, reg.resolutions, reg.n_step,
                                   reg.zoom_grid, False, inverse)

    return np.sum(reg.resulting_metric) / len(reg.resolutions), in_one_res


# @profile
def template_pipeline_derivatives(reg, similarity, regularizer, data, template, a, b,
                                  epsilon, shape, vf0, inverse, optim_template, path,
                                  n_jobs, window):
    in_one_res = to_one_resolution(resulting_vector_fields=reg.resulting_vector_fields,
                                   resolutions=reg.resolutions,
                                   n_steps=reg.n_step, zoom_grid=reg.zoom_grid, vf0=vf0, inverse=inverse)

    vf_all_in_one_res = to_one_resolution(resulting_vector_fields=reg.resulting_vector_fields,
                                          resolutions=reg.resolutions,
                                          n_steps=reg.n_step, zoom_grid=reg.zoom_grid,
                                          vf0=False, inverse=inverse)

    regularizer.set_operator(shape)

    if vf0:
        # find Lv(t=1,ndim, img_shape)
        Lvf = regularizer(in_one_res[0])[None]
        # get derivatives of Lv
        Deltas_v = get_derivative_Lv(A=reg.As[-1], v=in_one_res, a=a, b=b)
        dLv_da, dLv_db = Deltas_v[0], Deltas_v[1]
        del Deltas_v
    else:
        Lvf = np.stack([regularizer(in_one_res[i]) for i in range(reg.n_step + 1)], 0)
        Deltas_v = np.array([get_derivative_Lv(A=reg.As[-1], v=vf_all_in_one_res[i], a=a, b=b)
                                                    for i in range(reg.n_step + 1)])
        dLv_da, dLv_db = Deltas_v[:, 0], Deltas_v[:, 1]
        del Deltas_v
    # after that will be change reg so, save everything what you need
    dv_da, dv_db = get_derivative_v(a=a, b=b, reg=copy.deepcopy(reg),
                                    epsilon=epsilon, vf0=vf0, inverse=inverse, data=data,
                                    template=template)

    if optim_template:
        dv_dJ = get_derivative_template(data=data, template=template, n_steps=reg.n_step,
                                        vf_all_in_one_resolution=vf_all_in_one_res,
                                        similarity=similarity, regularizer=regularizer,
                                        vf0=vf0, inverse=inverse, path=path, n_jobs=n_jobs, window=window)
        gc.collect()
        return Lvf, in_one_res, dv_da, dv_db, dLv_da, dLv_db, dv_dJ
    
    gc.collect()
    return Lvf, in_one_res, dv_da, dv_db, dLv_da, dLv_db


# @profile
def one_to_one(data1, data2, a, b, epsilon, ssd_var, n_steps,
               n_iters, resolutions, smoothing_sigmas,
               delta_phi_threshold, unit_threshold, learning_rate, n_jobs,
               vf0, inverse, optim_template, data_type='path', path_to_dJ=None,
               change_res=True, init_resolution=4., pipe_template=True,
               add_padding=False, pad_size=2, window=3):
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
        learning_rate=learning_rate, n_jobs=1)

    if data_type == 'path':
        data1 = load_nii(data1)

    if path_to_dJ is None:
        raise  TypeError('Variable `path` should be initialized.')

    if isinstance(data2, (str, np.str, np.string_, np.unicode_)):
        data2 = load_nii(data2)

    if change_res:
        data1 = np.squeeze(change_resolution(data1, init_resolution, multiple=False))

    if add_padding:
        #should be add by all iterations! (if two times!! fix it by add pad size!)
        data1 = padding(data1, ndim=data1.ndim, pad_size=pad_size, mode='edge')

    # template supposed to be right shape
    # if data2.shape != data1.shape:
    #     data2 = np.squeeze(change_resolution(data2, init_resolution, multiple=False))

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
        gc.collect()
        return template_pipeline_derivatives(reg=reg, similarity=similarity, regularizer=regularizer,
                                             data=data1, template=data2, a=a, b=b,
                                             epsilon=epsilon, shape=data1.shape, vf0=vf0,
                                             inverse=inverse, optim_template=optim_template,
                                             n_jobs=n_jobs, path=path_to_dJ, window=window)

    else:
        gc.collect()
        return pairwise_pipeline_derivatives(reg, inverse)


# @profile
def count_dist_matrix_to_template(data, template, a, b, train_idx, epsilon=0.1, n_job=5, ssd_var=1000.,
                                  n_steps=30, n_iters=(50, 20, 10), resolutions=(4, 2, 1),
                                  smoothing_sigmas=(2., 1., 0.), delta_phi_threshold=0.1,
                                  unit_threshold=0.01, learning_rate=0.01, exp_path=None,
                                  change_res=True, init_resolution=4, data_type='path', vf0=True,
                                  inverse=False, optim_template=True, add_padding=False, pad_size=2,
                                  window=3):
    n = len(data)
    if exp_path is None:
        raise TypeError('exp path cannot be None')


    path_to_dJ = os.path.join(exp_path, 'derivative/')
    path = [path_to_dJ + 'template_dev_' + data[i].split('/nii/')[0].split('/data/')[-1] + '.pkl'
                                                                                for i in range(n)]

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
                                                                   path=path[i],
                                                                   change_res=change_res,
                                                                   init_resolution=init_resolution,
                                                                   pipe_template=True,
                                                                   add_padding=add_padding,
                                                                   pad_size=pad_size, window=window)
                                                               for i in tqdm(range(n), desc="registration"))

    if optim_template:
        gc.collect()
        return derivatives_of_pipeline_with_template(result, train_idx, n)

    else:
        gc.collect()
        return derivatives_of_pipeline_without_template(result, n)




def count_dist_matrix(data, y, a, b, idx_train=None, idx_test=None, n_job=5, ssd_var=1000., n_steps=20,
                      n_iters=(50, 20, 10), resolutions=(4, 2, 1), smoothing_sigmas=(2., 1., 0.),
                      delta_phi_threshold=0.1, unit_threshold=0.01, learning_rate=0.01,
                      inverse=True, change_res=True, init_resolution=4, data_type='path', add_padding=False,
                      pad_size=2):
    if idx_train is None:
        idx_train = np.arange(len(data))

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
                            init_resolution=init_resolution, pipe_template=False,
                            add_padding=add_padding, pad_size=pad_size)
        for i, idx1 in tqdm(enumerate(idx_train), desc='train') for idx2 in idx_train[i:])

    train_metric, train_vector_fields = map(np.concatenate, zip(*train_result))

    train_K = count_K_pairwise(train_metric, n_train)
    shape = train_vector_fields.shape[2:]
    train_da, train_db = count_da_db_pairwise(train_vector_fields, a, b, shape, n_job, n_train)

    if n_test != 0:
        test_result = Parallel(n_jobs=n_job, temp_folder=joblib_folder)(
            delayed(one_to_one)(data1=data[idx1], data2=data[idx2], a=a, b=b,
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
