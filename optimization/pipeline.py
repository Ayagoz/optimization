import os
import rtk

import numpy as np
import copy
import gc
from tqdm import tqdm
import ntpath
# from memory_profiler import profile
from joblib import Parallel, delayed

from RegOptim.optimization.derivatives import to_one_resolution, get_derivative_Lv, get_derivative_v, \
    get_derivative_template, derivatives_of_pipeline_with_template, \
    derivatives_of_pipeline_without_template
from RegOptim.utils import load_nii
from RegOptim.preprocessing import change_resolution
from RegOptim.image_utils import padding
from RegOptim.optimization.pipeline_utils import count_K_pairwise, count_da_db_pairwise, check_correctness_of_dict

joblib_folder = '~/JOBLIB_TMP_FOLDER/'


# @profile
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
        gc.collect()
        return Lvf, in_one_res, dv_da, dv_db, dLv_da, dLv_db, dv_dJ

    gc.collect()
    return Lvf, in_one_res, dv_da, dv_db, dLv_da, dLv_db


# @profile
def one_to_one(data1, data2, path_to_dJ, **kwargs):
    kwargs = check_correctness_of_dict(kwargs)

    reg_params = kwargs['reg_params']
    a, b = kwargs['a'], kwargs['b']
    # registration
    similarity = rtk.SSD(variance=kwargs['ssd_var'])
    regularizer = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b)

    reg = rtk.LDDMM(regularizer=regularizer, similarity=similarity, n_jobs=1, **reg_params)

    if kwargs['data_type'] == 'path':
        data1 = load_nii(data1)

    if kwargs['path_to_dJ'] is None:
        raise TypeError('Variable `path` should be initialized.')

    if isinstance(kwargs['data2'], (str, np.str, np.string_, np.unicode_)):
        data2 = load_nii(kwargs['data2'])

    if kwargs['change_res']:
        data1 = np.squeeze(change_resolution(data1, kwargs['init_resolution'], multiple=False))

    if kwargs['add_padding']:
        # should be add by all iterations! (if two times!! fix it by add pad size!)
        data1 = padding(data1, ndim=data1.ndim, pad_size=kwargs['pad_size'], mode='edge')
    kwargs['shape'] = data1.shape
    kwargs['ndim'] = data1.ndim
    # template supposed to be right shape
    # if data2.shape != data1.shape:
    #     data2 = np.squeeze(change_resolution(data2, init_resolution, multiple=False))

    # inverse means that we would like to find path from X to template
    # it means that we would like to find vf[-1] ~ vf^(-1)[0]

    if kwargs['inverse']:
        # first image = fixed, second = moving
        reg.set_images(rtk.ScalarImage(data=data2), rtk.ScalarImage(data=data1))
    else:
        reg.set_images(rtk.ScalarImage(data=data1), rtk.ScalarImage(data=data2))

    reg.execute()

    # get resulting vector field in one resolution
    # if vf0=True, get (1, ndim, img_shape), else get (t, ndim, img_shape)

    if kwargs['pipe_template']:
        gc.collect()
        return template_pipeline_derivatives(reg=reg, similarity=similarity, regularizer=regularizer,
                                             data=data1, template=data2, a=a, b=b, epsilon=kwargs['epsilon'],
                                             shape=data1.shape, inverse=kwargs['inverse'],
                                             optim_template=kwargs['optim_template'], n_jobs=kwargs['n_jobs'],
                                             path=path_to_dJ, window=kwargs['window']
                                             )

    else:
        gc.collect()
        return pairwise_pipeline_derivatives(reg, kwargs['inverse'])


# @profile
def count_dist_matrix_to_template(**kwargs):
    '''
    kwargs should consist of:
        - exp_path (experiment path, path to folder of experiment)
        - data ( full path or array of images you would like to registrate)
        - n_jobs (number of thread to parallel along subjects(data))
        - template (template on which you want to registrate)
        - jobllib_folder (not necessary, just to prevent some bugs)
        - train_idx (if you want to count derivative of dJ)
        -
    '''
    exp_path = kwargs['exp_path']
    data = kwargs['data']

    if exp_path is None:
        raise TypeError('exp path cannot be None')

    n = len(data)

    path_to_dJ = os.path.join(exp_path, 'derivative/')
    path = [os.path.join(path_to_dJ, ntpath.basename(data[i]).split('.')[0] + '.npz') for i in range(n)]

    result = Parallel(n_jobs=kwargs['n_jobs'], temp_folder=kwargs.get('joblib_folder', '~/JOBLIB_TMP_FOLDER/'))(
        delayed(one_to_one)(
            data1=data[i], data2=kwargs['template'], path_to_dJ=path[i], **kwargs
        )
        for i in tqdm(range(n), desc="registration")
    )

    if kwargs['optim_template']:
        gc.collect()
        return derivatives_of_pipeline_with_template(result=result, train_idx=kwargs.get('train_idx'), n_total=n,
                                                     img_shape=kwargs['shape']
                                                     )

    else:
        gc.collect()
        return derivatives_of_pipeline_without_template(result, n)


def count_dist_matrix(**kwargs):
    idx_train, idx_test = kwargs['idx_train'], kwargs['idx_test']
    data = kwargs['data']
    n_jobs = kwargs['n_jobs']

    if idx_train is None:
        idx_train = np.arange(len(data))

    n_train = len(idx_train)

    if idx_test is None:
        idx_test = []

    n_test = len(idx_test)

    train_result = Parallel(n_jobs=n_jobs, temp_folder=joblib_folder)(
        delayed(one_to_one)(
            data1=copy.deepcopy(data[idx1]), data2=copy.deepcopy(data[idx2]), **kwargs
        )
        for i, idx1 in tqdm(enumerate(idx_train), desc='train') for idx2 in idx_train[i:]
    )

    train_metric, train_vector_fields = map(np.concatenate, zip(*train_result))

    train_K = count_K_pairwise(train_metric, n_train)

    train_da, train_db = count_da_db_pairwise(train_vector_fields, kwargs['a'], kwargs['b'],
                                              kwargs['shape'], n_jobs, n_train)

    if n_test != 0:
        test_result = Parallel(n_jobs=n_jobs, temp_folder=joblib_folder)(
            delayed(one_to_one)(
                data1=copy.deepcopy(data[idx1]), data2=copy.deepcopy(data[idx2]),**kwargs)
            for idx1 in tqdm(idx_test, desc='test') for idx2 in idx_train
        )

        test_metric, test_vector_fields = map(np.concatenate, zip(*test_result))
        test_K = np.array(test_metric).reshape(n_test, n_train)

        K = np.zeros((n_train + n_test, n_train + n_test))

        K[np.ix_(idx_train, idx_train)] = train_K
        K[np.ix_(idx_test, idx_train)] = test_K
        K[np.ix_(idx_train, idx_test)] = test_K.T

        return K, train_da, train_db
    else:
        return train_K, train_da, train_db
