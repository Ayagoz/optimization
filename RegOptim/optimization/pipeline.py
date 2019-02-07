import rtk

import copy
import gc

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import rtk
from RegOptim.image_utils import padding
from RegOptim.optimization.derivatives import template_pipeline_derivatives, pairwise_pipeline_derivatives
from RegOptim.optimization.metrics import count_K_pairwise, count_da_db_pairwise, count_dJ, \
    count_K_to_template, count_da_db_to_template
from RegOptim.optimization.pipeline_utils import get_shape
from RegOptim.preprocessing import change_resolution
from RegOptim.utils import load_nii

joblib_folder = '~/JOBLIB_TMP_FOLDER/'


def derivatives_of_pipeline_without_template(result, n_total):
    Lvfs, vfs, dv_da, dv_db, dL_da, dL_db = map(np.concatenate, zip(*result))
    K = count_K_to_template(Lvfs, vfs, n_total)
    da, db = count_da_db_to_template(Lvfs, vfs, dv_da, dv_db, dL_da, dL_db, n_total)
    gc.collect()
    return K, da, db


def derivatives_of_pipeline_with_template(result, train_idx, n_total, img_shape):
    n_train = len(train_idx)
    Lvfs, vfs, dv_da, dv_db, dL_da, dL_db, dv_dJ = map(np.concatenate, zip(*result))
    shape = np.array(Lvfs).shape[2:]
    ndim = len(shape)

    # (t=1, ndim, img_shape)-for v and (img_shape,)- for template img J

    shape_res = (ndim,) + img_shape + img_shape

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


def one_to_one(data1, data2,  **kwargs):
    a, b = kwargs['a'], kwargs['b']
    # registration
    similarity = rtk.SSD(variance=kwargs['ssd_var'])
    regularizer = rtk.BiharmonicRegularizer(convexity_penalty=a, norm_penalty=b)

    reg = rtk.LDDMM(regularizer=regularizer, similarity=similarity, n_jobs=1, **kwargs['reg_params'])

    if kwargs['file_type'] == 'path':
        data1 = load_nii(data1)

    if isinstance(data2, (str, np.str, np.string_, np.unicode_)):
        data2 = load_nii(data2)

    if kwargs['change_res']:
        data1 = np.squeeze(change_resolution(data1, kwargs['init_resolution'], multiple=False))

    if kwargs['add_padding']:
        # should be add by all iterations! (if two times!! fix it by add pad size!)
        data1 = padding(data1, ndim=data1.ndim, pad_size=kwargs['pad_size'], mode='edge')
    kwargs['shape'] = data1.shape
    kwargs['ndim'] = data1.ndim


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
                                              window=kwargs['window']
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
        - reg_params (params of LDDMM)
        - epsilon, optim_template, inverse, n_jobs, window, add_padding, pad_size ...
    '''
    exp_path = kwargs['exp_path']
    data = kwargs['data']

    if exp_path is None:
        raise TypeError('exp path cannot be None')

    n = len(data)

    # path_to_dJ = os.path.join(exp_path, 'derivative/')
    # path = [os.path.join(path_to_dJ, ntpath.basename(data[i]).split('.')[0] + '.npz') for i in range(n)]

    result = Parallel(n_jobs=kwargs['n_jobs'], temp_folder=kwargs.get('joblib_folder', '~/JOBLIB_TMP_FOLDER/'))(
        delayed(one_to_one)(
            data1=data[i], data2=kwargs['template'], **kwargs
        )
        for i in tqdm(range(n), desc="registration")
    )

    if not kwargs.get('shape'):
        kwargs['shape'] = get_shape(kwargs['template'])

    if kwargs['optim_template']:
        gc.collect()
        return derivatives_of_pipeline_with_template(result=result, train_idx=kwargs['train_idx'], n_total=n,
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
                data1=copy.deepcopy(data[idx1]), data2=copy.deepcopy(data[idx2]), **kwargs)
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
