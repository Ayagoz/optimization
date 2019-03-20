from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from RegOptim.utils import import_func
from RegOptim.optimization.pipeline import count_dist_matrix_to_template
from RegOptim.ml.ml_utils import find_pipeline_params, adam_step
from RegOptim.optimization.pipeline_utils import update_template, preprocess_delta_template, \
    optim_template_strategy, pad_template_data_after_loop
from RegOptim.image_utils import check_for_padding

import pandas as pd
import numpy as np
import time
import os
import gc


def create_kwargs(params, data, template, a, b, idx_out_train, optim_template, add_padding, pad_size):
    kwargs = {'file_type': params['load_params']['file_type'],
              'exp_path': os.path.join(params['path_to_exp'], params['experiment_name']), 'data': data,
              'template': template, 'ndim': params['ndim'], 'n_jobs': params['n_jobs']}

    kwargs.update(params['pipeline_optimization_params'])

    if params.get('test_idx'):
        kwargs['test_idx'] = params['test_idx']

    kwargs['train_idx'] = idx_out_train

    kwargs['resolution'] = params['resolution']

    kwargs['random_state'] = params['random_state']
    kwargs['window'] = params['window']

    kwargs['pipe_template'] = params['pipe_template']

    # updated params
    kwargs['a'], kwargs['b'] = a, b
    kwargs['optim_template'] = optim_template
    kwargs['add_padding'] = add_padding
    kwargs['pad_size'] = pad_size
    kwargs['params_der'] = params['derivative_J_params']

    return kwargs


def pipeline_main_loop_template_only(data, template, y, idx_out_train, idx_out_test,
                                     experiment_path, path_to_template, template_name,
                                     pipeline_params, pad_size):
    # initialize learning rate changing strategy
    lr_change = import_func(**pipeline_params['lr_type'])
    lr_params = pipeline_params['lr_change_params'][pipeline_params['lr_type']['func']]
    template_updates = pipeline_params['template_updates']
    reg = pipeline_params['pipeline_optimization_params']
    lr =  template_updates['lr']
    it = 1
    a = pipeline_params['pipeline_optimization_params']['a']
    b = pipeline_params['pipeline_optimization_params']['b']

    # start param for optimization
    optim_template = pipeline_params['start_optim_template']
    # create a resulting data frame
    if pipeline_params['kernel']:
        results = pd.DataFrame(columns=["iter", "a", "b", "kernel gamma", "LR C ", "train_score",
                                        "train_loss", "test_score", "test_loss", "one_loop_time",
                                        "pad_size"])
    else:
        results = pd.DataFrame(columns=["iter", "a", "b", "LR C ", "train_score", "train_loss",
                                        "test_score", "test_loss", "one_loop_time", "pad_size"])

    print('For params a {} and b {}'.format(a, b))
    test_score_prediction = import_func(**pipeline_params['prediction_func'])
    count_grads_template = import_func(**pipeline_params['count_grads_template'])



    add_padding = reg['add_padding']
    print('add_pad', add_padding)
    print('pad size', pad_size)
    kwargs = create_kwargs(pipeline_params, data, template, a, b, idx_out_train, True, add_padding, pad_size)
    y_out_train = y[idx_out_train]

    while it < pipeline_params['Number_of_iterations']:

        st = time.time()

        print('For iter {}'.format(int(it)))

        K, da, db, dJ = count_dist_matrix_to_template(**kwargs)

        K_path = os.path.join(os.path.join(pipeline_params['path_to_exp'], pipeline_params['experiment_name']),
                              'kernel')

        np.savez(os.path.join(K_path, 'kernel_' + str(it) + '.npz'), K)

        K_out_train = K[np.ix_(idx_out_train, idx_out_train)]

        best_params = find_pipeline_params(
            K_out_train, y_out_train, pipeline_params['ml_params'], pipeline_params['n_jobs'],
            random_state=pipeline_params['random_state'], scaled=pipeline_params['scaled'], scoring='roc_auc',
            n_splits=pipeline_params['n_splits'], kernel=pipeline_params['kernel']
        )

        test_score, test_loss = test_score_prediction(K=K, y=y, idx_train=idx_out_train, idx_test=idx_out_test,
                                                      params=best_params)

        grads_da, grads_db, grads_dJ, train_score, train_loss = count_grads_template(
            exp_K=K_out_train, y=y_out_train, da=da[np.ix_(idx_out_train, idx_out_train)],
            db=db[np.ix_(idx_out_train, idx_out_train)], dJ=dJ, params=best_params,
            n_splits=pipeline_params['n_splits'], ndim=pipeline_params['ndim'],
            random_state=pipeline_params['random_state'], kernel=pipeline_params['kernel']
        )

        delta = preprocess_delta_template(grads_dJ, axis=template_updates['template_axis'],
                                          contour_color=template_updates['color'],
                                          width=template_updates['width'],
                                          ndim=pipeline_params['ndim'])

        template_name = template_name.split('_')[0] + '_' + str(it) + '.nii'
        template = update_template(template, path_to_template, template_name,
                                   delta, lr)

        if check_for_padding(template):
            template = pad_template_data_after_loop(template.copy(),
                                                    os.path.join(path_to_template, template_name),
                                                    pad_size=reg['pad_size'], ndim=pipeline_params['ndim'])

            kwargs['add_padding'] = True
            kwargs['pad_size'] += reg['pad_size']

        if add_padding:
            pipeline_params['pipeline_optimization_params']['add_padding'] = add_padding

        if pipeline_params['kernel']:
            results.loc[it - 1] = [it, a, b, best_params['kernel__gamma'],
                                   best_params['ml__C'], train_score, train_loss, test_score,
                                   test_loss, time.time() - st, pad_size]
        else:
            results.loc[it - 1] = [it, a, b, best_params['ml__C'], train_score,
                                   train_loss, test_score, test_loss, time.time() - st, pad_size]

        it += 1

        lr = lr_change(prev_lr=lr, it=it, step=lr_params['step'], decay=lr_params['decay'])

        results.to_csv(os.path.join(experiment_path, 'results.csv'))

        gc.collect()


def pipeline_main_loop(data, template, y, idx_out_train, idx_out_test,
                       experiment_path, path_to_template, template_name,
                       pipeline_params, pad_size):
    # initialize learning rate changing strategy
    lr_change = import_func(**pipeline_params['lr_type'])
    lr_params = pipeline_params['lr_change_params'][pipeline_params['lr_type']['func']]
    lr = lr_params['init_lr']
    it = 1
    a_it = [0., pipeline_params['pipeline_optimization_params']['a']]
    b_it = [0., pipeline_params['pipeline_optimization_params']['b']]

    mta, vta = 0., 0.
    mtb, vtb = 0., 0.
    # start param for optimization
    optim_template = pipeline_params['start_optim_template']
    # create a resulting data frame
    if pipeline_params['kernel']:
        results = pd.DataFrame(columns=["iter", "a", "b", "kernel gamma", "LR C ", "train_score",
                                        "train_loss", "test_score", "test_loss", "one_loop_time",
                                        "pad_size"])
    else:
        results = pd.DataFrame(columns=["iter", "a", "b", "LR C ", "train_score", "train_loss",
                                        "test_score", "test_loss", "one_loop_time", "pad_size"])

    while (abs(a_it[-1] - a_it[-2]) + abs(b_it[-1] - b_it[-2])) > 1e-10 or \
            it < pipeline_params['Number_of_iterations']:

        st = time.time()

        print('For iter {}'.format(int(it)))
        print('For params a {} and b {}'.format(a_it[-1], b_it[-1]))

        if optim_template:

            template, best_params, grads_da, grads_db, train_score, test_score, train_loss, test_loss, add_padding, pad_size = optimize_template_step(
                data.copy(), template, y.copy(), a_it[-1], b_it[-1], idx_out_train, idx_out_test,
                pipeline_params, template_name, path_to_template, pad_size, it
            )
            if add_padding:
                pipeline_params['pipeline_optimization_params']['add_padding'] = add_padding

        else:
            best_params, grads_da, grads_db, train_score, test_score, train_loss, test_loss = optimize_a_b_step(
                data.copy(), template, y.copy(), a_it[-1], b_it[-1], idx_out_train, idx_out_test,
                pipeline_params, pad_size
            )

        adam_grad_da, mta, vta = adam_step(grads_da, mta, vta, it)
        adam_grad_db, mtb, vtb = adam_step(grads_db, mtb, vtb, it)

        if pipeline_params['kernel']:
            results.loc[it - 1] = [it, a_it[-1], b_it[-1], best_params['kernel__gamma'],
                                   best_params['ml__C'], train_score, train_loss, test_score,
                                   test_loss, time.time() - st, pad_size]
        else:
            results.loc[it - 1] = [it, a_it[-1], b_it[-1], best_params['ml__C'], train_score,
                                   train_loss, test_score, test_loss, time.time() - st, pad_size]

        print("one loop time: ", time.time() - st)

        a_it += [a_it[-1] - lr * adam_grad_da]
        b_it += [b_it[-1] - lr * adam_grad_db]

        it += 1

        lr = lr_change(prev_lr=lr, it=it, step=lr_params['step'], decay=lr_params['decay'])

        optim_template = optim_template_strategy(it, pipeline_params['step_size_optim_template'])

        results.to_csv(os.path.join(experiment_path, 'results.csv'))

        gc.collect()


def optimize_template_step(data, template, y, a, b, idx_out_train, idx_out_test,
                           pipeline_params, template_name, path_to_template,
                           pad_size, it):
    test_score_prediction = import_func(**pipeline_params['prediction_func'])
    count_grads_template = import_func(**pipeline_params['count_grads_template'])

    template_updates = pipeline_params['template_updates']
    reg = pipeline_params['pipeline_optimization_params']

    add_padding = reg['add_padding']
    y_out_train = y[idx_out_train]

    kwargs = create_kwargs(pipeline_params, data, template, a, b, idx_out_train, True, add_padding, pad_size)
    K, da, db, dJ = count_dist_matrix_to_template(**kwargs)

    K_path = os.path.join(os.path.join(pipeline_params['path_to_exp'], pipeline_params['experiment_name']), 'kernel')
    np.savez(os.path.join(K_path, 'kernel_' + str(it) + '.npz'), K)

    K_out_train = K[np.ix_(idx_out_train, idx_out_train)]

    best_params = find_pipeline_params(
        K_out_train, y_out_train, pipeline_params['ml_params'], pipeline_params['n_jobs'],
        random_state=pipeline_params['random_state'], scaled=pipeline_params['scaled'], scoring='roc_auc',
        n_splits=pipeline_params['n_splits'], kernel=pipeline_params['kernel']
    )

    test_score, test_loss = test_score_prediction(K=K, y=y, idx_train=idx_out_train, idx_test=idx_out_test,
                                                  params=best_params)

    grads_da, grads_db, grads_dJ, train_score, train_loss = count_grads_template(
        exp_K=K_out_train, y=y_out_train, da=da[np.ix_(idx_out_train, idx_out_train)],
        db=db[np.ix_(idx_out_train, idx_out_train)], dJ=dJ, params=best_params,
        n_splits=pipeline_params['n_splits'], ndim=pipeline_params['ndim'],
        random_state=pipeline_params['random_state'], kernel=pipeline_params['kernel']
    )

    delta = preprocess_delta_template(grads_dJ, axis=template_updates['template_axis'],
                                      contour_color=template_updates['color'],
                                      width=template_updates['width'],
                                      ndim=pipeline_params['ndim'])

    template_name = template_name.split('_')[0] + '_' + str(it) + '.nii'
    template = update_template(template, path_to_template, template_name,
                               delta, template_updates['lr'])

    if check_for_padding(template):
        template = pad_template_data_after_loop(template.copy(),
                                                os.path.join(path_to_template, template_name),
                                                pad_size=reg['pad_size'], ndim=pipeline_params['ndim'])

        add_padding = True
        pad_size += reg['pad_size']

    gc.collect()
    return template, best_params, grads_da, grads_db, train_score, test_score, train_loss, test_loss, add_padding, pad_size


def optimize_a_b_step(data, template, y, a, b, idx_out_train, idx_out_test,
                      pipeline_params, pad_size):
    test_score_prediction = import_func(**pipeline_params['prediction_func'])
    count_grads_a_b = import_func(**pipeline_params['count_grads_a_b'])
    reg = pipeline_params['pipeline_optimization_params']

    y_out_train = y[idx_out_train]
    kwargs = create_kwargs(pipeline_params, data, template, a, b, idx_out_train, False,
                           reg['add_padding'], pad_size)
    K, da, db = count_dist_matrix_to_template(**kwargs)

    K_out_train = K[np.ix_(idx_out_train, idx_out_train)]

    best_params = find_pipeline_params(
        K_out_train, y_out_train, pipeline_params['ml_params'], pipeline_params['n_jobs'],
        random_state=pipeline_params['random_state'], scaled=pipeline_params['scaled'], scoring='roc_auc',
        n_splits=pipeline_params['n_splits'], kernel=pipeline_params['kernel']
    )

    test_score, test_loss = test_score_prediction(K=K, y=y, idx_train=idx_out_train, idx_test=idx_out_test,
                                                  params=best_params)

    grads_da, grads_db, train_score, train_loss = count_grads_a_b(
        exp_K=K_out_train, y=y_out_train, da=da[np.ix_(idx_out_train, idx_out_train)],
        db=db[np.ix_(idx_out_train, idx_out_train)],
        params=best_params, n_splits=pipeline_params['n_splits'], random_state=pipeline_params['random_state']
    )

    gc.collect()

    return best_params, grads_da, grads_db, train_score, test_score, train_loss, test_loss
