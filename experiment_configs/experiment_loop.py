from RegOptim.experiment_configs.utils import import_func
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


def pipeline_main_loop(data, template, y, idx_out_train, idx_out_test,
                       experiment_path, path_to_template, template_name,
                       pipeline_params, pad_size ):

    #split target
    y_out_test, y_out_train = y[idx_out_test], y[idx_out_train]

    # initialize learning rate changing strategy
    lr_change = import_func(**pipeline_params['lr_type'])
    #
    count_grads_a_b = import_func(**pipeline_params['count_grads_a_b'])
    count_grads_template = import_func(**pipeline_params['count_grads_template'])
    test_score_prediction = import_func(**pipeline_params['prediction_func'])
    #import registration params
    reg = pipeline_params['registration_params']

    #init pathes
    template_updates = pipeline_params['template_updates']
    #initialize all other numerical params
    n_jobs = pipeline_params['n_jobs']
    resolution = pipeline_params['resolution']
    lr_params = pipeline_params['lr_change_params'][pipeline_params['lr_type']]
    lr = lr_params['init_lr']
    it = 1
    a_it = [0., pipeline_params['a_b'][0]]
    b_it = [0., pipeline_params['a_b'][1]]

    mta, vta = 0., 0.
    mtb, vtb = 0., 0.
    #start param for optimization
    optim_template = pipeline_params['start_optim_template']
    #create a resulting data frame
    if pipeline_params['kernel']:
        results = pd.DataFrame(columns=["iter", "a", "b", "kernel gamma", "LR C ", "train_score",
                                        "train_loss", "test_score", "test_loss", "one_loop_time",
                                        "pad_size"])
    else:
        results = pd.DataFrame(columns=["iter", "a", "b", "LR C ", "train_score", "train_loss",
                                        "test_score", "test_loss", "one_loop_time", "pad_size"])


    while (abs(a_it[-1] - a_it[-2]) + abs(b_it[-1] - b_it[-2])) > 1e-10 or \
            it > pipeline_params['Number_of_iterations']:

        st = time.time()

        print 'For iter {}'.format(int(it))
        print 'For params a {} and b {}'.format(a_it[-1], b_it[-1])

        if optim_template:
            K, da, db, dJ = count_dist_matrix_to_template(data, template, a_it[-1], b_it[-1],
                                                          idx_out_train, epsilon=reg['epsilon'],
                                                          n_job=reg['n_job'], ssd_var=reg['ssd_var'],
                                                          n_steps=reg['n_steps'],
                                                          n_iters=reg['n_iters'],
                                                          resolutions=reg['resolutions'],
                                                          smoothing_sigmas=reg['smoothing_sigmas'],
                                                          delta_phi_threshold=reg['delta_phi_threshold'],
                                                          unit_threshold=reg['unit_threshold'],
                                                          learning_rate=reg['learning_rate'],
                                                          change_res=reg['change_res'],
                                                          init_resolution=resolution,
                                                          exp_path=experiment_path,
                                                          data_type='path', vf0=reg['vf0'],
                                                          inverse=reg['inverse'],
                                                          optim_template=True,
                                                          add_padding=reg['add_padding'],
                                                          pad_size=pad_size,
                                                          window=pipeline_params['window'])

        else:
            K, da, db = count_dist_matrix_to_template(data, template, a_it[-1], b_it[-1],
                                                      idx_out_train, epsilon=reg['epsilon'],
                                                      n_job=reg['n_job'], ssd_var=reg['ssd_var'],
                                                      n_steps=reg['n_steps'],
                                                      n_iters=reg['n_iters'],
                                                      resolutions=reg['resolutions'],
                                                      smoothing_sigmas=reg['smoothing_sigmas'],
                                                      delta_phi_threshold=reg['delta_phi_threshold'],
                                                      unit_threshold=reg['unit_threshold'],
                                                      learning_rate=reg['learning_rate'],
                                                      change_res=reg['change_res'],
                                                      init_resolution=resolution,
                                                      exp_path=experiment_path,
                                                      data_type='path', vf0=reg['vf0'],
                                                      inverse=reg['inverse'],
                                                      optim_template=False,
                                                      add_padding=reg['add_padding'],
                                                      pad_size=pad_size,
                                                      window=pipeline_params['window'])

        K_out_train = K[np.ix_(idx_out_train, idx_out_train)]

        best_params = find_pipeline_params(K_out_train, y_out_train, pipeline_params['ml_params'],
                                           pipeline_params['n_jobs'],
                                           random_state=pipeline_params['random_state'],
                                           scaled=pipeline_params['scaled'], scoring='roc_auc',
                                           n_splits=pipeline_params['n_splits'],
                                           kernel=pipeline_params['kernel'])



        if optim_template:
            grads_da, grads_db, grads_dJ, train_score, train_loss  = count_grads_template(K_out_train,
                                                        y_out_train,
                                                        da[np.ix_(idx_out_train, idx_out_train)],
                                                        db[np.ix_(idx_out_train, idx_out_train)],
                                                        best_params, dJ, pipeline_params['scaled'],
                                                        with_template=True,
                                                        n_splits=pipeline_params['n_splits'],
                                                        ndim=pipeline_params['ndim'],
                                                        random_state=pipeline_params['random_state'],
                                                        kernel=pipeline_params['kernel'])
        else:
            grads_da, grads_db, train_score, train_loss = count_grads_a_b(K_out_train, y_out_train,
                                                          da[np.ix_(idx_out_train, idx_out_train)],
                                                          db[np.ix_(idx_out_train, idx_out_train)],
                                                          best_params, None, pipeline_params['scaled'],
                                                          with_template=False,
                                                          n_splits=pipeline_params['n_splits'],
                                                          ndim=pipeline_params['ndim'],
                                                          random_state=pipeline_params['random_state'],
                                                          kernel=pipeline_params['kernel'])

        adam_grad_da, mta, vta = adam_step(grads_da, mta, vta, it)
        adam_grad_db, mtb, vtb = adam_step(grads_db, mtb, vtb, it)

        test_score, test_loss = test_score_prediction(K, y, idx_out_train, idx_out_test,
                                                      best_params, pipeline_params['scaled'])

        if pipeline_params['kernel']:
            results.loc[it - 1] = [it, a_it[-1], b_it[-1], best_params['kernel__gamma'],
                                   best_params['ml__C'], train_score, train_loss, test_score,
                                   test_loss, time.time() - st, pad_size]
        else:
            results.loc[it - 1] = [it, a_it[-1], b_it[-1], best_params['ml__C'], train_score,
                                   train_loss, test_score, test_loss, time.time() - st, pad_size]

        print "one loop time: ", time.time() - st

        a_it += [a_it[-1] - lr * adam_grad_da]
        b_it += [b_it[-1] - lr * adam_grad_db]

        if optim_template:
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

                reg['add_padding'] = True
                # was with this if, do i really need it?
                # if exp_data['load_type'] == 'path':
                pad_size += reg['pad_size']

        it += 1

        lr = lr_change(prev_lr=lr, it=it, step=lr_params['step'], decay=lr_params['decay'])

        optim_template = optim_template_strategy(it, pipeline_params['step_size_optim_template'])

        results.to_csv(os.path.join(experiment_path, 'results.csv'))

        gc.collect()