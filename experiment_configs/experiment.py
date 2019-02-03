from RegOptim.image_utils import check_for_padding
from RegOptim.utils import get_subset
from RegOptim.utils import save_params, import_func
from RegOptim.experiment_configs.experiment_loop import pipeline_main_loop
from RegOptim.optimization.pipeline_utils import create_exp_folders, create_template, \
    pad_template_data_after_loop

import numpy as np
import json
import os

from sklearn.model_selection import StratifiedShuffleSplit


def metric_learning_to_template(PATH):
    # path to experiment config
    print "START EXPERIMENT"
    pipeline_params = json.load(open(PATH, 'r'))
    # extract params to shorter usage
    path = pipeline_params['path']

    random_state = pipeline_params['random_state']
    exp_data = pipeline_params['experiment_data']
    experiment_name = exp_data['data_type'] + '_resolution' + str(pipeline_params['resolution']) + \
                      '_rs' + str(random_state)
    experiment_path = os.path.join(path['path_to_exp'], experiment_name)
    path_to_template = os.path.join(experiment_path, 'templates/')
    template_name = 'template_0.nii'

    load_data = import_func(**pipeline_params['load_func'])
    # create folder and path
    create_exp_folders(experiment_path, params=pipeline_params)

    print 'experiment name: ', experiment_name

    load_params = {'path_to_data': path['path_to_data'],
                   'file_type': pipeline_params['experiment_data']['file_type'],
                   'target_type': pipeline_params['experiment_data']['target_type']}

    if path.get('path_to_meta'):
        load_params.update({'path_to_meta':pipeline_params['path']['path_to_meta']})
        load_params.update({'data_type': pipeline_params['experiment_data']['data_type']})

    data, y = load_data(**load_params)

    if pipeline_params['subset'] is not None:
        data, y = get_subset(data, y, pipeline_params['subset'], pipeline_params['random_state'])
    print "Data size: ", data.shape, " target mean: ", y.mean()

    # create splits for (train+val) and test
    idx_out_train, idx_out_test = list(StratifiedShuffleSplit(n_splits=1, test_size=0.3,
                                                              random_state=random_state).split(
        np.arange(len(data)), y))[0]

    splits = {'train_val': idx_out_train.tolist(), 'test': idx_out_test.tolist()}
    save_params(experiment_path, 'splits_indices', splits)

    # create template on train data and save it
    template = create_template(path_to_data=data, train_idx=idx_out_train,
                               path_to_template=os.path.join(experiment_path, 'templates/'),
                               template_name=template_name, resolution=pipeline_params['resolution'],
                               load_func_template=pipeline_params['load_func_template'])

    # check if template needs padding
    if check_for_padding(template):
        template = pad_template_data_after_loop(template.copy(),
                                                os.path.join(path_to_template, template_name),
                                                pad_size=pipeline_params['registration_params']['pad_size'],
                                                ndim=pipeline_params['ndim'])

        pipeline_params['registration_params']['add_padding'] = True
        pad_size = pipeline_params['registration_params']['pad_size']
    else:
        pad_size = 0

    pipeline_main_loop(data=data, template=template, y=y, idx_out_train=idx_out_train,
                       idx_out_test=idx_out_test, experiment_path=experiment_path,
                       path_to_template=path_to_template, template_name=template_name,
                       pipeline_params=pipeline_params, pad_size=pad_size)

    print "FINISHED"
