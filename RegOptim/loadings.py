import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from RegOptim.utils import load_nii, balanced_fold


def load_data_from_one_dir(path_to_data, file_type, target_type):
    data, target = [], []
    subj = sorted(os.listdir(path_to_data))

    for subj in tqdm(subj, desc='loading subjects'):
        for k, i in target_type.items():
            if k in subj:
                target += [i]
                target += [i]

        path_to_subj = os.path.join(path_to_data, subj)
        if file_type == 'nii':
            data += [load_nii(path_to_subj, file_type)]
        if file_type == 'path':
            data += [path_to_subj]
    print('data shape {}, target shape {}'.format(np.array(data).shape, np.array(target).shape))
    return np.array(data), np.array(target)


def load_data_from_dirs(path_to_data, data_type, file_type):
    data = []
    subj_names = []
    path = sorted(os.listdir(path_to_data))

    for subj in tqdm(path, desc='loading subjects'):
        path_to_subj = os.path.join(os.path.join(path_to_data, subj), 'nii')

        image = load_nii(path_to_subj, data_type)
        if image.sum() != 0:
            if file_type == 'nii':
                data.append(image)
            if file_type == 'path':
                data.append(os.path.join(path_to_subj, data_type + '.nii'))
            subj_names.append(subj)

    return data, subj_names


def load_images(path):
    images = []
    for subj in path:
        images.append(load_nii(subj, None))
    return np.array(images)


def load_data_from_nii(global_path, data_type, file_type):
    if data_type is not None:
        data, subj_names = load_data_from_dirs(global_path, data_type, file_type)
        return np.array(data), np.array(subj_names)
    else:
        return load_images(global_path)




def load_target(path, idx, target_type):
    meta = pd.read_csv(path)[['SubjID', target_type]]
    target = []
    for one in idx:
        if isinstance(one, (str, np.str, np.string_, np.unicode_)):
            target.append(int(meta[meta.SubjID == int(one)][target_type]))
    return np.array(target)


def load_data(path_to_data, file_type, target_type, data_type, path_to_meta, path_to_exp, balanced=True, save=True):
    '''
    :param path_to_data:
    :param balanced:
    :param data_type: can be one the subcortical structures of brain
                    such as: hippo(hippocamp), thalamus, pallidum, putamen, etc.
    :param target_type: different columns in meta file, or if None means that there exist file '_target.pkl'
    :param path_to_meta: default None, otherwise is full path to existing csv file
    :param file_type: supports just 'nii' and 'pkl'
            if file_type is .pkl the data and target should be specifized by names: data_type and data_type + _target
    :return: data and target
    '''
    assert file_type is not None and path_to_meta is not None, 'provide target path'
    if file_type == 'nii' or file_type == 'path':
        data, names = load_data_from_nii(path_to_data, data_type, file_type)
        if save:
            np.savez(os.path.join(path_to_exp,'data_idx.npz'), names)
        target = load_target(path_to_meta, names, target_type)

        if balanced:
            idx = balanced_fold(target)
            if save:
                np.savez(os.path.join(path_to_exp, 'data_idx.npz'), names[idx])
            return data[np.ix_(idx)], target[np.ix_(idx)]
        return data, target

