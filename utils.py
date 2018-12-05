import numpy as np
import pandas as pd
import nibabel as nib
import pickle
import os

from tqdm import tqdm


def load_nii(path_to_nii, data_type=None):
    if '.nii' in data_type:
        np.squeeze(np.array(nib.load(os.path.join(path_to_nii, data_type)).get_data()))
    elif data_type is not None:
        return np.squeeze(np.array(nib.load(os.path.join(path_to_nii, data_type + '.nii')).get_data()))
    else:
        return np.squeeze(np.array(nib.load(path_to_nii).get_data()))


def save_nii(image, path, name=None):
    image_nii = nib.Nifti1Image(image, np.eye(4))
    if name is None:
        nib.save(image, path)
    if name is not None:
        nib.save(image_nii, os.path.join(path, name + '.nii'))


def load_data_from_dirs(global_path, data_type, file_type):
    data = []
    subj_names = []
    path = os.listdir(global_path)

    for subj in tqdm(path, decs='loading subjects'):
        path_to_subj = os.path.join(os.path.join(global_path, subj), 'nii')

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


def balanced_fold(y):
    '''
    :param y: binary array
    :return: sample of sorted indexes of y with mean(y) = 0.5
    '''

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    if len(idx0) > len(idx1):
        new_idx = np.random.choice(idx0, replace=False, size=len(idx1))
        idx = idx1
    else:
        new_idx = np.random.choice(idx1, replace=False, size=len(idx0))
        idx = idx0
    return np.concatenate([idx, new_idx], 0)


def load_target(path, idx, target_type):
    meta = pd.read_csv(path)[['SubjID', target_type]]
    target = []
    for one in idx:
        target.append(int(meta[meta.SubjId == one][target_type]))
    return np.array(target)


def load_data(path, data_type=None, target_type=None, path_to_meta=None, file_type='nii', balanced=True):
    '''
    :param path: path to data, architecture of folder proposed:
                    path - main folder contain subject's folders
                    path - subj folder - nii folder - nii images
    :param data_type: can be one the subcortical structures of brain
                    such as: hippo(hippocamp), thalamus, pallidum, putamen, etc.
    :param target_type: different columns in meta file, or if None means that there exist file '_target.pkl'
    :param path_to_meta: default None, otherwise is full path to existing csv file
    :param file_type: supports just 'nii' and 'pkl'
            if file_type is .pkl the data and target should be specifized by names: data_type and data_type + _target
    :return: data and target
    '''

    if file_type == 'nii':
        data, names = load_data_from_nii(path, data_type, file_type)
        if data_type is None:
            return data
        if path_to_meta is not None:
            target = load_target(path_to_meta, names, target_type)
            if balanced:
                idx = balanced_fold(target)
                return data[np.ix_(idx)], target[np.ix_(idx)]
            return data, target

    if file_type == 'pkl':
        with open(os.path.join(path, data_type + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        with open(os.path.join(path, data_type + '_target.pkl'), 'rb') as f:
            target = pickle.load(f)
        if balanced:
            idx = balanced_fold(target)
            return data[np.ix_(idx)], target[np.ix_(idx)]
        return data, target
