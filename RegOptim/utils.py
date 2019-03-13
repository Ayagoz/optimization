import numpy as np
import nibabel as nib
import os
import json
import importlib


from sklearn.model_selection import StratifiedShuffleSplit

def show_grid(grid, shape, interval=1):
    import matplotlib.pyplot as plt
    for x in range(0, shape[0], interval):
        plt.plot(grid[1, x, :], grid[0, x, :], 'k')
    for y in range(0, shape[1], interval):
        plt.plot(grid[1, :, y], grid[0, :, y], 'k')
    plt.show()

def save_params(path, name, params):
    json.dump(params, open(os.path.join(path, name + '.txt'), 'w'))


def import_func(lib, func):
    return getattr(importlib.import_module(lib), func)

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

def load_nii(path_to_nii, data_type=None):
    if data_type is not None:
        if 'nii' in data_type:
            np.squeeze(np.array(nib.load(os.path.join(path_to_nii, data_type)).get_data()))
        else:
            return np.squeeze(np.array(nib.load(os.path.join(path_to_nii, data_type + '.nii')).get_data()))
    else:
        return np.squeeze(np.array(nib.load(path_to_nii).get_data()))


def save_nii(image, path, name=None):
    image_nii = nib.Nifti1Image(image, np.eye(4))
    if name is None:
        nib.save(image_nii, path)
    if name is not None:
        nib.save(image_nii, os.path.join(path, name + '.nii'))


def get_subset(X, y, size=0.5, random_state=0):
    assert X.shape[0] == y.shape[0], 'WTF'
    idx_train, idx_test = list(StratifiedShuffleSplit(n_splits=1, random_state=random_state,
                                                      test_size=1-size).split(range(len(X)), y))[0]
    return X[np.ix_(idx_train)], y[np.ix_(idx_train)]

