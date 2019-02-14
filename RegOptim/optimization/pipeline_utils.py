from __future__ import print_function

import json
import os

import numpy as np

from RegOptim.image_utils import get_contour2D, get_contour3D, padding, binarize, get_outside_filled, \
    find_threshold_gray_scale
from RegOptim.preprocessing import change_resolution
from RegOptim.utils import load_nii, save_nii, import_func


def get_shape(img):
    if isinstance(img, (str, np.str, np.string_, np.unicode_)):
        tmp_img = load_nii(img)
        return tmp_img.shape
    if isinstance(img, (np.ndarray)):
        return img.shape


def optim_template_strategy(it, k=10):
    if it % k == 0:
        return True
    else:
        return False


def create_exp_folders(exp_path, params=None):
    os.makedirs(exp_path, exist_ok=True)

    if params is not None:
        json.dump(params, open(os.path.join(exp_path, 'pipeline_params.txt'), 'w'))

    path_to_template = os.path.join(exp_path, 'templates/')
    os.makedirs(path_to_template, exist_ok=True)

    path_to_kernels = os.path.join(exp_path, 'kernel/')
    os.makedirs(path_to_kernels, exist_ok=True)


def create_template(path_to_data, train_idx, path_to_template, template_name,
                    resolution, sigma=0.01, load_func_template=None):
    images = []

    if isinstance(path_to_data[0], (str, np.string_, np.unicode_)):
        assert load_func_template is not None, "if data given by full path, you should provide loader"
        load_images = import_func(**load_func_template)
        images = load_images(path_to_data[np.ix_(train_idx)])

    if isinstance(path_to_data[0], np.ndarray):
        images = path_to_data[np.ix_(train_idx)]

    if resolution != 1:
        images = change_resolution(images, resolution, sigma)

    images = np.array(images)
    template = np.mean(images, axis=0)

    save_nii(template, os.path.join(path_to_template, template_name))
    return template


def update_template(template, template_path, template_name, delta, learning_rate=0.1):
    if isinstance(template, (str, np.string_, np.str, np.unicode_)):
        image = load_nii(template)
    else:
        image = template.copy()

    if image.shape != delta.shape:
        print('Error not correct shape or resolution')
        raise TypeError

    image -= learning_rate * delta

    save_nii(image, os.path.join(template_path, template_name))

    if isinstance(template, np.ndarray):
        return image

    elif isinstance(template, (str, np.string_, np.str, np.unicode_)):
        return os.path.join(template_path, template_name)
    else:
        raise TypeError('Unknown type of template')


def pad_template_data_after_loop(template, path_to_template, pad_size=2, save=True, ndim=3):
    if isinstance(template, (str, np.str, np.string_, np.unicode_)):
        image = load_nii(template)
    else:
        image = template.copy()

    padded_template = padding(image, pad_size=pad_size, ndim=ndim)
    if save:
        save_nii(padded_template, path_to_template)
    return padded_template


def preprocess_delta_template(delta, axis=0, contour_color=150, width=4, ndim=3):
    bin_delta = binarize(delta, find_threshold_gray_scale(delta))

    if ndim == 3:
        contour_delta = get_contour3D(image=bin_delta, axis=axis, contour_color=contour_color,
                                      width=width, mask=True)
    elif ndim == 2:
        contour_delta = get_contour2D(image=bin_delta, contour_color=contour_color,
                                      width=width, mask=True)
    else:
        raise TypeError('Do not support images of ndim not equal 2 or 3')

    #filled_contour = get_outside_filled(bin_delta, contour_delta)

    return delta * contour_delta
