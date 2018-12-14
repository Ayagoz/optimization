import numpy as np
import itertools

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
from joblib import Parallel, delayed

import cv2

from .utils import load_nii, save_nii

def padding(image, ndim=2, pad_shape=3, mode='constant', c_val=0):

    if image.ndim > ndim:
        pad = [(pad_shape, pad_shape)] * ndim + [(0, 0)]
    else:
        pad = [(pad_shape, pad_shape)] * ndim

    if isinstance(image, (str, np.str, np.string_)):
        data = load_nii(image)
        if mode == 'constant':
            padded_data = np.pad(data.copy(), tuple(pad), mode=mode, constant_values=c_val)
        else:
            padded_data = np.pad(data.copy(), tuple(pad), mode=mode)

        return  padded_data

    if isinstance(image, np.ndarray):
        if mode =='constant':
            padded_data = np.pad(image.copy(), tuple(pad), mode=mode, constant_values=c_val)
        else:
            padded_data = np.pad(image.copy(), tuple(pad), mode=mode)

        return  padded_data

def get_contour3D(image, axis=0, contour_color=150, width=3, mask=True):
    '''

    :param image: last axis of image should be channel if it is
                  and it should be N * W * H
    :param axis: along which axis to find contours, should be one of N, W, H
    :return: 3D binary contours
    '''

    # if len(image.shape) > 3:
    #     raise TypeError('image should be in binary!')

    data = image.copy()

    if axis != 0:
        data = np.swapaxes(data, 0, axis)
        shape = data.shape
    else:
        shape = data.shape
    if mask:
        if image.ndim > 3:
            mask_img = np.zeros(shape[:-1])
        else:
            mask_img = np.zeros(shape)
    else:
        mask_img = np.zeros(shape)


    for ax in range(mask_img.shape[0]):
         mask_img[ax, ...] = get_contour2D(data[ax, ...], contour_color, width, mask)

    if axis != 0:
        return np.swapaxes(mask_img, 0, axis)

    return mask_img


def find_threshold_gray_scale(img):
    bins, x = np.histogram(img.reshape(-1))
    idx = np.where(bins == 0)[0]
    idx = idx[len(idx) / 2]
    return x[idx]


def get_contour2D(image, contour_color=150, width=3, mask=True):
    if image.ndim == 3:
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        imgray = image.copy()

    t = find_threshold_gray_scale(imgray)
    ret, thresh = cv2.threshold(imgray, t, 255, 0)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if mask:
        if image.ndim == 3:
            result = cv2.drawContours(image.copy(), contours, -1, (0, contour_color, 0), width)
            return (result[..., 1] == contour_color).astype(int)
        else:
            result = cv2.drawContours(thresh.copy(), contours, -1, contour_color, width)
            return (result == contour_color).astype(int)
    else:
        if image.ndim == 3:
            result = cv2.drawContours(image.copy(), contours, -1, (0, contour_color, 0), width)
            return result
        else:
            result = cv2.drawContours(thresh.copy(), contours, -1, contour_color, width)
            return result




def to_one_resolution(resulting_vector_fields, resolutions, n_steps,
                      zoom_grid, vf0, inverse):
    '''
    returns in_one_res = v_res4 + v_res2 + v_res1 * for first timestep
    and vf = sum(in_one_res, by timesteps) -- to count <Lv_i,v_j>
    '''
    #we have v_i in res_i of shape (t, ndim, img_shape)
    #v =[...v_i...] ~ (nres, t, ndim, img_shape)
    #we want sum(v, axis=nres)
    if vf0:
        if inverse:
            T = -1
        else:
            T = 0
        in_one_res_vf = np.stack([zoom_grid(resulting_vector_fields[i].vector_fields[T],
                        resolutions[i]) for i in range(len(resolutions))], 0)
        return np.sum(in_one_res_vf, axis=0)[None,]
    else:
        in_one_res_vf = np.stack([np.stack([zoom_grid(resulting_vector_fields[i].vector_fields[j],
                        resolutions[i]) for j in range(n_steps+1)], 0) for i in range(len(resolutions))], 0)

        return  np.sum(in_one_res_vf, axis=0)


def change_resolution(images, resolution, sigma=0.01, order=1, multiple=True):
    N = len(images)
    blurred_images = gaussian_filter(images, sigma)

    if multiple:
        ratio = [1 / float(resolution)] * images[0].ndim
        images_another_resolution = np.array([zoom(blurred_images[i], ratio, order=order) for i in range(N)])
    else:
        ratio = [1 / float(resolution)] * images.ndim
        images_another_resolution = zoom(blurred_images, ratio, order=order)
    return images_another_resolution


def get_croped_img(path, save=True):
    images = []
    for one in path:
        images.append(load_nii(one,None))

    croped_images = crop_img(np.array(images))

    if save:
        for i, one in enumerate(croped_images):
            save_nii(one, path[i], name=None)
    else:
        return croped_images

def crop_img(images, space=2, parallel=True, n_jobs=10):
    N = len(images)

    if parallel:
        bounds = Parallel(n_jobs=n_jobs)(delayed(count_bounds)(images[i]) for i in range(N))
    else:
        bounds = get_bounds(images)

    bounds = np.array(bounds)

    left_bound =  bounds.min(axis=0).T[0] - space
    right_bound = bounds.max(axis=0).T[1] + space + 1

    left_bound = np.where(left_bound < 0, 0, left_bound)
    right_bound = np.where(right_bound > images[0].shape, images[0].shape, right_bound)

    coords = []
    for ax in images[0].ndim:
        coords.append(np.arange(left_bound[ax], right_bound[ax]))

    return np.array([images[i][np.ix_(*coords)] for i in range(N)])

def get_bounds(images):
    bounds = []
    for img in images:
        bounds.append(count_bounds(img))
    return bounds

def count_bounds(image):
    N = image.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(image, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return np.array(out).reshape(N, 2)[::-1].astype(int)