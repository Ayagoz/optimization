import numpy as np
import itertools

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
from joblib import Parallel, delayed

import cv2

from .utils import load_nii, save_nii

def get_contour(image, channel=3, width=3, ndim=3, lower=15, upper=250):
    '''
    works just for one subject on image
    :param image: input image
    :param channel: boundary color
    :param width: of boundary
    :param type_of_channels:
    :param ndim: ndim of image
    :param lower: color of object greater than lower
    :param upper: color of object less that upper
    :return: mask of contour
    '''


    if len(image.shape) > ndim:
        lower_ = np.array([lower] * (ndim + 1))
        upper_ = np.array([upper] * (ndim + 1))
        shapeMask = cv2.inRange(image, lower_, upper_)
    else:
        shapeMask = image.copy()

    shapeMask = shapeMask.astype(np.uint8)

    cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    if channel == 1:
        bound_color = (upper, 0, 0)
    elif channel == 2:
        bound_color = (0, upper, 0)
    else:
        bound_color = (0, 0, upper)


    color = cv2.cvtColor(shapeMask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(color, cnts, -1, bound_color, width)

    return (color[..., channel - 1] == upper).astype(int)





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