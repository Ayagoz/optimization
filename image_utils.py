import numpy as np
import cv2

from .utils import load_nii, save_nii


def check_for_padding(image):
    bin_img = image.copy()
    bin_img[bin_img != 0] = 1.

    idx = [slice(1, -1)] * image.ndim
    borders = np.sum(bin_img) - np.sum(bin_img[idx])

    if borders > 0:
        return True
    else:
        return False


def pad_images(data, pad_size=2, ndim=3):
    if isinstance(data, np.ndarray):
        images = data.copy()
    if isinstance(data[0], (str, np.str, np.string_)):
        images = np.array([load_nii(subj, None) for subj in data])

    padded_image = np.array([padding(img, ndim, pad_size, mode='edge') for img in images])

    [save_nii(padded_image[i], data[i], None) for i in range(len(data))]

    if isinstance(data, np.ndarray):
        return padded_image
    if isinstance(data[0], (str, np.str, np.string_)):
        return data


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

        return padded_data

    if isinstance(image, np.ndarray):
        if mode == 'constant':
            padded_data = np.pad(image.copy(), tuple(pad), mode=mode, constant_values=c_val)
        else:
            padded_data = np.pad(image.copy(), tuple(pad), mode=mode)

        return padded_data


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
