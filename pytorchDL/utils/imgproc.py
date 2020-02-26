import numpy as np
import cv2


def normalize(img, min_val=0.0, max_val=1.0):
    img = img.astype(float)
    return (img - img.min()) * (max_val - min_val) / (img.max() - img.min()) + min_val


def normalize_per_channel(img, min_val=0.0, max_val=1.0):
    ch = img.shape[-1]
    norm_img = np.concatenate([normalize(img[..., i], min_val, max_val) for i in range(ch)], axis=-1)
    return norm_img


def standardize(img):
    return (img - img.mean()) / img.std()


def standardize_per_channel(img):
    ch = img.shape[-1]
    norm_img = np.concatenate([standardize(img[..., i]) for i in range(ch)], axis=-1)
    return norm_img


def transform_image(image, rot_angle, x_shift=0, y_shift=0):
    h, w = image.shape[0:2]
    sx = w * x_shift
    sy = h * y_shift
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -rot_angle, 1.0)
    M[0, 2] += sx
    M[1, 2] += sy
    trf_img = cv2.warpAffine(image, M, dsize=(w, h))
    return trf_img, M
