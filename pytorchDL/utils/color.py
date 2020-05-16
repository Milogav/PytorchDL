import numpy as np


def rgb_to_yuv(img_rgb):
    """ Converts a rgb image array of dims [H, W, CH] to YUV color space.
    Input RGB ranges are assumed to be in [0, 1]
    Output YUV values ranges are:
        Y: [0, 1]
        U: [-0.436, 0.436]
        V: [-0.615, 0.615]
    """

    r = img_rgb[..., 0]
    g = img_rgb[..., 1]
    b = img_rgb[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.493 * (b - y)
    v = 0.877 * (r - y)

    img_yuv = np.stack((y, u, v), axis=2)
    return img_yuv


def yuv_to_rgb(img_yuv):
    """ Converts a YUV image array of dims [H, W, CH] to RGB color space.
    Input YUV ranges are assumed to be:
        Y: [0, 1]
        U: [-0.436, 0.436]
        V: [-0.615, 0.615]
    Output RGB range is [0, 1]
    """

    y = img_yuv[..., 0]
    u = img_yuv[..., 1]
    v = img_yuv[..., 2]

    r = y + 1.14 * v
    g = y - 0.396 * u - 0.581 * v
    b = y + 2.029 * u

    img_rgb = np.stack((r, g, b), axis=2)
    img_rgb = np.clip(img_rgb, 0, 1)
    return img_rgb

