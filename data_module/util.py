# Copyright 2022 Mayo Clinic. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy as np
import pandas as pd
import pydicom as dcm
from PIL import Image
from skimage import feature
from scipy.ndimage.filters import gaussian_filter


def load_dicom_file(cfg):
    """_summary_

    Args:
        cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    dcm = dcm.dcmread(cfg.dcm_file_path)
    return dcm


def get_first_of_dicom_field_as_int(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == dcm.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


# Function to take care of teh translation and windowing.
def default_window_image(dcm, cfg):
    """_summary_

    Args:
        dcm (_type_): _description_
        rescale (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    dicom_fields = [
        dcm.WindowCenter,  # window center
        dcm.WindowWidth,  # window width
        dcm.RescaleIntercept,  # intercept
        dcm.RescaleSlope,  # slope
    ]
    window_center, window_width, intercept, slope = [
        get_first_of_dicom_field_as_int(x) for x in dicom_fields
    ]

    img = dcm.pixel_array
    img = (
        img * slope + intercept
    )  # for translation adjustments given in the dicom file.
    img_min = window_center - window_width // 2  # minimum HU level
    img_max = window_center + window_width // 2  # maximum HU level
    img[
        img < img_min
    ] = img_min  # set img_min for all HU levels less than minimum HU level
    img[
        img > img_max
    ] = img_max  # set img_max for all HU levels higher than maximum HU level
    if cfg.rescale:
        img = (img - img_min) / (img_max - img_min) * 255.0
    return img


def save_img_png(img_ary, img_path):
    """_summary_

    Args:
        img_ary (_type_): _description_
        img_path (_type_): _description_
    """
    img = Image.fromarray(np.uint8(img_ary))
    img.save(img_path)


def dcm_bit_standardize(dcm):
    """_summary_

    Args:
        dcm (_type_): _description_
    """
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


def custom_window_image(dcm, window_center, window_width):
    """_summary_

    Args:
        dcm (_type_): _description_
        window_center (_type_): _description_
        window_width (_type_): _description_

    Returns:
        _type_: _description_
    """
    if (
        (dcm.BitsStored == 12)
        and (dcm.PixelRepresentation == 0)
        and (int(dcm.RescaleIntercept) > -100)
    ):
        print("Bit Standardization")
        dcm_bit_standardize(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img


def channel_combine(dcm):
    """_summary_

    Args:
        dcm (_type_): _description_

    Returns:
        _type_: _description_
    """
    brain_img = custom_window_image(dcm, 40, 80)
    subdural_img = custom_window_image(dcm, 80, 200)
    soft_img = custom_window_image(dcm, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380

    combined_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return combined_img


def grayscale_win_img_blur(dcm, cfg):
    """_summary_

    Args:
        win_img_type (_type_): _description_
        blur_sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
    if cfg.win_img_type == "default":
        win_img = default_window_image(dcm=dcm, rescale=cfg.rescale)
    else:
        win_img = custom_window_image(
            dcm=dcm,
            window_center=cfg.window_center[0],
            window_width=cfg.window_width[0],
        )
    win_blurred_img = gaussian_filter(win_img, sigma=cfg.blur_sigma)
    return win_blurred_img


def rgb_comb_img_blur(win_img1, win_img2, win_img3):
    """_summary_

    Args:
        win_img1 (_type_): _description_
        win_img2 (_type_): _description_
        win_img3 (_type_): _description_

    Returns:
        _type_: _description_
    """
    combined_blurred_img = np.array([win_img1, win_img2, win_img3]).transpose(1, 2, 0)
    return combined_blurred_img


def canny_edge_detect(dcm, cfg):
    """_summary_

    Args:
        dcm (_type_): _description_
        cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    if cfg.win_img_type == "default":
        win_img = default_window_image(dcm=dcm, rescale=cfg.rescale)
    else:
        win_img = custom_window_image(
            dcm=dcm,
            window_center=cfg.window_center[0],
            window_width=cfg.window_width[0],
        )
    edges_img = feature.canny(win_img, sigma=cfg.canny_sigma)
    return edges_img


def dcm_metadata_df(dcm, cfg):
    """_summary_

    Args:
        dcm (_type_): _description_

    Returns:
        _type_: _description_
    """
    metadata_info = [
        dcm[cfg.dcm_metadata_tags[i]].value for i in range(len(cfg.dcm_metadata_tags))
    ]
    metadata_dict = dict(zip(cfg.dcm_metadata_tags, metadata_info))
    metadata_df = pd.DataFrame([metadata_dict])
    return metadata_df
