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
import hydra
from omegaconf import DictConfig

from util import *


def preprocess_ich_imgs(cfg):
    """_summary_

    Args:
        cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    dcm = load_dicom_file(cfg)
    if cfg.task == "source_img":
        source_img = np.fromarray(np.uint8(dcm.pixel_array))
        return source_img
    elif cfg.task == "default_win_img":
        default_win_ary = default_window_image(dcm=dcm, rescale=cfg.rescale)
        default_win_img = np.fromarray(np.uint8(default_win_ary))
        return default_win_img
    elif cfg.task == "customize_win_img":
        custom_win_ary = custom_window_image(
            dcm=dcm,
            window_center=cfg.window_center[0],
            window_width=cfg.window_width[0]
        )
        custom_win_img = np.fromarray(np.uint8(custom_win_ary))
        return custom_win_img
    elif cfg.task == "combine_win_img":
        combined_win_ary = channel_combine(dcm)
        combined_win_img = np.fromarray(np.uint8(combined_win_ary))
        return combined_win_img
    elif cfg.task == "blur_win_img":
        blurred_win_ary = grayscale_win_img_blur(dcm=dcm, cfg=cfg)
        blurred_win_img = np.fromarray(np.uint8(blurred_win_ary))
        return blurred_win_img
    elif cfg.task == "blur_comb_img":
        win_img1 = custom_window_image(
            dcm=dcm,
            window_center=cfg.window_center[0],
            window_width=cfg.window_width[0]
        )
        win_img2 = custom_window_image(
            dcm=dcm,
            window_center=cfg.window_center[1],
            window_width=cfg.window_width[1]
        )
        win_img3 = custom_window_image(
            dcm=dcm,
            window_center=cfg.window_center[2],
            window_width=cfg.window_width[2]
        )

        blurred_comb_ary = rgb_comb_img_blur(
            win_img1=win_img1,
            win_img2=win_img2,
            win_img3=win_img3
        )
        blurred_comb_img = np.fromarray(np.uint8(blurred_comb_ary))
        return blurred_comb_img
    elif cfg.task == "edge_detect":
        detected_edge_ary = canny_edge_detect(dcm=dcm, cfg=cfg)
        detected_edge_img = np.fromarray(np.uint8(detected_edge_ary))
        return detected_edge_img
    else:
        metadata_df = dcm_metadata_df(dcm=dcm, cfg=cfg)
        return metadata_df

@hydra.main(version_base=None, config_path="configs", config_name="preprocess")
def main(cfg: DictConfig) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_

    Returns:
        _type_: _description_
    """
    for key, value in cfg.items():
        if value == "None":
            cfg[key] = eval(value)
    preprocess_ich_imgs(cfg)


if __name__=="__main__":
    print("Start to Preprocess Images from the RSNA ICH Dataset")
    main()