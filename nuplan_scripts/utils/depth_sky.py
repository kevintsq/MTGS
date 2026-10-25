"""Sky mask aligned with the dense-depth PNG layout (used by generate_dense_depth).

Training feeds the dense depth into the depth-NCC loss (0.1-80 m) and the normal-from-
depth loss (< 50 m); neither excludes sky. On nb_20 (overcast, 2026-10-25) UniDepth put
most of the sky inside 80 m in 25% of the frames that show sky and DA3 multi-view in 40%,
so the sky was supervised as a near surface. The mask stage already labels sky (cityscapes
class 10) in masks/raw/cityscape/<image>.png, in RAW image space; it is undistorted here
exactly like the image the depth model saw (nearest neighbour, same ROI), so its pixels
line up with the depth PNG (ROI pasted back into the full frame at the same place).
MTGS_DEPTH_SKY_MASK=0 disables it.
"""
import os

import cv2
import numpy as np

from nuplan_scripts.utils.camera_utils import undistort_image_with_cam_info
from nuplan_scripts.utils.constants import cityscape_label

SKY = cityscape_label['sky']


def enabled():
    return os.environ.get('MTGS_DEPTH_SKY_MASK', '1') == '1'


def sky_mask(video_scene, cam_info, undistort_mode):
    """bool [H, W] in the depth PNG's full-frame layout, or None when no semantic mask exists"""
    p = os.path.join(video_scene.raw_mask_path, video_scene.mask_suffix_cityscape,
                     cam_info['data_path']).replace('.jpg', '.png')
    seg = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if seg is None:
        return None
    if seg.ndim == 3:
        seg = seg[..., 0]
    H, W = seg.shape
    if undistort_mode == 'optimal':
        und, _K, (x, y, w, h) = undistort_image_with_cam_info(seg, cam_info, return_mask=False,
                                                              interpolation='nearest', mode='optimal')
        full = np.zeros((H, W), bool)
        full[y:y + h, x:x + w] = und[y:y + h, x:x + w] == SKY
        return full
    und = undistort_image_with_cam_info(seg, cam_info, return_mask=False,
                                        interpolation='nearest', mode='keep_focal_length')
    return und == SKY
