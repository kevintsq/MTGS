"""Sky mask aligned with the dense-depth PNG layout (used by generate_dense_depth).

Training feeds the dense depth into the depth-NCC loss (0.1-80 m) and the normal-from-
depth loss (< 50 m); neither excludes sky. On nb_20 (overcast, 2026-10-25) UniDepth put
most of the sky inside 80 m in 25% of the frames that show sky and DA3 multi-view in 40%,
so the sky was supervised as a near surface. The mask stage writes skyseg sky masks to
masks/raw/sky/<image>.png (default since 2026-11-09; the cityscapes sky class 10 in
masks/raw/cityscape/<image>.png otherwise), in RAW image space; it is undistorted here
exactly like the image the depth model saw (nearest neighbour, same ROI), so its pixels
line up with the depth PNG (ROI pasted back into the full frame at the same place).
"""
import os

import cv2
import numpy as np

from nuplan_scripts.utils.camera_utils import undistort_image_with_cam_info
from nuplan_scripts.utils.constants import cityscape_label

SKY = cityscape_label['sky']


_FALLBACK_SAID = set()


def _raw_sky(video_scene, cam_info):
    """bool [H, W] sky in RAW image space, or None: the skyseg mask (masks/raw/sky, generate_semantic_mask.py,
    MTGS_SKY_SEG=skyseg default since 2026-11-09), else the cityscapes class 10 (blocks prepped before skyseg)."""
    dp = cam_info['data_path']
    if os.environ.get('MTGS_SKY_SEG', 'skyseg') == 'skyseg':
        m = cv2.imread(os.path.join(video_scene.raw_mask_path, video_scene.mask_suffix_sky, dp).replace('.jpg', '.png'),
                       cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return m > 127
        if video_scene.raw_mask_path not in _FALLBACK_SAID:
            _FALLBACK_SAID.add(video_scene.raw_mask_path)
            print(f"[depth_sky] no skyseg masks under {video_scene.raw_mask_path}/{video_scene.mask_suffix_sky}: "
                  f"cityscapes sky class used (rerun generate_semantic_mask.py to add them)", flush=True)
    seg = cv2.imread(os.path.join(video_scene.raw_mask_path, video_scene.mask_suffix_cityscape, dp).replace('.jpg', '.png'),
                     cv2.IMREAD_UNCHANGED)
    if seg is None:
        return None
    if seg.ndim == 3:
        seg = seg[..., 0]
    return seg == SKY


def sky_mask(video_scene, cam_info, undistort_mode):
    """bool [H, W] in the depth PNG's full-frame layout, or None when no sky / semantic mask exists"""
    raw = _raw_sky(video_scene, cam_info)
    if raw is None:
        return None
    seg = raw.astype(np.uint8)
    H, W = seg.shape
    if undistort_mode == 'optimal':
        und, _K, (x, y, w, h) = undistort_image_with_cam_info(seg, cam_info, return_mask=False,
                                                              interpolation='nearest', mode='optimal')
        full = np.zeros((H, W), bool)
        full[y:y + h, x:x + w] = und[y:y + h, x:x + w] == 1
        return full
    und = undistort_image_with_cam_info(seg, cam_info, return_mask=False,
                                        interpolation='nearest', mode='keep_focal_length')
    return und == 1
