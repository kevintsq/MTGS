"""Dense metric depth from Depth Anything 3 in multi-view mode (MTGS_DEPTH_MODEL=da3mv).

Benchmark against lidar (2026-10-25, TTA scripts/hugsim/gapfix/depth_bench.py, 129 images
of nb_20 and fj_loc_0120): DA3NESTED-GIANT-LARGE-1.1 fed 9 consecutive frames of one
camera with their metric poses and intrinsics (align_to_input_ext_scale) has median
relative error 0.033 on nb_20 vs 0.101 for UniDepthV2-L (the stage's default), scale
0.99 vs 0.92, and keeps thin structure (trees, poles) that UniDepth turns into blobs.
Single-view DA3 is far worse (scale 0.72-0.83), so the poses are what make it metric.

Per camera stream (one traversal, one camera, time order): overlapping windows of
MTGS_DA3_WINDOW frames (9) every MTGS_DA3_STRIDE frames (4); each frame keeps the depth
of the window in which it sits most centrally. Images are undistorted exactly as the
UniDepth path does (same ROI / intrinsics), and the PNG is written in the same encoding
(single-channel uint16, cm, ROI pasted into the full frame), so nothing downstream changes.
Camera poses: colmap_param's BA-refined sensor2global when present (what training uses
with use_colmap_pose), else ego2global x sensor2ego.
"""
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from nuplan_scripts.utils.camera_utils import undistort_image_with_cam_info


def _c2w(frame_info, cam_info):
    cp = cam_info.get('colmap_param')
    if cp is not None and 'sensor2global_rotation' in cp:
        T = np.eye(4)
        T[:3, :3] = Quaternion(cp['sensor2global_rotation']).rotation_matrix
        T[:3, 3] = np.asarray(cp['sensor2global_translation'], float)
        return T
    c2e = np.eye(4)
    c2e[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
    c2e[:3, 3] = np.asarray(cam_info['sensor2ego_translation'], float)
    return np.asarray(frame_info['ego2global'], float) @ c2e


def _load(video_scene, cam_info, undistort_mode):
    rgb = np.array(Image.open(os.path.join(video_scene.raw_image_path, cam_info['data_path'])).convert('RGB'))
    H, W = rgb.shape[:2]
    if undistort_mode == 'optimal':
        rgb, K, roi = undistort_image_with_cam_info(rgb, cam_info, return_mask=False,
                                                    interpolation='cubic', mode='optimal')
        x, y, w, h = roi
        rgb = rgb[y:y + h, x:x + w]
        K = np.asarray(K, float).copy(); K[0, 2] -= x; K[1, 2] -= y
    else:
        rgb = undistort_image_with_cam_info(rgb, cam_info, return_mask=False,
                                            interpolation='cubic', mode='keep_focal_length')
        K = np.asarray(cam_info['cam_intrinsic'], float); x, y, w, h = 0, 0, W, H
    return np.ascontiguousarray(rgb), K, (H, W), (x, y, w, h)


def _windows(n, W, S):
    """window starts covering 0..n-1; every frame then takes its most central window"""
    if n <= W:
        return [0]
    starts = list(range(0, n - W + 1, S))
    if starts[-1] != n - W:
        starts.append(n - W)
    return starts


def run_da3_multiview(video_scene, video_scene_dict, undistort_mode, max_range, out_root):
    from depth_anything_3.api import DepthAnything3
    model_id = os.environ.get('MTGS_DA3_MODEL', 'depth-anything/DA3NESTED-GIANT-LARGE-1.1')
    Wn = int(os.environ.get('MTGS_DA3_WINDOW', 9)); St = int(os.environ.get('MTGS_DA3_STRIDE', 4))
    model = DepthAnything3.from_pretrained(model_id).to(device='cuda')
    print(f'[dense_depth_da3] {model_id}, window {Wn}, stride {St}')

    streams = defaultdict(list)            # (video, cam) -> [(timestamp, frame_info, cam_info)]
    for vt, v in video_scene_dict.items():
        for fi in v['frame_infos']:
            if fi.get('skipped', False):
                continue
            for cam_name, ci in fi['cams'].items():
                streams[(vt, cam_name)].append((ci.get('timestamp', fi['timestamp']), fi, ci))

    total = sum(len(s) for s in streams.values()); done = 0
    pbar = tqdm(total=total, ncols=120, desc='DA3 multi-view depth')
    for key in sorted(streams):
        seq = sorted(streams[key], key=lambda t: t[0])
        outs = [os.path.join(out_root, ci['data_path'].replace('.jpg', '.png')) for _, _, ci in seq]
        if all(os.path.exists(o) for o in outs):
            pbar.update(len(seq)); continue
        loaded = [_load(video_scene, ci, undistort_mode) for _, _, ci in seq]
        w2c = [np.linalg.inv(_c2w(fi, ci)).astype(np.float32) for _, fi, ci in seq]
        n = len(seq); best = [None] * n; best_off = [1e9] * n
        for s0 in _windows(n, Wn, St):
            idx = list(range(s0, min(n, s0 + Wn)))
            pred = model.inference(image=[loaded[i][0] for i in idx],
                                   extrinsics=np.stack([w2c[i] for i in idx]),
                                   intrinsics=np.stack([loaded[i][1] for i in idx]).astype(np.float32),
                                   align_to_input_ext_scale=True, ref_view_strategy='saddle_balanced')
            depth = np.asarray(pred.depth).astype(np.float32)
            c = s0 + (len(idx) - 1) / 2.0
            for k, i in enumerate(idx):
                if abs(i - c) < best_off[i]:
                    best_off[i] = abs(i - c); best[i] = depth[k]
        for i in range(n):
            (H, W), (x, y, w, h) = loaded[i][2], loaded[i][3]
            d = best[i]
            if d.shape != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            d = np.where((d > max_range) | (d < 0.1), 0, d)
            img = np.zeros((H, W), np.uint16)
            img[y:y + h, x:x + w] = np.clip(np.trunc(d * 100), 0, 65535).astype(np.uint16)
            os.makedirs(os.path.dirname(outs[i]), exist_ok=True)
            cv2.imwrite(outs[i], img)
        pbar.update(n)
    pbar.close()
    del model; torch.cuda.empty_cache()
