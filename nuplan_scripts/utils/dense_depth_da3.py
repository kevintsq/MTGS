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
from nuplan_scripts.utils import depth_sky


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
    lidar_scale = os.environ.get('MTGS_DA3_LIDAR_SCALE', '1') == '1'
    scale_tol = float(os.environ.get('MTGS_DA3_SCALE_TOL', '0.15'))
    lidar_cache = {}; ratios = []; n_scaled_windows = 0
    with tqdm(total=total, ncols=120, desc='DA3 multi-view depth') as pbar:
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
                # Scale check against the lidar (2026-11-05): align_to_input_ext_scale trusts DA3's own relative
                # poses; on side cameras with little structure (and in near-static windows) they are wrong and the
                # whole window comes out 5-10x too close (pn_w2_02: 119 of 3336 maps, max 3-8 m where lidar sees 80 m;
                # the depth loss then drove one camera's pose to NaN and the model with it). The lidar median ratio
                # per window fixes the scale; windows without lidar coverage are left as they are.
                if lidar_scale:
                    rs = []
                    for k, i in enumerate(idx):
                        r = _lidar_ratio(video_scene, seq[i][1], seq[i][2], w2c[i], loaded[i], depth[k], lidar_cache)
                        if r is not None:
                            rs.append(r)
                    if rs:
                        rw = float(np.exp(np.median(np.log(np.concatenate(rs)))))
                        n_scaled_windows += 1 if abs(np.log(rw)) > np.log(1.0 + scale_tol) else 0
                        ratios.append(rw)
                        if abs(np.log(rw)) > np.log(1.0 + scale_tol):
                            depth = depth * rw
                            print(f'[dense_depth_da3] {key[1]} window {s0}: lidar/DA3 scale {rw:.2f} -> rescaled', flush=True)
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
                if depth_sky.enabled():
                    sky = depth_sky.sky_mask(video_scene, seq[i][2], undistort_mode)
                    if sky is not None and sky.shape == img.shape:
                        img[sky] = 0                      # sky is not a surface: invalid for the depth losses
                os.makedirs(os.path.dirname(outs[i]), exist_ok=True)
                cv2.imwrite(outs[i], img)
            pbar.update(n)
    if lidar_scale and ratios:
        r = np.array(ratios)
        print(f'[dense_depth_da3] lidar scale check: {len(r)} windows, median ratio {np.median(r):.3f}, '
              f'p5 {np.percentile(r, 5):.2f}, p95 {np.percentile(r, 95):.2f}, rescaled {n_scaled_windows} '
              f'(|log ratio| > log {1.0 + scale_tol:.2f})', flush=True)
    del model; torch.cuda.empty_cache()


def _lidar_ratio(video_scene, frame_info, cam_info, w2c, loaded, depth, cache, min_pts=100):
    """lidar depth / DA3 depth at the lidar hits of one frame (same undistorted crop as the output), or None."""
    from nuplan_scripts.utils.nuplan_utils_custom import load_lidar
    lp = frame_info.get('lidar_path')
    if not lp:
        return None
    if lp not in cache:
        if len(cache) > 64:
            cache.clear()
        try:
            cache[lp] = load_lidar(os.path.join(video_scene.raw_lidar_path, lp))
        except Exception:
            cache[lp] = None
    pts = cache[lp]
    if pts is None or len(pts) == 0:
        return None
    K = loaded[1]; (x, y, w, h) = loaded[3]
    lidar2global = np.asarray(frame_info['ego2global'], float) @ np.asarray(frame_info['lidar2ego'], float)
    lidar2cam = np.asarray(w2c, float) @ lidar2global
    pc = (np.concatenate([pts[:, :3], np.ones((len(pts), 1))], 1) @ lidar2cam.T)[:, :3]
    pc = pc[pc[:, 2] > 0.5]
    uv = pc @ K.T; uv[:, :2] /= uv[:, 2:]
    d = depth if depth.shape == (h, w) else cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (uv[:, 2] < 80.0)
    if ok.sum() < min_pts:
        return None
    dz = d[v[ok], u[ok]]; lz = uv[ok, 2]
    good = dz > 0.1
    if good.sum() < min_pts:
        return None
    return lz[good] / dz[good]
