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

Pipeline (2026-11-07): the GPU used to idle through every CPU step. Frames are decoded,
undistorted and preprocessed (DA3's own InputProcessor._process_one) once each in loader
threads, one stream ahead, and kept on the GPU across the ~2.25 windows each sits in; the
forward goes through the same internals model.inference uses (normalised extrinsics,
umeyama scale) but the depth stays on the GPU for the upsample, the lidar scale check
(sweeps cached per video on the GPU) and the cm quantisation; writer threads paste,
sky-mask and encode the PNGs.
"""
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

from nuplan_scripts.utils.camera_utils import undistort_image_with_cam_info
from nuplan_scripts.utils import depth_sky

PROCESS_RES = 504                       # model.inference defaults
PROCESS_RES_METHOD = 'upper_bound_resize'


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


def _lens_sig(cam_info):
    """the intrinsics + distortion undistort_image_with_cam_info will use for this camera, as a sortable key"""
    if 'colmap_param' in cam_info and os.environ.get('MTGS_USE_COLMAP_INTRINSICS', '1') != '0':
        K, d = cam_info['colmap_param']['cam_intrinsic'], cam_info['colmap_param']['distortion']
    else:
        K, d = cam_info['cam_intrinsic'], cam_info['distortion']
    return tuple(np.round(np.concatenate([np.asarray(K, float).ravel(), np.asarray(d, float).ravel()]), 6).tolist())


def _prep_frame(video_scene, frame_info, cam_info, undistort_mode, processor, lidar_cache):
    """Everything model.inference does to one image before the forward, once (it redid it in every window):
    the undistorted crop goes through DA3's InputProcessor._process_one (resize, /14 rounding, normalisation,
    K rescale); the result is small (3 x ~280 x 504) and the full-res image is dropped. The frame's lidar sweep
    is parsed here too, into the per-video GPU cache."""
    rgb, K, size, roi = _load(video_scene, cam_info, undistort_mode)
    img_t, _hw, K_p, _ = processor._process_one(rgb, None, K.astype(np.float32),
                                                process_res=PROCESS_RES, process_res_method=PROCESS_RES_METHOD)
    lp = frame_info.get('lidar_path')
    if lp and lp not in lidar_cache:
        from nuplan_scripts.utils.nuplan_utils_custom import load_lidar
        try:
            pts = load_lidar(os.path.join(video_scene.raw_lidar_path, lp))
            lidar_cache[lp] = torch.from_numpy(np.ascontiguousarray(pts[:, :3], dtype=np.float32)).cuda()
        except Exception:
            lidar_cache[lp] = None
    return img_t, K_p, K, size, roi


def _windows(n, W, S):
    """window starts covering 0..n-1; every frame then takes its most central window"""
    if n <= W:
        return [0]
    starts = list(range(0, n - W + 1, S))
    if starts[-1] != n - W:
        starts.append(n - W)
    return starts


def _div_like_numpy(depth_t, scale):
    """prediction.depth /= scale of the inference path: numpy divides a float32 array by a float64 scalar in
    float64 and casts back, by a float32 / python scalar in float32 -- reproduced on the GPU."""
    if np.result_type(np.float32, scale) == np.float64:
        return (depth_t.double() / float(scale)).float()
    return depth_t / float(np.float32(scale))


def _forward_window(model, imgs, exts, ixts, ref_view_strategy):
    """model.inference(image, extrinsics, intrinsics, align_to_input_ext_scale=True) on preprocessed GPU inputs,
    returning the metric depth [N, h, w] float32 on the GPU instead of a CPU Prediction, and whether its scale is
    metric (False: the pose alignment was degenerate and the depth is still in DA3's own scale)."""
    ex_t = torch.from_numpy(np.asarray(exts)).float().to(imgs.device, non_blocking=True)[None]
    in_t = torch.from_numpy(np.asarray(ixts)).float().to(imgs.device, non_blocking=True)[None]
    ex_t_norm = model._normalize_extrinsics(ex_t.clone())
    raw = model._run_model_forward(imgs[None], ex_t_norm, in_t, [], False, False, ref_view_strategy)
    depth = raw['depth'].squeeze(0).squeeze(-1).float()                  # OutputProcessor._extract_depth, on the GPU
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    pred_ext = raw['extrinsics'].squeeze(0).cpu().numpy()
    ext_np = np.asarray(exts, dtype=np.float32)
    try:
        _, _, scale, _ = align_poses_umeyama(pred_ext, ext_np, ransac=len(ext_np) >= 10, return_aligned=True, random_state=42)
    except Exception:
        # degenerate Sim3 (a 1-2 frame stream: the few cameras of a pass the BA left unregistered): no pose scale,
        # the caller takes the lidar's
        return depth, False
    return _div_like_numpy(depth, scale), True


def _upsample(depth, h, w):
    """[N, h0, w0] -> [N, h, w] bilinear with half-pixel centres like cv2.INTER_LINEAR (only float rounding differs)"""
    if depth.shape[-2:] == (h, w):
        return depth
    return F.interpolate(depth[:, None], size=(h, w), mode='bilinear', align_corners=False)[:, 0]


def run_da3_multiview(video_scene, video_scene_dict, undistort_mode, max_range, out_root):
    from depth_anything_3.api import DepthAnything3
    model_id = os.environ.get('MTGS_DA3_MODEL', 'depth-anything/DA3NESTED-GIANT-LARGE-1.1')
    Wn = int(os.environ.get('MTGS_DA3_WINDOW', 9)); St = int(os.environ.get('MTGS_DA3_STRIDE', 4))
    model = DepthAnything3.from_pretrained(model_id).to(device='cuda')
    print(f'[dense_depth_da3] {model_id}, window {Wn}, stride {St}')

    # (video, cam, lens) -> [(timestamp, frame_info, cam_info)]. The lens is part of the key: a camera the BA did
    # not register has no colmap_param and is undistorted with the calibration, so its crop has another size than
    # its registered neighbours' and one window could not be stacked (ru_loc_0195, 2026-11-10: 266 vs 252 rows).
    streams = defaultdict(list)
    for vt, v in video_scene_dict.items():
        for fi in v['frame_infos']:
            if fi.get('skipped', False):
                continue
            for cam_name, ci in fi['cams'].items():
                streams[(vt, cam_name, _lens_sig(ci))].append((ci.get('timestamp', fi['timestamp']), fi, ci))

    total = sum(len(s) for s in streams.values())
    scale_tol = float(os.environ.get('MTGS_DA3_SCALE_TOL', '0.15'))
    ratios = []; n_scaled_windows = 0
    loaders = ThreadPoolExecutor(max_workers=int(os.environ.get('MTGS_DA3_LOADERS', 8)))
    writer = ThreadPoolExecutor(max_workers=8)          # png encode drops the GIL
    lidar_caches = {}                                   # video -> {lidar_path: GPU points}; a video's streams are consecutive

    def _write(path, packed, cam_info, size, roi):
        (H, W), (x, y, w, h) = size, roi
        img = np.zeros((H, W), np.uint16)
        img[y:y + h, x:x + w] = packed
        sky = depth_sky.sky_mask(video_scene, cam_info, undistort_mode)
        if sky is not None and sky.shape == img.shape:
            img[sky] = 0                      # sky is not a surface: invalid for the depth losses
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)

    def _submit(key):
        """(frames, output paths, loader futures) of one stream; futures None when its PNGs all exist"""
        seq = sorted(streams[key], key=lambda t: t[0])
        outs = [os.path.join(out_root, ci['data_path'].replace('.jpg', '.png')) for _, _, ci in seq]
        if all(os.path.exists(o) for o in outs):
            return seq, outs, None
        cache = lidar_caches.setdefault(key[0], {})
        futs = [loaders.submit(_prep_frame, video_scene, fi, ci, undistort_mode, model.input_processor, cache)
                for _, fi, ci in seq]
        return seq, outs, futs

    keys = sorted(streams)
    pending = _submit(keys[0]) if keys else None
    with tqdm(total=total, ncols=120, desc='DA3 multi-view depth') as pbar:
        for ki, key in enumerate(keys):
            seq, outs, futs = pending
            pending = _submit(keys[ki + 1]) if ki + 1 < len(keys) else None       # the next stream loads meanwhile
            if futs is None:
                pbar.update(len(seq)); continue
            n = len(seq)
            w2c = [np.linalg.inv(_c2w(fi, ci)).astype(np.float32) for _, fi, ci in seq]
            loaded = [None] * n                     # (img_t on the GPU, K_p, K, size, roi)
            best = [None] * n; best_off = [1e9] * n
            lidar_cache = lidar_caches[key[0]]
            for s0 in _windows(n, Wn, St):
                idx = list(range(s0, min(n, s0 + Wn)))
                for i in idx:
                    if loaded[i] is None:
                        img_t, K_p, K, size, roi = futs[i].result()
                        loaded[i] = (img_t.to('cuda', non_blocking=True), K_p, K, size, roi)
                imgs = torch.stack([loaded[i][0] for i in idx])
                depth, metric = _forward_window(model, imgs, [w2c[i] for i in idx], [loaded[i][1] for i in idx],
                                                'saddle_balanced')
                # Scale check against the lidar (2026-11-05): align_to_input_ext_scale trusts DA3's own relative
                # poses; on side cameras with little structure (and in near-static windows) they are wrong and the
                # whole window comes out 5-10x too close (pn_w2_02: 119 of 3336 maps, max 3-8 m where lidar sees 80 m;
                # the depth loss then drove one camera's pose to NaN and the model with it). The lidar median ratio
                # per window fixes the scale; windows without lidar coverage are left as they are.
                (x, y, w, h) = loaded[idx[0]][4]
                full = _upsample(depth, h, w)
                rs = []
                for k, i in enumerate(idx):
                    r = _lidar_ratio(seq[i][1], w2c[i], loaded[i][2], (x, y, w, h), full[k], lidar_cache)
                    if r is not None:
                        rs.append(r)
                if rs:
                    rw = float(np.exp(np.median(np.log(torch.cat(rs).cpu().numpy()))))
                    n_scaled_windows += 1 if abs(np.log(rw)) > np.log(1.0 + scale_tol) else 0
                    ratios.append(rw)
                    if not metric:
                        depth = depth * rw
                    elif abs(np.log(rw)) > np.log(1.0 + scale_tol):
                        depth = depth * rw
                        print(f'[dense_depth_da3] {key[1]} window {s0}: lidar/DA3 scale {rw:.2f} -> rescaled', flush=True)
                elif not metric:
                    depth = torch.zeros_like(depth)         # no pose scale and no lidar: invalid, not a guessed scale
                c = s0 + (len(idx) - 1) / 2.0
                for k, i in enumerate(idx):
                    if abs(i - c) < best_off[i]:
                        best_off[i] = abs(i - c); best[i] = depth[k]
            for i in range(n):
                size, (x, y, w, h) = loaded[i][3], loaded[i][4]
                d = _upsample(best[i][None], h, w)[0]
                d = torch.where((d > max_range) | (d < 0.1), torch.zeros((), device=d.device), d)
                # trunc then clamp on the GPU, as the UniDepth path: the same uint16 cm encoding
                packed = (d * 100).trunc().clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
                writer.submit(_write, outs[i], packed, seq[i][2], size, (x, y, w, h))
            loaded = best = None
            if ki + 1 >= len(keys) or keys[ki + 1][0] != key[0]:
                lidar_caches.pop(key[0], None)          # last stream of this video: free its sweeps
            pbar.update(n)
    loaders.shutdown(wait=True); writer.shutdown(wait=True)
    if ratios:
        r = np.array(ratios)
        print(f'[dense_depth_da3] lidar scale check: {len(r)} windows, median ratio {np.median(r):.3f}, '
              f'p5 {np.percentile(r, 5):.2f}, p95 {np.percentile(r, 95):.2f}, rescaled {n_scaled_windows} '
              f'(|log ratio| > log {1.0 + scale_tol:.2f})', flush=True)
    del model; torch.cuda.empty_cache()


def _lidar_ratio(frame_info, w2c, K, roi, depth, cache, min_pts=100):
    """lidar depth / DA3 depth at the lidar hits of one frame (depth: the frame's upsampled map on the GPU, same
    crop as the output) as a float64 GPU tensor like the numpy original, or None."""
    lp = frame_info.get('lidar_path')
    pts = cache.get(lp) if lp else None
    if pts is None or len(pts) == 0:
        return None
    (x, y, w, h) = roi
    lidar2global = np.asarray(frame_info['ego2global'], float) @ np.asarray(frame_info['lidar2ego'], float)
    lidar2cam = torch.from_numpy(np.asarray(w2c, float) @ lidar2global).to(pts.device)
    pc = pts.double() @ lidar2cam[:3, :3].T + lidar2cam[:3, 3]
    pc = pc[pc[:, 2] > 0.5]
    uv = pc @ torch.from_numpy(np.asarray(K, float)).to(pts.device).T
    z = uv[:, 2]
    u = torch.round(uv[:, 0] / z).long(); v = torch.round(uv[:, 1] / z).long()
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z < 80.0)
    if int(ok.sum()) < min_pts:
        return None
    dz = depth[v[ok], u[ok]]; lz = z[ok]
    good = dz > 0.1
    if int(good.sum()) < min_pts:
        return None
    return lz[good] / dz[good].double()
