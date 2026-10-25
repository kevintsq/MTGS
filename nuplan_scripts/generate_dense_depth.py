#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
from tqdm import tqdm

import numpy as np
from pyquaternion import Quaternion
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import tqdm as accelerate_tqdm

from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.camera_utils import undistort_image_with_cam_info
from nuplan_scripts.utils.nuplan_utils_custom import load_lidar
from unidepth.models import UniDepthV2


model_path = 'lpiccinelli/unidepth-v2-vitl14'

class MetricDepthError:

    class PseudoDataset(Dataset):
        H, W = 1080, 1920

        def __init__(
                self,
                video_scene: VideoScene,
                data_infos,
                undistort_mode,
                max_depth=80,
                min_depth=0.1):
            self.video_scene = video_scene
            self.data_infos = data_infos
            self.undistort_mode = undistort_mode
            self.max_depth = max_depth
            self.min_depth = min_depth

        def __len__(self):
            return len(self.data_infos)

        def __getitem__(self, idx):
            info = self.data_infos[idx]
            ref_errors = {}
            lidar_pts = load_lidar(
                os.path.join(video_scene.raw_lidar_path, info['lidar_path'])
            )
            lidar_pts_xyz1 = np.concatenate([lidar_pts, np.ones((lidar_pts.shape[0], 1))], axis=1)
            lidar2ego = info['lidar2ego']

            for cam_name, cam_info in info['cams'].items():

                ego2cam = np.eye(4)
                ego2cam[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix.T
                ego2cam[:3, 3] = -ego2cam[:3, :3] @ cam_info['sensor2ego_translation']
                lidar2cam = ego2cam @ lidar2ego

                if 'colmap_param' in cam_info:
                    intrinsic = cam_info['colmap_param']['cam_intrinsic']
                    distortion = cam_info['colmap_param']['distortion']
                    intrinsic = intrinsic.copy()
                    intrinsic[0, 2] = intrinsic[0, 2] - 0.5
                    intrinsic[1, 2] = intrinsic[1, 2] - 0.5
                else:
                    intrinsic = cam_info['cam_intrinsic']
                    distortion = cam_info['distortion']

                if self.undistort_mode == 'optimal':
                    new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
                        intrinsic, distortion, (self.W, self.H), 1
                    )
                else:
                    new_intrinsic = intrinsic
                    roi = (0, 0, self.W, self.H)

                # load depth
                if self.undistort_mode == 'optimal':
                    depth_path = os.path.join(
                        self.video_scene.optimal_undistorted_depth_path,
                        cam_info['data_path'].replace('.jpg', '.png'))
                else:
                    depth_path = os.path.join(
                        self.video_scene.undistorted_depth_path,
                        cam_info['data_path'].replace('.jpg', '.png'))

                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth.ndim == 2:   # single-channel uint16, depth in cm
                    depth = depth.astype(np.float32)
                else:                 # legacy 3-channel (lo, hi, 0) uint8
                    depth = depth.astype(np.float32)
                    depth = depth[:,:,0] + (depth[:,:,1] * 256)
                depth = depth * 0.01

                # project lidar points to image
                point_cam = (lidar_pts_xyz1 @ lidar2cam.T)[:, :3]
                uvz = point_cam[point_cam[:, 2] > 0]
                uvz = uvz @ new_intrinsic.T
                uvz[:, :2] /= uvz[:, 2:]
                uvz = uvz[(uvz[:, 0] >= roi[0]) & (uvz[:, 0] < roi[2]) &
                          (uvz[:, 1] >= roi[1]) & (uvz[:, 1] < roi[3]) &
                          (uvz[:, 2] > self.min_depth) & (uvz[:, 2] < self.max_depth)]
                # filter out out of range depth
                uvz = uvz[(depth[uvz[:, 1].astype(int), uvz[:, 0].astype(int)] > self.min_depth) & 
                          (depth[uvz[:, 1].astype(int), uvz[:, 0].astype(int)] < self.max_depth)]
                uv = uvz[:, :2]
                uv = uv.astype(int)

                pts_depth = np.zeros((self.H, self.W, 1), dtype=np.float32)
                pts_depth[uv[:, 1], uv[:, 0], 0] = uvz[:, 2]

                depth_pts = depth[uv[:, 1], uv[:, 0]]
                valid_mask = depth_pts != 0

                # calculate error
                ref_error = np.abs(depth_pts - uvz[:, 2])[valid_mask].mean()
                ref_errors[cam_name] = ref_error

            return ref_errors

    @staticmethod
    def run(
        video_scene: VideoScene,
        data_infos,
        undistort_mode,
        max_range,
        num_workers):

        dataset = MetricDepthError.PseudoDataset(
            video_scene, 
            data_infos,
            undistort_mode,
            max_depth=max_range)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x, pin_memory=False, drop_last=False)

        ref_errors = {}
        for batch in tqdm(dataloader, desc='processing frames', ncols=120):
            ref_error = batch[0]
            for cam_name, error in ref_error.items():
                if cam_name not in ref_errors:
                    ref_errors[cam_name] = []
                ref_errors[cam_name].append(error)

        return ref_errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--undistort_mode', type=str, default='optimal', choices=['optimal', 'keep_focal_length'])
    parser.add_argument('--max_range', type=int, default=80)
    parser.add_argument('--report_errors', action='store_true')
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)

    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path)

    total_frame_infos = []
    total_cams = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]
        total_frame_infos.extend(frame_infos)
        for info in frame_infos:
            total_cams.extend(list(info['cams'].values()))

    # MTGS_DEPTH_MODEL=da3mv: Depth Anything 3 multi-view with the block's metric poses
    # (median relative error vs lidar 0.033 vs 0.101 for UniDepth on nb_20, 2026-10-25);
    # same undistortion, PNG encoding and paths, so training is unaffected otherwise.
    if os.environ.get('MTGS_DEPTH_MODEL', 'unidepth') == 'da3mv':
        from nuplan_scripts.utils.dense_depth_da3 import run_da3_multiview
        out_root = (video_scene.optimal_undistorted_depth_path if args.undistort_mode == 'optimal'
                    else video_scene.undistorted_depth_path)
        run_da3_multiview(video_scene, video_scene_dict, args.undistort_mode, args.max_range, out_root)
        exit()

    model = UniDepthV2.from_pretrained(model_path)

    distributed_state = Accelerator()
    distributed_state.prepare_model(model, evaluation_mode=True)
    model.eval()

    class _DepthDataset(torch.utils.data.Dataset):
        """Image load + undistortion (the expensive cv2 remap) in DataLoader
        workers; skips items whose depth png already exists (resumable)."""

        def __init__(self, cams):
            self.items = []
            seen = set()
            for cam_info in cams:
                dp = cam_info['data_path']
                if dp in seen:
                    continue
                seen.add(dp)
                if args.undistort_mode == 'optimal':
                    output_path = os.path.join(
                        video_scene.optimal_undistorted_depth_path, dp.replace('.jpg', '.png'))
                else:
                    output_path = os.path.join(
                        video_scene.undistorted_depth_path, dp.replace('.jpg', '.png'))
                if not os.path.exists(output_path):
                    self.items.append((cam_info, output_path))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            cam_info, output_path = self.items[idx]
            image = Image.open(os.path.join(video_scene.raw_image_path, cam_info['data_path']))
            rgb = np.array(image)
            raw_height, raw_width = rgb.shape[:2]

            if args.undistort_mode == 'optimal':
                rgb, intrinsic, roi = undistort_image_with_cam_info(
                    rgb, cam_info,
                    return_mask=False, interpolation='cubic', mode='optimal')
                x, y, w, h = roi
                rgb = rgb[y:y+h, x:x+w]
                intrinsic = intrinsic.copy()
                intrinsic[0, 2] = intrinsic[0, 2] - x
                intrinsic[1, 2] = intrinsic[1, 2] - y
            else:  # keep_focal_length
                rgb = undistort_image_with_cam_info(
                    rgb, cam_info,
                    return_mask=False, interpolation='cubic', mode='keep_focal_length')
                intrinsic = np.asarray(cam_info['cam_intrinsic'])
                x, y, w, h = 0, 0, raw_width, raw_height

            inputs = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            intrinsic = torch.from_numpy(np.asarray(intrinsic)).float()
            return inputs, intrinsic, output_path, (raw_height, raw_width), (x, y, w, h)

    from concurrent.futures import ThreadPoolExecutor
    writer = ThreadPoolExecutor(max_workers=8)  # png encode is ~39 ms and drops the GIL

    def _write_depth(depth_image, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, depth_image)

    def _write_depth_packed(packed, raw_size, roi, output_path):
        """packed: [h,w] uint16 (depth in cm) straight off the GPU -> paste into
        the full frame and store as a single-channel 16-bit PNG."""
        raw_height, raw_width = raw_size
        x, y, w, h = roi
        depth_image = np.zeros((raw_height, raw_width), dtype=np.uint16)
        depth_image[y:y+h, x:x+w] = packed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, depth_image)

    with distributed_state.split_between_processes(total_cams) as partial_frames:
        dataset = _DepthDataset(partial_frames)

        # UniDepth one frame at a time leaves the GPU ~half idle. Frames only
        # stack if they share a shape, and the undistorted ROI is fixed per
        # camera (same intrinsics + distortion), so batch within a camera.
        # DEPTH_BATCH pins it; otherwise size from the memory actually free right
        # now (this GPU may be shared with trainers / other users).
        batch_size = int(os.environ.get('DEPTH_BATCH', 0))
        if batch_size <= 0:
            try:
                free_gb = torch.cuda.mem_get_info()[0] / 2**30
            except Exception:
                free_gb = 0
            # measured on a Pro6000: bs=4 @ 24 workers is the knee (21.2 it/s vs
            # 15.2 at bs=1/6w); bs=8 is slightly worse, so 4 is the cap.
            batch_size = 4 if free_gb >= 34 else 2 if free_gb >= 18 else 1
            print(f'[generate_dense_depth] GPU free {free_gb:.0f} GiB -> DEPTH_BATCH={batch_size}')
        # The ROI follows from the calibration that undistort_image_with_cam_info
        # actually uses — which is the BA-refined colmap_param when present, and it
        # differs per camera. Key on exactly that, or the stack hits mismatched shapes.
        def _calib_key(cam_info):
            src = cam_info.get('colmap_param', cam_info)
            return (np.asarray(src['cam_intrinsic']).tobytes(),
                    np.asarray(src['distortion']).tobytes())

        by_calib = {}
        for i, (cam_info, _) in enumerate(dataset.items):
            by_calib.setdefault(_calib_key(cam_info), []).append(i)
        batches = [idxs[i:i + batch_size]
                   for idxs in by_calib.values()
                   for i in range(0, len(idxs), batch_size)]

        def _collate(items):
            inputs = torch.stack([it[0] for it in items])
            intrinsics = torch.stack([it[1] for it in items])
            return inputs, intrinsics, [it[2] for it in items], \
                   [it[3] for it in items], [it[4] for it in items]

        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batches, num_workers=max(args.num_workers, 6),
            collate_fn=_collate, prefetch_factor=4)
        batched_infer = None   # probed on the first multi-frame batch
        pbar = accelerate_tqdm(total=len(dataset), ncols=120,
                               desc=f"Generating dense depths (bs={batch_size}, prefetch+fp16)")
        for inputs, intrinsics, output_paths, raw_sizes, rois in loader:
            inputs = inputs.to(distributed_state.device, non_blocking=True)
            intrinsics = intrinsics.to(distributed_state.device)

            def _infer_per_sample():
                return {'depth': torch.cat(
                    [model.infer(inputs[b:b + 1], intrinsics[b:b + 1])['depth']
                     for b in range(inputs.shape[0])], dim=0)}

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                # Stock UniDepth's camera path assumes batch=1 (get_rays builds a
                # batch-1 uv grid; B>1 intrinsics broadcast against it and
                # unproject's reshape blows up). A patched camera.py fixes that
                # (reshape(-1,...)) and true batching is ~8% faster e2e — probe it
                # once on the first real batch, per-sample where unsupported.
                # Frames in a batch share calibration (_calib_key), so both paths
                # are result-identical.
                if inputs.shape[0] == 1:
                    outputs = model.infer(inputs, intrinsics)
                elif batched_infer is None:
                    try:
                        outputs = model.infer(inputs, intrinsics)
                        batched_infer = True
                        print('[generate_dense_depth] batched UniDepth inference OK')
                    except RuntimeError:
                        batched_infer = False
                        print('[generate_dense_depth] stock UniDepth: per-sample forward')
                        outputs = _infer_per_sample()
                elif batched_infer:
                    outputs = model.infer(inputs, intrinsics)
                else:
                    outputs = _infer_per_sample()

            # Quantize depth (cm as uint16) ON THE GPU. Doing this in numpy cost
            # 22.8 ms/frame in the main loop — serialized with the 23.6 ms of
            # inference, it halved throughput. uint16 is also 2x cheaper to copy
            # back than the float map, and single-channel 16-bit PNGs are ~40%
            # smaller on disk than the legacy 3-channel (lo,hi,0) encoding.
            depth_t = outputs['depth'].squeeze(1).float()          # [B,h,w]
            if depth_t.ndim == 2:                                  # a batch of one
                depth_t = depth_t[None]
            depth_t = torch.where(
                (depth_t > args.max_range) | (depth_t < 0.1),
                torch.zeros((), device=depth_t.device), depth_t)
            # trunc matches the legacy uint8-pair encoding bit for bit
            packed = (depth_t * 100).trunc().clamp(0, 65535).to(torch.int32)
            packed = packed.cpu().numpy().astype(np.uint16)        # [B,h,w]

            for pk, output_path, (raw_height, raw_width), (x, y, w, h) in zip(
                    packed, output_paths, raw_sizes, rois):
                # Guard against degenerate model outputs (e.g. a frame whose
                # undistorted ROI produces a 1-D / mismatched depth map). Skip the
                # frame instead of crashing the whole stage.
                if pk.shape[0] != h or pk.shape[1] != w:
                    print(f"[generate_dense_depth] skip {output_path}: "
                          f"depth shape {pk.shape[:2]} != roi ({h},{w})")
                    pbar.update(distributed_state.num_processes)
                    continue

                # the paste + png encode run in the writer threads (imwrite drops
                # the GIL), so they no longer stall the main loop
                writer.submit(_write_depth_packed, pk, (raw_height, raw_width),
                              (x, y, w, h), output_path)
                pbar.update(distributed_state.num_processes)

    writer.shutdown(wait=True)
    distributed_state.wait_for_everyone()
    pbar.close()

    if not distributed_state.is_main_process:
        print(f'exiting subprocess {distributed_state.process_index}')
        exit()

    if not args.report_errors:
        exit()

    # Clear memory
    import gc
    # del image_processor, model, distributed_state
    del model, distributed_state
    gc.collect()
    torch.cuda.empty_cache()

    ref_errors = MetricDepthError.run(
        video_scene,
        total_frame_infos,
        args.undistort_mode,
        args.max_range,
        num_workers=args.num_workers)

    from prettytable import PrettyTable
    table = PrettyTable()
    table.field_names = ["Camera", "Mean Error"]
    for key, value in ref_errors.items():
        table.add_row([key, f"{np.mean(value):.4f}"])

    print(table)

