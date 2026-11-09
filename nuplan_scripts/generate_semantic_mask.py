#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse


import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from accelerate import Accelerator
from accelerate.utils import tqdm

from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import NUPLAN_SENSOR_ROOT

model_path = "facebook/mask2former-swin-large-cityscapes-semantic"

# Sky masks for the depth supervision (utils/depth_sky.py): skyseg (U2Net sky segmenter, the dynamic-batch export of VGGT's
# skyseg.onnx) is sharper at tree / building edges than the cityscapes sky class (2026-11-09, user). Written next to the
# cityscapes masks as masks/raw/sky/<image>.png (255 = sky). MTGS_SKY_SEG=cityscapes skips it (depth_sky then uses class 10).
SKY_SEG = os.environ.get('MTGS_SKY_SEG', 'skyseg')
SKYSEG_ONNX = os.environ.get('MTGS_SKYSEG_ONNX', os.path.expanduser('~/.cache/skyseg/skyseg_batch.onnx'))
SKY_THRESH = 0.4        # on the per-frame min-max stretched sky confidence (the threshold used for skyseg elsewhere)


class SkySeg:
    """skyseg on onnxruntime, fed and read straight from CUDA tensors (IO binding, no host round trip). TensorRT EP (fp32:
    fp16 moved 0.12 % of the pixels across the threshold) with a dynamic-batch profile when the tensorrt package is present,
    else the CUDA EP; onnxruntime's silent CPU fallback is an error."""

    def __init__(self, device, batch):
        import onnxruntime as ort
        try:
            import tensorrt  # noqa: F401  (loads libnvinfer for the TensorRT EP)
            trt_ok = 'TensorrtExecutionProvider' in ort.get_available_providers()
        except ImportError:
            trt_ok = False
        ort.preload_dlls()   # CUDA / cuDNN from the pip nvidia packages torch brought
        if not os.path.exists(SKYSEG_ONNX):
            raise SystemExit(f"skyseg model missing: {SKYSEG_ONNX} (MTGS_SKYSEG_ONNX; MTGS_SKY_SEG=cityscapes to skip)")
        dev = device.index or 0
        provs = []
        if trt_ok:
            # node-local and per GPU model: torch / civo nodes differ in GPU and share an NFS home, where concurrent
            # engine writes could collide; a rebuild costs 20-80 s once per node
            import tempfile
            cache = os.environ.get('MTGS_SKYSEG_TRT_CACHE') or os.path.join(
                tempfile.gettempdir(), f'skyseg_trt_{os.getuid()}', torch.cuda.get_device_name(dev).replace(' ', '_'))
            os.makedirs(cache, exist_ok=True)
            provs.append(('TensorrtExecutionProvider', {
                'device_id': dev, 'trt_engine_cache_enable': True, 'trt_engine_cache_path': cache, 'trt_fp16_enable': False,
                'trt_profile_min_shapes': 'input.1:1x3x320x320', 'trt_profile_opt_shapes': f'input.1:{batch}x3x320x320',
                'trt_profile_max_shapes': f'input.1:{max(batch, 32)}x3x320x320'}))
        provs.append(('CUDAExecutionProvider', {'device_id': dev}))
        self.sess = ort.InferenceSession(SKYSEG_ONNX, providers=provs)
        if self.sess.get_providers()[0] == 'CPUExecutionProvider':
            raise SystemExit("skyseg: onnxruntime fell back to the CPU (no CUDA / cuDNN for its CUDA EP)")
        print(f"skyseg on {self.sess.get_providers()[0]} ({SKYSEG_ONNX})")
        self.inp, self.out = self.sess.get_inputs()[0].name, self.sess.get_outputs()[0].name
        self.dev = dev
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, pixels):
        """[B, H, W, 3] uint8 RGB on the device -> [B, H, W] bool sky."""
        B, H, W, _ = pixels.shape
        x = F.interpolate(pixels.permute(0, 3, 1, 2).float().div_(255.0), (320, 320), mode='bilinear')
        x = x.sub_(self.mean).div_(self.std).contiguous()
        y = torch.empty((B, 1, 320, 320), dtype=torch.float32, device=x.device)
        io = self.sess.io_binding()
        io.bind_input(self.inp, 'cuda', self.dev, np.float32, tuple(x.shape), x.data_ptr())
        io.bind_output(self.out, 'cuda', self.dev, np.float32, tuple(y.shape), y.data_ptr())
        torch.cuda.synchronize(x.device)          # x written by torch's stream before onnxruntime reads it
        self.sess.run_with_iobinding(io)
        torch.cuda.synchronize(x.device)
        r = y.view(B, -1)
        lo, hi = r.min(1, keepdim=True).values, r.max(1, keepdim=True).values
        r = ((r - lo) / (hi - lo).clamp_min(1e-8)).view(B, 1, 320, 320)
        return F.interpolate(r, (H, W), mode='bilinear')[:, 0] >= SKY_THRESH

class _CamMaskDataset(torch.utils.data.Dataset):
    """Loads + preprocesses images in DataLoader workers so the GPU never
    waits on (NFS) I/O. Items already having a mask on disk are filtered out
    up-front, which also makes the step resumable."""

    def __init__(self, cams, processor, mask_root, sky_root=None):
        self.processor = processor
        self.items = []
        seen = set()
        for cam_info in cams:
            dp = cam_info['data_path']
            if dp in seen:
                continue
            seen.add(dp)
            mask_path = os.path.join(mask_root, dp).replace('.jpg', '.png')
            sky_path = os.path.join(sky_root, dp).replace('.jpg', '.png') if sky_root else None
            need_sem, need_sky = not os.path.exists(mask_path), bool(sky_path) and not os.path.exists(sky_path)
            if need_sem or need_sky:          # (blocks prepped before skyseg get only their sky masks)
                self.items.append((dp, mask_path if need_sem else None, sky_path if need_sky else None))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        dp, mask_path, sky_path = self.items[idx]
        image = Image.open(os.path.join(NUPLAN_SENSOR_ROOT, dp))
        # raw uint8 HWC: 6 MB per image across the worker queue instead of the 25 MB normalised float32 map
        # (the processor's rescale + normalise, identical arithmetic, runs on the GPU in _normalize)
        pixel = torch.from_numpy(np.asarray(image))
        return pixel, mask_path, sky_path, (image.height, image.width)


def _collate(batch):
    pixels = torch.stack([b[0] for b in batch])
    return pixels, [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch]


def _normalize(pixels, processor):
    """[B,H,W,3] uint8 on the device -> [B,3,H,W] float32, bit for bit what Mask2FormerImageProcessor(do_resize=False)
    produces: float64 * rescale_factor, cast to float32, then (x - mean) / std in float32."""
    x = (pixels.permute(0, 3, 1, 2).double() * processor.rescale_factor).float()
    mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=int(os.environ.get('SEM_MASK_BATCH', 4)))
    parser.add_argument('--num_workers', type=int, default=int(os.environ.get('SEM_MASK_WORKERS', 8)))
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)

    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path)

    total_cams = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        # generate semantic mask for stacking RGB point cloud
        frame_infos = [info for info in frame_infos if info.get('skipped', False) != 'low_velocity']
        for info in frame_infos:
            total_cams.extend(list(info['cams'].values()))

    processor = Mask2FormerImageProcessor(do_resize=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
    distributed_state = Accelerator()

    distributed_state.prepare_model(model, evaluation_mode=True)
    model.eval()

    mask_root = os.path.join(video_scene.raw_mask_path, video_scene.mask_suffix_cityscape)
    sky_root = os.path.join(video_scene.raw_mask_path, video_scene.mask_suffix_sky) if SKY_SEG == 'skyseg' else None
    skyseg = SkySeg(distributed_state.device, args.batch_size) if sky_root else None

    from concurrent.futures import ThreadPoolExecutor
    writer = ThreadPoolExecutor(max_workers=4)

    def _write_mask(seg_map, mask_path):
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, seg_map)

    with distributed_state.split_between_processes(total_cams) as partial_frames:
        dataset = _CamMaskDataset(partial_frames, processor, mask_root, sky_root)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=_collate, pin_memory=True)
        # context-managed bar: closed on the way out (also on exceptions) instead of in tqdm.__del__
        # at interpreter shutdown, which segfaulted on py3.13 (kaitain 2026-10-28)
        with tqdm(total=len(dataset), ncols=120,
                  desc=f"Generating semantic masks (bs={args.batch_size}, fp16)") as pbar:
            for raw, mask_paths, sky_paths, sizes in loader:
                raw = raw.to(distributed_state.device, non_blocking=True)
                sem = [i for i, m in enumerate(mask_paths) if m]
                if sem:
                    pixels = _normalize(raw[sem], processor)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        outputs = model(pixels)
                    seg_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=[sizes[i] for i in sem])
                    for seg, i in zip(seg_maps, sem):
                        # uint8 on the GPU before the copy: 2 MB per map instead of the 16 MB int64 argmax
                        writer.submit(_write_mask, seg.to(torch.uint8).cpu().numpy(), mask_paths[i])
                sky = [i for i, m in enumerate(sky_paths) if m]
                if sky:
                    with torch.inference_mode():
                        sky_maps = skyseg(raw[sky]).to(torch.uint8).mul_(255).cpu().numpy()
                    for m, i in zip(sky_maps, sky):
                        writer.submit(_write_mask, m, sky_paths[i])
                pbar.update(len(mask_paths) * distributed_state.num_processes)

    writer.shutdown(wait=True)
