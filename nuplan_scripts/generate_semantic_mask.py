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

from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from accelerate import Accelerator
from accelerate.utils import tqdm

from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import NUPLAN_SENSOR_ROOT

model_path = "facebook/mask2former-swin-large-cityscapes-semantic"

class _CamMaskDataset(torch.utils.data.Dataset):
    """Loads + preprocesses images in DataLoader workers so the GPU never
    waits on (NFS) I/O. Items already having a mask on disk are filtered out
    up-front, which also makes the step resumable."""

    def __init__(self, cams, processor, mask_root):
        self.processor = processor
        self.items = []
        seen = set()
        for cam_info in cams:
            dp = cam_info['data_path']
            if dp in seen:
                continue
            seen.add(dp)
            mask_path = os.path.join(mask_root, dp).replace('.jpg', '.png')
            if not os.path.exists(mask_path):
                self.items.append((dp, mask_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        dp, mask_path = self.items[idx]
        image = Image.open(os.path.join(NUPLAN_SENSOR_ROOT, dp))
        pixel = self.processor(images=image, return_tensors="pt").pixel_values[0]
        return pixel, mask_path, (image.height, image.width)


def _collate(batch):
    pixels = torch.stack([b[0] for b in batch])
    return pixels, [b[1] for b in batch], [b[2] for b in batch]


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

    from concurrent.futures import ThreadPoolExecutor
    writer = ThreadPoolExecutor(max_workers=4)

    def _write_mask(seg_map, mask_path):
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, seg_map)

    with distributed_state.split_between_processes(total_cams) as partial_frames:
        dataset = _CamMaskDataset(partial_frames, processor, mask_root)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            collate_fn=_collate, pin_memory=True)
        pbar = tqdm(total=len(dataset), ncols=120,
                    desc=f"Generating semantic masks (bs={args.batch_size}, fp16)")
        for pixels, mask_paths, sizes in loader:
            pixels = pixels.to(distributed_state.device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outputs = model(pixels)
            seg_maps = processor.post_process_semantic_segmentation(outputs, target_sizes=sizes)
            for seg, mask_path in zip(seg_maps, mask_paths):
                writer.submit(_write_mask, seg.cpu().numpy().astype(np.uint8), mask_path)
            pbar.update(len(mask_paths) * distributed_state.num_processes)

    writer.shutdown(wait=True)
