#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse
import shutil
from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)
    video_scene = VideoScene(config)

    data_root = video_scene.data_root

    folders_to_remove = [
        "masks",
        "depth",
        "registration_results",
        "colmap",
        "instance_point_cloud",
        "rgb_point_cloud",
        "sfm_point_cloud"
    ]

    # colmap/: whitelist-checked per model dir (ba_multi_traversal.safe_rmtree_colmap); the other folders are
    # pipeline-only trees. A refusal here is deliberate (2026-11-04: experiments living next to the pipeline's files).
    from nuplan_scripts.ba_multi_traversal import safe_rmtree_colmap
    for folder in folders_to_remove:
        dir_path = os.path.join(data_root, config.road_block_name, folder)
        if not os.path.exists(dir_path):
            continue
        if folder == "colmap":
            for sub in os.listdir(dir_path):
                p_ = os.path.join(dir_path, sub)
                if os.path.isdir(p_):
                    safe_rmtree_colmap(p_, "clean_temp_files")
            if os.listdir(dir_path):
                print(f"[clean] {dir_path} kept: {os.listdir(dir_path)}")
                continue
        shutil.rmtree(dir_path)
