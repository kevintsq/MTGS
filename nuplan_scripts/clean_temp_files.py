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

    # MTGS_KEEP_PREP=1: keep the whole prep tree of this block (masks, depth, registration, colmap, point clouds),
    # e.g. to inspect what a training was fed. Tens of GB per block: set it per block, not in a recipe.
    if os.environ.get('MTGS_KEEP_PREP') == '1':
        print(f"[clean] MTGS_KEEP_PREP=1: prep tree of {config.road_block_name} kept")
        raise SystemExit(0)

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
    # The SfM RESULT outlives the clean-up (2026-11-10: ps_loc_0191's 20 h COLMAP solution was gone when a far-building
    # ghost had to be traced back to it): the model folders of every colmap/<video> tree (sparse_model, sfm_model,
    # sfm_model_raw: cameras / images / points3D, MBs) and its name_to_token.json move to <block>/colmap_models/<video>/.
    # The bulk (database, images, masks, depth ...) is deleted as before.
    colmap_root = os.path.join(data_root, config.road_block_name, "colmap")
    keep_root = os.path.join(data_root, config.road_block_name, "colmap_models")
    if os.path.isdir(colmap_root):
        for sub in sorted(os.listdir(colmap_root)):
            for name in ("sparse_model", "sfm_model", "sfm_model_raw", "name_to_token.json"):
                src = os.path.join(colmap_root, sub, name)
                if not os.path.exists(src) or (os.path.isdir(src) and not any(fs for _, _, fs in os.walk(src))):
                    continue
                dst = os.path.join(keep_root, sub, name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
                shutil.move(src, dst)
        if os.path.isdir(keep_root):
            print(f"[clean] SfM models kept in {keep_root}: {sorted(os.listdir(keep_root))}")

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
