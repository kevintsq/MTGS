#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import argparse

os.environ["NERFSTUDIO_DATAPARSER_CONFIGS"] = "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser"
os.environ["NERFSTUDIO_METHOD_CONFIGS"] = "mtgs_st=mtgs.config.MTGS_ST:method,mtgs=mtgs.config.MTGS:method"

from nerfstudio.scripts.train import main
from nuplan_scripts.utils.config import load_config, FrameCentralConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    recon_subdir = os.environ.get('MTGS_RECON_SUBDIR', 'background_reconstruction')
    output_dir = os.path.join(config.data_root, recon_subdir, config.road_block_name)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(config, FrameCentralConfig):
        # official navsim auto flow: single (central) traversal
        from mtgs.config.MTGS_ST import config as ns_config
        travels = (0,)
        ns_config.pipeline.datamanager.dataparser.only_moving = True
    else:
        # RoadBlockConfig: full multi-traversal training on every traversal
        # that survived registration/BA (read them from the final vsd).
        from mtgs.config.MTGS import config as ns_config
        video_scene = VideoScene(config)
        vsd = video_scene.load_pickle(video_scene.pickle_path)
        travels = tuple(sorted(int(k.rsplit('-', 1)[1]) for k in vsd))
        assert travels, "no traversals left in the video_scene_dict"

    ns_config.experiment_name = config.road_block_name
    ns_config.pipeline.datamanager.cache_strategy = 'prefetch'
    ns_config.pipeline.datamanager.dataparser.road_block_config = args.config
    ns_config.pipeline.datamanager.dataparser.train_scene_travels = travels
    ns_config.pipeline.datamanager.dataparser.eval_scene_travels = travels
    ns_config.pipeline.model.color_corrected_metrics = False
    ns_config.pipeline.model.lpips_metric = False
    ns_config.pipeline.model.model_config['background'].verbose = False
    if os.environ.get('MTGS_DENSIFY_GRAD_THRESH'):
        ns_config.pipeline.model.control.densify_grad_thresh = float(os.environ['MTGS_DENSIFY_GRAD_THRESH'])
    if os.environ.get('MTGS_DENSIFY_SIZE_THRESH'):
        ns_config.pipeline.model.control.densify_size_thresh = float(os.environ['MTGS_DENSIFY_SIZE_THRESH'])
    if os.environ.get('MTGS_MAX_ITERS'):
        ns_config.max_num_iterations = int(os.environ['MTGS_MAX_ITERS'])
    if os.environ.get('MTGS_EVAL_ALL_STEPS'):
        ns_config.steps_per_eval_all_images = int(os.environ['MTGS_EVAL_ALL_STEPS'])
    if os.environ.get('MTGS_STEPS_PER_SAVE'):
        ns_config.steps_per_save = int(os.environ['MTGS_STEPS_PER_SAVE'])
    if os.environ.get('MTGS_DM_WORKERS'):
        ns_config.pipeline.datamanager.num_workers = int(os.environ['MTGS_DM_WORKERS'])
    ns_config.output_dir = output_dir
    ns_config.set_base_dir(output_dir)

    main(ns_config)
