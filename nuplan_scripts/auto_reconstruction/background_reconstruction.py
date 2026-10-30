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

    if isinstance(config, FrameCentralConfig) or os.environ.get('MTGS_FORCE_ST') == '1':
        # official navsim auto flow: single (central) traversal
        from mtgs.config.MTGS_ST import config as ns_config
        if os.environ.get('MTGS_FORCE_ST') == '1' and not isinstance(config, FrameCentralConfig):
            from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
            _vs = VideoScene(config)
            _vsd = _vs.load_pickle(_vs.pickle_path)
            travels = tuple(sorted(int(k.rsplit('-', 1)[1]) for k in _vsd))
        else:
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
    # COLMAP re-estimates intrinsics + a 4-param OPENCV distortion per *video*, so one
    # physical camera gets a different focal (±1%) and k1/k2 in every traversal, and the
    # calibrated k3 (-0.052) is unrepresentable. Undistorting each traversal with its own
    # fit breaks multi-view consistency on thin structures (bent poles). 0 = calibrated 5-param.
    # BA-refined per-camera poses were never consumed by training (use_colmap_pose
    # defaults False); joint BA only pays off if they are.
    if os.environ.get('MTGS_USE_COLMAP_POSE') == '1':
        ns_config.pipeline.datamanager.dataparser.use_colmap_pose = True
    if os.environ.get('MTGS_USE_COLMAP_INTRINSICS', '1') == '0':
        ns_config.pipeline.datamanager.dataparser.use_colmap_intrinsics = False
    # MTGS trains at half resolution by default (camera_res_scale_factor=0.5); renders
    # come out ~4x less sharp than the photos. 1.0 = full 1920x1080.
    if os.environ.get('MTGS_CAM_RES_SCALE'):
        ns_config.pipeline.datamanager.camera_res_scale_factor = float(os.environ['MTGS_CAM_RES_SCALE'])
    # Pose A/B 2026-10-27: the stock camera optimiser (SO3xR3, lr 1e-4, L2 penalties) moves
    # cameras by millimetres; MTGS_CAM_OPT_LR / MTGS_CAM_OPT_TRANS_PENALTY / MTGS_CAM_OPT_ROT_PENALTY
    # let a run refine poses photometrically (lr 1e-3, penalties 0), MTGS_CAM_OPT=off disables it.
    if os.environ.get('MTGS_CAM_OPT'):
        ns_config.pipeline.model.camera_optimizer.mode = os.environ['MTGS_CAM_OPT']
    if os.environ.get('MTGS_CAM_OPT_TRANS_PENALTY'):
        ns_config.pipeline.model.camera_optimizer.trans_l2_penalty = float(os.environ['MTGS_CAM_OPT_TRANS_PENALTY'])
    if os.environ.get('MTGS_CAM_OPT_ROT_PENALTY'):
        ns_config.pipeline.model.camera_optimizer.rot_l2_penalty = float(os.environ['MTGS_CAM_OPT_ROT_PENALTY'])
    if os.environ.get('MTGS_CAM_OPT_LR'):
        ns_config.optimizers['camera_opt']['optimizer'].lr = float(os.environ['MTGS_CAM_OPT_LR'])
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
    # learning-rate knobs (2026-10-30, user A/B "position 8e-5 -> 8e-6, scales 1e-3"); MTGS works in metres, so
    # these are absolute values, NOT vanilla 3DGS's extent-relative position_lr_init
    if os.environ.get('MTGS_LR_MEANS'):
        ns_config.optimizers['means']['optimizer'].lr = float(os.environ['MTGS_LR_MEANS'])
    if os.environ.get('MTGS_LR_MEANS_FINAL'):
        ns_config.optimizers['means']['scheduler'].lr_final = float(os.environ['MTGS_LR_MEANS_FINAL'])
    if os.environ.get('MTGS_LR_SCALES'):
        ns_config.optimizers['scales']['optimizer'].lr = float(os.environ['MTGS_LR_SCALES'])
    if os.environ.get('MTGS_STEPS_PER_SAVE'):
        ns_config.steps_per_save = int(os.environ['MTGS_STEPS_PER_SAVE'])
    if os.environ.get('MTGS_DM_WORKERS'):
        ns_config.pipeline.datamanager.num_workers = int(os.environ['MTGS_DM_WORKERS'])
    ns_config.output_dir = output_dir
    ns_config.set_base_dir(output_dir)
    # MTGS_CKPT_MIRROR (preemptible jobs, see CustomTrainer): resume from the mirrored checkpoint if there is one
    mirror = os.environ.get('MTGS_CKPT_MIRROR')
    if mirror and os.path.isdir(mirror):
        steps = sorted(int(f[5:14]) for f in os.listdir(mirror) if f.startswith('step-') and f.endswith('.ckpt'))
        if steps:
            from pathlib import Path
            ns_config.load_dir = Path(mirror)
            ns_config.load_step = steps[-1]
            os.environ['MTGS_RESUMED'] = '1'
            print(f"[resume] {config.road_block_name}: resuming from {mirror}/step-{steps[-1]:09d}.ckpt", flush=True)

    main(ns_config)
