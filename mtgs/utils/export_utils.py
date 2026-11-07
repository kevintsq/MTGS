#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from mtgs.scene_model.mtgs_scene_graph import MTGSSceneModel, merge_control_cfg
from mtgs.utils.ckpt_io import load_ckpt

def extract_portable_ckpt(pipeline: VanillaPipeline, output_path: Path) -> torch.nn.Module:
    model: MTGSSceneModel = pipeline.model
    model.eval()
    gaussians = getattr(model, "gaussian_models", None)
    if gaussians is None:
        CONSOLE.log("[WARNING] No Gaussian models found in the model.", "red")
        return None

    video_scene = pipeline.datamanager.dataparser.video_scene
    train_scene_travels = pipeline.datamanager.dataparser.config.train_scene_travels
    video_scene.video_scene_dict_process(
        {'type': 'filter_by_video_idx', 'kwargs': {'video_idxs': train_scene_travels}}, inline=True
    )

    recon2world_translation = pipeline.datamanager.dataparser.recon2world_translation

    portable_model = OrderedDict()
    for submodel_name, submodel in gaussians.items():
        portable_config = submodel.portable_config

        if submodel_name == "background":
            portable_config["recon2world_translation"] = recon2world_translation

        portable_model[submodel_name] = OrderedDict({
            "config": portable_config,
            "state_dict": submodel.state_dict()
        })

    torch.save(portable_model, output_path)
    return portable_model


def _train_frame_timestamps(video_scene, dataparser_config) -> dict:
    """raw_timestamps of the train split per travel id, as NuplanDataParser builds them (frame filtering only)."""
    video_scene.load_pickle(video_scene.pickle_path, verbose=False)
    travels = dataparser_config.train_scene_travels
    filters = [{'type': 'filter_by_video_idx', 'kwargs': {'video_idxs': travels}}] if travels else []
    vsd = video_scene.video_scene_dict_process(filters + ['filter_skipped_frames'])
    out = {}
    for video_token, video in vsd.items():
        frame_infos = video['frame_infos']
        if dataparser_config.manual_split:
            frame_infos = frame_infos[:-1][::2] + frame_infos[-1:]
        out[int(video_token.split("-")[-1])] = torch.from_numpy(np.array([info['timestamp'] for info in frame_infos], dtype=np.int64))
    return out


def extract_portable_ckpt_from_file(ckpt_path: Path, ns_config, video_scene, output_path: Path) -> OrderedDict:
    """The same portable checkpoint as extract_portable_ckpt, built from the checkpoint file alone: no dataset, no
    initial point cloud, no GPU. The submodel configs come from the run's config.yml, the rigid objects' log
    timestamps from the block's video scene dict; the tensors are saved as CPU tensors (a loader passes
    map_location; the training checkpoint carried cuda:0 tags, which a CPU-only host could not read as saved)."""
    state = load_ckpt(ckpt_path)["pipeline"]
    per_model = OrderedDict()
    for key, value in state.items():
        if key.startswith("_model.gaussian_models."):
            model_name, subkey = key[len("_model.gaussian_models."):].split(".", maxsplit=1)
            per_model.setdefault(model_name, OrderedDict())[subkey] = value
    if not per_model:
        CONSOLE.log("[WARNING] No Gaussian models found in the checkpoint.", "red")
        return None

    model_cfg = ns_config.pipeline.model
    dp_cfg = ns_config.pipeline.datamanager.dataparser
    travel_ids = list(dp_cfg.train_scene_travels)
    raw_timestamps = _train_frame_timestamps(video_scene, dp_cfg)
    recon2world_translation = video_scene.video_scene_dict[next(iter(video_scene.video_scene_dict))]['global2world_translation']

    portable_model = OrderedDict()
    for model_id, (model_name, state_dict) in enumerate(per_model.items()):
        node_type = next(k for k in model_cfg.model_config if model_name == k or model_name.startswith(k + "_"))
        cfg = merge_control_cfg(model_cfg.control, model_cfg.model_config[node_type])
        submodel = cfg.setup(model_name=model_name, model_id=model_id, travel_ids=travel_ids, nearest_train_travel_of_eval=None)
        if node_type == "rigid_object":   # portable_config of a moving object lists its log timestamps
            submodel.is_static = state_dict["instance_quats"].ndim == 1
            submodel.dataframe_dict = {"raw_timestamps": raw_timestamps[int(model_name[len(node_type) + 1:].split("_")[0])]}
        portable_config = submodel.portable_config
        if model_name == "background":
            portable_config["recon2world_translation"] = recon2world_translation
        portable_model[model_name] = OrderedDict({"config": portable_config, "state_dict": state_dict})

    torch.save(portable_model, output_path)
    return portable_model
