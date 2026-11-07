#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
"""Reads of our own checkpoint files."""
import functools
import os

import torch


def load_ckpt(path, map_location="cpu"):
    """torch.load of an MTGS checkpoint through mmap: only the pages that are copied out get read (the state
    dicts are copied into the model's parameters), no full host copy of the file first (~20 GB for a 6-pass
    block with its Adam state). weights_only=False: the optimizer param groups hold numpy scalars."""
    try:
        return torch.load(path, map_location=map_location, mmap=True, weights_only=False)
    except RuntimeError:   # legacy (non-zip) file: mmap is not supported
        return torch.load(path, map_location=map_location, weights_only=False)


@functools.lru_cache(maxsize=4)
def _tensor_cached(path: str, mtime: float, key: str) -> torch.Tensor:
    return load_ckpt(path)["pipeline"][key].clone()   # the clone materialises this tensor only; the mapping goes


def load_pipeline_tensor(path, key: str) -> torch.Tensor:
    """One tensor of a checkpoint's pipeline state (e.g. the camera optimizer's pose_adjustment), read once per file:
    the dataparser asks for it for the train and the eval split, and render.py loads the same file right after."""
    path = os.fspath(path)
    return _tensor_cached(path, os.path.getmtime(path), key)
