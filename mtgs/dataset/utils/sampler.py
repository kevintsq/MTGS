#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from nuplan_scripts.utils.console import say
import atexit
import json
import os
import random
import numpy as np
import torch

# EXPERIMENT 2026-11-13 (MTGS_VIEW_SAMPLE=cover): coverage-aware view sampling.  With the once-per-epoch sampler a region seen by
# 22 of 6,188 images gets ~430 of 120k optimisation steps.  The scene model reports, per training step, the image's
# "excess rarity" e = mean over the gaussians that contributed to it of (rarity - 1) (rarity = clamp(f_ref / f_g, 1, cap), f_g the
# long-term contribution frequency; see scene_model/gaussian_model/vanilla_gaussian_splatting.py COVER_*).  e is ~0 for an image
# of common content and grows with the SHARE of rarely optimised gaussians in it, so the draw goes to images that hold rare
# content rather than to images that merely have a few rare gaussians.  MTGS_VIEW_SAMPLE_FRAC (0.5) of the steps draw images in
# proportion to (e + floor) ** MTGS_VIEW_SAMPLE_ALPHA; the others keep the base sampler, so every image is still visited once per
# base epoch.  Scores stay on the GPU; one host sync per block of 256 draws.  MTGS_VIEW_SAMPLE_LOG=<json>: visits per image.
ACTIVE = None


def report(value):
    """called once per training step (after the step) with the image's excess rarity, a GPU scalar"""
    if ACTIVE is not None:
        ACTIVE.report(value)


class CoverAwareSampler:

    def __init__(self, base, n, block=256):
        self.base, self.n, self.block = base, n, block
        self.frac = float(os.environ.get("MTGS_VIEW_SAMPLE_FRAC", "0.5"))
        self.alpha = float(os.environ.get("MTGS_VIEW_SAMPLE_ALPHA", "1.0"))
        # mining starts once (almost) every image has a score: scores exist from MTGS_COVER_START on, one base epoch later all have one
        self.start = int(os.environ.get("MTGS_VIEW_SAMPLE_START", str(int(os.environ.get("MTGS_COVER_START", "9000")) + n)))
        self.log = os.environ.get("MTGS_VIEW_SAMPLE_LOG", "")
        self.score = None; self.queue = []; self.last = None; self.k = 0
        self.visits = [0] * n; self.mined = [0] * n
        if self.log:
            atexit.register(self.dump)

    def get_next_image_idx(self) -> int:
        self.k += 1
        idx = None
        if self.score is not None and self.k > self.start and random.random() < self.frac:
            if not self.queue:
                known = self.score >= 0
                e = torch.where(known, self.score, (self.score * known).sum() / known.sum().clamp_min(1))
                w = (e + 0.05 * e.mean() + 1e-8) ** self.alpha
                self.queue = torch.multinomial(w, self.block, replacement=True).tolist()   # the one host sync per block
                if sum(self.mined) % (40 * self.block) == 0:
                    q = e.quantile(torch.tensor([0.5, 0.9, 0.99, 1.0], device=e.device)).tolist()
                    say(f"[view-cover] step {self.k}: excess rarity p50/90/99/max {[round(x, 4) for x in q]}, scored {int(known.sum())}/{self.n}", flush=True)
            idx = self.queue.pop(); self.mined[idx] += 1
        if idx is None:
            idx = self.base.get_next_image_idx()
        self.visits[idx] += 1
        self.last = idx
        if self.log and self.k % 10000 == 0:
            self.dump()
        return idx

    def peek(self, k):
        """Only while mining cannot start within the window (mining draws depend on scores not reported yet)."""
        if self.score is not None and self.k + k > self.start:
            return []
        return self.base.peek(k) if hasattr(self.base, "peek") else []

    def report(self, value):
        if self.last is None:
            return
        v = value.detach().float()
        if self.score is None:
            self.score = torch.full((self.n,), -1.0, device=v.device)
        old = self.score[self.last]
        self.score[self.last] = torch.where(old < 0, v, 0.5 * old + 0.5 * v)

    def dump(self):
        if self.k == 0:   # an eval / render process builds the datamanager too: never overwrite the training's log
            return
        try:
            d = dict(k=self.k, start=self.start, frac=self.frac, alpha=self.alpha, visits=self.visits, mined=self.mined,
                     score=None if self.score is None else self.score.tolist())
            json.dump(d, open(self.log + ".tmp", "w")); os.replace(self.log + ".tmp", self.log)
        except Exception as e:   # never take the training down for a log
            say(f"[view-cover] dump failed: {e}", flush=True)

def _peek(sampler, k, state_attrs):
    """The next k indices `sampler` will return, without changing it or python's random state (the training's only
    other user of `random` is the in-order eval sampler, which draws nothing): for decoding samples ahead."""
    import copy
    rnd = random.getstate(); saved = {a: copy.deepcopy(getattr(sampler, a)) for a in state_attrs}
    try:
        return [sampler.get_next_image_idx() for _ in range(k)]
    finally:
        for a, v in saved.items():
            setattr(sampler, a, v)
        random.setstate(rnd)


class CameraSampler:
    
    def __init__(self, dataset, in_order: bool = False):
        self.dataset = dataset
        self.unseen_cameras = [i for i in range(len(self.dataset))]
        self.in_order = in_order

    def peek(self, k):
        return _peek(self, k, ("unseen_cameras",))

    def get_next_image_idx(self) -> int:

        if self.in_order:
            image_idx = self.unseen_cameras.pop(0)
        else:
            image_idx = self.unseen_cameras.pop(random.randint(0, len(self.unseen_cameras) - 1))
        if len(self.unseen_cameras) == 0:
            self.unseen_cameras = [i for i in range(len(self.dataset))]

        return image_idx

class MultiTraversalBalancedSampler:

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.dataparser_outputs = self.dataset._dataparser_outputs

        self.traversals = set(self.dataparser_outputs.travel_ids)
        
        travel_ids = np.array(self.dataparser_outputs.travel_ids)
        self.traversal_counts = {
            traversal: (travel_ids == traversal).sum() for traversal in self.traversals}
        self.traversal_indices = {
            traversal: (np.where(travel_ids == traversal)[0]).tolist() for traversal in self.traversals
        }

        self.unseen_traversals = list(self.traversals)
        self.unseen_per_traversal_images = {
            traversal: self.traversal_indices[traversal].copy() for traversal in self.traversals
        }

    def peek(self, k):
        return _peek(self, k, ("unseen_traversals", "unseen_per_traversal_images"))

    def get_next_traversal(self):
        traversal = self.unseen_traversals.pop(random.randint(0, len(self.unseen_traversals) - 1))
        if len(self.unseen_traversals) == 0:
            self.unseen_traversals = list(self.traversals)
        return traversal
    
    def get_next_image_idx(self):
        travel_id = self.get_next_traversal()
        image_idx = self.unseen_per_traversal_images[travel_id].pop(random.randint(0, len(self.unseen_per_traversal_images[travel_id]) - 1))
        if len(self.unseen_per_traversal_images[travel_id]) == 0:
            self.unseen_per_traversal_images[travel_id] = self.traversal_indices[travel_id].copy()
        return image_idx
