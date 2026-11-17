#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from nuplan_scripts.utils.console import say
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
import torch

from torch import Tensor
from torch.nn import Parameter

try:
    from gsplat.cuda._wrapper import spherical_harmonics
except ImportError:
    say("Please install gsplat>=1.0.0")

from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.configs.base_config import InstantiateConfig, PrintableConfig

from .utils import num_sh_bases, random_quat_tensor, matrix_to_quaternion, rotate_vector_to_vector, RGB2SH, quat_to_rotmat
import os as _os

# EXPERIMENT exp/density-1113 (not deployed): density-control variants for the BACKGROUND model, default off.
#   MTGS_DENSITY_MODE=decay        Revising Densification (Rota Bulo et al., ECCV 2024, arXiv 2404.06109; no official code): no hard
#                                  opacity reset; every refinement while densifying lowers each opacity by MTGS_DENSITY_DECAY (0.001)
#   MTGS_DENSITY_MODE=decay_pixgs  decay + Pixel-GS (Zhang et al., ECCV 2024; official github.com/zhengzhang01/Pixel-GS @05c9d55,
#                                  gaussian_model.add_densification_stats): view gradients averaged weighted by the pixels each
#                                  gaussian covers (sum |g|*pix / sum pix); pix approximated by pi*r^2 of gsplat's radius (the official
#                                  rasterizer counts contributing pixels); its depth factor min(1,(d/(0.37*extent))^2) is NOT used
#   MTGS_DENSITY_MODE=mcmc         3DGS-MCMC (Kheradmand et al., NeurIPS 2024; gsplat 1.4.0 strategy/mcmc.py + strategy/ops.py): dead
#                                  (opacity <= 0.005) gaussians relocated to opacity-sampled live ones, 5 % growth per refinement up to
#                                  MTGS_MCMC_CAP, position noise every step (lr * 5e5), opacity / scale L1 regularisers 0.01; no reset,
#                                  no split / clone / cull heuristics
DENSITY_MODE = _os.environ.get("MTGS_DENSITY_MODE", "")
DENSITY_DECAY = float(_os.environ.get("MTGS_DENSITY_DECAY", "0.001"))
MCMC_CAP = int(_os.environ.get("MTGS_MCMC_CAP", "2400000"))
# MTGS_DENSITY_CAP=N (0 = off): gaussian budget for the gradient-driven modes (A / decay / decay_pixgs): a refinement may add at
# most N - current gaussians, spent on the highest average gradients (budgeted densification; culled gaussians free budget).
DENSITY_CAP = int(_os.environ.get("MTGS_DENSITY_CAP", "0"))
MCMC_NOISE_LR = float(_os.environ.get("MTGS_MCMC_NOISE_LR", "5e5"))
MCMC_MIN_OPACITY = 0.005
# 2026-11-14 fixes found by the validation on pn_w2_02 / pw_w2_19 (the converter's occupancy grid ran out of memory, 54 % of the
# budget was dead at the end, blobs in the sky):
#   MTGS_MCMC_STOP_FRAC (0.85)        relocation / growth run until this fraction of the schedule (was stop_split_at = 50 %); the
#                                     refinements after it, and always the last one of the run, only REMOVE dead / invalid
#                                     gaussians, so a checkpoint holds live ones.  1 = relocate to the end and cull once.
#   MTGS_MCMC_NOISE_MAX_SIGMA (1.0)   the position noise is covariance x noise, i.e. it grows with scale^2: a barely alive 20 m
#                                     gaussian jumped ~200 m per step and random-walked up to 29 km (1,849 km in pw_w2_19).  The
#                                     displacement of a step is clamped to this many of the gaussian's own largest sigma (0 = off).
#   MTGS_MCMC_BOUND_M (1000)          scene bound [m from the origin] = the configured skybox radius: background gaussians beyond it
#                                     get no noise and count as dead (0 = off).  The gradient recipe ends with 3 / 84 background
#                                     gaussians beyond 1 km (pn_w2_02 / pw_w2_19), MCMC had 889 / 6,649: a second, semi-transparent
#                                     sky layer in front of the skybox; the skybox under it learns saturated colours that show as
#                                     blue blobs where the layer is thin.  (Not derived from the init points: those end at ~290 m
#                                     in pn_w2_02 while real far buildings are reconstructed out to ~1 km, and reach 2.1 km in
#                                     pw_w2_19 because of a few strays.)
#   MTGS_MCMC_SCALE_CULL (far)        far = gaussians larger than the recipe's far scale limit (cull_scale_thresh x
#                                     far_background_cull_scale_factor = 20 m) count as dead; recipe = the full rule incl. 0.5 m
#                                     within 100 m (3 % of the live gaussians of an MCMC model break it); off.
MCMC_STOP_FRAC = float(_os.environ.get("MTGS_MCMC_STOP_FRAC", "0.85"))
# Culling after densification (2026-11-16, continue_cull_post_densification, on in the MTGS recipe unless MTGS_CULL_POST=0):
# gaussians the second half of the schedule drives transparent were never removed (13-20 % of a served model's rows ended
# below 0.005). MTGS_CULL_POST_ALPHA: the opacity threshold of the culls after stop_split_at (0 = the config's
# cull_alpha_thresh); during densification the config's threshold applies as before.
CULL_POST_ALPHA = float(_os.environ.get("MTGS_CULL_POST_ALPHA", "0"))
MCMC_NOISE_MAX_SIGMA = float(_os.environ.get("MTGS_MCMC_NOISE_MAX_SIGMA", "1.0"))
# MTGS_MCMC_CLEANUP: each = every refinement after the relocation phase removes the dead / invalid gaussians; final = only the last
# refinement of the run does (dead gaussians keep random-walking until then, as in gsplat).
MCMC_CLEANUP = _os.environ.get("MTGS_MCMC_CLEANUP", "each")
MCMC_BOUND_M = float(_os.environ.get("MTGS_MCMC_BOUND_M", "1000"))
MCMC_SCALE_CULL = _os.environ.get("MTGS_MCMC_SCALE_CULL", "far")


def _density_on(model) -> bool:
    return bool(DENSITY_MODE) and getattr(model, "model_name", "") == "background"


# MTGS_MCMC_SAMPLE=grad (experiment 2026-11-13): MCMC draws relocation / growth sources with probability opacity x (pixel-weighted
# average screen gradient / its median) ** MTGS_MCMC_GRAD_POW instead of opacity alone, so weakly reconstructed regions (few large
# gaussians, low opacity, high error) are not starved by the opacity-proportional draw (blurry far side-street in pn_w2_02 f38 R0).
MCMC_SAMPLE = _os.environ.get("MTGS_MCMC_SAMPLE", "opacity")
MCMC_GRAD_POW = float(_os.environ.get("MTGS_MCMC_GRAD_POW", "1.0"))


# EXPERIMENT 2026-11-13 (rare regions: the far side-street of pn_w2_02 f38 CAM_R0 is seen unoccluded by 22 of 6,188 images and is
# blurry at those very training views): the background model keeps a long-term per-gaussian CONTRIBUTION frequency
# f_g = (steps with a non-zero screen-space gradient) / (steps alive); a frustum hit is not enough, the region sits in ~1,900
# frusta but is occluded in most.  Relocated / added MCMC gaussians inherit the counters of their source.  Used by
#   MTGS_VIEW_SAMPLE=cover   coverage-aware view sampling (dataset/utils/sampler.py): image score = mean over its contributing
#                            gaussians of (rarity - 1), rarity = clamp(f_ref / f_g, 1, MTGS_COVER_CAP)
#   MTGS_VIS_LR=1            the Adam step of a contributing gaussian is multiplied by clamp((f_ref / f_g) ** MTGS_VIS_LR_BETA, 1,
#                            MTGS_VIS_LR_MAX) (mtgs_scene_graph._vis_lr_extra)
#   MTGS_COVER_STATS=1       counters only.   f_ref = the MTGS_COVER_REF_Q quantile of f_g (median), refreshed every 500 steps.
# Nothing is used before MTGS_COVER_START steps (frequencies need ~1.5 epochs).  Supported with MTGS_DENSITY_MODE=mcmc only (the
# split / dup / cull path does not carry the counters; they restart on a size change).
COVER_VIEW = _os.environ.get("MTGS_VIEW_SAMPLE", "") == "cover"
VIS_LR = _os.environ.get("MTGS_VIS_LR", "0") == "1"
COVER_ON = COVER_VIEW or VIS_LR or _os.environ.get("MTGS_COVER_STATS", "0") == "1"
COVER_START = int(_os.environ.get("MTGS_COVER_START", "9000"))
COVER_CAP = float(_os.environ.get("MTGS_COVER_CAP", "8"))
COVER_REF_Q = float(_os.environ.get("MTGS_COVER_REF_Q", "0.5"))
VIS_LR_BETA = float(_os.environ.get("MTGS_VIS_LR_BETA", "1.0"))
VIS_LR_MAX = float(_os.environ.get("MTGS_VIS_LR_MAX", "4"))
# A contribution counts only when the gaussian is resolved (projected radius >= MTGS_COVER_MIN_RADIUS px at the training
# resolution): offline on the full-length MCMC 2.4 M model the weak region's gaussians contribute to 209 images (median 493 for
# all), the images that see it rank at the 74-95th percentile of the image score with any radius and at the 90-96th with >= 4 px.
COVER_MIN_RADIUS = float(_os.environ.get("MTGS_COVER_MIN_RADIUS", "4"))
COVER_DUMP = _os.environ.get("MTGS_COVER_DUMP", "")
COVER_LOG_EVERY = int(_os.environ.get("MTGS_COVER_LOG_EVERY", "10000"))   # multiple of 500
_COVER_LAST = int(_os.environ.get("MTGS_MAX_ITERS", "0")) - 1

# EXPERIMENT 2026-11-15 (baked parked cars, px_loc_0195): with MTGS_MASK_VEHICLES=1 the car pixels are out of every loss, so the
# VOLUME of a car that stands in (almost) every pass is seen only through masked pixels and the background is free to grow a blurred
# car there (94-99 % of the pixels such gaussians cover are masked). MTGS_UCULL=1: the background keeps two per-gaussian counters,
#   vis  = training steps in which its centre projects into the image (radius > 0),
#   free = of those, steps in which the centre pixel is UNMASKED and at least MTGS_UCULL_DILATE px from any masked pixel
#          (the mask rim, where the leak and the silhouette features live, counts as masked).
# A gaussian with vis >= MTGS_UCULL_MIN_VIS and free <= MTGS_UCULL_FRAC * vis is unconstrained: it is culled at every refinement
# (also after densification stops, every MTGS_UCULL_EVERY steps) and never split / duplicated, unless its centre lies in the
# ground band (< MTGS_UCULL_GROUND m above the 5th height percentile of the opaque gaussians of its MTGS_UCULL_CELL m cell: the
# road under an always-parked car is unobservable too and must stay). Anything that ANY pass sees through an empty spot gets free
# hits and stays. Counters follow the row edits of _rebuild_gaussians (new rows start at 0); they restart on a resume.
UCULL_ON = _os.environ.get("MTGS_UCULL", "0") == "1"
UCULL_DILATE = int(_os.environ.get("MTGS_UCULL_DILATE", "4"))
UCULL_MIN_VIS = float(_os.environ.get("MTGS_UCULL_MIN_VIS", "50"))
UCULL_FRAC = float(_os.environ.get("MTGS_UCULL_FRAC", "0.05"))
UCULL_EVERY = int(_os.environ.get("MTGS_UCULL_EVERY", "0"))           # after stop_split_at; 0 = every refinement
UCULL_GROUND = float(_os.environ.get("MTGS_UCULL_GROUND", "0.25"))
UCULL_CELL = float(_os.environ.get("MTGS_UCULL_CELL", "3.0"))
UCULL_DUMP = _os.environ.get("MTGS_UCULL_DUMP", "")
# MTGS_UCULL_DECAY < 1: after every judgement both counters are multiplied by it, so the free share is that of the recent past
# (window ~ refine_every / (1 - decay) steps) instead of the gaussian's whole life: with lifetime counters a gaussian that sat on the
# road for a while and then drifted into a parked car's volume kept its early free steps (px retrain 2, spot 26: 1,241 opaque body
# gaussians with a current free share of 0.012 were never culled). 1 = lifetime counters (round-1 behaviour).
UCULL_DECAY = float(_os.environ.get("MTGS_UCULL_DECAY", "1"))
# MTGS_UCULL_OCC=1 (default): a step counts for a gaussian only when its centre is not hidden behind the rendered surface at its
# pixel (camera depth <= rendered depth * (1 + MTGS_UCULL_OCC_REL) + MTGS_UCULL_OCC_ABS m, or the pixel is mostly empty). The
# frustum-only count (0) also counted every distant view in which the centre projects onto a wall or another car in front of
# it: an unmasked pixel = "free" (pw_loc_0182, long straight streets: the gaussians of always-parked cars had an online free
# share of 0.3-0.45, 0.07-0.13 over the views within 40 m, and nothing was culled; px_loc_0195 has few such views).
UCULL_OCC = _os.environ.get("MTGS_UCULL_OCC", "1") == "1"
UCULL_OCC_REL = float(_os.environ.get("MTGS_UCULL_OCC_REL", "0.05"))
UCULL_OCC_ABS = float(_os.environ.get("MTGS_UCULL_OCC_ABS", "0.5"))                    # torch.save of the counters at every cull (diagnostics)


def _pixgs_on(model) -> bool:
    """Pixel-GS weighted gradient statistics: mode decay_pixgs, or MCMC with gradient-aware sampling."""
    return _density_on(model) and (DENSITY_MODE == "decay_pixgs" or (DENSITY_MODE == "mcmc" and MCMC_SAMPLE == "grad"))


def _pix_weight(radii, image_area=None):
    """Pixel-GS pixel count per view, approximated by the area of gsplat's bounding radius (0 where not visible)."""
    r = radii.float()
    w = math.pi * r * r
    if image_area is not None:
        w = w.clamp(max=float(image_area))
    return w


@dataclass
class GaussianSplattingControlConfig(PrintableConfig):
    densify_from_iter: Optional[int] = None
    """period of steps where refinement is turned off"""
    refine_every: Optional[int] = None
    """period of steps where gaussians are culled and densified"""
    stop_split_at: Optional[int] = None
    """stop splitting at this step"""
    reset_alpha_every: Optional[int] = None
    """Every this many refinement steps, reset the alpha"""
    continue_cull_post_densification: Optional[bool] = None
    """If True, continue to cull gaussians post refinement"""

    cull_alpha_thresh: Optional[float] = None
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: Optional[float] = None
    """threshold of scale for culling huge gaussians"""
    far_background_distance_thresh: float = 100.0
    """distance threshold for far background gaussians to avoid culling"""
    far_background_cull_scale_factor: float = 40.0
    """scale multiplier for far background gaussian culling threshold"""
    densify_grad_thresh: Optional[float] = None
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: Optional[float] = None
    """below this size, gaussians are *duplicated*, otherwise split, unit: meters"""
    n_split_samples: Optional[int] = None
    """number of samples to split gaussians into"""
    clone_sample_means: Optional[bool] = None
    """When duplicating gaussians, whether to sample new means or clone the existing ones."""

    cull_scale_ratio: Optional[float] = None
    """minimum scale threshold for culling or splitting gaussians."""
    densify_size_ratio: Optional[float] = None
    """minimum scale threshold for splitting gaussians."""

    stop_screen_size_at: Optional[int] = None
    """stop culling/splitting at this step WRT screen size of gaussians"""
    cull_screen_size: Optional[float] = None
    """if a gaussian is more than this pixel of screen space, cull it"""
    split_screen_size: Optional[float] = None
    """if a gaussian is more than this pixel of screen space, split it"""

    sh_degree: Optional[int] = None
    """maximum degree of spherical harmonics to use"""
    sh_degree_interval: Optional[int] = None
    """every n intervals turn on another sh degree"""
    use_abs_grad: Optional[bool] = None
    """Whether to use absolute gradient or not"""

    scale_dim: int = 3
    """dimension of scale for gaussians
    for scale_dim == 3, original 3D Gaussian splatting.
    for scale_dim == 1, Gaussian marbles.
    """


@dataclass
class VanillaGaussianSplattingModelConfig(InstantiateConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: VanillaGaussianSplattingModel)
    model_type: str = 'vanilla'
    control: Optional[GaussianSplattingControlConfig] = None
    verbose: bool = False


class VanillaGaussianSplattingModel(torch.nn.Module):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: VanillaGaussianSplattingModelConfig

    def __init__(
        self,
        config: VanillaGaussianSplattingModelConfig,
        model_name: Optional[str] = None,
        model_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.model_name = model_name
        self.model_type = self.config.model_type
        self.model_id = model_id
        self.frozen = False

    def populate_modules(self, points_3d=None, features_dc_dim=None, **kwargs):
        assert points_3d is not None
        assert (features_dc_dim is None) or (features_dc_dim is not None and features_dc_dim > 1), \
            "invalid features_dc_dim [{}], must be set to `None` or larger than 1".format(features_dc_dim)
        # Drop exact duplicate points (first occurrence kept). The joint COLMAP cloud (sfm_point_cloud/joint.pcd) is
        # written at centimetre precision, so some points coincide; with >= 4 copies all three nearest-neighbour
        # distances below are 0, the log scale is -inf and the NaN spread to the whole model within 1000 steps
        # (pn_w2_02: 4 copies of one point, pn_w2_05: 9 copies, 2026-11-04).
        xyz = points_3d['xyz']
        if xyz.shape[0] > 1:
            _, first = np.unique(xyz.detach().cpu().numpy(), axis=0, return_index=True)
            if len(first) < xyz.shape[0]:
                keep = torch.from_numpy(np.sort(first)).to(xyz.device)
                say(f"init points: dropped {xyz.shape[0] - len(keep)} exact duplicates of {xyz.shape[0]}")
                # in place: subclasses (MultiColorGaussianSplatting) size their own parameters from the caller's dict
                for k, v in list(points_3d.items()):
                    if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == xyz.shape[0]:
                        points_3d[k] = v[keep]
        means = torch.nn.Parameter(points_3d['xyz'])  # (Location, Color)

        self.xys_grad_norm = None
        self.max_2Dsize = None
        self.vis_counts = None
        num_points = means.shape[0]
        dim_sh = num_sh_bases(self.sh_degree)
        if num_points == 0:
            self._skip_current_model(dim_sh)
            return

        shs = torch.zeros((points_3d['rgb'].shape[0], dim_sh, 3)).float().cuda()
        if self.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(points_3d['rgb'] / 255)
            shs[:, 1:, 3:] = 0.0
        else:
            say("use color only optimization with sigmoid activation")
            shs[:, 0, :3] = torch.logit(points_3d['rgb'] / 255, eps=1e-10)
        
        # .contiguous(): these are views into shs; the fused Adam kernel
        # rejects non-contiguous params (multi-tensor Adam tolerated them)
        if features_dc_dim is None:
            features_dc = torch.nn.Parameter(shs[:, 0, :].contiguous())
        else:
            features_dc = torch.zeros(num_points, features_dc_dim, 3)
            features_dc[:, 0, :] = shs[:, 0, :]
            features_dc = torch.nn.Parameter(features_dc)
        features_rest = torch.nn.Parameter(shs[:, 1:, :].contiguous())

        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

        if self.scale_dim == 3:
            if points_3d.get('normals', None) is not None:
                normals_seed = points_3d['normals'].float()
                normals_seed = normals_seed / torch.norm(
                    normals_seed, dim=-1, keepdim=True
                )
                scales = torch.log(avg_dist.repeat(1, 3))
                scales[:, 2] = torch.log((avg_dist / 10)[:, 0])
                scales = torch.nn.Parameter(scales.detach())
                quats = torch.zeros(len(normals_seed), 4)
                mat = rotate_vector_to_vector(
                    torch.tensor(
                        [0, 0, 1], dtype=torch.float, device=normals_seed.device
                    ).repeat(normals_seed.shape[0], 1),
                    normals_seed,
                )
                quats = matrix_to_quaternion(mat)
                quats = torch.nn.Parameter(quats.detach())
            else:
                scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
                quats = torch.nn.Parameter(random_quat_tensor(num_points))
            
            self.gauss_params = torch.nn.ParameterDict(
                {
                    "means": means,
                    "scales": scales,
                    "quats": quats,
                    "features_dc": features_dc,
                    "features_rest": features_rest,
                    "opacities": opacities,
                }
            )

        elif self.scale_dim == 1:
            scales = torch.nn.Parameter(torch.log(avg_dist))

            self.gauss_params = torch.nn.ParameterDict(
                {
                    "means": means,
                    "scales": scales,
                    "features_dc": features_dc,
                    "features_rest": features_rest,
                    "opacities": opacities,
                }
            )

    def __empty_gaussians(self, num_points=0, dim_sh=None, features_dc_dim=None):
        dim_sh = num_sh_bases(self.sh_degree) if dim_sh is None else dim_sh
        features_dc = torch.zeros(num_points, 3) if features_dc_dim is None else torch.zeros(num_points, features_dc_dim, 3)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self.vis_counts = None
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(num_points, 3)),
                "scales": torch.nn.Parameter(torch.zeros(num_points, self.scale_dim)),
                "quats": torch.nn.Parameter(torch.zeros(num_points, 4)),
                "features_dc": torch.nn.Parameter(features_dc),
                "features_rest": torch.nn.Parameter(torch.zeros(num_points, dim_sh - 1, 3)),
                "opacities": torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1))),
            }
        )
        return self.gauss_params

    def _skip_current_model(self, dim_sh):
        self.__empty_gaussians(0, dim_sh)
        # if we skip a model, there is no need to use time-aware fourier embedding

    def freeze(self):
        for param in self.gauss_params.values():
            param.requires_grad = False
        self.frozen = True

    def update_statistics(self, xys_grad: torch.Tensor, radii: torch.Tensor):
        self.xys_grad = xys_grad
        self.radii = radii

    def get_isotropic_quats(self, num_points):
        quats = torch.zeros(num_points, 4)
        quats[:, 0] = 1.0
        return quats.to(self.device)

    @property
    def device(self):
        return self.gauss_params["means"].device

    @property
    def sh_degree(self):
        return self.config.control.sh_degree
    
    @property
    def ctrl_config(self):
        return self.config.control
    
    @property
    def scale_dim(self):
        return self.config.control.scale_dim

    @property
    def portable_config(self):
        return {
            "type": type(self).__name__,
            "sh_degree": self.sh_degree,
            "scale_dim": self.scale_dim,
        }
    
    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self) -> torch.nn.Parameter:
        return self.gauss_params["means"]

    @property
    def scales(self) -> torch.nn.Parameter:
        if self.scale_dim == 3:
            return self.gauss_params["scales"]
        elif self.scale_dim == 1:
            return self.gauss_params["scales"].repeat(1, 3)

    @property
    def quats(self) -> torch.nn.Parameter:
        if self.scale_dim == 3:
            return self.gauss_params["quats"]
        elif self.scale_dim == 1:
            return self.get_isotropic_quats(self.num_points)

    @property
    def features_dc(self) -> torch.nn.Parameter:
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self) -> torch.nn.Parameter:
        return self.gauss_params["features_rest"]

    @property
    def opacities(self) -> torch.nn.Parameter:
        return self.gauss_params["opacities"]

    @property
    def model_name_abbr(self):
        return self.model_name.split("_")[-1]

    def get_means(self):
        return self.gauss_params['means']

    def get_scales(self):
        return torch.exp(self.scales)

    def get_quats(self): 
        quats = self.quats
        return quats / quats.norm(dim=-1, keepdim=True)

    def get_opacity(self):
        return torch.sigmoid(self.gauss_params['opacities']).squeeze(-1)
    
    def get_rgbs(self, camera_to_worlds):
        assert self.features_dc.dim() == 2, \
            "if Fourier embedding is used in models derived from `VanillaGaussianSplattingModel`, a new `get_rgbs` function must be implemented"
        colors = torch.cat((self.features_dc[:, None, :], self.features_rest), dim=1)
        if self.sh_degree > 0:
            # Camera position detached (as in nerfstudio's splatfacto) and the norm floored: with the position
            # attached, a gaussian within ~1e-4 m of an image's camera centre gave 1/|d| gradients that turned that
            # image's pose_adjustment (3 translation values) to NaN while every gaussian stayed finite (means are
            # detached here); the NaN then spread through camera_opt_regularizer and the model (pn_w2_02 / pn_w2_05,
            # 2026-11-05, first seen with joint.pcd initial points that lie on the camera trajectories).
            viewdirs = self.means.detach() - camera_to_worlds[..., :3, 3].detach()  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            n = min(self.step // self.ctrl_config.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
        
        return rgbs

    def get_gaussians(self, camera_to_worlds, return_features=False, return_v=False, **kwargs):

        return_dict = {
            "means": self.get_means(),
            "scales": self.get_scales(),
            "quats": self.get_quats(),
            "opacities": self.get_opacity(),
        }

        if return_features:
            return_dict["features_dc"] = self.features_dc
            return_dict["features_rest"] = self.features_rest
        else:
            return_dict["rgbs"] = self.get_rgbs(camera_to_worlds)

        if return_v:
            return_dict["velocities"] = torch.zeros_like(return_dict["means"])
        return return_dict

    def get_gaussian_params(self, **kwargs):
        return self.gauss_params

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            f"{self.model_name}.{self.model_type}.{name}": [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    def load_state_dict(self, dict: dict, **kwargs):  # type: ignore

        # resize the parameters to match the new number of points
        newp = dict["gauss_params.means"].shape[0]
        if getattr(self, "num_points", None) is None:
            self.__empty_gaussians(newp)
        else:
            for name, param in self.gauss_params.items():
                old_shape = param.shape
                new_shape = (newp,) + old_shape[1:]
                self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))

        if 'strict' in kwargs:
            kwargs.pop('strict')

        super().load_state_dict(dict, strict=False, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def _rebuild_gaussians(self, optimizers: Optimizers, new_chunks, keep):
        """Replaces every gaussian parameter and its Adam state by the rows `keep` of the virtual
        concatenation [current params] + new_chunks (the split and dup rows: dicts name -> tensor),
        one tensor at a time: a final-size tensor is allocated, the kept rows are index_select'ed
        straight into it, the new rows are filled (zeros for the Adam moments) and the old tensor is
        freed before the next one. Peak = the old plus ONE new copy of the current tensor. The former
        cat + mask-index sequence held old and new copies of param, exp_avg and exp_avg_sq at once
        (pn_gapC_010 OOMed at 90 GB in remove_from_optim: features_rest [N,T,15,3], N=6.5M).
        Row order is unchanged: kept old rows, then kept split rows, then kept dup rows."""
        if UCULL_ON and getattr(self, "_uc_vis", None) is not None and self._uc_vis.shape[0] == self.num_points:
            n_new = sum(c["means"].shape[0] for c in new_chunks)
            z = self._uc_vis.new_zeros(n_new)
            self._uc_vis = torch.cat([self._uc_vis, z])[keep]; self._uc_free = torch.cat([self._uc_free, z])[keep]
        elif UCULL_ON:
            self._uc_vis = self._uc_free = None
        sizes = [self.num_points] + [c["means"].shape[0] for c in new_chunks]
        offsets = [0]
        for n in sizes:
            offsets.append(offsets[-1] + n)
        keep_idx = torch.nonzero(keep).squeeze(1)  # the host sync of this refine: the final size
        ends = torch.searchsorted(keep_idx, torch.tensor(offsets[1:], device=keep_idx.device)).tolist()
        starts = [0] + ends[:-1]
        local_idx = [keep_idx[s:e] - o for s, e, o in zip(starts, ends, offsets)]  # kept rows, per chunk
        n_final = ends[-1]

        def gather(name, old, zero_new):
            out = torch.empty((n_final,) + old.shape[1:], dtype=old.dtype, device=old.device)
            torch.index_select(old, 0, local_idx[0], out=out[:ends[0]])
            if zero_new:
                out[ends[0]:].zero_()
            else:
                for chunk, s, e, idx in zip(new_chunks, starts[1:], ends[1:], local_idx[1:]):
                    torch.index_select(chunk.pop(name), 0, idx, out=out[s:e])  # pop: the chunk's copy dies here
            return out

        # parameter name -> optimizer name (comprehensions: no stray reference to an old parameter)
        names = {id(p): name for name, p in self.gauss_params.items()}
        groups = {names[id(p)]: group for group, (p,) in self.get_gaussian_param_groups().items() if id(p) in names}
        for name in list(self.gauss_params.keys()):
            old = self.gauss_params[name]
            new = torch.nn.Parameter(gather(name, old.data, zero_new=False))
            self.gauss_params[name] = new
            group = groups.get(name)
            if group is None:
                del old
                continue
            optim = optimizers.optimizers[group]
            state = optim.state.pop(optim.param_groups[0]["params"][0], {})
            optim.param_groups[0]["params"] = [new]
            optim.state[new] = state
            # nerfstudio's Optimizers keeps its own [param] list (for grad clipping); never updated before,
            # it pinned the INITIAL parameters for the whole training (features_rest alone: GBs)
            if group in getattr(optimizers, "parameters", {}):
                optimizers.parameters[group] = [new]
            del old  # the last reference to the old parameter: freed before its Adam moments are rebuilt
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = gather(name, state[key], zero_new=True)  # new rows start with zero moments

    def after_train(self, step: int):
        assert step == self.step
        if DENSITY_MODE == "mcmc" and _density_on(self) and getattr(self, "_mcmc_optimizers", None) is not None and not self.frozen:
            self._mcmc_inject_noise()
        if getattr(self, "_stats_step", None) == step:   # accumulated by MTGSSceneModel._update_submodel_statistics_batched
            return
        if self.frozen:
            return
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.ctrl_config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            # Same updates as indexing with the boolean mask, written with torch.where:
            # a boolean index needs the number of hits on the host, i.e. a GPU sync,
            # and this runs for every submodel every step (~5 syncs x ~50 models).
            visible_mask = (self.radii > 0).flatten()

            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)
            assert self.vis_counts is not None
            if _pixgs_on(self):   # Pixel-GS: sum |g| * pix / sum pix
                w = torch.where(visible_mask, _pix_weight(self.radii.reshape(-1)), 0.0)
                self.vis_counts += w
                self.xys_grad_norm += torch.where(visible_mask, self.xys_grad, 0.0) * w
            else:
                self.vis_counts += visible_mask
                self.xys_grad_norm += torch.where(visible_mask, self.xys_grad, 0.0)

            # update the max screen size, as number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            self.max_2Dsize = torch.where(
                visible_mask, torch.maximum(self.max_2Dsize, self.radii.detach()), self.max_2Dsize)

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.frozen:
            return
        if self.step <= self.ctrl_config.densify_from_iter:
            return
        if DENSITY_MODE == "mcmc" and _density_on(self):
            self._mcmc_optimizers = optimizers
            if self.num_points > 0:
                # the last refinement of the run always cleans up (MTGS_MCMC_STOP_FRAC=1: relocation to the very end, one cull)
                last = MCMC_STOP_FRAC > 0 and self.step + self.ctrl_config.refine_every >= self._mcmc_total_steps()
                if self.step < self._mcmc_stop_step() and not last:
                    self._mcmc_refine(optimizers)
                elif MCMC_STOP_FRAC > 0 and (last or MCMC_CLEANUP == "each"):
                    self._mcmc_cleanup(optimizers)
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None
            return
        with torch.no_grad():
            if self.num_points <= 0:
                return
            reset_interval = self.ctrl_config.reset_alpha_every * self.ctrl_config.refine_every
            counts = {}  # GPU scalars for the verbose log, read back once at the end
            new_chunks = []
            extra_cull = None
            n_old = n_before = self.num_points
            uc = None
            if self.step < self.ctrl_config.stop_split_at:
                # then we densify
                if self.config.verbose:
                    say(f"[=================== INFO for {self.model_name_abbr} <{self.model_type}> ===================]")
                    say(f"Do densification at step {self.step}")
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                # avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                if self.config.verbose:
                    counts["avg_grad_norm"] = avg_grad_norm.mean()

                high_grads = (avg_grad_norm > self.ctrl_config.densify_grad_thresh).reshape(-1)
                uc = self._ucull_mask()
                if uc is not None:
                    high_grads &= ~uc                                   # never densify from an unconstrained gaussian
                if DENSITY_CAP > 0 and self.model_name == "background":
                    budget = max(0, DENSITY_CAP - self.num_points)
                    if self.step < self.ctrl_config.stop_screen_size_at:   # screen-size splits (below) are not gradient-driven: they come out of the budget first
                        budget = max(0, budget - int((self.max_2Dsize > self.ctrl_config.split_screen_size).sum()))
                    hg = torch.nonzero(high_grads).squeeze(1)
                    if hg.numel() > budget:
                        high_grads = torch.zeros_like(high_grads)
                        if budget > 0:
                            high_grads[hg[torch.topk(avg_grad_norm.reshape(-1)[hg], budget).indices]] = True
                        if self.step % (10 * self.ctrl_config.refine_every) == 0:
                            say(f"[cap {self.model_name}] step {self.step}: {hg.numel()} candidates, budget {budget}, n {self.num_points}")

                splits = (self.scales.exp().max(dim=-1).values > self.ctrl_config.densify_size_thresh).reshape(-1)
                splits &= high_grads

                if self.step < self.ctrl_config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_config.split_screen_size).reshape(-1)

                nsamps = self.ctrl_config.n_split_samples
                split_idx = torch.nonzero(splits).squeeze(1)  # host sync: the number of split rows
                new_chunks.append(self.split_gaussians(split_idx, nsamps))

                # (unchanged) the dup test sees the scales split_gaussians has just shrunk
                dups = (self.scales.exp().max(dim=-1).values <= self.ctrl_config.densify_size_thresh).reshape(-1)
                dups &= high_grads
                dup_idx = torch.nonzero(dups).squeeze(1)
                new_chunks.append(self.dup_gaussians(dup_idx))

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                extra_cull = splits
                n_before -= split_idx.numel()
                rebuild = True
            elif self.step >= self.ctrl_config.stop_split_at and self.ctrl_config.continue_cull_post_densification:
                rebuild = True
                if UCULL_ON and self.model_name == "background" and (UCULL_EVERY <= 0 or self.step % UCULL_EVERY < self.ctrl_config.refine_every):
                    uc = self._ucull_mask()                     # joins the alpha / scale cull of this rebuild below
            elif (UCULL_ON and self.model_name == "background"
                  and (UCULL_EVERY <= 0 or self.step % UCULL_EVERY < self.ctrl_config.refine_every)):
                uc = self._ucull_mask()
                rebuild = False
                if uc is not None:
                    self._rebuild_gaussians(optimizers, [], ~uc)
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                rebuild = False

            if rebuild:
                # cull mask over the virtual layout [old rows, split rows, dup rows]; the new rows carry
                # no screen-size statistics (the old code appended zeros to max_2Dsize for them)
                if self.step < self.ctrl_config.stop_screen_size_at:
                    assert self.max_2Dsize is not None
                culls = [self.cull_mask(self.gauss_params, self.max_2Dsize, counts)]
                if extra_cull is not None:
                    culls[0] |= extra_cull
                if uc is not None:
                    culls[0] |= uc
                culls += [self.cull_mask(chunk, None, counts) for chunk in new_chunks]
                n_before += sum(c["means"].shape[0] for c in new_chunks)
                self._rebuild_gaussians(optimizers, new_chunks, ~torch.cat(culls))

            if DENSITY_MODE in ("decay", "decay_pixgs") and _density_on(self):
                if self.step < self.ctrl_config.stop_split_at:   # Revising Densification: small decay instead of the hard reset
                    op = torch.sigmoid(self.opacities.data) - DENSITY_DECAY
                    self.opacities.data.copy_(torch.logit(op.clamp(min=1e-6, max=1.0 - 1e-6)))
            elif self.step < self.ctrl_config.stop_split_at and self.step % reset_interval == self.ctrl_config.refine_every:
                if self.config.verbose:
                    say(f"Reset opacity at step {self.step}")
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.ctrl_config.cull_alpha_thresh * 2.0
                # math.log: same float32 bound as torch.logit(...).item() without the GPU round trip; in place
                self.opacities.data.clamp_(max=math.log(reset_value / (1.0 - reset_value)))
                # reset the exp of optimizer
                gaussian_param_groups = self.get_gaussian_param_groups()
                for name, param in gaussian_param_groups.items():
                    if "opacities" in name:
                        optim = optimizers.optimizers[name]
                        param = optim.param_groups[0]["params"][0]
                        param_state = optim.state[param]
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in param_state:
                                param_state[key].zero_()  # in place: no second copy
                        break

            if self.config.verbose and rebuild:
                keys = list(counts.keys())
                read = dict(zip(keys, torch.stack([counts[k].float() for k in keys]).tolist()))  # one sync
                if extra_cull is not None:
                    n_split, n_dup = split_idx.numel(), dup_idx.numel()
                    say(f"avg_grad_norm: {read['avg_grad_norm']}")
                    say(f"Splitting {n_split / n_old} gaussians: {n_split}/{n_old}")
                    say(f"Duplicating {n_dup / n_old} gaussians: {n_dup}/{n_old}")
                say(
                    f"Culled {n_before - self.num_points} gaussians "
                    f"({int(read.get('below_alpha', 0))} below alpha thresh, {int(read.get('toobigs', 0))}"
                    f"({int(read.get('toobigs_ws', 0))}, {int(read.get('toobigs_vs', 0))}) too bigs, {self.num_points} remaining)"
                )

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    # ---- 3DGS-MCMC (experiment): ported from gsplat 1.4.0 strategy/ops.py relocate / sample_add / inject_noise_to_position
    # ---- unconstrained-gaussian culling (experiment, see UCULL_* above)
    @torch.no_grad()
    def _ucull_update(self, xys, radii, free_map, valid_map=None, occ=None):
        """one training step: xys [N, 2] projected centres (render px), radii [N], free_map [H, W] bool (valid, not class-masked, off the
        class-mask rim), valid_map [H, W] bool (lens / ego validity: a centre outside it counts neither as visible nor as free)."""
        n = xys.shape[0]
        if getattr(self, "_uc_vis", None) is None or self._uc_vis.shape[0] != n:
            if getattr(self, "_uc_vis", None) is not None:
                say(f"[ucull {self.model_name}] step {self.step}: {self._uc_vis.shape[0]} -> {n} gaussians, counters restarted")
            self._uc_vis = torch.zeros(n, device=xys.device, dtype=torch.float32)
            self._uc_free = torch.zeros(n, device=xys.device, dtype=torch.float32)
        H, W = free_map.shape
        x = xys[:, 0].floor().long(); y = xys[:, 1].floor().long()
        vis = (radii > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        yc, xc = y.clamp(0, H - 1), x.clamp(0, W - 1)
        if valid_map is not None:
            vis &= valid_map[yc, xc]
        if occ is not None:   # (centre depths [N], rendered depth [H, W], alpha [H, W]): hidden behind the rendered surface = not visible
            gdepth, depth_im, alpha_im = occ
            d = depth_im[yc, xc]
            vis &= (gdepth <= d * (1.0 + UCULL_OCC_REL) + UCULL_OCC_ABS) | (alpha_im[yc, xc] < 0.5)
        free = free_map[yc, xc] & vis
        self._uc_vis += vis; self._uc_free += free

    @torch.no_grad()
    def _ucull_ground(self):
        """per gaussian: the 5th height percentile (z) of the opaque gaussians of its UCULL_CELL m xy cell (-inf: no opaque one)."""
        m = self.means.detach(); xy = m[:, :2]; z = m[:, 2]
        ij = ((xy - xy.min(0).values) / UCULL_CELL).long(); nj = int(ij[:, 1].max()) + 1
        key = ij[:, 0] * nj + ij[:, 1]; K = int(key.max()) + 1
        o = torch.sigmoid(self.opacities.detach().flatten()) > 0.3
        ko, zo = key[o], z[o]
        p = torch.argsort(zo); ko, zo = ko[p], zo[p]
        p = torch.argsort(ko, stable=True); ko, zo = ko[p], zo[p]
        cnt = torch.bincount(ko, minlength=K); start = torch.cumsum(cnt, 0) - cnt
        q = start + ((cnt - 1).clamp_min(0).float() * 0.05).long()
        gz = torch.full((K,), float("-inf"), device=m.device)
        has = cnt > 0
        gz[has] = zo[q[has]]
        return gz[key]

    @torch.no_grad()
    def _ucull_mask(self):
        """[N] bool: unconstrained gaussians (see UCULL_*), or None when the experiment is off / no counters yet."""
        if not UCULL_ON or self.model_name != "background" or getattr(self, "_uc_vis", None) is None \
                or self._uc_vis.shape[0] != self.num_points:
            return None
        v, f = self._uc_vis, self._uc_free
        cand = (v >= UCULL_MIN_VIS) & (f <= UCULL_FRAC * v)
        cand &= self.means.detach()[:, 2] > self._ucull_ground() + UCULL_GROUND
        n = int(cand.sum())
        say(f"[ucull {self.model_name}] step {self.step}: {n} unconstrained of {self.num_points} "
            f"(judged {int((v >= UCULL_MIN_VIS).sum())})")
        if UCULL_DECAY < 1:
            v.mul_(UCULL_DECAY); f.mul_(UCULL_DECAY)                 # in place: the rebuild below re-indexes these tensors
        if UCULL_DUMP:
            torch.save({"step": self.step, "vis": v, "free": f, "cand": cand, "means": self.means.detach(),
                        "opac": torch.sigmoid(self.opacities.detach().flatten())}, UCULL_DUMP)
        return cand

    # ---- contribution-frequency counters (experiment, see COVER_* above)
    @torch.no_grad()
    def _cover_update(self, grad_norm, radii, step):
        """one training step: grad_norm / radii [N] = screen-space gradient norm and projected radius of this model's gaussians.
        Returns the contribution mask."""
        n = grad_norm.shape[0]; dev = grad_norm.device
        if getattr(self, "_cov_cnt", None) is None or self._cov_cnt.shape[0] != n:
            if getattr(self, "_cov_cnt", None) is not None:
                say(f"[cover {self.model_name}] step {step}: {self._cov_cnt.shape[0]} -> {n} gaussians, counters restarted")
            self._cov_cnt = torch.zeros(n, device=dev, dtype=torch.float32)
            self._cov_age = torch.zeros(n, device=dev, dtype=torch.float32)
            self._cov_ref = torch.full((), 1e-3, device=dev)
        c = (grad_norm > 0) & (radii >= COVER_MIN_RADIUS)
        self._cov_cnt += c
        self._cov_age += 1
        if step % 500 == 0:   # reference frequency over a subsample of the gaussians old enough to judge
            idx = torch.randint(0, n, (min(n, 400000),), device=dev)
            f = self._cov_cnt[idx] / self._cov_age[idx].clamp_min(1)
            # the reference is the quantile over LIVE gaussians: MCMC leaves dead ones (opacity <= MCMC_MIN_OPACITY) in the model once
            # relocation stops (41 % of the 2.4 M at the end of the 11-13 run), and they would drag the reference down
            live = torch.sigmoid(self.opacities.detach().flatten()[idx]) > MCMC_MIN_OPACITY
            f = torch.where((self._cov_age[idx] >= 0.5 * COVER_START) & live, f, torch.nan)
            ref = torch.nanquantile(f, COVER_REF_Q)
            self._cov_ref = torch.where(torch.isfinite(ref) & (ref > 0), ref, self._cov_ref)
            if step % COVER_LOG_EVERY == 0:   # log: one sync per 10k steps
                q = torch.nanquantile(f, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=dev))
                say(f"[cover {self.model_name}] step {step}: f_g p5/25/50/75/95 {[round(float(x), 5) for x in q]} ref {float(self._cov_ref):.5f}", flush=True)   # print: rich eats the [tag] and wraps
        if COVER_DUMP and (step == _COVER_LAST or (step > 0 and step % 20000 == 0)):
            torch.save({"step": step, "cnt": self._cov_cnt, "age": self._cov_age, "ref": self._cov_ref, "means": self.means.detach()}, COVER_DUMP)
        return c

    def _cover_ratio(self):
        """f_ref / f_g per gaussian (1 for gaussians too young to judge); a gaussian that never contributed counts as half a hit."""
        r = self._cov_ref * self._cov_age / self._cov_cnt.clamp_min(0.5)
        return torch.where(self._cov_age >= 0.5 * COVER_START, r, 1.0)

    def _cover_rarity(self):
        return self._cover_ratio().clamp(1.0, COVER_CAP)

    def _cover_step_scale(self):
        return self._cover_ratio().clamp_min(1.0).pow(VIS_LR_BETA).clamp(max=VIS_LR_MAX)

    def _mcmc_groups(self, optimizers):
        names = {id(p): name for name, p in self.gauss_params.items()}
        return {names[id(p)]: group for group, (p,) in self.get_gaussian_param_groups().items() if id(p) in names}

    def _mcmc_update(self, optimizers, param_fn, state_fn):
        """gsplat's _update_param_with_optimizer for MTGS's named groups (+ nerfstudio's Optimizers.parameters list)."""
        groups = self._mcmc_groups(optimizers)
        for name in list(self.gauss_params.keys()):
            old = self.gauss_params[name]
            new = torch.nn.Parameter(param_fn(name, old.data))
            self.gauss_params[name] = new
            group = groups.get(name)
            if group is None:
                continue
            optim = optimizers.optimizers[group]
            st = optim.state.pop(optim.param_groups[0]["params"][0], {})
            for key in list(st.keys()):
                if key != "step":
                    st[key] = state_fn(key, st[key])
            optim.param_groups[0]["params"] = [new]
            optim.state[new] = st
            if group in getattr(optimizers, "parameters", {}):
                optimizers.parameters[group] = [new]

    def _mcmc_total_steps(self):
        return int(_os.environ.get("MTGS_MAX_ITERS", "0")) or 2 * self.ctrl_config.stop_split_at

    def _mcmc_stop_step(self):
        if MCMC_STOP_FRAC <= 0:
            return self.ctrl_config.stop_split_at
        return int(MCMC_STOP_FRAC * self._mcmc_total_steps())

    def _mcmc_bound(self):
        """Scene bound [m from the origin] of the background model (None = no bound)."""
        return MCMC_BOUND_M if MCMC_BOUND_M > 0 else None

    @torch.no_grad()
    def _mcmc_invalid(self):
        """Gaussians MCMC must not keep: beyond the scene bound, or larger than the recipe's scale-cull rule."""
        m = self.means.detach()
        bad = torch.zeros(m.shape[0], dtype=torch.bool, device=m.device)
        bound = self._mcmc_bound()
        if bound is not None:
            bad |= ~(m.norm(dim=-1) <= bound)   # also catches non-finite positions
        if MCMC_SCALE_CULL == "recipe":
            bad |= torch.exp(self.scales.detach()).amax(dim=-1) > self.cull_scale_thresh(m).reshape(-1)
        elif MCMC_SCALE_CULL == "far":
            bad |= torch.exp(self.scales.detach()).amax(dim=-1) > self.ctrl_config.cull_scale_thresh * self.ctrl_config.far_background_cull_scale_factor
        return bad

    @torch.no_grad()
    def _mcmc_cleanup(self, optimizers):
        """After the relocation phase: drop dead / invalid gaussians (rows and their Adam moments) instead of relocating them."""
        opac = torch.sigmoid(self.opacities.detach()).flatten()
        drop = (opac <= MCMC_MIN_OPACITY) | self._mcmc_invalid()
        n_drop = int(drop.sum())
        if n_drop == 0:
            return
        keep = ~drop
        n_before = self.num_points
        self._mcmc_update(optimizers, lambda name, p: p[keep], lambda key, v: v[keep])
        if getattr(self, "_cov_cnt", None) is not None and self._cov_cnt.shape[0] == n_before:
            self._cov_cnt = self._cov_cnt[keep]; self._cov_age = self._cov_age[keep]
        say(f"[mcmc {self.model_name}] step {self.step}: cleanup removed {n_drop} dead / invalid gaussians, {self.num_points} left")

    @torch.no_grad()
    def _mcmc_refine(self, optimizers):
        from gsplat.relocation import compute_relocation
        if not hasattr(self, "_mcmc_binoms"):
            n_max = 51
            b = torch.zeros((n_max, n_max))
            for n in range(n_max):
                for k in range(n + 1):
                    b[n, k] = math.comb(n, k)
            self._mcmc_binoms = b.to(self.device)
        eps = torch.finfo(torch.float32).eps

        def sample(probs, n):
            return torch.multinomial(probs, n, replacement=True)

        # relocate dead gaussians (gsplat ops.relocate)
        opac = torch.sigmoid(self.opacities.detach()).flatten()
        invalid = self._mcmc_invalid()
        dead = (opac <= MCMC_MIN_OPACITY) | invalid
        if self.step % (10 * self.ctrl_config.refine_every) == 0:
            say(f"[mcmc {self.model_name}] step {self.step}: {int(dead.sum())} dead of {opac.shape[0]}, of them {int(invalid.sum())} out of bound / oversized")
        dead_idx = dead.nonzero(as_tuple=True)[0]; alive_idx = (~dead).nonzero(as_tuple=True)[0]
        n_dead = int(dead_idx.numel())
        gw = None   # gradient weight of the draw (None = opacity only, as gsplat)
        if MCMC_SAMPLE == "grad" and self.xys_grad_norm is not None and self.vis_counts is not None and self.xys_grad_norm.shape[0] == opac.shape[0]:
            g = (self.xys_grad_norm / self.vis_counts).flatten()
            seen = g > 0
            if bool(seen.any()):
                gw = (g / g[seen].median().clamp_min(1e-12)).clamp(0.1, 10.0) ** MCMC_GRAD_POW
        if MCMC_SAMPLE == "grad" and self.step % (10 * self.ctrl_config.refine_every) == 0:
            say(f"[mcmc-grad {self.model_name}] step {self.step}: " + ("NO gradient statistics, opacity draw" if gw is None else
                        f"weight p10/50/90 {[round(float(x), 2) for x in gw.quantile(torch.tensor([0.1, 0.5, 0.9], device=gw.device))]}"))
        if n_dead > 0 and alive_idx.numel() > 0:
            src = alive_idx[sample(opac[alive_idx] if gw is None else opac[alive_idx] * gw[alive_idx], n_dead)]
            new_op, new_sc = compute_relocation(opacities=opac[src], scales=torch.exp(self.scales.detach())[src],
                                                ratios=torch.bincount(src)[src] + 1, binoms=self._mcmc_binoms)
            new_op = new_op.clamp(max=1.0 - eps, min=MCMC_MIN_OPACITY)

            def pf(name, p):
                if name == "opacities":
                    p[src] = torch.logit(new_op).reshape(p[src].shape)
                elif name == "scales":
                    p[src] = torch.log(new_sc)
                p[dead_idx] = p[src]
                return p

            def sf(key, v):
                v[src] = 0
                return v
            self._mcmc_update(optimizers, pf, sf)
            if getattr(self, "_cov_cnt", None) is not None and self._cov_cnt.shape[0] == opac.shape[0]:   # relocated rows inherit the source's counters
                self._cov_cnt[dead_idx] = self._cov_cnt[src]; self._cov_age[dead_idx] = self._cov_age[src]
        # add new gaussians (gsplat ops.sample_add), 5 % per refinement up to the cap
        n_cur = self.num_points
        n_add = max(0, min(MCMC_CAP, int(1.05 * n_cur)) - n_cur)
        if n_add > 0:
            opac = torch.sigmoid(self.opacities.detach()).flatten()
            opac = torch.where(self._mcmc_invalid(), torch.zeros_like(opac), opac)   # never grow from an invalid gaussian
            src = sample(opac if gw is None else opac * gw, n_add)   # gw: statistics of the gaussians before this refinement (relocated rows keep their own)
            new_op, new_sc = compute_relocation(opacities=opac[src], scales=torch.exp(self.scales.detach())[src],
                                                ratios=torch.bincount(src)[src] + 1, binoms=self._mcmc_binoms)
            new_op = new_op.clamp(max=1.0 - eps, min=MCMC_MIN_OPACITY)

            def pf2(name, p):
                if name == "opacities":
                    p[src] = torch.logit(new_op).reshape(p[src].shape)
                elif name == "scales":
                    p[src] = torch.log(new_sc)
                return torch.cat([p, p[src]])

            def sf2(key, v):
                return torch.cat([v, torch.zeros((n_add,) + v.shape[1:], dtype=v.dtype, device=v.device)])
            self._mcmc_update(optimizers, pf2, sf2)
            if getattr(self, "_cov_cnt", None) is not None and self._cov_cnt.shape[0] == n_cur:
                self._cov_cnt = torch.cat([self._cov_cnt, self._cov_cnt[src]]); self._cov_age = torch.cat([self._cov_age, self._cov_age[src]])
        if self.config.verbose:
            say(f"[mcmc {self.model_name}] step {self.step}: relocated {n_dead}, added {n_add}, now {self.num_points}")

    @torch.no_grad()
    def _mcmc_inject_noise(self):
        from gsplat import quat_scale_to_covar_preci
        groups = self._mcmc_groups(self._mcmc_optimizers)
        g = groups.get("means")
        if g is None:
            return
        lr = self._mcmc_optimizers.optimizers[g].param_groups[0]["lr"]
        opac = torch.sigmoid(self.opacities.detach()).flatten()
        q = self.quats.detach(); q = q / q.norm(dim=-1, keepdim=True)
        covars, _ = quat_scale_to_covar_preci(q, torch.exp(self.scales.detach()), compute_covar=True, compute_preci=False, triu=False)
        k, x0 = 100.0, 0.995
        noise = torch.randn_like(self.means) * (1.0 / (1.0 + torch.exp(-k * ((1.0 - opac) - x0)))).unsqueeze(-1) * (lr * MCMC_NOISE_LR)
        disp = torch.einsum("bij,bj->bi", covars, noise)
        if MCMC_NOISE_MAX_SIGMA > 0:   # at most this many own sigmas per step (the raw step grows with scale^2)
            lim = MCMC_NOISE_MAX_SIGMA * torch.exp(self.scales.detach()).amax(dim=-1, keepdim=True)
            disp = disp * (lim / disp.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
        bound = self._mcmc_bound()
        if bound is not None:          # nothing beyond the scene bound moves further out
            disp = torch.where((self.means.detach().norm(dim=-1, keepdim=True) <= bound), disp, torch.zeros_like(disp))
        self.means.data.add_(disp)

    def cull_scale_thresh(self, means):
        """World-space scale above which a gaussian is culled, per gaussian (subclasses change the far rule)."""
        # Temporal fix for far background gaussians.
        # Do not cull far background gaussians.
        far_mask = means.norm(dim=-1) > self.ctrl_config.far_background_distance_thresh
        return (
            torch.where(
                far_mask,
                self.ctrl_config.far_background_cull_scale_factor,
                1.0,
            )
            * self.ctrl_config.cull_scale_thresh
        )

    def cull_mask(self, params, max_2Dsize, counts):
        """
        Mask of the rows of `params` (a mapping name -> tensor) to delete: the ones under the opacity
        threshold and, from the first opacity reset until stop_split_at, the huge ones in world space and (until
        stop_screen_size_at) in screen space; after densification (continue_cull_post_densification) only the
        transparent ones. max_2Dsize None: rows without screen statistics.
        The counts for the verbose log are summed into `counts` as GPU scalars: no host sync here.
        """
        # cull transparent ones
        thresh = self.ctrl_config.cull_alpha_thresh
        if CULL_POST_ALPHA > 0 and self.step >= self.ctrl_config.stop_split_at:
            thresh = CULL_POST_ALPHA
        culls = (torch.sigmoid(params["opacities"]) < thresh).reshape(-1)
        counts["below_alpha"] = counts.get("below_alpha", 0) + culls.sum()

        # cull huge ones: only while densifying. After stop_split_at nothing splits, and a gaussian that has grown past the
        # size limit is a surface the model needs (pn_w2_02, 2026-11-16: removing the 26k = 0.6 % such rows of the final
        # model drops the route NCC 0.642 -> 0.588, removing the 1.29 M rows below 0.005 changes nothing); culled at every
        # refinement, its neighbours grow to fill the gap and are culled in turn (continue_cull with the size rule: 0.557)
        if self.ctrl_config.refine_every * self.ctrl_config.reset_alpha_every < self.step < self.ctrl_config.stop_split_at:
            toobigs = (torch.exp(params["scales"]).max(dim=-1).values > self.cull_scale_thresh(params["means"])).reshape(-1)
            counts["toobigs_ws"] = counts.get("toobigs_ws", 0) + toobigs.sum()
            if self.step < self.ctrl_config.stop_screen_size_at and max_2Dsize is not None:
                # cull big screen space
                toobigs_vs = (max_2Dsize > self.ctrl_config.cull_screen_size).reshape(-1)
                counts["toobigs_vs"] = counts.get("toobigs_vs", 0) + toobigs_vs.sum()
                toobigs |= toobigs_vs

            culls |= toobigs
            counts["toobigs"] = counts.get("toobigs", 0) + toobigs.sum()
        return culls

    def split_gaussians(self, split_idx, samps):
        """
        This function splits gaussians that are too large
        split_idx: long index of the gaussians to split; returns `samps` new rows per gaussian,
        sample-major ([all, all, ...]: the order of the former mask index + repeat(samps, 1))
        """
        n_splits = split_idx.numel()
        # one index tensor instead of ~12 boolean-mask indexings (each a nonzero + host sync)
        idx = split_idx.repeat(samps)
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales.index_select(0, idx)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats.index_select(0, idx)
        quats = quats / quats.norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats)  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze(-1)
        new_means = rotated_samples + self.means.index_select(0, idx)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.gauss_params["scales"].index_select(0, idx)) / size_fac)
        self.scales[split_idx] = torch.log(torch.exp(self.scales[split_idx]) / size_fac)
        out = {"means": new_means, "scales": new_scales}
        # step 2 & 3, every other property (colours, opacities, quats, per-traversal features, any rank) is copied
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param.index_select(0, idx)
        return out

    def dup_gaussians(self, dup_idx):
        """
        This function duplicates gaussians that are too small
        dup_idx: long index of the gaussians to duplicate
        """
        n_dups = dup_idx.numel()
        new_dups = {}
        for name, param in self.gauss_params.items():

            if self.ctrl_config.clone_sample_means and name == 'means':
                centered_samples = torch.randn((n_dups, 3), device=self.device)  # Nx3 of axis-aligned scales
                scaled_samples = (
                    torch.exp(self.scales.index_select(0, dup_idx)) * centered_samples
                )  # how these scales are rotated
                quats = self.quats.index_select(0, dup_idx)
                quats = quats / quats.norm(dim=-1, keepdim=True)  # normalize them first
                rots = quat_to_rotmat(quats)  # how these scales are rotated
                rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze(-1)
                new_means = rotated_samples + self.means.index_select(0, dup_idx)
                new_dups[name] = new_means
                continue

            new_dups[name] = param.index_select(0, dup_idx)
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.ctrl_config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step
    
    def translate(self, translate_vector: Tensor):
        assert translate_vector.shape == (3,)
        new_means = self.gauss_params['means'] + translate_vector
        self.gauss_params['means'].data = new_means
