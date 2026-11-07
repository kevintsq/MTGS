#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
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
    print("Please install gsplat>=1.0.0")

from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.configs.base_config import InstantiateConfig, PrintableConfig
from nerfstudio.utils.rich_utils import CONSOLE

from .utils import num_sh_bases, random_quat_tensor, matrix_to_quaternion, rotate_vector_to_vector, RGB2SH, quat_to_rotmat

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
                CONSOLE.log(f"init points: dropped {xyz.shape[0] - len(keep)} exact duplicates of {xyz.shape[0]}")
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
            CONSOLE.log("use color only optimization with sigmoid activation")
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
        with torch.no_grad():
            if self.num_points <= 0:
                return
            reset_interval = self.ctrl_config.reset_alpha_every * self.ctrl_config.refine_every
            counts = {}  # GPU scalars for the verbose log, read back once at the end
            new_chunks = []
            extra_cull = None
            n_old = n_before = self.num_points
            if self.step < self.ctrl_config.stop_split_at:
                # then we densify
                if self.config.verbose:
                    CONSOLE.log(f"[=================== INFO for {self.model_name_abbr} <{self.model_type}> ===================]")
                    CONSOLE.log(f"Do densification at step {self.step}")
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                # avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                if self.config.verbose:
                    counts["avg_grad_norm"] = avg_grad_norm.mean()

                high_grads = (avg_grad_norm > self.ctrl_config.densify_grad_thresh).reshape(-1)

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
                culls += [self.cull_mask(chunk, None, counts) for chunk in new_chunks]
                n_before += sum(c["means"].shape[0] for c in new_chunks)
                self._rebuild_gaussians(optimizers, new_chunks, ~torch.cat(culls))

            if self.step < self.ctrl_config.stop_split_at and self.step % reset_interval == self.ctrl_config.refine_every:
                if self.config.verbose:
                    CONSOLE.log(f"Reset opacity at step {self.step}")
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
                    CONSOLE.log(f"avg_grad_norm: {read['avg_grad_norm']}")
                    CONSOLE.log(f"Splitting {n_split / n_old} gaussians: {n_split}/{n_old}")
                    CONSOLE.log(f"Duplicating {n_dup / n_old} gaussians: {n_dup}/{n_old}")
                CONSOLE.log(
                    f"Culled {n_before - self.num_points} gaussians "
                    f"({int(read.get('below_alpha', 0))} below alpha thresh, {int(read.get('toobigs', 0))}"
                    f"({int(read.get('toobigs_ws', 0))}, {int(read.get('toobigs_vs', 0))}) too bigs, {self.num_points} remaining)"
                )

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

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
        threshold and, after the first opacity reset, the huge ones in world space and (until
        stop_screen_size_at) in screen space. max_2Dsize None: rows without screen statistics.
        The counts for the verbose log are summed into `counts` as GPU scalars: no host sync here.
        """
        # cull transparent ones
        culls = (torch.sigmoid(params["opacities"]) < self.ctrl_config.cull_alpha_thresh).reshape(-1)
        counts["below_alpha"] = counts.get("below_alpha", 0) + culls.sum()

        if self.step > self.ctrl_config.refine_every * self.ctrl_config.reset_alpha_every:
            # cull huge ones
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
