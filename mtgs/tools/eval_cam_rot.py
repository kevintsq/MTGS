#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
"""Optional eval mode: eval PSNR / SSIM after removing one constant camera rotation per (drive, camera).

Recipes that train on SfM camera poses can leave a constant per-camera rotation (0.2-0.4 deg) between a vehicle's
cameras and the multi-pass model; it is invisible in renders but costs 2-4 dB of eval PSNR, so the raw number is
not comparable with lidar-pose recipes. With this mode the eval reports BOTH the raw metrics and the metrics with
that rotation fitted and removed, plus the table of fitted rotations.

Fit, per (drive = log date + vehicle, camera) over its eval images: a rotation of the camera about its own centre is
a homography of the rendered image, so every image is rendered once and all candidate rotations are scored as one
batch of warps on the GPU (masked PSNR against the photo on a fixed interior crop). Coarse grid +-`bound` deg in
yaw / pitch / roll, then a fine grid around the best. The aligned metrics are then computed by rendering at the
fitted rotation through the model (exact, not the warp).

A group is NOT fitted (identity kept, said loudly, listed in the table) when it has too few eval images
(MTGS_EVAL_FIT_CAM_ROT_MIN_FRAMES, default 8), when its coarse optimum lies on the search bound
(MTGS_EVAL_FIT_CAM_ROT_BOUND, default 1.0 deg), or when a rotation larger than one fine step does not improve
most of its images.

Needs the full training checkpoint (per-image camera deltas, appearance): run it in the eval of the training run
(`MTGS_EVAL_FIT_CAM_ROT=1` for nuplan_scripts.auto_reconstruction.render_reconstruction) or on a kept run dir:
    python -m mtgs.tools.eval_cam_rot --recon_dir <.../background_reconstruction/<block>> [--out result.json]
"""
import argparse
import json
import math
import os
from math import isnan
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from nuplan_scripts.utils.console import say

_AXES = ("yaw", "pitch", "roll")


def _rot_nf(ypr_deg: torch.Tensor) -> torch.Tensor:
    """[N, 3] yaw / pitch / roll in degrees -> [N, 3, 3] rotation in the nerfstudio camera frame
    (x right, y up, z back): R = Ry(yaw) @ Rx(pitch) @ Rz(roll); the camera becomes c2w @ R."""
    a = torch.deg2rad(ypr_deg)
    cy, sy, cp, sp, cr, sr = a[:, 0].cos(), a[:, 0].sin(), a[:, 1].cos(), a[:, 1].sin(), a[:, 2].cos(), a[:, 2].sin()
    z, o = torch.zeros_like(cy), torch.ones_like(cy)
    ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).reshape(-1, 3, 3)
    rx = torch.stack([o, z, z, z, cp, -sp, z, sp, cp], -1).reshape(-1, 3, 3)
    rz = torch.stack([cr, -sr, z, sr, cr, z, z, z, o], -1).reshape(-1, 3, 3)
    return ry @ rx @ rz


def _grid(centre: torch.Tensor, half: float, step: float) -> torch.Tensor:
    """All yaw / pitch / roll combinations centre +- half in steps of step: [n^3, 3]."""
    n = int(round(half / step))
    off = torch.arange(-n, n + 1, device=centre.device, dtype=centre.dtype) * step
    g = torch.stack(torch.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    return centre[None] + g


def _warp_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, K: torch.Tensor, rots_nf: torch.Tensor,
               margin: int, chunk: int) -> torch.Tensor:
    """Masked PSNR of `pred` seen from the camera rotated by each candidate, against `gt`.

    pred, gt: [H, W, 3]; mask: [H, W, 1] bool; K: [3, 3] pinhole of the render; rots_nf: [C, 3, 3]. Returns [C].
    The new image at pixel x' is the old one at x ~ K R_cv K^-1 x' (R_cv = F R_nf F, F = diag(1, -1, -1)).
    """
    pred, gt = pred.float(), gt.float()      # the warps stay float32 on the device whatever the render precision
    H, W = pred.shape[:2]
    dev, dt = pred.device, pred.dtype
    flip = torch.tensor([1.0, -1.0, -1.0], device=dev, dtype=dt)
    Hm = K[None] @ (flip[None, :, None] * rots_nf * flip[None, None, :]) @ torch.linalg.inv(K)[None]     # [C, 3, 3]
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=dt) + 0.5, torch.arange(W, device=dev, dtype=dt) + 0.5,
                          indexing="ij")
    xh = torch.stack([u, v, torch.ones_like(u)], -1).reshape(-1, 3)                                       # [HW, 3]
    inner = torch.zeros((H, W), dtype=torch.bool, device=dev)
    inner[margin:H - margin, margin:W - margin] = True
    m = (mask[..., 0] & inner)[None, None].to(dt)                                                         # [1, 1, H, W]
    denom = (3.0 * m.sum()).clamp_min(1.0)
    src = pred.permute(2, 0, 1)[None]                                                                     # [1, 3, H, W]
    tgt = gt.permute(2, 0, 1)[None]
    out = []
    for s in range(0, rots_nf.shape[0], chunk):
        h = Hm[s:s + chunk]
        x = xh[None] @ h.transpose(1, 2)                                                                  # [c, HW, 3]
        g = torch.stack([2.0 * x[..., 0] / x[..., 2] / W - 1.0, 2.0 * x[..., 1] / x[..., 2] / H - 1.0], -1)
        w = F.grid_sample(src.expand(h.shape[0], -1, -1, -1), g.reshape(-1, H, W, 2), mode="bilinear",
                          padding_mode="border", align_corners=False)
        mse = (((w - tgt) ** 2) * m).sum(dim=(1, 2, 3)) / denom
        out.append(-10.0 * torch.log10(mse.clamp_min(1e-12)))
    return torch.cat(out)


def _average(metrics_list: List[Dict[str, float]], get_std: bool) -> Dict[str, float]:
    """The averaging of CustomPipeline.get_average_eval_image_metrics (NaN = image of another traversal)."""
    res = {}
    for key in metrics_list[0].keys():
        vals = [float(m[key]) for m in metrics_list]
        vals = torch.tensor([v for v in vals if not isnan(v)])
        if get_std:
            std, mean = torch.std_mean(vals)
            res[key], res[f"{key}_std"] = float(mean), float(std)
        else:
            res[key] = float(vals.mean())
    return res


def _group_of(camera) -> Tuple[str, str]:
    log_name, cam = camera.metadata["data_path"][0], camera.metadata["data_path"][1]
    return "_".join(str(log_name).split("_")[:2]), str(cam)


def _render(model, camera, batch):
    """One eval render: prediction, photo and metric mask exactly as get_image_metrics_and_images uses them."""
    outputs = model.get_outputs_for_camera(camera=camera)
    gt = model.composite_with_background(model.get_gt_img(batch["image"]), outputs["background"])
    pred = outputs["rgb_appearance"].to(model.device)
    mask = model.combine_visible_mask(pred, gt, batch=batch)
    return outputs, pred, gt, mask


def _pinhole(camera, pred: torch.Tensor) -> torch.Tensor:
    K = torch.eye(3, device=pred.device, dtype=torch.float32)
    K[0, 0], K[1, 1] = camera.fx.reshape(-1)[0], camera.fy.reshape(-1)[0]
    K[0, 2], K[1, 2] = camera.cx.reshape(-1)[0], camera.cy.reshape(-1)[0]
    return K


def _margin(K: torch.Tensor, max_deg: float) -> int:
    """Border (px) left out of the fit so every candidate is scored on the same, fully rendered pixels."""
    return int(math.ceil(2.0 * float(K[0, 0]) * math.tan(math.radians(max_deg)))) + 2


def eval_fit_cam_rot(pipeline, get_std: bool = True, bound: Optional[float] = None, coarse_step: float = 0.25,
                     fine_step: float = 0.05, min_frames: Optional[int] = None, min_improved: float = 0.7,
                     chunk: int = 64) -> Dict:
    """Raw and rotation-aligned eval metrics of `pipeline` plus the fitted per-(drive, camera) rotations.

    Returns {"raw": metrics, "aligned": metrics, "fit": {"<drive>/<camera>": {...}}, "settings": {...}}.
    """
    bound = float(os.environ.get("MTGS_EVAL_FIT_CAM_ROT_BOUND", 1.0)) if bound is None else bound
    min_frames = int(os.environ.get("MTGS_EVAL_FIT_CAM_ROT_MIN_FRAMES", 8)) if min_frames is None else min_frames
    model, dm = pipeline.model, pipeline.datamanager
    dev = model.device
    travel_ids = getattr(dm.eval_dataparser_outputs, "travel_ids", None)
    tset = list(set(travel_ids)) if travel_ids is not None else None
    was_training = pipeline.training
    pipeline.eval()
    zero = torch.zeros(3, device=dev)
    coarse = _grid(zero, bound, coarse_step)                    # the same candidates for every group
    coarse_R = _rot_nf(coarse)
    i_zero = int((coarse.abs().sum(-1)).argmin())
    max_deg = bound + 3 * fine_step
    groups: Dict[Tuple[str, str], Dict] = {}
    raw_list: List[Dict[str, float]] = []

    with torch.inference_mode():
        # pass 1: raw metrics (identity, exact) + coarse candidate scores per group
        for camera, batch in dm.fixed_indices_eval_dataloader:
            outputs, pred, gt, mask = _render(model, camera, batch)
            m, _ = model.get_image_metrics_and_images(outputs, batch, travel_id_set=tset, return_image_dict=False)
            raw_list.append(m)
            K = _pinhole(camera, pred)
            margin = _margin(K, max_deg)
            sc = _warp_psnr(pred, gt, mask, K, coarse_R, margin, chunk)
            g = groups.setdefault(_group_of(camera), dict(n=0, sum=torch.zeros_like(sc), own=[]))
            g["n"] += 1
            g["sum"] += sc
            g["own"].append(sc.argmax())
        # coarse optimum per group, bound and frame-count checks
        for key, g in groups.items():
            mean = g["sum"] / g["n"]
            ib = int(mean.argmax())
            g["coarse"] = coarse[ib]
            g["reason"] = None
            if g["n"] < min_frames:
                g["reason"] = f"only {g['n']} eval images (< {min_frames})"
            elif bool((coarse[ib].abs() > bound - 0.5 * coarse_step).any()):
                g["reason"] = f"coarse optimum {[round(float(v), 2) for v in coarse[ib]]} deg on the +-{bound} deg bound"
            own = coarse[torch.stack(g["own"])]
            g["agree"] = float(((own - coarse[ib]).abs().amax(-1) <= coarse_step + 1e-6).float().mean())
            g["fine"] = _grid(coarse[ib], 3 * fine_step, fine_step)
            g["fine_R"] = _rot_nf(g["fine"])
            g["fsum"] = torch.zeros(g["fine"].shape[0], device=dev)
            g["zsum"] = torch.zeros((), device=dev)
            g["per_image"] = []
        # pass 2: fine grid around each group's coarse optimum (+ the identity score on the same crop)
        for camera, batch in dm.fixed_indices_eval_dataloader:
            g = groups[_group_of(camera)]
            if g["reason"] is not None:
                continue
            _, pred, gt, mask = _render(model, camera, batch)
            K = _pinhole(camera, pred)
            margin = _margin(K, max_deg)
            sc = _warp_psnr(pred, gt, mask, K, torch.cat([g["fine_R"], coarse_R[i_zero:i_zero + 1]]), margin, chunk)
            g["fsum"] += sc[:-1]
            g["zsum"] += sc[-1]
            g["per_image"].append(sc)
        fit = {}
        for key, g in sorted(groups.items()):
            rot = zero
            entry = dict(drive=key[0], camera=key[1], n=g["n"], fitted=False, frames_agreeing_coarse=round(g["agree"], 3))
            if g["reason"] is None:
                jb = int(g["fsum"].argmax())
                per = torch.stack(g["per_image"])                                       # [n, fine + identity]
                improved = float((per[:, jb] > per[:, -1]).float().mean())
                cand = g["fine"][jb]
                entry.update(psnr_crop_raw=round(float(g["zsum"] / g["n"]), 3),
                             psnr_crop_fit=round(float(g["fsum"][jb] / g["n"]), 3), frames_improved=round(improved, 3))
                small = float(cand.abs().max()) < 1.5 * fine_step
                if float(cand.abs().max()) < 0.5 * fine_step or (small and improved < min_improved):
                    entry["fitted"] = True          # the optimum is no rotation (one fine step that helps too few images = none)
                elif improved < min_improved:
                    g["reason"] = (f"rotation {[round(float(v), 2) for v in cand]} deg improves only "
                                   f"{improved:.0%} of the images (< {min_improved:.0%})")
                else:
                    entry["fitted"], rot = True, cand
            if g["reason"] is not None:
                entry["not_fitted_reason"] = g["reason"]
                say(f"[eval cam rot] NOT FITTED {key[0]} {key[1]}: {g['reason']} -> identity kept for this camera",
                    style="bold red")
            entry.update({a: round(float(v), 3) for a, v in zip(_AXES, rot)})
            g["rot"] = rot
            fit[f"{key[0]}/{key[1]}"] = entry
        # pass 3: aligned metrics, rendered at the fitted rotation through the model
        aligned_list: List[Dict[str, float]] = []
        for camera, batch in dm.fixed_indices_eval_dataloader:
            g = groups[_group_of(camera)]
            if bool((g["rot"] != 0).any()):
                c2w = camera.camera_to_worlds
                Rm = _rot_nf(g["rot"][None].to(c2w.dtype))[0].to(c2w.device)
                camera.camera_to_worlds = torch.cat([c2w[..., :3, :3] @ Rm, c2w[..., :3, 3:]], dim=-1)
            outputs = model.get_outputs_for_camera(camera=camera)
            m, _ = model.get_image_metrics_and_images(outputs, batch, travel_id_set=tset, return_image_dict=False)
            aligned_list.append(m)
    if was_training:
        pipeline.train()
    res = dict(raw=_average(raw_list, get_std), aligned=_average(aligned_list, get_std), fit=fit,
               settings=dict(bound_deg=bound, coarse_step_deg=coarse_step, fine_step_deg=fine_step,
                             min_frames=min_frames, min_improved=min_improved,
                             rotation="camera frame (x right, y up, z back): c2w @ Ry(yaw) Rx(pitch) Rz(roll)"))
    _report(res)
    return res


def _report(res: Dict) -> None:
    raw, al = res["raw"], res["aligned"]
    say("[eval cam rot] fitted rotations per (drive, camera), deg:", style="bold magenta")
    for name, e in res["fit"].items():
        tag = "ok " if e["fitted"] else "NOT FITTED"
        say(f"  {name:44s} n {e['n']:3d}  yaw {e['yaw']:+.2f} pitch {e['pitch']:+.2f} roll {e['roll']:+.2f}  {tag}"
            + (f"  crop PSNR {e['psnr_crop_raw']:.2f} -> {e['psnr_crop_fit']:.2f}, improved {e['frames_improved']:.0%}"
               if "psnr_crop_fit" in e else "") + (f"  ({e['not_fitted_reason']})" if not e["fitted"] else ""))
    keys = [k for k in raw if (k.endswith("psnr") or k.endswith("ssim")) and "cc_" not in k]
    say("[eval cam rot] metric: raw -> rotation-aligned", style="bold magenta")
    for k in sorted(keys, key=lambda s: (s not in ("psnr", "ssim"), s)):
        say(f"  {k:16s} {raw[k]:.3f} -> {al[k]:.3f}")
    nf = [n for n, e in res["fit"].items() if not e["fitted"]]
    if nf:
        say(f"[eval cam rot] {len(nf)} of {len(res['fit'])} cameras NOT fitted (identity kept): {', '.join(nf)}",
            style="bold red")


if __name__ == "__main__":
    os.environ.setdefault("NERFSTUDIO_DATAPARSER_CONFIGS", "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser")
    os.environ.setdefault("NERFSTUDIO_METHOD_CONFIGS", "mtgs_st=mtgs.config.MTGS_ST:method,mtgs=mtgs.config.MTGS:method")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--recon_dir", type=str, required=True, help="run dir with config.yml and nerfstudio_models/")
    parser.add_argument("--out", type=str, default=None, help="result json (default <recon_dir>/eval_cam_rot.json)")
    args = parser.parse_args()
    import nuplan_scripts.utils.ckpt_compat  # noqa: F401  (torch.load weights_only=False for our own checkpoints)
    from nerfstudio.utils.eval_utils import eval_setup

    def _skip_init_points(cfg):   # the checkpoint replaces the initial gaussians: no point-cloud filtering
        cfg.pipeline.datamanager.dataparser.load_3D_points = False
        return cfg
    _, pipe, ckpt, _ = eval_setup(Path(args.recon_dir) / "config.yml", update_config_callback=_skip_init_points)
    result = eval_fit_cam_rot(pipe)
    result["checkpoint"] = str(ckpt)
    out = Path(args.out) if args.out else Path(args.recon_dir) / "eval_cam_rot.json"
    out.write_text(json.dumps(result, indent=2), "utf8")
    say(f"Saved results to: {out}")
