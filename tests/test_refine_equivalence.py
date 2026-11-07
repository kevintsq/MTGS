"""Equivalence + peak-memory test for the gaussian refinement (densify / cull / optimizer surgery).

Runs one refinement sequence on a synthetic model in a given source tree and dumps the result
(content hashes of every parameter and Adam moment, num_points, peak memory, time); `compare`
asserts that two dumps are identical. Usage (the reference tree is a detached worktree at the
commit before the rewrite: `git -C /home/kevin/MTGS worktree add --detach /home/kevin/MTGS_w2_old b751a3c`):

    python tests/test_refine_equivalence.py run --tree /home/kevin/MTGS_w2_old --out /tmp/old.pt --model multicolor --n 200000 --t 6
    python tests/test_refine_equivalence.py run --tree /home/kevin/MTGS_w2     --out /tmp/new.pt --model multicolor --n 200000 --t 6
    python tests/test_refine_equivalence.py compare /tmp/old.pt /tmp/new.pt
"""
import argparse
import hashlib
import math
import os
import sys
import time

import torch


def build(model_kind, n, t, device):
    from mtgs.scene_model.gaussian_model.vanilla_gaussian_splatting import (
        GaussianSplattingControlConfig, VanillaGaussianSplattingModelConfig)
    from mtgs.scene_model.gaussian_model.multi_color_gaussian_splatting import MultiColorGaussianSplattingModelConfig
    from mtgs.scene_model.gaussian_model.skybox_gaussian_splatting import SkyboxGaussianSplattingModelConfig
    from mtgs.scene_model.gaussian_model.utils import num_sh_bases

    f = 1  # iteration factor; the MTGS.py values
    control = GaussianSplattingControlConfig(
        densify_from_iter=500 * f, refine_every=100 * f, stop_split_at=15000 * f, reset_alpha_every=30,
        continue_cull_post_densification=False, cull_alpha_thresh=0.005, cull_scale_thresh=0.5,
        densify_size_thresh=0.2, densify_grad_thresh=0.001, n_split_samples=2, clone_sample_means=True,
        stop_screen_size_at=15000 * f, cull_screen_size=150, split_screen_size=100, sh_degree=3,
        sh_degree_interval=1000 * f, use_abs_grad=True)
    travel_ids = [f"tr{i}" for i in range(t)]
    if model_kind == "vanilla":
        cfg = VanillaGaussianSplattingModelConfig(model_type="vanilla", control=control, verbose=True)
        model = cfg.setup(model_name="background", model_id=0)
    elif model_kind == "multicolor":
        cfg = MultiColorGaussianSplattingModelConfig(model_type="multicolor", control=control, verbose=True,
                                                    multi_feature_rest=True)
        model = cfg.setup(model_name="background", model_id=0, travel_ids=travel_ids, nearest_train_travel_of_eval={})
    elif model_kind == "skybox":
        cfg = SkyboxGaussianSplattingModelConfig(model_type="multicolor", control=control, verbose=False,
                                                 skybox_radius=1000, num_sky_gaussians=100000, skybox_type="spheric",
                                                 skybox_scale_factor=1000.0, mono_sky=False, multi_feature_rest=True)
        model = cfg.setup(model_name="skybox", model_id=1, travel_ids=travel_ids, nearest_train_travel_of_eval={})
    elif model_kind == "rigid":
        from mtgs.scene_model.gaussian_model.rigid_node import RigidSubModelConfig
        cfg = RigidSubModelConfig(model_type="rigid", control=control, verbose=False, fourier_features_dim=4)
        model = cfg.setup(model_name="obj_0", model_id=2)
    else:
        raise ValueError(model_kind)

    # a tiny cloud through the real populate_modules (sets up the ParameterDict layout), then synthetic
    # parameters of the requested size in the same layout
    g = torch.Generator().manual_seed(0)
    small = {"xyz": torch.rand(16, 3, generator=g) * 10, "rgb": torch.rand(16, 3, generator=g) * 255}
    if model_kind == "skybox":
        model.populate_modules(max_distance=100.0, num_traversals=t)
    elif model_kind == "rigid":
        nf = 10
        inst = {"class_name": "vehicle", "token": "x", "pts": small["xyz"], "colors": small["rgb"],
                "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(nf, 1), "trans": torch.zeros(nf, 3),
                "size": torch.tensor([4.0, 2.0, 1.5]), "in_frame_indices": torch.arange(nf),
                "in_frame_mask": torch.ones(nf, dtype=torch.bool), "num_frames_cur_travel": nf, "travel_id": 0}
        frames = {0: {"raw_timestamps": torch.arange(nf).float(), "frame_timestamps": torch.arange(nf).float(),
                      "min_ts": 0.0, "max_ts": float(nf - 1)}}
        model.populate_modules(inst, frames)
    elif model_kind == "multicolor":
        model.populate_modules(points_3d=small, num_traversals=t)
    else:
        model.populate_modules(points_3d=small)

    dim_sh = num_sh_bases(3)
    g = torch.Generator(device=device).manual_seed(1)
    def rnd(*shape, lo=0.0, hi=1.0):
        return torch.rand(shape, generator=g, device=device) * (hi - lo) + lo
    radius = 1000.0 if model_kind == "skybox" else 300.0
    new = {}
    for name, p in model.gauss_params.items():
        shape = (n,) + tuple(p.shape[1:])
        if name == "means":
            v = rnd(n, 3, lo=-1, hi=1) * radius          # some beyond the far/skybox radius
        elif name == "scales":
            v = rnd(*shape, lo=math.log(0.01), hi=math.log(0.6))  # dup (<0.2), split (>0.2), too big (>0.5)
        elif name == "opacities":
            v = rnd(n, 1, lo=-7.0, hi=3.0)                 # logit(0.005) = -5.29: some below the cull threshold
        elif name == "quats":
            v = rnd(n, 4, lo=-1, hi=1)
        else:
            v = rnd(*shape, lo=-1, hi=1)
        new[name] = torch.nn.Parameter(v.contiguous())
    model.gauss_params = torch.nn.ParameterDict(new)
    return model


def make_optimizers(model):
    from nerfstudio.engine.optimizers import Optimizers
    from mtgs.config.MTGS import FusedAdamOptimizerConfig
    groups = model.get_gaussian_param_groups()
    config = {name: {"optimizer": FusedAdamOptimizerConfig(lr=1e-3, eps=1e-15), "scheduler": None} for name in groups}
    return Optimizers(config, groups)


def adam_step(model, optimizers, seed):
    g = torch.Generator(device=model.device).manual_seed(seed)
    for name, (p,) in model.get_gaussian_param_groups().items():
        p.grad = torch.randn(p.shape, generator=g, device=p.device)
        optimizers.optimizers[name].step()
        p.grad = None


def set_stats(model, seed):
    g = torch.Generator(device=model.device).manual_seed(seed)
    n = model.num_points
    model.xys_grad_norm = torch.rand(n, generator=g, device=model.device) * 0.00105  # ~5% above 0.001
    model.vis_counts = torch.ones(n, device=model.device)
    model.max_2Dsize = torch.rand(n, generator=g, device=model.device) * 104.0       # ~4% above 100 (split)
    model.max_2Dsize[::100] = 200.0                                                  # 1% above 150 (cull)


def hashes(model, optimizers):
    out = {"num_points": model.num_points, "order": list(model.gauss_params.keys())}
    def h(x):
        return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
    for name, p in model.gauss_params.items():
        out[f"param.{name}"] = (tuple(p.shape), str(p.dtype), h(p))
        assert p.is_contiguous(), name
    for gname, (p,) in model.get_gaussian_param_groups().items():
        opt = optimizers.optimizers[gname]
        assert opt.param_groups[0]["params"][0] is p, gname
        st = opt.state[p]
        for k in ("exp_avg", "exp_avg_sq"):
            assert st[k].shape == p.shape, (gname, k)
            out[f"{k}.{gname}"] = (tuple(st[k].shape), h(st[k]))
        out[f"step.{gname}"] = float(st["step"])
    return out


def run(args):
    sys.path.insert(0, args.tree)
    os.chdir(args.tree)
    device = "cuda"
    torch.manual_seed(1234)
    model = build(args.model, args.n, args.t, device)
    model.to(device)
    optimizers = make_optimizers(model)
    adam_step(model, optimizers, 10)
    adam_step(model, optimizers, 11)

    stats = {"model": args.model, "n": args.n, "t": args.t, "tree": args.tree}
    # step 3100: > refine_every * reset_alpha_every (world/screen-size culls active), < stop_split_at,
    # and 3100 % 3000 == refine_every: the opacity reset runs as well
    steps = [3100, 3200]
    for i, step in enumerate(steps):
        model.step = step
        set_stats(model, 20 + i)
        torch.manual_seed(100 + i)  # the split / dup samples
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        model.refinement_after(optimizers, step)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated()
        stats[f"refine{i}"] = {"step": step, "allocated_before_GB": before / 2**30, "peak_GB": peak / 2**30,
                               "extra_peak_GB": (peak - before) / 2**30, "time_s": dt, "num_points": model.num_points}
        adam_step(model, optimizers, 30 + i)   # the rebuilt optimizer must still step the new parameters
    result = {"stats": stats, "hashes": hashes(model, optimizers)}
    if args.n <= 500_000:
        result["tensors"] = {name: p.detach().cpu() for name, p in model.gauss_params.items()}
        result["states"] = {g: {k: optimizers.optimizers[g].state[p][k].cpu() for k in ("exp_avg", "exp_avg_sq")}
                            for g, (p,) in model.get_gaussian_param_groups().items()}
    torch.save(result, args.out)
    for k, v in stats.items():
        print(k, v)


def compare(args):
    a = torch.load(args.a, weights_only=False)
    b = torch.load(args.b, weights_only=False)
    ha, hb = a["hashes"], b["hashes"]
    assert ha.keys() == hb.keys(), (ha.keys() ^ hb.keys())
    bad = [k for k in ha if ha[k] != hb[k]]
    assert not bad, f"hash mismatch: {bad}"
    assert ha["num_points"] == hb["num_points"]
    assert ha["order"] == hb["order"]
    if "tensors" in a and "tensors" in b:
        for k in a["tensors"]:
            assert torch.equal(a["tensors"][k], b["tensors"][k]), k
        for g in a["states"]:
            for k in a["states"][g]:
                assert torch.equal(a["states"][g][k], b["states"][g][k]), (g, k)
        print(f"full tensors equal: {len(a['tensors'])} params, {len(a['states'])} optimizer states")
    print(f"IDENTICAL: num_points={ha['num_points']}, {len(ha) - 2} hashed tensors/scalars match")
    for i in range(2):
        sa, sb = a["stats"][f"refine{i}"], b["stats"][f"refine{i}"]
        print(f"refine{i} step {sa['step']}: peak {sa['peak_GB']:.3f} GB -> {sb['peak_GB']:.3f} GB "
              f"(extra over allocated {sa['extra_peak_GB']:.3f} -> {sb['extra_peak_GB']:.3f} GB, "
              f"ratio {sb['peak_GB'] / sa['peak_GB']:.3f}); time {sa['time_s']*1e3:.1f} -> {sb['time_s']*1e3:.1f} ms; "
              f"num_points {sa['num_points']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tree", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default="multicolor", choices=["vanilla", "multicolor", "skybox", "rigid"])
    r.add_argument("--n", type=int, default=200_000)
    r.add_argument("--t", type=int, default=6)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else compare(args)
