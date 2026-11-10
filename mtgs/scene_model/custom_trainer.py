#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import time
import copy
import dataclasses
import os
import signal
import shutil
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Type

import torch
from torch.optim.adam import adam as _adam_functional
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.utils.decorators import check_main_thread
from nerfstudio.engine.callbacks import TrainingCallbackAttributes
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.rich_utils import CONSOLE
from mtgs.custom_viewer.viewer import Viewer
from mtgs.utils.ckpt_io import load_ckpt

class CustomOptimizers(Optimizers):
    """Optimizers with batched zero_grad / Adam steps.

    MTGS gives every rigid object its own ~8 parameter groups, each with its own fused
    Adam, so a block with 320 objects steps ~2500 optimizers per iteration. Calling
    zero_grad()/step() on each one is pure Python overhead: on torch (A100) it was 34%
    of the main thread and the GPU sat at 30% utilisation. Here all groups that share
    the same hyper-parameters are stepped with ONE fused Adam call on the optimizers'
    own state tensors -- the same kernel and the same math as stepping them one by one.
    Groups with a grad clip, or any call while the GradScaler is enabled, take the
    original per-optimizer path."""

    def zero_grad_some(self, param_groups) -> None:
        for name in param_groups:
            for pg in self.optimizers[name].param_groups:
                for p in pg["params"]:
                    p.grad = None

    def optimizer_scaler_step_some(self, grad_scaler, param_groups) -> None:
        if grad_scaler.is_enabled():
            return super().optimizer_scaler_step_some(grad_scaler, param_groups)
        fallback, buckets = [], {}
        for name in param_groups:
            opt = self.optimizers[name]
            if self.config[name]["optimizer"].max_norm is not None or not isinstance(opt, torch.optim.Adam):
                fallback.append(name)
                continue
            for pg in opt.param_groups:
                if pg.get("amsgrad") or pg.get("differentiable") or not pg.get("fused"):
                    fallback.append(name)
                    break
            else:
                for pg in opt.param_groups:
                    key = (float(pg["lr"]), tuple(pg["betas"]), pg["eps"], pg["weight_decay"],
                           bool(pg.get("maximize", False)), bool(pg.get("decoupled_weight_decay", False)))
                    b = buckets.setdefault(key, ([], [], [], [], []))
                    for p in pg["params"]:
                        if p.grad is None:
                            continue
                        st = opt.state[p]
                        if len(st) == 0:   # same initialisation as Adam._init_group (fused)
                            st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                            st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        b[0].append(p); b[1].append(p.grad); b[2].append(st["exp_avg"])
                        b[3].append(st["exp_avg_sq"]); b[4].append(st["step"])
                # LR schedulers check that optimizer.step() ran before scheduler.step()
                opt._opt_called = True
        for (lr, (beta1, beta2), eps, wd, maximize, decoupled), (ps, gs, m1, m2, steps) in buckets.items():
            if not ps:
                continue
            _adam_functional(ps, gs, m1, m2, [], steps, foreach=False, capturable=False,
                                  differentiable=False, fused=True, grad_scale=None, found_inf=None,
                                  has_complex=False, decoupled_weight_decay=decoupled, amsgrad=False,
                                  beta1=beta1, beta2=beta2, lr=lr, weight_decay=wd, eps=eps,
                                  maximize=maximize)
        if fallback:
            super().optimizer_scaler_step_some(grad_scaler, fallback)

    def scheduler_step_all(self, step: int) -> None:
        """Run step for all schedulers.
        We disable the lr logging for MTGS model.

        Args:
            step: the current step
        """
        if not self._batched:
            for param_group_name, scheduler in self.schedulers.items():
                scheduler.step()
            return
        # ~2500 LambdaLR schedulers, but only a handful of distinct (config, lr_init)
        # pairs: the multiplier depends on nothing else. Step one scheduler per pair and
        # hand its learning rates to the others -- the same values LambdaLR would set.
        cache = self.__dict__.setdefault("_sched_keys", {})
        leaders = {}
        for name, s in self.schedulers.items():
            ent = cache.get(name)
            if ent is None or ent[0] is not s:
                ok = type(s) is torch.optim.lr_scheduler.LambdaLR
                ent = (s, (repr(self.config[name]["scheduler"]), tuple(s.base_lrs)) if ok else None)
                cache[name] = ent
            if ent[1] is None:
                s.step()
                continue
            key = (ent[1], s.last_epoch)
            lead = leaders.get(key)
            if lead is None:
                s.step()
                leaders[key] = s._last_lr
                continue
            s.last_epoch += 1
            s._step_count += 1
            for pg, lr in zip(s.optimizer.param_groups, lead):
                pg["lr"] = lr
            s._last_lr = list(lead)

@dataclass
class CustomTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: CustomTrainer)

    def get_base_dir(self) -> Path:
        # once the base_dir is set, use it
        if hasattr(self, 'base_dir'):
            return Path(self.base_dir)
        else:
            return super().get_base_dir()

    def set_base_dir(self, base_dir: Path) -> None:
        self.base_dir = base_dir

class CustomTrainer(Trainer):
    config: CustomTrainerConfig
    optimizers: CustomOptimizers

    def __init__(self, config: CustomTrainerConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Preemption safety (2026-10-29). MTGS_CKPT_MIRROR=<dir on persistent storage>: every periodic
        # checkpoint is also copied there (latest only, written as .tmp then renamed), and on SIGTERM
        # (Slurm preemption: SIGTERM, then SIGKILL after KillWait = 30 s on torch) the state after the
        # last finished step is saved straight into the mirror and the process exits 143.
        # background_reconstruction.py resumes from the mirror when it holds a checkpoint.
        self._mirror = Path(os.environ["MTGS_CKPT_MIRROR"]) if os.environ.get("MTGS_CKPT_MIRROR") else None
        self._term = False
        self._writer = None        # background thread writing the last checkpoint (local file, then mirror)
        self._save_error = None    # its failure, raised by the next save so a dead disk still stops the run
        if self._mirror is not None:
            self._mirror.mkdir(parents=True, exist_ok=True)
            for f in self._mirror.glob(".step-*.partial"):   # left by a copy that was killed mid-way
                f.unlink()
            signal.signal(signal.SIGTERM, self._on_sigterm)

    def train(self) -> None:
        # resumed from the run's own final checkpoint (export-only rerun): no loop, no log writer -- just write the
        # final checkpoint back so the export finds it (nerfstudio's writer divides by max_iter = 0 otherwise)
        if os.environ.get("MTGS_RESUMED") == "1" and self.config.max_num_iterations == 0:
            self.training_state = "completed"
            self.save_checkpoint(self.step)
            CONSOLE.print(f"[resume] run already finished at step {self.step}: final checkpoint rewritten, nothing to train", style="bold yellow")
            return
        super().train()

    def _after_train(self) -> None:
        # a run resumed from its final checkpoint performs no step, so nerfstudio's log writer never saw a step and
        # write_out_storage() raises KeyError('step') right after the final save (manco it_w2_13, 2026-10-30)
        from nerfstudio.utils import writer as _w
        _w.GLOBAL_BUFFER.setdefault("step", self.step)
        if not _w.GLOBAL_BUFFER.get("max_iter"):   # 0 steps left -> its progress line divides by max_iter (Civo w2_02)
            _w.GLOBAL_BUFFER["max_iter"] = 1
        super()._after_train()

    def _on_sigterm(self, signum, frame):
        # A full checkpoint (model + Adam state) is ~12 GB for a 6-pass block: it cannot reach network storage
        # inside Slurm's 30 s KillWait (measured on Civo 2026-10-29: 1.1 of 11.8 GB written). By default just
        # stop; the run resumes from the last periodic mirror. MTGS_SIGTERM_SAVE=1 tries the save anyway.
        if os.environ.get("MTGS_SIGTERM_SAVE") == "1":
            self._term = True
            CONSOLE.print("[preempt] SIGTERM received: saving to the mirror at the next step boundary", style="bold yellow")
        else:
            CONSOLE.print("[preempt] SIGTERM received: exiting; resume will use the last mirrored checkpoint", style="bold yellow")
            os._exit(143)

    def _ckpt_dict(self, step: int, completed: bool) -> dict:
        d = {
            "step": step,
            "pipeline": self.pipeline.module.state_dict() if hasattr(self.pipeline, "module") else self.pipeline.state_dict(),  # type: ignore
        }
        if not completed:
            d["optimizers"] = {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()}
            d["schedulers"] = {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()}
            d["scalers"] = self.grad_scaler.state_dict()
        return d

    @staticmethod
    def _host_snapshot(obj):
        """The checkpoint dict with every tensor on the host. CUDA tensors go into pinned buffers by non-blocking
        copies queued behind the step's kernels (one D2H pass); the writer thread then serialises the local file
        and the mirror from these buffers, so the training thread never waits for the disk and the mirror is not
        a read-back of the local file."""
        if torch.is_tensor(obj):
            if obj.is_cuda:
                try:
                    h = torch.empty_like(obj, device="cpu", pin_memory=True)
                except RuntimeError:   # no pinned memory left: pageable buffer, blocking copy
                    h = torch.empty_like(obj, device="cpu")
                h.copy_(obj, non_blocking=h.is_pinned())
                return h
            return obj.clone()
        if isinstance(obj, dict):
            return type(obj)((k, CustomTrainer._host_snapshot(v)) for k, v in obj.items())
        if isinstance(obj, (list, tuple)):
            return type(obj)(CustomTrainer._host_snapshot(v) for v in obj)
        return copy.deepcopy(obj)

    @staticmethod
    def _bf16_moments(host: dict) -> dict:
        """MTGS_MIRROR_BF16=1: the mirror stores Adam's exp_avg / exp_avg_sq as bf16 (half the mirror bytes); the
        optimizer casts them back to the parameter dtype on load. Off by default: a run resumed from such a mirror is
        not bit-identical to an uninterrupted one."""
        d = dict(host)
        if "optimizers" in d:
            d["optimizers"] = {
                name: {**sd, "state": {i: {k: v.to(torch.bfloat16) if k in ("exp_avg", "exp_avg_sq") else v
                                           for k, v in st.items()} for i, st in sd["state"].items()}}
                for name, sd in d["optimizers"].items()}
        return d

    def _write_checkpoint(self, host: dict, copied: torch.cuda.Event, ckpt_path: Path, step: int) -> None:
        """Writer thread: local file from the host buffers, then the mirror, then the old local files."""
        copied.synchronize()   # the D2H copies of the snapshot have landed
        for attempt in range(5):
            try:
                torch.save(host, ckpt_path)
                break
            except Exception as e:
                CONSOLE.log(f"[ERROR] saving checkpoint on attempt {attempt + 1}", style="bold red")
                CONSOLE.print(e)
                if attempt == 4:
                    self._save_error = e
                    return
                time.sleep(2 ** attempt)  # Exponential backoff
        if self._mirror is not None:
            self._mirror_install(ckpt_path, step, host)
        if self.config.save_only_latest_checkpoint:
            # delete every other checkpoint in the checkpoint folder
            for f in self.checkpoint_dir.glob("*.ckpt"):
                if f != ckpt_path:
                    f.unlink()

    def _mirror_install(self, src: Path, step: int, host: dict = None) -> None:
        # src -> mirror/step-<step>.ckpt via a temp name, then drop older mirror checkpoints
        dst = self._mirror / f"step-{step:09d}.ckpt"
        tmp = self._mirror / f".step-{step:09d}.partial"
        if src != tmp:
            if host is not None and os.environ.get("MTGS_MIRROR_BF16") == "1":
                torch.save(self._bf16_moments(host), tmp)
            elif os.stat(src).st_dev == os.stat(self._mirror).st_dev:
                os.link(src, tmp)   # same filesystem: one inode, nothing written twice
            elif host is not None:
                torch.save(host, tmp)   # from the host buffers, not a read-back of the local file
            else:
                shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        for f in self._mirror.glob("step-*.ckpt"):
            if f != dst:
                f.unlink()

    def _preempt_save_and_exit(self, step: int) -> None:
        if self._writer is not None:
            self._writer.join()
        t0 = time.time()
        tmp = self._mirror / f".step-{step:09d}.partial"
        torch.save(self._ckpt_dict(step, completed=False), tmp)
        self._mirror_install(tmp, step)
        CONSOLE.print(f"[preempt] saved step {step} to {self._mirror} in {time.time() - t0:.1f} s; exiting for requeue", style="bold yellow")
        os._exit(143)

    # MTGS_SYNC_DEBUG=<step>: report every host<->GPU synchronisation made during steps
    # <step> .. <step>+2 (each one drains the GPU queue while Python catches up).
    _sync_debug = int(os.environ["MTGS_SYNC_DEBUG"]) if os.environ.get("MTGS_SYNC_DEBUG") else None

    # MTGS_PROFILE_STEPS=a:b brackets steps [a, b) with cudaProfilerStart/Stop, for
    # `nsys profile --capture-range=cudaProfilerApi`.
    _prof = tuple(int(x) for x in os.environ["MTGS_PROFILE_STEPS"].split(":")) if os.environ.get("MTGS_PROFILE_STEPS") else None

    def _install_nan_debug(self):
        """MTGS_DEBUG_NAN=1: log the loss dict and camera of the step whose gradient first goes non-finite.
        (pn_w2_02 2026-11-05: camera_optimizer.pose_adjustment of one image turned NaN at step ~4000 while every
        gaussian stayed finite; the loss table is not logged during training, so this hook names the term.)"""
        import math
        model = self.pipeline.model
        self._dbg = {"step": None, "loss": {}, "cam": {}}
        orig = self.pipeline.get_train_loss_dict

        def wrapped(step):
            model_outputs, loss_dict, metrics_dict = orig(step)
            self._dbg = {"step": step, "cam": dict(getattr(model, "cur_cam_metas", {}) or {}),
                         "loss": {k: float(v) for k, v in loss_dict.items()}}
            bad = {k: v for k, v in self._dbg["loss"].items() if not math.isfinite(v)}
            if bad:
                print(f"[nan-debug] step {step} NON-FINITE LOSS {bad} cam {self._dbg['cam']}", flush=True)
            return model_outputs, loss_dict, metrics_dict
        self.pipeline.get_train_loss_dict = wrapped

        def make_hook(name):
            def hook(grad):
                fin = torch.isfinite(grad)
                if not bool(fin.all()):
                    rows = (~fin.reshape(grad.shape[0], -1).all(1)).nonzero().flatten().tolist() if grad.dim() > 1 else []
                    print(f"[nan-debug] step {self._dbg['step']} {name} grad non-finite: {int((~fin).sum())} values, rows {rows[:8]} "
                          f"(of {grad.shape[0]}); loss {self._dbg['loss']}; cam {self._dbg['cam']}", flush=True)
                return grad
            return hook
        co = getattr(model, "camera_optimizer", None)
        if co is not None and hasattr(co, "pose_adjustment"):
            co.pose_adjustment.register_hook(make_hook("camera_optimizer.pose_adjustment"))
        for mname, gm in getattr(model, "gaussian_models", {}).items():
            gp = getattr(gm, "gauss_params", None)
            if gp is not None and "means" in gp:
                gp["means"].register_hook(make_hook(f"{mname}.means"))
        print("[nan-debug] hooks installed", flush=True)

    def train_iteration(self, step: int):
        if self._term:   # SIGTERM arrived during the previous step: its state is complete, save it and leave
            self._preempt_save_and_exit(step - 1)
        if self._prof is not None and step == self._prof[0]:
            torch.cuda.synchronize(); torch.cuda.profiler.start()
        if self._prof is not None and step == self._prof[1]:
            torch.cuda.synchronize(); torch.cuda.profiler.stop()
        if self._sync_debug is not None and step == self._sync_debug:
            import warnings
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode(1)
        out = super().train_iteration(step)
        if self._sync_debug is not None and step == self._sync_debug + 3:
            torch.cuda.set_sync_debug_mode(0)
        return out

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = Viewer(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self._load_checkpoint()

        if os.environ.get("MTGS_DEBUG_NAN") == "1":   # 2026-11-05: find which image / loss term produces a non-finite gradient
            self._install_nan_debug()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline, trainer=self
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

    def setup_optimizers(self) -> CustomOptimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        param_groups = self.pipeline.get_param_groups()
        optimizer_config = {}
        for key, value in param_groups.items():
            if key not in self.config.optimizers:    # in the format of {model_name}.{model_type}.{param_name}
                sub_key = key
                while '.' in sub_key:
                    sub_key = sub_key.split('.', maxsplit=1)[-1]
                    if sub_key in self.config.optimizers:
                        break

                assert sub_key in self.config.optimizers, f"Optimizer for {key} not found in config"
                optimizer_config[key] = self.config.optimizers[sub_key].copy()
            else:
                optimizer_config[key] = self.config.optimizers[key].copy()

        return CustomOptimizers(optimizer_config, param_groups)

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        completed = self.training_state == "completed"
        if self._writer is not None:   # the previous save (local file and mirror) must be on disk first
            self._writer.join()
        if self._save_error is not None:
            raise self._save_error
        # one D2H pass into pinned host buffers; the writer thread waits on the event, not the training thread
        host = self._host_snapshot(self._ckpt_dict(step, completed))
        copied = torch.cuda.Event()
        copied.record()
        self._writer = threading.Thread(target=self._write_checkpoint, args=(host, copied, ckpt_path, step), daemon=False)
        self._writer.start()
        if completed:   # the final file: the stages after training need it, nothing left to overlap with
            self._writer.join()
            if self._save_error is not None:
                raise self._save_error

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = load_ckpt(load_path)   # mmap: copied file -> GPU, no full host copy first
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            load_path = load_checkpoint
            loaded_state = load_ckpt(load_path)
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
            return
        # load the checkpoints for pipeline, optimizers, and gradient scalar
        self._start_step = loaded_state["step"] + 1
        self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        # VanillaGaussianSplattingModel.load_state_dict replaces every gauss_params Parameter object (new point
        # count), but the optimizers were built in setup() from the OLD objects: a resumed run stepped dead tensors
        # until the next densification re-linked them, and past stop_split_at never -- pn_v2_04 (2026-11-07,
        # resumed at step 60000) trained its gaussians for zero steps, loss flat for 60k steps. Rebuild the
        # optimizers from the live parameters before loading their state.
        self.optimizers = self.setup_optimizers()
        if "optimizers" in loaded_state:
            self.optimizers.load_optimizers(loaded_state["optimizers"])
        if "schedulers" in loaded_state and self.config.load_scheduler:
            self.optimizers.load_schedulers(loaded_state["schedulers"])
        if "scalers" in loaded_state:
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
        CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        if os.environ.get("MTGS_RESUMED") == "1":
            # the run's config.yml must not keep pointing at the mirror checkpoint it resumed from: the render /
            # eval / export stages read config.yml and would look for that step in nerfstudio_models (pa_loc_0191,
            # 2026-10-30: eval died on step-45000 after training to 90002)
            self.config.load_dir = None
            self.config.load_step = None
            try:
                self.config.save_config()
            except Exception as e:   # never let bookkeeping stop a resumed run
                CONSOLE.print(f"[resume] could not rewrite config.yml: {e}")
        if os.environ.get("MTGS_RESUMED") == "1":
            # nerfstudio trains range(start, start + max_num_iterations): shorten it so a resumed run ends at
            # the original final step instead of running a full schedule on top of the checkpoint
            total = self.config.max_num_iterations
            self.config.max_num_iterations = max(0, total - self._start_step)
            # nerfstudio's final save uses self.step, which the loop sets; a run resumed from its final checkpoint
            # runs no step at all, so without this the finished model was saved as step 0 (Civo v2_07, 2026-10-30)
            # and a later resume would have retrained the whole schedule on top of it
            self.step = self._start_step - 1
            CONSOLE.print(f"[resume] step {self._start_step} of {total}: {self.config.max_num_iterations} steps left", style="bold yellow")
