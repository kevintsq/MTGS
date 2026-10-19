#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import time
import dataclasses
import os
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

class CustomOptimizers(Optimizers):
    """Optimizers with batched zero_grad / Adam steps.

    MTGS gives every rigid object its own ~8 parameter groups, each with its own fused
    Adam, so a block with 320 objects steps ~2500 optimizers per iteration. Calling
    zero_grad()/step() on each one is pure Python overhead: on torch (A100) it was 34%
    of the main thread and the GPU sat at 30% utilisation. Here all groups that share
    the same hyper-parameters are stepped with ONE fused Adam call on the optimizers'
    own state tensors -- the same kernel and the same math as stepping them one by one.
    Groups with a grad clip, or any call while the GradScaler is enabled, take the
    original per-optimizer path. MTGS_BATCHED_OPTIM=0 disables the batching."""

    _batched = os.environ.get("MTGS_BATCHED_OPTIM", "1") != "0"

    def zero_grad_some(self, param_groups) -> None:
        if not self._batched:
            return super().zero_grad_some(param_groups)
        for name in param_groups:
            for pg in self.optimizers[name].param_groups:
                for p in pg["params"]:
                    p.grad = None

    def optimizer_scaler_step_some(self, grad_scaler, param_groups) -> None:
        if not self._batched or grad_scaler.is_enabled():
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

    # MTGS_SYNC_DEBUG=<step>: report every host<->GPU synchronisation made during steps
    # <step> .. <step>+2 (each one drains the GPU queue while Python catches up).
    _sync_debug = int(os.environ["MTGS_SYNC_DEBUG"]) if os.environ.get("MTGS_SYNC_DEBUG") else None

    def train_iteration(self, step: int):
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
        ckpt_dict = {
            "step": step,
            "pipeline": self.pipeline.module.state_dict()  # type: ignore
            if hasattr(self.pipeline, "module")
            else self.pipeline.state_dict(),
            "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
            "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
            "scalers": self.grad_scaler.state_dict(),
        }
        if self.training_state == "completed":
            ckpt_dict.pop("optimizers")
            ckpt_dict.pop("schedulers")
            ckpt_dict.pop("scalers")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                torch.save(ckpt_dict, ckpt_path)
                break
            except Exception as e:
                CONSOLE.log(f"[ERROR] saving checkpoint on attempt {attempt + 1}", style="bold red")
                CONSOLE.print(e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete every other checkpoint in the checkpoint folder
            for f in self.checkpoint_dir.glob("*.ckpt"):
                if f != ckpt_path:
                    f.unlink()

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
            loaded_state = torch.load(load_path, map_location="cpu")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            load_path = load_checkpoint
            loaded_state = torch.load(load_path, map_location="cpu")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
            return
        # load the checkpoints for pipeline, optimizers, and gradient scalar
        self._start_step = loaded_state["step"] + 1
        self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
        if "optimizers" in loaded_state:
            self.optimizers.load_optimizers(loaded_state["optimizers"])
        if "schedulers" in loaded_state and self.config.load_scheduler:
            self.optimizers.load_schedulers(loaded_state["schedulers"])
        if "scalers" in loaded_state:
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
        CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
