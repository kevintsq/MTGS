"""torch >= 2.6 loads checkpoints with weights_only=True by default and refuses the numpy float64 scalars that
nerfstudio's ExponentialDecayScheduler leaves in every optimizer param_group ('lr' = np.exp(...)) -- so a mid-run
checkpoint (rh_w2_06 step 48000, 2026-10-30) cannot be read by nerfstudio's eval_load_checkpoint / eval_setup.
Our checkpoints are our own files: importing this module makes torch.load default to weights_only=False for the
process (an explicit weights_only=... in a call still wins)."""
import functools
import torch

if not getattr(torch.load, "_mtgs_weights_only_patch", False):
    _orig = torch.load

    @functools.wraps(_orig)
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._mtgs_weights_only_patch = True
    torch.load = _load
