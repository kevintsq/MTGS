"""torch >= 2.6 loads checkpoints with weights_only=True by default and refuses the numpy float64 scalars that
nerfstudio's ExponentialDecayScheduler leaves in every optimizer param_group ('lr' = np.exp(...)) -- so a mid-run
checkpoint (rh_w2_06 step 48000, 2026-10-30) cannot be read by nerfstudio's eval_load_checkpoint / eval_setup.
Our checkpoints are our own files: importing this module makes torch.load default to weights_only=False for the
process (an explicit weights_only=... in a call still wins), and to mmap=True for a path argument, so nerfstudio's
eval_load_checkpoint (the PSNR stage) copies the file's pages into the model instead of a full host copy of the
~8 GB file first (a legacy non-zip file falls back to a plain read)."""
import functools
import os
import torch

if not getattr(torch.load, "_mtgs_weights_only_patch", False):
    _orig = torch.load

    @functools.wraps(_orig)
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        f = args[0] if args else kwargs.get("f")
        if "mmap" not in kwargs and isinstance(f, (str, os.PathLike)):
            try:
                return _orig(*args, mmap=True, **kwargs)
            except RuntimeError:
                pass
        return _orig(*args, **kwargs)

    _load._mtgs_weights_only_patch = True
    torch.load = _load
