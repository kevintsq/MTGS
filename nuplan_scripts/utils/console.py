#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
"""The one output channel of MTGS (mtgs/ and nuplan_scripts/), 2026-11-16.

Messages go through `say` (or, for rich renderables such as tables, CONSOLE directly): one console, one style,
printed as written. Rule: a status line is `say("[tag] text")`; it gets the time in front and is never wrapped, so a
line in a Slurm log file is the message in one piece; rich markup is off unless asked for (markup=True), so "[tag]"
prefixes appear literally (rich used to swallow "[sfm reuse]" / "[ba input]" as unknown markup).

No random state is touched by printing. rich gives every style that carries a link (CONSOLE.log renders its caller's
file:line as one, tracebacks link their frames) an id from `random.randint`, i.e. from python's GLOBAL random state,
which the training sampler (mtgs/dataset/utils/sampler.py) draws the images from: every log line changed which image
later steps trained on. Importing this module points rich's link ids at a private generator; mtgs/__init__.py and
nuplan_scripts/__init__.py import it, so it is in place before anything renders. Data written for another program
(e.g. nuplan_scripts/topup_state.py, sel_count.py read by shell scripts) stays plain print.
"""
import random
import sys
import time

_LINK_IDS = random.Random(20261116)          # rich's link ids only; nothing else draws from it


def _private_rich_randomness():
    """rich.style imports randint at module level (13.x - 15.x) and looks it up at call time: rebind it. Every other
    use of random in rich is in `if __name__ == "__main__"` demos."""
    try:
        import rich.style
    except ImportError:
        return False
    if getattr(rich.style, "randint", None) is not _LINK_IDS.randint:
        rich.style.randint = _LINK_IDS.randint
    return True


_HAVE_RICH = _private_rich_randomness()

if _HAVE_RICH:
    try:                                     # one console object for MTGS and nerfstudio's own messages
        from nerfstudio.utils.rich_utils import CONSOLE
    except Exception:                        # environments without nerfstudio (prep stages): the same settings
        from rich.console import Console
        CONSOLE = Console(width=120)
else:
    CONSOLE = None


def say(*objects, sep=" ", markup=False, style=None, **_ignored):
    """One status line (or a multi-line message), "[HH:MM:SS] " in front, printed as written: no markup, no
    highlighting, no wrapping. Rich renderables (tables, panels) are printed as they are. Keyword arguments of the
    builtin print (end, flush, file) are accepted and ignored: the console writes whole lines to stdout."""
    if not _HAVE_RICH:
        print(time.strftime("[%H:%M:%S] ") + sep.join(str(o) for o in objects), flush=True)
        return
    if objects and all(isinstance(o, str) or not hasattr(o, "__rich_console__") and not hasattr(o, "__rich__")
                       for o in objects):
        CONSOLE.print(time.strftime("[%H:%M:%S] ") + sep.join(str(o) for o in objects), markup=markup,
                      highlight=False, soft_wrap=True, style=style)
    else:
        CONSOLE.print(*objects, markup=markup, soft_wrap=True, style=style)
    try:
        CONSOLE.file.flush()
    except Exception:
        pass


def check_random_free():
    """(ok, detail): rendering through CONSOLE leaves python's, numpy's and torch's random states untouched."""
    import io
    import contextlib
    states = [random.getstate()]
    try:
        import numpy as np
        states.append(np.random.get_state()[1].tobytes())
    except ImportError:
        np = None
    try:
        import torch
        states.append(torch.random.get_rng_state().numpy().tobytes())
    except ImportError:
        torch = None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if _HAVE_RICH:
            from rich.console import Console
            c = Console(file=buf, force_terminal=True, width=120)
            c.log("log line"); c.print("[link=https://x]link[/link]"); CONSOLE.log("shared console log line")
            say("[tag] status line")
            try:
                raise ValueError("traceback test")
            except ValueError:
                c.print_exception()
            from rich.progress import track
            for _ in track(range(3), console=c, description="progress"):
                pass
    after = [random.getstate()]
    if np is not None:
        after.append(np.random.get_state()[1].tobytes())
    if torch is not None:
        after.append(torch.random.get_rng_state().numpy().tobytes())
    names = ["python random", "numpy", "torch"][:len(states)]
    bad = [n for n, a, b in zip(names, states, after) if a != b]
    return not bad, ("unchanged: " + ", ".join(names)) if not bad else ("CHANGED: " + ", ".join(bad))


if __name__ == "__main__":
    ok, detail = check_random_free()
    print(f"rich random-free: {ok} ({detail})")
    sys.exit(0 if ok else 1)
