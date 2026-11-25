#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from nuplan_scripts.utils.console import say
from concurrent.futures import ThreadPoolExecutor
import os
from collections import OrderedDict
import time

import torch
import threading

class PinnedUploader:
    """Moves one cached sample (data dict + Cameras) to the device without a stream sync.

    A .to(device) from pageable memory does a cudaStreamSynchronize per tensor, draining the GPU queue;
    the model did that for the image, mask, depth and lidar depth (several times each) every step, and
    the whole image cache cannot simply be pinned (page-locking tens of GB, 1.5-2x with the allocator's
    power-of-two buckets). Each tensor is memcpy'd into a pinned staging buffer and uploaded with
    non_blocking=True; the buffers rotate over a few slots and a slot is reused only once the event
    recorded after its uploads has completed (checked with query(), so no sync in the normal case).
    The cache keeps its host tensors: the returned dict is a new one."""

    def __init__(self, device, slots: int = 3):
        self.device = torch.device(device)
        self.slots = [[{}, None] for _ in range(slots)]
        self.next = 0

    def upload(self, data: dict, camera):
        if self.device.type != "cuda":
            return data, camera.to(self.device)
        slot = self.slots[self.next]
        self.next = (self.next + 1) % len(self.slots)
        bufs, event = slot
        if event is not None and not event.query():
            event.synchronize()

        def up(key, t):
            if t.device.type == "cuda":
                return t
            # one flat buffer per (key, dtype) and slot, grown to the largest sample seen and used through a view:
            # keyed by shape as well, every per-traversal undistorted lens size got its own pinned block (about 1000
            # blocks / 5 GB of page-locked host memory per training on pn_w2_09, never freed)
            k = (key, t.dtype); n = t.numel()
            buf = bufs.get(k)
            if buf is None or buf.numel() < n:   # this slot's previous uploads have completed (event above)
                buf = bufs[k] = torch.empty(n, dtype=t.dtype, pin_memory=True)
            view = buf[:n].view(t.shape)
            view.copy_(t)
            return view.to(self.device, non_blocking=True)

        out = {k: up(k, v) if torch.is_tensor(v) else v for k, v in data.items()}
        counter = [0]

        def cam_fn(t):   # fields are visited in a fixed order, so the counter identifies the buffer
            counter[0] += 1
            return up(("camera", counter[0]), t)

        camera = camera._apply_fn_to_fields(cam_fn)
        slot[1] = torch.cuda.Event()
        slot[1].record()
        return out, camera


class FixedIndicesPseudoDataloader:
    def __init__(self, datamanager):
        self.datamanager = datamanager
        self.cached_func = self.datamanager.cached_train if self.datamanager.same_train_eval else self.datamanager.cached_eval

    def __iter__(self):
        loader = getattr(self.datamanager, "eval_dataloader", None)
        ahead = loader is not None and getattr(loader, "mode", "lazy") != "lazy"
        for idx in range(len(self)):
            if ahead:   # the evaluation over all images is in order: decode the next ones first
                loader.hint(list(range(idx, min(idx + 32, len(self)))))
            data, camera = self.datamanager.cached_eval(idx)
            camera, data = self.datamanager.upload(data, camera)
            yield camera, data

    def __len__(self):
        return len(self.datamanager.eval_dataset)

class ImageCache:
    """The datamanager's ONE image cache (2026-11-25, user: "最后应该只保留一套最优代码"; it replaces upstream's
    OnDemandDataLoader / AsyncDataLoader and the blocking prefetch). Every sample comes from datamanager._load_data,
    so a returned sample is byte-identical in every mode; only WHEN it is decoded changes.

      background (training default): `num_workers` threads decode every sample in index order and training starts at
          once (pq_area0806, 19k images, had left an H100 idle ~60 min in the blocking prefetch). hint() (the sampler's
          next draws, sampler.peek) moves those to a 2-thread priority pool; a draw that is not resident yet is decoded
          on the spot (its queued decode is cancelled) or, if a worker already holds it, waited for.
      capped (MTGS_CACHE_MAX_IMAGES=N < number of samples, 2026-11-15): at most N samples stay resident (a 12k-image
          merged block is ~100 GB of host RAM); hint() decodes the sampler's next draws ahead; eviction is least
          recently used but never a hinted sample.
      lazy (cache strategy "on_demand": inference / test, or num_workers=0): decoded on first access and kept.

    Threads, not processes: the JPEG decode and undistortion (OpenCV, numpy) release the GIL, and a decoded sample
    (~10 MB) would otherwise be pickled across processes."""

    def __init__(self, datamanager, dataset, scale_factor: float = 1.0, lazy: bool = False):
        self.datamanager = datamanager
        self.dataset = dataset
        self.scale_factor = scale_factor
        self.n_total = len(dataset)
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.pending = {}                       # idx -> Future of a decode that has not finished
        self.upcoming = set()                   # capped: the hinted next draws, never evicted
        self.prio = set()                       # background: indices moved to the priority pool
        self.stats = dict(hit=0, wait=0, spot=0, wait_s=0.0, spot_s=0.0)
        self.n_done = 0
        self.t_start = time.perf_counter()
        workers = int(datamanager.config.num_workers)
        cap = int(os.environ.get('MTGS_CACHE_MAX_IMAGES', '0') or 0)
        self.cache_max = cap if 0 < cap < self.n_total else 0
        self.background = False
        if self.cache_max:
            self.mode = "capped"
            self.pool = ThreadPoolExecutor(max_workers=max(1, workers or 4), thread_name_prefix="mtgs_cache")
            say(f"[image cache] capped at {self.cache_max} of {self.n_total} samples (MTGS_CACHE_MAX_IMAGES): "
                "decoded ahead of the sampler", flush=True)
        elif lazy or workers == 0:
            self.mode = "lazy"
        else:
            self.mode = "background"
            self.background = True
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mtgs_prefetch")
            with self.lock:
                for idx in range(self.n_total):
                    self.pending[idx] = pool.submit(self._decode_bg, idx)
            pool.shutdown(wait=False)           # the workers finish the queue; nothing new is submitted
            self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mtgs_prefetch_next")
            say(f"[image cache] background prefetch of {self.n_total} samples started ({workers} workers); "
                "training does not wait for it", flush=True)

    # ------------------------------------------------------------------ decoding
    def _store(self, idx, data):
        with self.lock:
            self.pending.pop(idx, None)
            first = idx not in self.cache
            if first:
                self.cache[idx] = data
                self.n_done += 1
            else:
                data = self.cache[idx]          # someone else stored it first: keep one copy
            if self.cache_max:
                self.cache.move_to_end(idx)
                while len(self.cache) > self.cache_max:
                    victim = next((k for k in self.cache if k not in self.upcoming), None)
                    if victim is None or victim == idx:
                        break
                    del self.cache[victim]
            done = self.n_done if (first and self.background) else 0
        if done and (done == self.n_total or done % max(1, self.n_total // 10) == 0):
            say(f"[image cache] background prefetch {done}/{self.n_total} ({time.perf_counter() - self.t_start:.0f} s)",
                flush=True)
            if done == self.n_total:
                self._report(final=True)
        return data

    def _decode(self, idx):
        return self._store(idx, self.datamanager._load_data(self.dataset, idx, scale_factor=self.scale_factor))

    def _decode_bg(self, idx):
        data = self.cache.get(idx)
        return data if data is not None else self._decode(idx)

    def hint(self, indices):
        """The indices the sampler will most likely ask for next (sampler.peek): decode them first."""
        if self.mode == "lazy":
            return
        with self.lock:
            if self.mode == "capped":
                self.upcoming = set(indices)
                for i in indices:
                    if i not in self.cache and i not in self.pending:
                        self.pending[i] = self.pool.submit(self._decode, i)
                return
            for i in indices:                   # background: move them to the priority pool
                if i in self.cache or i in self.prio:
                    continue
                fut = self.pending.get(i)
                if fut is not None and not fut.cancel():     # a worker already has it
                    continue
                self.prio.add(i)
                self.pending[i] = self.pool.submit(self._decode_bg, i)

    # ------------------------------------------------------------------ access
    def get_cache(self, idx):
        with self.lock:
            data = self.cache.get(idx)
            if data is not None and self.cache_max:
                self.cache.move_to_end(idx)
            fut = self.pending.get(idx) if data is None else None
        if data is not None:
            self.stats["hit"] += 1
        else:
            t0 = time.perf_counter()
            # capped: always wait for a decode in flight; background: a queued decode is cancelled and done here
            if fut is not None and (self.mode == "capped" or not fut.cancel()):
                data = fut.result()
                self.stats["wait"] += 1; self.stats["wait_s"] += time.perf_counter() - t0
            else:
                data = self._decode(idx)
                if self.mode != "lazy":
                    self.stats["spot"] += 1; self.stats["spot_s"] += time.perf_counter() - t0
        if self.mode == "capped":
            self._report()
        return data

    def _report(self, final=False):
        """capped: every 2000 draws; background: once the prefetch finished. wait = a decode was in flight,
        spot = decoded by the training thread itself."""
        st = self.stats
        n = st["hit"] + st["wait"] + st["spot"]
        if final or (n and n % 2000 == 0):
            what = "background prefetch finished; training draws meanwhile" if final else f"{n} draws"
            say(f"[image cache] {what}: resident {st['hit']}, waited for a worker {st['wait']} ({st['wait_s']:.1f} s), "
                f"decoded on the spot {st['spot']} ({st['spot_s']:.1f} s)", flush=True)
