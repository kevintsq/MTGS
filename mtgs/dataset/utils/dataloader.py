#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from nuplan_scripts.utils.console import say
from concurrent.futures import ThreadPoolExecutor
import os
from collections import OrderedDict
from functools import lru_cache

import torch
from rich.progress import track
import threading
import torch.multiprocessing as mp

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
        ahead = getattr(loader, "cache_max", 0) and hasattr(loader, "hint")
        for idx in range(len(self)):
            if ahead:   # capped image cache: the evaluation over all images is in order, decode the next ones ahead
                loader.hint(list(range(idx, min(idx + 32, len(self)))))
            data, camera = self.datamanager.cached_eval(idx)
            camera, data = self.datamanager.upload(data, camera)
            yield camera, data

    def __len__(self):
        return len(self.datamanager.eval_dataset)

class AsyncDataLoader:

    class PseudoDataset:
        def __init__(self, length):
            self.length = length
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            return idx

    def __init__(self, datamanager, dataset, scale_factor: float = 1.0):
        manager = mp.Manager()
        self.cached_idx = manager.dict()
        # Holding every decoded frame costs frames x cameras x resolution of host
        # RAM, which on a large joint block overshot a whole node's fair share (one
        # 1075-frame block peaked at 66 GB and was OOM-killed). MTGS_CACHE_MAX_IMAGES
        # caps how many stay resident; the least recently used are dropped and
        # reloaded on demand. 0 keeps the original unbounded behaviour.
        self.cache = OrderedDict()
        self.cache_max = int(os.environ.get('MTGS_CACHE_MAX_IMAGES', '0'))
        self.cache_lock = threading.Lock()
        self.dataset = dataset
        self.datamanager = datamanager
        self.scale_factor = scale_factor

        def collate_fn(batch):
            idx = batch[0]
            if idx in self.cached_idx:
                return None
            # If not in cache, load it
            data = self.datamanager._load_data(self.dataset, idx, scale_factor=self.scale_factor)
            return data

        self.dataloader = torch.utils.data.DataLoader(
            self.PseudoDataset(len(dataset)),
            batch_size=1,
            shuffle=False,
            num_workers=self.datamanager.config.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        say(f'Start async loading images.')
        self._start_loading()

    def _start_loading(self):
        def load_data():
            for idx, batch in enumerate(self.dataloader):
                with self.cache_lock:
                    if idx not in self.cache:  # Only cache if not already cached
                        assert batch is not None
                        self._remember(idx, batch)
            say(f'Async loading images finished.', style='bold green')

        self.thread = threading.Thread(target=load_data, daemon=True)
        self.thread.start()

    def _remember(self, idx, data):
        """Insert under the lock, evicting the oldest entry when over the cap."""
        self.cache[idx] = data
        self.cache.move_to_end(idx)
        self.cached_idx[idx] = True
        if self.cache_max > 0:
            while len(self.cache) > self.cache_max:
                old, _ = self.cache.popitem(last=False)
                self.cached_idx.pop(old, None)

    def get_cache(self, idx):
        with self.cache_lock:
            if idx in self.cache:
                self.cache.move_to_end(idx)
                return self.cache[idx]

        # If not in cache, load it
        data = self.datamanager._load_data(self.dataset, idx, scale_factor=self.scale_factor)
        # Cache the newly loaded data
        with self.cache_lock:
            if idx not in self.cache:  # Double-check in case background thread cached it
                self._remember(idx, data)
        return data

class OnDemandDataLoader:
    def __init__(self, datamanager, dataset, scale_factor: float = 1.0):
        self.dataset = dataset
        self.datamanager = datamanager
        self.scale_factor = scale_factor

    @lru_cache(maxsize=None)
    def get_cache(self, idx):
        data = self.datamanager._load_data(self.dataset, idx, scale_factor=self.scale_factor)
        return data

class PrefetchDataLoader:

    class PseudoDataset:
        def __init__(self, length):
            self.length = length
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            return idx

    def __init__(self, datamanager, dataset, scale_factor: float = 1.0):
        self.dataset = dataset
        self.datamanager = datamanager
        self.scale_factor = scale_factor
        self.cache = {}
        # MTGS_CACHE_MAX_IMAGES=N > 0 (2026-11-15): at most N decoded samples stay resident instead of every training
        # image (12k images of a merged block = ~100 GB of host RAM before the model). Samples come from the same
        # _load_data, so every returned sample is byte-identical to the unbounded cache; only when they are decoded
        # changes. The training sampler's next draws are predicted without consuming its random state (peek) and
        # decoded ahead by a thread pool (hint), so the GPU normally finds the sample resident. Unset / 0 = the
        # original prefetch of everything.
        self.cache_max = int(os.environ.get('MTGS_CACHE_MAX_IMAGES', '0') or 0)
        self.background = False
        if self.cache_max > 0 and self.cache_max < len(dataset):
            self._init_capped()
        else:
            self.cache_max = 0
            # 2026-11-25 (user: "应该直接默认改为async"): the full prefetch runs in the BACKGROUND and training starts at
            # once; a sample the sampler asks for before its turn is decoded on the spot (or waited for when a worker
            # already holds it). Same samples in the same order, only the decode timing changes; host RAM ends the
            # same (every image resident). pq_area0806 (19k images) had sat ~60 min on an idle GPU in the blocking
            # prefetch. Only num_workers=0 (no thread pool) keeps the inline prefetch.
            if int(self.datamanager.config.num_workers) == 0:
                self._prefetch()
            else:
                self._start_background()

    def _init_capped(self):
        from concurrent.futures import ThreadPoolExecutor as _Pool
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.pending = {}                       # idx -> Future of a decode in flight
        self.upcoming = set()                   # predicted next indices: never evicted
        self.pool = _Pool(max_workers=max(1, int(self.datamanager.config.num_workers) or 4), thread_name_prefix="mtgs_cache")
        self.stats = dict(hit=0, wait=0, miss=0, wait_s=0.0, miss_s=0.0)
        say(f"[image cache] capped at {self.cache_max} of {len(self.dataset)} samples (MTGS_CACHE_MAX_IMAGES): "
              "decoded ahead of the sampler instead of prefetching everything", flush=True)

    def _store(self, idx, data):
        with self.lock:
            self.pending.pop(idx, None)
            self.cache[idx] = data
            self.cache.move_to_end(idx)
            while len(self.cache) > self.cache_max:
                victim = next((k for k in self.cache if k not in self.upcoming), None)
                if victim is None or victim == idx:
                    break
                del self.cache[victim]
        return data

    def _decode(self, idx):
        return self._store(idx, self.datamanager._load_data(self.dataset, idx, scale_factor=self.scale_factor))

    def hint(self, indices):
        """The indices the sampler will most likely ask for next (sampler.peek): decode them in the background."""
        if not self.cache_max:
            return
        with self.lock:
            self.upcoming = set(indices)
            todo = [i for i in indices if i not in self.cache and i not in self.pending]
            for i in todo:
                self.pending[i] = self.pool.submit(self._decode, i)
    
    def _prefetch(self):
        def fetch_data(idx):
            return self.datamanager._load_data(self.dataset, idx, scale_factor=self.scale_factor)
        
        if self.datamanager.config.num_workers == 0:
            for idx in track(
                range(len(self.dataset)), 
                description="Prefetching data", 
                total=len(self.dataset)
            ):
                self.cache[idx] = fetch_data(idx)
        else:
            with ThreadPoolExecutor(max_workers=self.datamanager.config.num_workers) as executor:
                cached_data = list(
                    track(
                        executor.map(
                            fetch_data,
                            range(len(self.dataset)),
                        ),
                        description="Prefetching data",
                        total=len(self.dataset),
                    )
                )
            for idx, batch in enumerate(cached_data):
                self.cache[idx] = batch

    def _start_background(self):
        import time as _t
        self.background = True
        self.lock = threading.Lock()
        self.pending = {}                       # idx -> Future of a background decode that has not finished
        self.n_total = len(self.dataset)
        self.n_done = 0
        self.bg_stats = dict(hit=0, wait=0, spot=0, wait_s=0.0, spot_s=0.0)
        self.t_start = _t.perf_counter()
        pool = ThreadPoolExecutor(max_workers=int(self.datamanager.config.num_workers), thread_name_prefix="mtgs_prefetch")
        with self.lock:
            for idx in range(self.n_total):
                self.pending[idx] = pool.submit(self._bg_decode, idx)
        pool.shutdown(wait=False)               # the workers finish the queue; nothing new is submitted
        say(f"[image cache] background prefetch of {self.n_total} samples started "
            f"({self.datamanager.config.num_workers} workers); training does not wait for it", flush=True)

    def _bg_store(self, idx, data):
        import time as _t
        with self.lock:
            first = idx not in self.cache
            if first:
                self.cache[idx] = data
                self.n_done += 1
            self.pending.pop(idx, None)
            done = self.n_done if first else 0
        if done and (done == self.n_total or done % max(1, self.n_total // 10) == 0):
            say(f"[image cache] background prefetch {done}/{self.n_total} ({_t.perf_counter() - self.t_start:.0f} s)",
                flush=True)
            if done == self.n_total:
                st = self.bg_stats
                say(f"[image cache] background prefetch finished; training draws meanwhile: resident {st['hit']}, "
                    f"waited for a worker {st['wait']} ({st['wait_s']:.1f} s), decoded on the spot {st['spot']} "
                    f"({st['spot_s']:.1f} s)", flush=True)
        return self.cache[idx]

    def _bg_decode(self, idx):
        data = self.cache.get(idx)
        if data is not None:                    # decoded on the spot by the training thread first
            return data
        return self._bg_store(idx, self.datamanager._load_data(self.dataset, idx, scale_factor=self.scale_factor))

    def _get_background(self, idx):
        data = self.cache.get(idx)
        if data is not None:
            self.bg_stats["hit"] += 1
            return data
        import time as _t
        t0 = _t.perf_counter()
        with self.lock:
            data = self.cache.get(idx)
            fut = self.pending.get(idx)
        if data is not None:
            self.bg_stats["hit"] += 1
            return data
        if fut is not None and not fut.cancel():    # a worker is decoding it right now: wait for that one
            data = fut.result()
            self.bg_stats["wait"] += 1; self.bg_stats["wait_s"] += _t.perf_counter() - t0
            return data
        data = self._bg_store(idx, self.datamanager._load_data(self.dataset, idx, scale_factor=self.scale_factor))
        self.bg_stats["spot"] += 1; self.bg_stats["spot_s"] += _t.perf_counter() - t0
        return data

    def get_cache(self, idx):
        if self.background:
            return self._get_background(idx)
        if not self.cache_max:
            return self.cache[idx]
        with self.lock:
            data = self.cache.get(idx)
            if data is not None:
                self.cache.move_to_end(idx); self.stats["hit"] += 1
                self._report()
                return data
            fut = self.pending.get(idx)
        import time as _t
        t0 = _t.perf_counter()
        if fut is not None:
            data = fut.result(); self.stats["wait"] += 1; self.stats["wait_s"] += _t.perf_counter() - t0
        else:
            data = self._decode(idx); self.stats["miss"] += 1; self.stats["miss_s"] += _t.perf_counter() - t0
        self._report()
        return data

    def _report(self):
        """Every 2000 samples: how often the training thread waited (wait = decode in flight, miss = not predicted)."""
        st = self.stats; n = st["hit"] + st["wait"] + st["miss"]
        if n % 2000 == 0:
            say(f"[image cache] {n} samples: resident {st['hit']}, waited {st['wait']} ({st['wait_s']:.1f} s), "
                  f"decoded on the spot {st['miss']} ({st['miss_s']:.1f} s)", flush=True)   # print: see _init_capped
