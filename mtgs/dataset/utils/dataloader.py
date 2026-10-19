#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from concurrent.futures import ThreadPoolExecutor
import os
from collections import OrderedDict
from functools import lru_cache

import torch
from rich.progress import track
import threading
import torch.multiprocessing as mp

from nerfstudio.utils.rich_utils import CONSOLE


class FixedIndicesPseudoDataloader:
    def __init__(self, datamanager):
        self.datamanager = datamanager
        self.cached_func = self.datamanager.cached_train if self.datamanager.same_train_eval else self.datamanager.cached_eval

    def __iter__(self):
        for idx in range(len(self)):
            data, camera = self.datamanager.cached_eval(idx)
            data["image"] = data["image"].to(self.datamanager.device)
            camera = camera.to(self.datamanager.device)
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
        CONSOLE.log(f'Start async loading images.')
        self._start_loading()

    def _start_loading(self):
        def load_data():
            for idx, batch in enumerate(self.dataloader):
                with self.cache_lock:
                    if idx not in self.cache:  # Only cache if not already cached
                        assert batch is not None
                        self._remember(idx, batch)
            CONSOLE.log(f'Async loading images finished.', style='bold green')

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
        self._prefetch()
    
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

    def get_cache(self, idx):
        return self.cache[idx]
