"""Joint SfM split into a matching phase and a mapping phase (2026-11-16).

On torch a GPU job whose 2-h average GPU utilisation stays below ~50-60 % is cancelled, so a whole pure-SfM ba_stack
(feature extraction + matching on the GPU, then hours of mapper / BA on the CPU) cannot run on a GPU node; on a CPU node
the matching alone runs ~15x slower than on a GPU (civo h200: 127 vs ~8 pairs/s). MTGS_SFM_PHASE splits the stage:

    match  (GPU job)  model build (images, masks, cameras, pairs, rig) + feature extraction + matching, then the COLMAP
                      database is written to <block>/sfm_match/<model>/database.db.zst with meta.json and the stage stops:
                      no mapper, no alignment, no vsd written. Restartable: a snapshot whose key matches is kept.
    map    (CPU job)  model build again (cheap), the database restored from the snapshot instead of extraction + matching
                      (sha256 checked; the pair set must be the same), then mapper / BA / alignment / derived
                      products exactly as an unsplit run. After a successful stage the database is deleted and only
                      meta.json stays: a rerun of the stage (e.g. a later step failed) then finds the kept solution under
                      the same key (sfm_reuse) or the published model, as an unsplit rerun would.
    unset             today's single stage.

Only for the pure-SfM recipe (MTGS_JOINT_BA=1, MTGS_JOINT_SFM=1, use_colmap_ba). The key that identifies a snapshot is
sfm_reuse.compute_key computed with the COLMAP_USE_GPU of the MATCH job (it decides how features were made), so the
map job (no GPU, COLMAP_USE_GPU=0) and a later reuse of the solution see the same key as an unsplit GPU run.
"""
import hashlib
import json
import os
import sqlite3
import subprocess
import time

PHASES = ('', 'match', 'map')
DB = 'database.db.zst'
META = 'meta.json'


def phase():
    p = os.environ.get('MTGS_SFM_PHASE', '')
    if p not in PHASES:
        raise SystemExit(f'MTGS_SFM_PHASE={p!r}: one of match, map (or unset)')
    return p


def check_recipe(config):
    p = phase()
    if p and not (os.environ.get('MTGS_JOINT_BA') == '1' and os.environ.get('MTGS_JOINT_SFM') == '1' and config.use_colmap_ba):
        raise SystemExit(f'MTGS_SFM_PHASE={p}: only for the pure-SfM recipe (MTGS_JOINT_BA=1, MTGS_JOINT_SFM=1, '
                         'use_colmap_ba: true); per-video models have no separate matching step to split off')
    return p


def snap_dir(video_scene, model_name):
    """<block>/sfm_match/<model>: inside the block's workdir, so the kit's stage-out tar carries it to the map job."""
    return os.path.join(os.path.dirname(video_scene.colmap_path), 'sfm_match', model_name)


def read_meta(d):
    p = os.path.join(d, META)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except ValueError:
        return None


def has_db(d):
    return os.path.exists(os.path.join(d, DB))


class gpu_env_of:
    """compute_key with the COLMAP_USE_GPU value the features were made with (None = unset)."""
    def __init__(self, value):
        self.value, self.saved = value, None

    def __enter__(self):
        self.saved = os.environ.get('COLMAP_USE_GPU')
        if self.value is None:
            os.environ.pop('COLMAP_USE_GPU', None)
        else:
            os.environ['COLMAP_USE_GPU'] = self.value

    def __exit__(self, *a):
        if self.saved is None:
            os.environ.pop('COLMAP_USE_GPU', None)
        else:
            os.environ['COLMAP_USE_GPU'] = self.saved


def _sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return h.hexdigest()
            h.update(b)


def _pairs_sha256(path):
    """The pair SET: create_colmap_model writes image_pairs.txt by iterating a python set of name tuples, so the line
    order changes with every process (str hash randomisation); the pairs themselves are deterministic."""
    return hashlib.sha256(''.join(sorted(open(path).readlines())).encode()).hexdigest()


def _threads():
    return str(int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or os.cpu_count() or 8)


def write(colmap_path, d, key_sha, use_gpu, log=print):
    """Called by bundle_adjustment after matches_importer in the match phase."""
    db = os.path.join(colmap_path, 'database.db')
    c = sqlite3.connect(db)
    c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    n_img = c.execute('select count(*) from images').fetchone()[0]
    n_pairs = c.execute('select count(*) from two_view_geometries where rows > 0').fetchone()[0]
    c.close()
    for f in ('database.db-wal', 'database.db-shm'):
        if os.path.exists(os.path.join(colmap_path, f)) and os.path.getsize(os.path.join(colmap_path, f)) > 0:
            raise SystemExit(f'[sfm split] {f} not empty after the checkpoint: the database is still open elsewhere')
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    tmp = os.path.join(d, DB + '.tmp')
    subprocess.run(['zstd', '-q', '-f', '-3', f'-T{_threads()}', db, '-o', tmp], check=True)
    meta = dict(key=key_sha, db_sha256=_sha256(db), db_bytes=os.path.getsize(db),
                pairs_set_sha256=_pairs_sha256(os.path.join(colmap_path, 'image_pairs.txt')),
                colmap_use_gpu=use_gpu, images=n_img, verified_pairs=n_pairs,
                written=time.strftime('%Y-%m-%dT%H:%M:%S'), host=os.uname().nodename)
    os.replace(tmp, os.path.join(d, DB))
    json.dump(meta, open(os.path.join(d, META) + '.tmp', 'w'), indent=1)
    os.replace(os.path.join(d, META) + '.tmp', os.path.join(d, META))
    log(f'[sfm split] match phase done: database ({meta["db_bytes"] / 1e9:.2f} GB, {n_img} images, {n_pairs} verified pairs) '
        f'-> {d}/{DB} ({os.path.getsize(os.path.join(d, DB)) / 1e9:.2f} GB, {time.time() - t0:.0f} s)')
    return meta


def restore(colmap_path, d, log=print):
    """Called by bundle_adjustment in the map phase instead of extraction + matching."""
    meta = read_meta(d)
    if meta is None or not has_db(d):
        raise SystemExit(f'[sfm split] map phase: no match snapshot in {d} (run the ba_match stage first)')
    pairs = os.path.join(colmap_path, 'image_pairs.txt')
    if _pairs_sha256(pairs) != meta['pairs_set_sha256']:
        raise SystemExit('[sfm split] map phase: the pairs of image_pairs.txt differ from those the database was matched with')
    db = os.path.join(colmap_path, 'database.db')
    for f in ('database.db', 'database.db-wal', 'database.db-shm'):
        if os.path.exists(os.path.join(colmap_path, f)):
            os.remove(os.path.join(colmap_path, f))
    t0 = time.time()
    subprocess.run(['zstd', '-q', '-d', f'-T{_threads()}', os.path.join(d, DB), '-o', db], check=True)
    if _sha256(db) != meta['db_sha256']:
        raise SystemExit('[sfm split] map phase: restored database does not match its sha256')
    log(f'[sfm split] map phase: database restored from {d} ({meta["db_bytes"] / 1e9:.2f} GB, {meta["images"]} images, '
        f'matched on {meta["host"]} {meta["written"]}, {time.time() - t0:.0f} s); no extraction, no matching')
    return meta


def consume(d):
    """After a successful map phase: delete the database, keep meta.json (its key / COLMAP_USE_GPU serve a rerun)."""
    meta = read_meta(d)
    if os.path.exists(os.path.join(d, DB)):
        os.remove(os.path.join(d, DB))
    if meta is not None:
        meta['consumed'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        json.dump(meta, open(os.path.join(d, META) + '.tmp', 'w'), indent=1)
        os.replace(os.path.join(d, META) + '.tmp', os.path.join(d, META))
