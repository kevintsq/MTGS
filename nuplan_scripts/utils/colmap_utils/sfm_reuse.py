"""Reuse of a finished COLMAP solution (2026-11-15).

A block's SfM costs hours (Strip ru_area0806: 3.5 h on torch, px_loc_0195: 51 h on kaitain), while everything ba_stack
derives from it (aligned camera poses in the vsd, sfm_point_cloud/<model>.pcd) takes minutes. clean_temp_files.py keeps
the SfM model, but nothing could start from it: a retrain reran the whole COLMAP. Now every COLMAP run leaves, next to
the block, colmap_models/<model>/ with
    sfm_key.json          what the solution was computed from (hash of the inputs below + a readable summary)
    sfm_invalid.json      the cameras the model builder left out (mask-invalid / duplicate names), so a reuse marks the
                          same cameras invalid as the run that built the model
    name_to_token.json    COLMAP image name -> camera token
    sfm_model_compact/0   cameras.bin, points3D.bin and images.bin WITHOUT the 2-D observations (align_model reads image
                          poses, camera ids and names, and the point tracks' image ids / xyz / rgb only; px: 1.4 GB -> 0.3 GB)
and ba_stack takes the solution from there when the key matches instead of building and solving the model again.

The key covers what changes the SOLUTION: per camera of every non-skipped frame of the input vsd its token, image path,
intrinsics, distortion, extrinsics and the frame's ego pose (pairs, priors and the Sim3 come from those), the vehicle of
each pass, the block size (COLMAP frame scale), use_colmap_ba and the SfM_ENV switches. NOT in the key: masks (the mask
stage regenerates them before ba_stack; the model's own image set is kept in sfm_invalid.json), thread counts, scratch /
log dirs, the COLMAP version (recorded only), and everything align_model reads (MTGS_SIM3_ROT, MTGS_SIM3_YAW*, MTGS_SFM_ALIGN_THRESH,
MTGS_SFM_RIGID_M, MTGS_SFM_DISCONNECT_M, MTGS_SFM_LOCAL_SCALE, MTGS_JOINT_WRITE_EGO): the alignment runs again on reuse.

MTGS_SFM_REUSE: unset = reuse when the key matches; 0 = never reuse (full COLMAP run); 1 = also accept a model kept
before this change (no key file), if its images are all cameras of the current input vsd.
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess

import numpy as np

KEY_FILE = 'sfm_key.json'
INVALID_FILE = 'sfm_invalid.json'
NAME_MAP = 'name_to_token.json'
COMPACT = 'sfm_model_compact'
SFM_ENV = ('MTGS_JOINT_BA', 'MTGS_JOINT_SFM', 'MTGS_SFM_RIG', 'COLMAP_SFM_MAPPER', 'COLMAP_BA_MODE',
           'COLMAP_REFINE_INTRINSICS', 'COLMAP_PRIOR_STD', 'COLMAP_PRIOR_ROBUST', 'COLMAP_USE_GPU',
           'MTGS_SFM_PRIOR_MAX_ERR', 'MTGS_MAX_PAIRS_PER_IMAGE', 'MTGS_MAX_PAIR_DIST', 'MTGS_COLMAP_CAM_MODEL',
           'MTGS_SFM_IMPORT', 'MTGS_SFM_DB_IMPORT')


def keep_dir(video_scene, model_name):
    """<block>/colmap_models/<model>: the folder clean_temp_files.py keeps."""
    return os.path.join(os.path.dirname(video_scene.colmap_path), 'colmap_models', model_name)


def _colmap_version():
    try:
        out = subprocess.run(['colmap', 'help'], capture_output=True, text=True, timeout=30).stdout
        return next((l.strip() for l in out.splitlines() if 'COLMAP' in l), 'unknown')
    except Exception:
        return 'unknown'


def compute_key(video_scene, video_scene_dict, use_colmap_ba):
    h = hashlib.sha256()
    n_img = 0
    for video_token in video_scene_dict:
        v = video_scene_dict[video_token]
        h.update(f'{video_token}|{v["vehicle_name"]}'.encode())
        for info in v['frame_infos']:
            if info.get('skipped', False):
                continue
            h.update(info['token'].encode())
            h.update(np.asarray(info['ego2global'], np.float64).tobytes())
            for cam in sorted(info['cams']):
                c = info['cams'][cam]
                h.update(f'{cam}|{c["token"]}|{c["data_path"]}'.encode())
                for k in ('cam_intrinsic', 'distortion', 'sensor2ego_rotation', 'sensor2ego_translation'):
                    h.update(np.asarray(c[k], np.float64).tobytes())
                n_img += 1
    env = {k: os.environ.get(k) for k in SFM_ENV}
    h.update(json.dumps(env, sort_keys=True).encode())
    h.update(repr([float(x) for x in video_scene.block_size]).encode())
    h.update(repr(bool(use_colmap_ba)).encode())
    return dict(sha256=h.hexdigest(), images=n_img, passes=list(video_scene_dict), env=env,
                block_size=[float(x) for x in video_scene.block_size], use_colmap_ba=bool(use_colmap_ba))


def _compact_images_bin(src, dst):
    """images.bin without the 2-D observations (num_points2D = 0); poses, camera ids and names unchanged."""
    with open(src, 'rb') as fi, open(dst + '.tmp', 'wb') as fo:
        n = struct.unpack('<Q', fi.read(8))[0]
        fo.write(struct.pack('<Q', n))
        for _ in range(n):
            head = fi.read(64)                                       # image_id (i), qvec (4d), tvec (3d), camera_id (i)
            name = b''
            while True:
                ch = fi.read(1)
                name += ch
                if ch == b'\x00':
                    break
            npts = struct.unpack('<Q', fi.read(8))[0]
            fi.seek(24 * npts, 1)
            fo.write(head + name + struct.pack('<Q', 0))
    os.replace(dst + '.tmp', dst)


def save(video_scene, colmap_path, model_name, key, video_scene_dict):
    """Keep the solution just computed (or just accepted) under colmap_models/<model>/ for later reuse."""
    model = os.path.join(colmap_path, 'sfm_model', '0')
    if not os.path.exists(os.path.join(model, 'images.bin')):
        return None
    out = keep_dir(video_scene, model_name)
    comp = os.path.join(out, COMPACT, '0')
    os.makedirs(comp, exist_ok=True)
    for f in ('cameras.bin', 'points3D.bin'):
        shutil.copyfile(os.path.join(model, f), os.path.join(comp, f))
    _compact_images_bin(os.path.join(model, 'images.bin'), os.path.join(comp, 'images.bin'))
    if os.path.exists(os.path.join(colmap_path, NAME_MAP)):
        shutil.copyfile(os.path.join(colmap_path, NAME_MAP), os.path.join(out, NAME_MAP))
    invalid = sorted(c['token'] for v in video_scene_dict.values() for f in v['frame_infos'] if not f.get('skipped', False)
                     for c in f['cams'].values() if c.get('valid', True) is False)
    json.dump(invalid, open(os.path.join(out, INVALID_FILE), 'w'))
    key = dict(key, colmap_version=_colmap_version())
    json.dump(key, open(os.path.join(out, KEY_FILE), 'w'), indent=1)
    return out


def _model_in(d):
    for sub in ('sfm_model', COMPACT):
        m = os.path.join(d, sub, '0')
        if os.path.exists(os.path.join(m, 'images.bin')) and os.path.exists(os.path.join(m, 'points3D.bin')):
            return m
    return None


def find(video_scene, colmap_path, model_name, key, video_scene_dict, log=print):
    """(model dir, name_to_token path, set of invalid camera tokens, source dir) of a reusable solution, or None."""
    mode = os.environ.get('MTGS_SFM_REUSE', '')
    if mode == '0':
        log(f'[sfm reuse] {model_name}: MTGS_SFM_REUSE=0, full COLMAP run')
        return None
    for d in (colmap_path, keep_dir(video_scene, model_name)):
        model, nm = _model_in(d), os.path.join(d, NAME_MAP)
        if model is None or not os.path.exists(nm):
            continue
        kf = os.path.join(d, KEY_FILE)
        if os.path.exists(kf):
            old = json.load(open(kf))
            if old.get('sha256') != key['sha256']:
                log(f'[sfm reuse] {model_name}: model in {d} was computed from other inputs '
                    f'({old.get("images")} vs {key["images"]} images, env {old.get("env")} vs {key["env"]}): not reused')
                continue
            inv = os.path.join(d, INVALID_FILE)
            invalid = set(json.load(open(inv))) if os.path.exists(inv) else None
            return model, nm, invalid, d
        if mode != '1':
            log(f'[sfm reuse] {model_name}: model in {d} has no {KEY_FILE} (kept before 2026-11-15); '
                f'MTGS_SFM_REUSE=1 accepts it')
            continue
        tokens = set(json.load(open(nm)).values())
        cams = {c['token'] for v in video_scene_dict.values() for f in v['frame_infos'] if not f.get('skipped', False)
                for c in f['cams'].values()}
        extra = tokens - cams
        if extra:
            log(f'[sfm reuse] {model_name}: legacy model in {d} holds {len(extra)} images that are not cameras of the '
                f'input vsd: not reused')
            continue
        return model, nm, None, d
    return None


def mark_valid(video_scene_dict, invalid, name_map_path):
    """The camera validity create_colmap_model would have written: invalid = left out of the model by the builder
    (stored at build time; for a legacy model: not among its images)."""
    if invalid is None:
        named = set(json.load(open(name_map_path)).values())
    for v in video_scene_dict.values():
        for f in v['frame_infos']:
            if f.get('skipped', False):
                continue
            for c in f['cams'].values():
                c['valid'] = (c['token'] not in invalid) if invalid is not None else (c['token'] in named)


def adopt(video_scene, src, model_name, key, video_scene_dict):
    """After a reuse from a place that is not (yet) a keyed colmap_models/<model>/ (the full colmap tree kept with
    MTGS_KEEP_COLMAP, or a legacy model accepted with MTGS_SFM_REUSE=1): store it keyed, so the next run reuses it
    without asking. video_scene_dict must already carry the reused validity (mark_valid)."""
    out = keep_dir(video_scene, model_name)
    if os.path.realpath(src) != os.path.realpath(out):
        return save(video_scene, src, model_name, key, video_scene_dict)
    invalid = sorted(c['token'] for v in video_scene_dict.values() for f in v['frame_infos'] if not f.get('skipped', False)
                     for c in f['cams'].values() if c.get('valid', True) is False)
    json.dump(invalid, open(os.path.join(out, INVALID_FILE), 'w'))
    json.dump(dict(key, colmap_version='unknown (legacy model)'), open(os.path.join(out, KEY_FILE), 'w'), indent=1)
    return out
