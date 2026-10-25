#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import copy
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse

import multiprocessing
import subprocess

from tqdm import tqdm

import numpy as np
import cv2
from pyquaternion import Quaternion

from nuplan_scripts.utils.constants import CONSOLE
from nuplan_scripts.utils.config import load_config, RoadBlockConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.camera_utils import field_of_view_intrinsic
from nuplan_scripts.utils.colmap_utils.gen_colmap_db import create_colmap_database
from nuplan_scripts.utils.colmap_utils.read_write_model import read_model
from nuplan_scripts.utils.colmap_utils.align_model import compute_transformation_matrix_with_scaling
from nuplan_scripts.utils.constants import NUPLAN_SENSOR_ROOT


def copy_ego_masks(video_scene: VideoScene):
    data_root = video_scene.data_root
    raw_mask_path = os.path.join(data_root, 'ego_masks/raw')
    assert os.path.exists(raw_mask_path), f'Ego masks not found at {raw_mask_path}'
    output_raw_mask_path = os.path.join(video_scene.raw_mask_path, video_scene.mask_suffix_ego)
    if not os.path.exists(output_raw_mask_path):
        os.makedirs(os.path.dirname(output_raw_mask_path), exist_ok=True)
        shutil.copytree(raw_mask_path, output_raw_mask_path)

def create_colmap_folder(colmap_path):

    if os.path.exists(colmap_path):
        shutil.rmtree(colmap_path)

    os.makedirs(f"{colmap_path}/images", exist_ok=True)
    os.makedirs(f"{colmap_path}/masks", exist_ok=True)
    os.makedirs(f"{colmap_path}/sfm_model", exist_ok=True)
    os.makedirs(f"{colmap_path}/sparse_model", exist_ok=True)
    open(
        f"{colmap_path}/sparse_model/points3D.txt", "w"
    ).close()  # create an empty points3D.txt

def clean_colmap_folder(colmap_path):
    if os.path.exists(colmap_path):
        shutil.rmtree(colmap_path)

def create_cameras_per_vehicle(video_scene_dict, colmap_path):
    exsiting_veh_name = {}
    f_cameras = open(f"{colmap_path}/sparse_model/cameras.txt", "w")
    global_cam_id = 1
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]["frame_infos"]

        # same vehicle has same camera calibration
        veh_name = video_scene_dict[video_token]['vehicle_name']
        if veh_name not in exsiting_veh_name:
            exsiting_veh_name[veh_name] = {}
            cam_infos = frame_infos[0]["cams"]
            for cam in cam_infos:
                cam_info = cam_infos[cam]
                intrinsic = cam_info['cam_intrinsic']
                distortion = cam_info['distortion']
                line = (
                    f"{global_cam_id} OPENCV 1920 1080 "
                    + f"{intrinsic[0, 0]} {intrinsic[1, 1]} {intrinsic[0, 2]} {intrinsic[1, 2]} "
                    + f"{distortion[0]} {distortion[1]} {distortion[2]} {distortion[3]}\n"
                )
                f_cameras.write(line)
                exsiting_veh_name[veh_name][cam] = global_cam_id
                global_cam_id += 1
    f_cameras.close()

    cam_name_mapper = {}
    for veh_name in exsiting_veh_name:
        for cam in exsiting_veh_name[veh_name]:
            cam_name_colmap = f'{veh_name}_{cam}'
            cam_name_mapper[cam_name_colmap] = exsiting_veh_name[veh_name][cam]
            os.makedirs(f'{colmap_path}/images/{cam_name_colmap}', exist_ok=True)
            os.makedirs(f'{colmap_path}/masks/{cam_name_colmap}', exist_ok=True)

    return cam_name_mapper


def _prepare_one_image(task):
    """Read the two masks for one frame, write the colmap mask, link the image.

    All of this is per-frame and independent, but it used to run inline in the
    single-threaded model-building loop: two PNG reads and one PNG write per
    frame over NFS, thousands of times. That left a 152-core node at 26% of one
    core. Returning the validity decision lets the bookkeeping stay sequential
    (image ids and images.txt ordering must not change) while the I/O fans out.
    """
    (data_path, sem_mask_path, ego_mask_path,
     image_output_path, mask_output_path) = task

    sem_mask = cv2.imread(sem_mask_path, cv2.IMREAD_GRAYSCALE)
    if sem_mask is None:
        return ('missing', sem_mask_path)
    ego_mask = cv2.imread(ego_mask_path, cv2.IMREAD_GRAYSCALE) != 0
    invalid_mask = ego_mask | (sem_mask >= 10)   # sky, person, ...

    # TODO: this threshold should be further tuned.
    if invalid_mask.sum() > 0.8 * invalid_mask.shape[0] * invalid_mask.shape[1]:
        return ('invalid', None)

    os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
    # colmap only reads the image, so a hard link beats copying the bytes; falls
    # back to a copy when source and destination are on different filesystems.
    try:
        if os.path.lexists(image_output_path):
            os.remove(image_output_path)
        os.link(data_path, image_output_path)
    except OSError:
        shutil.copy2(data_path, image_output_path)
    cv2.imwrite(mask_output_path, np.logical_not(invalid_mask).astype(np.uint8) * 255)
    return ('ok', None)


def _restore_published(work, nfs_path):
    """Bring a previously archived colmap model back onto scratch.

    A joint bundle adjustment on a large block runs for half a day, so when the
    stage dies after it (a bad gate, a timeout in a later step) the model must not
    be rebuilt from nothing. If the archives from a previous run are on shared
    storage, unpack them into the working directory and report that the expensive
    part can be skipped.
    """
    if not os.path.isdir(nfs_path):
        return False
    restored = False
    for name in ('sfm_model', 'sparse_model'):
        for ext in ('.tar.zst', '.tar'):
            arc = os.path.join(nfs_path, name + ext)
            if not os.path.exists(arc) or os.path.getsize(arc) == 0:
                continue
            os.makedirs(work, exist_ok=True)
            cmd = ['tar', '-C', work, '-xf', arc]
            if ext == '.tar.zst':
                cmd = ['tar', '-C', work, '--zstd', '-xf', arc]
            try:
                subprocess.run(cmd, check=True)
                restored = True
                CONSOLE.log(f'[colmap] restored {name} from {os.path.basename(arc)}')
            except Exception as exc:
                CONSOLE.log(f'[colmap] could not restore {name}: {exc}')
            break
    return restored and os.path.isdir(os.path.join(work, 'sfm_model'))


def _colmap_workdir(nfs_path):
    """Where to actually build a colmap model.

    Everything colmap does here is small random I/O -- an 8 GB SQLite database
    with WAL, thousands of jpeg reads, mask writes -- which is the worst possible
    shape for a shared NFS/WekaFS mount. MTGS_COLMAP_SCRATCH redirects the model
    to node-local memory (/dev/shm) for the duration; only the finished artefacts
    go back over the network. Returns (workdir, nfs_path); they are equal when no
    scratch is configured.
    """
    scratch = os.environ.get('MTGS_COLMAP_SCRATCH')
    if not scratch:
        return nfs_path, nfs_path
    work = os.path.join(scratch, os.path.basename(os.path.dirname(nfs_path)),
                        os.path.basename(nfs_path))
    os.makedirs(work, exist_ok=True)
    return work, nfs_path


def _publish_colmap(work, nfs_path):
    """Move a scratch-built colmap model back to shared storage.

    Only the sparse/sfm models and the logs are worth keeping; the images are
    hard links to data that already lives on the share and the database is a
    rebuildable intermediate, so neither is copied back. Archiving is done with
    zstd when available (the sfm model is many small binary files, and one
    sequential stream beats thousands of NFS creates).
    """
    if work == nfs_path:
        return
    os.makedirs(nfs_path, exist_ok=True)
    for name in ('sfm_model', 'sparse_model'):
        src = os.path.join(work, name)
        if not os.path.isdir(src):
            continue
        tar = os.path.join(nfs_path, name + '.tar')
        cmd = ['tar', '-C', work, '-cf', tar, name]
        if shutil.which('zstd'):
            tar += '.zst'
            cmd = ['tar', '-C', work, '--zstd', '-cf', tar, name]
        try:
            subprocess.run(cmd, check=True)
            CONSOLE.log(f'[colmap] archived {name} -> {os.path.basename(tar)}')
        except Exception as exc:            # keep the run alive; the model is still on scratch
            CONSOLE.log(f'[colmap] archive of {name} failed ({exc}); copying instead')
            shutil.copytree(src, os.path.join(nfs_path, name), dirs_exist_ok=True)
    for log in ('colmap_ba.log', 'image_pairs.txt'):
        src = os.path.join(work, log)
        if os.path.exists(src):
            try:
                shutil.copy2(src, os.path.join(nfs_path, log))
            except OSError:
                pass


def _sfm_rig():
    """MTGS_SFM_RIG=1 (with MTGS_JOINT_SFM): one COLMAP rig per vehicle -- the 8 cameras of a
    frame share one rig pose, their cam_from_rig start at the calibration and are refined by
    the mapper (user 2026-10-25). COLMAP groups images into frames by the name after the
    camera folder, so images are named by the frame token in this mode."""
    return os.environ.get('MTGS_JOINT_SFM') == '1' and os.environ.get('MTGS_SFM_RIG') == '1'


def _colmap_image_name(veh, cam, frame_info, cam_info):
    if _sfm_rig():
        return f"{veh}_{cam}/{frame_info['token']}.jpg"
    return f"{veh}_{cam}/{os.path.basename(cam_info['data_path'])}"


def _write_rig_config(video_scene_dict, colmap_path, ref='CAM_F0'):
    """rig_config.json for colmap rig_configurator: per vehicle, ref camera = rig frame,
    cam_from_rig = inv(cam2ego_cam) @ cam2ego_ref from the first frame's calibration"""
    import json
    rigs = {}
    for vt, v in video_scene_dict.items():
        veh = v['vehicle_name']
        if veh in rigs:
            continue
        fi = next((f for f in v['frame_infos'] if not f.get('skipped', False)), None)
        if fi is None:
            continue
        c2e = {}
        for cam, ci in fi['cams'].items():
            T = np.eye(4); T[:3, :3] = Quaternion(ci['sensor2ego_rotation']).rotation_matrix
            T[:3, 3] = ci['sensor2ego_translation']; c2e[cam] = T
        r = ref if ref in c2e else sorted(c2e)[0]
        cams = []
        # COLMAP's rig_configurator needs the reference sensor FIRST in the list
        # ("The reference sensor needs to be added first before other sensors")
        for cam, T in sorted(c2e.items(), key=lambda kv: (kv[0] != r, kv[0])):
            entry = {'image_prefix': f'{veh}_{cam}/'}
            if cam == r:
                entry['ref_sensor'] = True
            else:
                cfr = np.linalg.inv(T) @ c2e[r]
                q = Quaternion(matrix=cfr[:3, :3], atol=1e-6)
                entry['cam_from_rig_rotation'] = [float(q.w), float(q.x), float(q.y), float(q.z)]
                entry['cam_from_rig_translation'] = [float(x) for x in cfr[:3, 3]]
            cams.append(entry)
        rigs[veh] = {'cameras': cams}
    with open(os.path.join(colmap_path, 'rig_config.json'), 'w') as f:
        json.dump(list(rigs.values()), f, indent=1)
    CONSOLE.log(f'[sfm rig] rig_config.json: {len(rigs)} rigs ({", ".join(rigs)}), ref {ref}')


def create_colmap_model(video_scene: VideoScene, video_scene_dict, colmap_path):

    create_colmap_folder(colmap_path)
    cam_name_mapper = create_cameras_per_vehicle(video_scene_dict, colmap_path)

    progress_bar = tqdm(
        total=VideoScene.count_total_frames(video_scene_dict),
        desc='Loading image infos', ncols=120)

    scale_factor = 10 / max(video_scene.block_size)

    f_images = open(f"{colmap_path}/sparse_model/images.txt", "w")
    global_image_id = 1
    invalid_images = 0
    cam_views = []
    image_names = set()
    name_to_token = {}          # colmap image name -> camera token (align_model)

    # Fan the per-frame image/mask I/O out first; the loop below then only does
    # bookkeeping, which must stay ordered. Threads (not processes) because the
    # work is cv2 read/write and filesystem calls, all of which drop the GIL.
    _tasks, _keys = [], []
    for _vt in video_scene_dict:
        for _info in video_scene_dict[_vt]['frame_infos']:
            if _info.get('skipped', False):
                continue
            _veh = video_scene_dict[_vt]['vehicle_name']
            for _cam in _info['cams']:
                _ci = _info['cams'][_cam]
                _dp = os.path.join(NUPLAN_SENSOR_ROOT, _ci['data_path'])
                _ip = _colmap_image_name(_veh, _cam, _info, _ci)
                _tasks.append((
                    _dp,
                    os.path.join(video_scene.raw_mask_path,
                                 video_scene.mask_suffix_cityscape,
                                 _ci['data_path'].replace('.jpg', '.png')),
                    os.path.join(video_scene.raw_mask_path,
                                 video_scene.mask_suffix_ego, f'{_cam}.png'),
                    f'{colmap_path}/images/{_ip}',
                    f'{colmap_path}/masks/{_ip}.png',
                ))
                _keys.append((_info['token'], _cam))

    _io_workers = int(os.environ.get('MTGS_IO_WORKERS',
                                     min(64, (os.cpu_count() or 8) * 2)))
    _prepared = {}
    with ThreadPoolExecutor(max_workers=_io_workers) as _ex:
        for _k, _res in zip(_keys, tqdm(
                _ex.map(_prepare_one_image, _tasks),
                total=len(_tasks), desc='Preparing images', ncols=120)):
            _prepared[_k] = _res
    for _k, (_st, _detail) in _prepared.items():
        if _st == 'missing':
            raise FileNotFoundError(
                f'semantic mask missing/unreadable: {_detail} '
                '(mask stage output incomplete - regenerate masks for this block)')

    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]["frame_infos"]
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]

        # same vehicle has same camera calibration
        veh_name = video_scene_dict[video_token]['vehicle_name']

        for idx, info in enumerate(frame_infos):

            ego2global = info['ego2global']

            for cam_name in info['cams']:
                cam_info = info['cams'][cam_name]

                data_path = os.path.join(NUPLAN_SENSOR_ROOT, cam_info['data_path'])

                sem_mask_path = os.path.join(
                    video_scene.raw_mask_path,
                    video_scene.mask_suffix_cityscape,
                    cam_info['data_path'].replace('.jpg', '.png')
                )
                ego_mask_path = os.path.join(
                    video_scene.raw_mask_path,
                    video_scene.mask_suffix_ego,
                    f'{cam_name}.png'
                )
                if _prepared.get((info['token'], cam_name), ('invalid', None))[0] != 'ok':
                    invalid_images += 1
                    cam_info['valid'] = False
                    continue
                cam_info['valid'] = True

                intrinsic = cam_info['cam_intrinsic']
                cam2ego = Quaternion(cam_info['sensor2ego_rotation']).transformation_matrix
                cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                cam2global = np.dot(ego2global, cam2ego)
                cam2global[:3, 3] *= scale_factor  # scale to about 10 meters for colmap

                # global2cam
                Q = Quaternion(matrix=cam2global[:3, :3].T).q
                T = -cam2global[:3, :3].T @ cam2global[:3, 3]

                cam_name_colmap = f'{veh_name}_{cam_name}'
                cam_id = cam_name_mapper[cam_name_colmap]
                image_path = _colmap_image_name(veh_name, cam_name, info, cam_info)
                name_to_token[image_path] = cam_info['token']

                # deduplicate images
                if image_path in image_names:
                    invalid_images += 1
                    cam_info['valid'] = False
                    continue
                image_names.add(image_path)

                view_field = field_of_view_intrinsic(intrinsic, 30 * scale_factor, cam2global)
                cam_views.append(dict(
                    token=image_path,
                    ego_token=info['token'],
                    view_field=view_field
                ))

                # write to images.txt
                line = f'{global_image_id} {Q[0]} {Q[1]} {Q[2]} {Q[3]} {T[0]} {T[1]} {T[2]} {cam_id} {image_path}' + '\n\n'
                f_images.write(line)

                global_image_id += 1

            progress_bar.update(1)
    progress_bar.close()
    f_images.close()

    print(f'Filter out invalid images: {invalid_images}/{len(frame_infos) * 8}')
    import json as _json
    with open(f'{colmap_path}/name_to_token.json', 'w') as _f:
        _json.dump(name_to_token, _f)
    if _sfm_rig():
        _write_rig_config(video_scene_dict, colmap_path)

    # compute image pairs
    #
    # Any frustum overlap at all makes a pair, which is fine for one traversal but
    # grows with the square of the image count once several traversals of the same
    # road go into one joint model: je_loc_0001 produced 7.5M pairs from 8679
    # images (866 per image) and could not finish matching inside a day, while the
    # blocks that did finish sat at 235-362 per image. Keeping each image's
    # strongest overlaps bounds the total without losing the geometry that matters,
    # and cross-traversal pairs -- the entire point of a joint model -- are kept
    # first, since those are what tie the traversals to each other.
    max_per_image = int(os.environ.get('MTGS_MAX_PAIRS_PER_IMAGE', '0'))

    def _travel_of(token):
        return token.split('/')[0] if '/' in token else token

    candidates = defaultdict(list)          # image -> [(iou, other_index)]
    for i, view1 in enumerate(tqdm(cam_views, desc='Computing camera pairs...', ncols=120)):
        for j in range(i + 1, len(cam_views)):
            view2 = cam_views[j]
            v1 = view1['view_field']
            v2 = view2['view_field']
            iou = v1.intersection(v2).area / v1.union(v2).area
            if iou > 0.0:
                candidates[i].append((iou, j))
                candidates[j].append((iou, i))

    pairs = set()
    if max_per_image <= 0:
        for i, lst in candidates.items():
            for _iou, j in lst:
                a, b = sorted((cam_views[i]['token'], cam_views[j]['token']))
                pairs.add((a, b))
    else:
        for i, lst in candidates.items():
            own = _travel_of(cam_views[i]['token'])
            cross = sorted((c for c in lst if _travel_of(cam_views[c[1]]['token']) != own),
                           key=lambda c: -c[0])
            same = sorted((c for c in lst if _travel_of(cam_views[c[1]]['token']) == own),
                          key=lambda c: -c[0])
            # give cross-traversal links the larger share, then fill with the rest
            keep = cross[:max(1, max_per_image // 2)]
            keep += same[:max_per_image - len(keep)]
            if len(keep) < max_per_image:
                keep += cross[len(keep):max_per_image - len(keep) + len(keep)]
            for _iou, j in keep:
                a, b = sorted((cam_views[i]['token'], cam_views[j]['token']))
                pairs.add((a, b))
        CONSOLE.log(f'[pairs] capped at {max_per_image} per image -> {len(pairs)} pairs '
                    f'for {len(cam_views)} images')

    f_pairs = open(f'{colmap_path}/image_pairs.txt', 'w')
    for pair in pairs:
        f_pairs.write(f'{pair[0]} {pair[1]}\n')
    f_pairs.close()

    # create colmap database after everything is done
    create_colmap_database(colmap_path)
    return video_scene_dict

def create_sparse_colmap_model(video_scene: VideoScene, video_scene_dict, colmap_path):

    create_colmap_folder(colmap_path)
    cam_name_mapper = create_cameras_per_vehicle(video_scene_dict, colmap_path)

    progress_bar = tqdm(
        total=VideoScene.count_total_frames(video_scene_dict),
        desc='Loading image infos', ncols=120)

    f_images = open(f"{colmap_path}/sparse_model/images.txt", "w")
    global_image_id = 1
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]

        # same vehicle has same camera calibration
        veh_name = video_scene_dict[video_token]['vehicle_name']

        for idx, info in enumerate(frame_infos):

            ego2global = info['ego2global']

            for cam in info['cams']:
                cam_info = info['cams'][cam]

                data_path = os.path.join(NUPLAN_SENSOR_ROOT, cam_info['data_path'])

                cam2ego = Quaternion(cam_info['sensor2ego_rotation']).transformation_matrix
                cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                cam2global = np.dot(ego2global, cam2ego)

                # global2cam
                Q = Quaternion(matrix=cam2global[:3, :3].T).q
                T = -cam2global[:3, :3].T @ cam2global[:3, 3]

                cam_name_colmap = f'{veh_name}_{cam}'
                cam_id = cam_name_mapper[cam_name_colmap]
                image_path = f'{cam_name_colmap}/{os.path.basename(data_path)}'

                # write to images.txt
                line = f'{global_image_id} {Q[0]} {Q[1]} {Q[2]} {Q[3]} {T[0]} {T[1]} {T[2]} {cam_id} {image_path}' + '\n\n'
                f_images.write(line)
                global_image_id += 1

            progress_bar.update(1)
    progress_bar.close()
    f_images.close()
    # create colmap database after everything is done
    create_colmap_database(colmap_path)
    return video_scene_dict


def align_model(video_scene: VideoScene, video_scene_dict, colmap_path, thresh=1.0):
    cameras, images, points3D = read_model(f'{colmap_path}/sfm_model/0', ext='.bin')

    # read points3D
    points3D_new = {}
    for key, val in points3D.items():
        if val.image_ids.shape[0] < 3:
            continue
        points3D_new[key] = val
    print('[Read Colmap] filter {}/{} points3D, min view = {}'.format(len(points3D_new), len(points3D), 3))

    # read camera
    camera_params = {}
    for key in cameras.keys():
        # print(cameras[key])
        p = cameras[key].params
        if cameras[key].model == 'PINHOLE':
            fx, fy, cx, cy = p
            K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
            dist = None
        elif cameras[key].model == 'OPENCV':
            K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
            dist = np.array([p[4], p[5], p[6], p[7], 0.])
        else:
            raise NotImplementedError
        camera_params[cameras[key].id] = (K, dist)

    # read image
    images_out = {}
    _nm = f'{colmap_path}/name_to_token.json'
    import json as _json
    _name_map = _json.load(open(_nm)) if os.path.exists(_nm) else {}
    for img in images.values():
        intrinsic, distortion = camera_params[img.camera_id]
        R = Quaternion(img.qvec)
        t = img.tvec

        c2w = np.eye(4)
        c2w[:3, :3] = R.rotation_matrix.T
        c2w[:3, 3] = -R.rotation_matrix.T @ t

        token = _name_map.get(img.name) or img.name.split('.')[0].split('/')[1]
        cam_info = {}
        cam_info['sensor2global_rotation'] = Quaternion(matrix=c2w[:3, :3])
        cam_info['sensor2global_translation'] = c2w[:3, 3]
        cam_info['cam_intrinsic'] = intrinsic
        cam_info['distortion'] = distortion
        cam_info['data_path'] = img.name
        images_out[token] = cam_info

    raw_cam_T = {}
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]
        for info in frame_infos:
            ego2global = info['ego2global']

            for cam in info['cams']:
                cam_info = info['cams'][cam]

                cam2ego = np.eye(4)
                cam2ego[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                cam2global = np.dot(ego2global, cam2ego)
                token = cam_info['token']

                if token not in images_out:
                    # Joint BA registers only images with enough triangulated support
                    # (2783/4560 on gapC_008); invalidating the rest throws away 39% of
                    # the training data, which cost more than the alignment gained.
                    # Keep them on their INS poses instead -- the BA frame is anchored
                    # to INS by the Sim3 (residual ~0.11m), so mixing is benign.
                    if os.environ.get('MTGS_JOINT_KEEP_UNREGISTERED', '1') != '1':
                        cam_info['valid'] = False
                    continue

                raw_cam_T[token] = cam2global[:3, 3]

    colmap_cam_array = []
    raw_cam_array = []
    for cam_token in images_out.keys():
        colmap_cam_array.append(images_out[cam_token]['sensor2global_translation'])
        raw_cam_array.append(raw_cam_T[cam_token])
    colmap_cam_array = np.asarray(colmap_cam_array)
    raw_cam_array = np.asarray(raw_cam_array)

    scale, R, t = compute_transformation_matrix_with_scaling(colmap_cam_array, raw_cam_array)

    images_aligned = {}
    for token, cam_info in images_out.items():
        cam_info = copy.deepcopy(cam_info)
        sensor2global_rotation = R @ Quaternion(cam_info['sensor2global_rotation']).rotation_matrix
        cam_info['sensor2global_rotation'] = Quaternion(matrix=sensor2global_rotation)
        cam_info['sensor2global_translation'] = scale * (R @ cam_info['sensor2global_translation']).T + t
        images_aligned[token] = cam_info

    pts3D_xyz = np.array([p.xyz for p in points3D_new.values()], dtype=np.float32)
    pts3D_xyz = scale * (pts3D_xyz @ R.T) + t
    pts3D_rgb = np.array([p.rgb for p in points3D_new.values()], dtype=np.float32)
    pts3D_rgb /= 255.0

    import open3d as o3d
    o3d_points3D = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3D_xyz))
    o3d_points3D.colors = o3d.utility.Vector3dVector(pts3D_rgb)
    os.makedirs(video_scene.sfm_point_cloud_path, exist_ok=True)
    o3d.io.write_point_cloud(f'{video_scene.sfm_point_cloud_path}/{os.path.basename(colmap_path)}.pcd', o3d_points3D)

    invalid_count = 0
    errors = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        frame_infos = [info for info in frame_infos if not info.get('skipped', False)]
        for info in frame_infos:
            ego2global = info['ego2global']
            for cam in info['cams']:
                cam_info = info['cams'][cam]
                token = cam_info['token']
                if cam_info.get('valid', True):
                    # colmap could not register this image at all; drop it the same
                    # way as one whose pose disagrees with INS by more than thresh
                    if token not in images_aligned:
                        cam_info['valid'] = False
                        invalid_count += 1
                        continue
                    colmap_cam_info = images_aligned[token]
                    colmap_cam2global_T = colmap_cam_info['sensor2global_translation']

                    cam2ego = np.eye(4)
                    cam2ego[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                    cam2ego[:3, 3] = cam_info['sensor2ego_translation']
                    cam2global = np.dot(ego2global, cam2ego)
                    cam2global_T = cam2global[:3, 3]

                    error = np.linalg.norm(colmap_cam2global_T - cam2global_T)
                    errors.append(error)
                    if error > thresh:
                        cam_info['valid'] = False
                        invalid_count += 1
                        continue
                    else:
                        cam_info['valid'] = True
                        cam_info['colmap_param'] = colmap_cam_info

    if os.environ.get('MTGS_JOINT_WRITE_EGO') == '1':
        from scipy.spatial.transform import Rotation as _Rot
        n_upd = 0
        for video_token in video_scene_dict:
            for info in video_scene_dict[video_token]['frame_infos']:
                if info.get('skipped', False):
                    continue
                egos = []
                for cam in info['cams']:
                    cam_info = info['cams'][cam]
                    if not cam_info.get('valid', True) or 'colmap_param' not in cam_info:
                        continue
                    cp = cam_info['colmap_param']
                    c2g = np.eye(4)
                    c2g[:3, :3] = Quaternion(cp['sensor2global_rotation']).rotation_matrix
                    c2g[:3, 3] = cp['sensor2global_translation']
                    c2e = np.eye(4)
                    c2e[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
                    c2e[:3, 3] = cam_info['sensor2ego_translation']
                    egos.append(c2g @ np.linalg.inv(c2e))
                if not egos:
                    continue
                R_mean = _Rot.from_matrix(np.stack([e[:3, :3] for e in egos])).mean().as_matrix()
                t_mean = np.mean([e[:3, 3] for e in egos], axis=0)
                pose = np.eye(4); pose[:3, :3] = R_mean; pose[:3, 3] = t_mean
                if 'ego2global_original' not in info:
                    info['ego2global_original'] = info['ego2global'].copy()
                info['ego2global'] = pose
                info['ego2global_translation'] = pose[:3, 3]
                info['ego2global_rotation'] = Quaternion(matrix=pose[:3, :3], atol=1e-6)
                info['lidar2global'] = pose @ info['lidar2ego']
                n_upd += 1
        CONSOLE.log(f'[Align Colmap] wrote BA ego poses for {n_upd} frames')
    CONSOLE.log(f'[Align Colmap] for {os.path.basename(colmap_path)}')
    CONSOLE.log(f'[Align Colmap] invalid count = {invalid_count}, total = {len(images_out)}')
    CONSOLE.log(f'[Align Colmap] error = {np.mean(errors):.3f}')
    return video_scene_dict

def split_threads(weights, total_threads, min_threads=2):
    """Apportion total_threads across concurrently-running tasks, in proportion to weights.

    Every task keeps at least min_threads; the remainder is handed out by largest
    remainder, so the counts sum to exactly total_threads -- no core is left idle
    and none is oversubscribed.
    """
    n = len(weights)
    total_threads = max(total_threads, n)  # a concurrent task is useless with 0 threads
    total_w = sum(weights)
    if total_w <= 0:  # no size information -- fall back to an even split
        weights = [1] * n
        total_w = n
    # min_threads is what each task holds back before the rest is shared out; drop
    # it if the allocation is too small to give everyone that much.
    min_threads = min(min_threads, total_threads // n)
    spare = total_threads - n * min_threads
    exact = [w / total_w * spare for w in weights]
    counts = [min_threads + int(e) for e in exact]
    fractions = sorted(range(n), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    for i in fractions[:total_threads - sum(counts)]:
        counts[i] += 1
    return counts

def _stage_log(colmap_path, name):
    """Where to write a colmap stage log.

    When the model is built on scratch (/dev/shm) the whole tree is removed on
    exit -- including on a timeout or scancel, which is precisely when the log is
    wanted. MTGS_COLMAP_LOGDIR keeps it on storage that outlives the job.
    """
    logdir = os.environ.get('MTGS_COLMAP_LOGDIR')
    if not logdir:
        return f'{colmap_path}/{name}'
    os.makedirs(logdir, exist_ok=True)
    return os.path.join(logdir, f'{os.path.basename(colmap_path)}_{name}')


def run_colmap_ba(colmap_path, num_workers=8):
    CONSOLE.log(f'Colmap BA for {os.path.basename(colmap_path)} started')
    log_file = _stage_log(colmap_path, 'colmap_ba.log')

    if os.path.exists(log_file):
        os.remove(log_file)

    command = f'python -m nuplan_scripts.utils.colmap_utils.bundle_adjustment --colmap_path {colmap_path} --num_workers {num_workers}'
    with open(log_file, 'w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, shell=True)
        process.communicate()
        if process.returncode != 0:
            CONSOLE.log(f'[ERROR] Colmap Triangulation for {os.path.basename(colmap_path)} failed')
            return False

    CONSOLE.log(f'Colmap BA for {os.path.basename(colmap_path)} done')
    return True

def run_colmap_triangulation(colmap_path, num_workers=8):
    CONSOLE.log(f'Colmap Triangulation for {os.path.basename(colmap_path)} started')
    log_file = _stage_log(colmap_path, 'colmap_triangulation.log')

    if os.path.exists(log_file):
        os.remove(log_file)

    command = f'python -m nuplan_scripts.utils.colmap_utils.point_triangulator --colmap_path {colmap_path} --num_workers {num_workers}'
    with open(log_file, 'w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, shell=True)
        process.communicate()
        if process.returncode != 0:
            CONSOLE.log(f'[ERROR] Colmap Triangulation for {os.path.basename(colmap_path)} failed')
            return False

    CONSOLE.log(f'Colmap Triangulation for {os.path.basename(colmap_path)} done')
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--skip_create_model', action='store_true')
    parser.add_argument('--skip_colmap_ba', action='store_true')
    args = parser.parse_args()

    skip_create_model = args.skip_create_model
    skip_colmap_ba = args.skip_colmap_ba

    # 1. load config and video scene dict
    config: RoadBlockConfig = load_config(args.config)
    video_scene = VideoScene(config)
    all_video_scene_dict = video_scene.load_pickle(video_scene.pickle_path)

    # copy ego mask to sub data root
    copy_ego_masks(video_scene)

    colmap_paths = []
    colmap_publish = []
    single_video_scene_dicts = []
    # Joint mode: one colmap model over every traversal, so bundle adjustment enforces
    # cross-traversal consistency photometrically (pairs are pose-proximity based over
    # the whole dict, so cross-video matches come for free). Per-video mode ties each
    # traversal only to INS (~0.2m / 0.3 deg), which is exactly the softness budget.
    if os.environ.get('MTGS_JOINT_BA') == '1':
        colmap_path = os.path.join(video_scene.colmap_path, 'joint')
        colmap_path, colmap_nfs_path = _colmap_workdir(colmap_path)
        colmap_publish.append((colmap_path, colmap_nfs_path))
        # a finished model from an earlier attempt makes both expensive steps moot
        if _restore_published(colmap_path, colmap_nfs_path):
            skip_create_model = True
            skip_colmap_ba = True
            CONSOLE.log('[colmap] reusing the archived joint model; '
                        'skipping model build and bundle adjustment')
        if not skip_create_model:
            create_colmap_model(video_scene, all_video_scene_dict, colmap_path)
        else:
            CONSOLE.log('Skipping create colmap model for joint')
        colmap_paths.append(colmap_path)
        single_video_scene_dicts.append(all_video_scene_dict)
    # 2. process each video scene
    for video_token in (all_video_scene_dict if os.environ.get('MTGS_JOINT_BA') != '1' else ()):
        video_idx = int(video_token.split('-')[-1])
        video_scene_dict = video_scene.video_scene_dict_process([
            {'type': 'filter_by_video_idx', 'kwargs': {'video_idxs': (video_idx,)}}
        ])

        colmap_path = os.path.join(video_scene.colmap_path, video_token)
        colmap_path, colmap_nfs_path = _colmap_workdir(colmap_path)
        colmap_publish.append((colmap_path, colmap_nfs_path))
        if not skip_create_model:
            create_colmap_model(video_scene, video_scene_dict, colmap_path)
        else:
            CONSOLE.log(f'Skipping craete colmap model for {video_token}')
        colmap_paths.append(colmap_path)
        single_video_scene_dicts.append(video_scene_dict)

    # 3. run colmap for each video scene
    if not skip_colmap_ba:
        num_tasks = len(colmap_paths)
        # Respect the SLURM allocation: cpu_count() reports the NODE's cores,
        # oversubscribing the cgroup by an order of magnitude on shared nodes.
        # MTGS_CPU_BUDGET caps the budget on shared boxes (e.g. manco, where trainers
        # may run alongside and their loop is CPU-bound); unset = take every core.
        total_threads = (int(os.environ.get('MTGS_CPU_BUDGET', 0))
                         or int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
                         or multiprocessing.cpu_count())
        # MTGS_BA_THREADS_PER_TASK: threads a concurrent traversal is sized for (4 on
        # ARC's CPU nodes; 1 on manco, where SIFT extraction/matching run on the GPU
        # and the CPU share per task matters much less -- every traversal then runs
        # at once).
        per_task = max(1, int(os.environ.get('MTGS_BA_THREADS_PER_TASK', 4)))
        pool_size = min(num_tasks, max(1, total_threads // per_task))

        # Traversals within one block differ in length by up to 7x, and colmap cost
        # is close to linear in image count. Splitting the cores evenly therefore
        # makes the LONGEST traversal the straggler: it gets 1/n of the allocation
        # to do most of the work, and every core freed by a short traversal then
        # sits idle until it finishes. That is how loc_0059 (6 videos, 216..1488
        # images) TIMEOUTed at 2h: 5 of 6 were done by 82 min, the 1488-image one
        # was still going at 120 with 26 of its 32 cores idle. Weighting each
        # traversal's threads by its image count instead makes concurrent
        # traversals land together, which is what actually bounds the stage.
        # Frame count is the weight (images = frames x cams, cams is per-rig
        # constant within a block), and it is available whether or not the colmap
        # models were just built.
        weights = [VideoScene.count_total_frames(vsd) for vsd in single_video_scene_dicts]
        # Longest-processing-time first: when there are more traversals than pool
        # slots, starting the big ones first keeps the tail short.
        order = sorted(range(num_tasks), key=lambda i: -weights[i])
        ordered_paths = [colmap_paths[i] for i in order]

        if num_tasks <= pool_size:
            # Every traversal runs concurrently, so give each a share of the cores
            # proportional to its work.
            thread_counts = split_threads([weights[i] for i in order], total_threads,
                                          min_threads=min(2, per_task))
        else:
            # Slots get reused, so each slot takes an equal, fully-used share --
            # an even split would silently drop total_threads % pool_size cores.
            base, remainder = divmod(total_threads, pool_size)
            slots = [base + (1 if i < remainder else 0) for i in range(pool_size)]
            thread_counts = [slots[i % pool_size] for i in range(num_tasks)]

        CONSOLE.log(f'Using threads {thread_counts} for videos '
                    f'{[os.path.basename(p) for p in ordered_paths]} with pool size {pool_size} '
                    f'(weights {[weights[i] for i in order]}, '
                    f'{sum(thread_counts[:pool_size])}/{total_threads} cores in use)')

        run_colmap = run_colmap_ba if config.use_colmap_ba else run_colmap_triangulation

        with multiprocessing.Pool(processes=pool_size) as pool:
            return_codes = pool.starmap(run_colmap, list(zip(ordered_paths, thread_counts)))

        if not all(return_codes):
            for colmap_path, return_code in zip(ordered_paths, return_codes):
                if not return_code:
                    CONSOLE.log(f'[ERROR] Colmap failed for {os.path.basename(colmap_path)}')
            exit(1)

        CONSOLE.log('All colmap done')
    else:
        CONSOLE.log('Skipping colmap')

    # 4. merge all the models
    video_scene_dict_new = {}
    for video_scene_dict, colmap_path in zip(single_video_scene_dicts, colmap_paths):
        video_scene_dict = align_model(video_scene, video_scene_dict, colmap_path)
        video_scene_dict_new.update(video_scene_dict)

    # Exporting a combined sparse model is a debugging aid, but it rebuilds the
    # whole image/mask tree a second time -- thousands of files over NFS after the
    # real work is already done and archived. Skip it unless asked for.
    if os.environ.get('MTGS_SKIP_SPARSE_ALL', '1') != '1':
        create_sparse_colmap_model(video_scene, video_scene_dict_new,
                                   os.path.join(video_scene.colmap_path, 'all'))

    for _work, _nfs in colmap_publish:
        if os.environ.get('MTGS_KEEP_COLMAP') == '1':
            _publish_colmap(_work, _nfs)
        if _work != _nfs:
            shutil.rmtree(_work, ignore_errors=True)   # scratch is per-job, never keep it
    if os.environ.get('MTGS_KEEP_COLMAP') != '1':
        for colmap_path in colmap_paths:
            clean_colmap_folder(colmap_path)

    video_scene.video_scene_dict = video_scene_dict_new
    video_scene.dump_pickle(video_scene.pickle_path_colmap)
    video_scene.update_pickle_link(video_scene.pickle_path_colmap)
