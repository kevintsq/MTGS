"""Ego poses for the frames a pose solution did not touch.

A joint SfM solution writes its ego pose only to the frames that took part in it; the frames skipped there (standing
ego: 'low_velocity', no SfM camera: 'sfm_unregistered', 'out_of_region') kept the pose of the stage before, which sits
0.3-1.6 m and up to 3 deg away from the SfM frame the block is trained in (pn_w2_08 pass 18: 188 of 326 frames, a
1.3 m / 2.8 deg step at every stop of the ego). The video_scene_dict then held two frames of reference in one pass:
replayed traffic and the camera jumped whenever the ego stopped, and renders of standing frames were scored from the
wrong place (2026-11-11).
"""
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation, Slerp


def carry_ego_correction(frame_infos, is_ref, is_target):
    """Give every target frame the pose that continues the reference frames next to it in time.

    A reference frame r holds the solved pose E_r and the pose before it, O_r ('ego2global_original'). A target frame
    s is placed by dead reckoning from its neighbours with the original relative motion, E_r @ inv(O_r) @ O_s, blended
    between the reference frame before and after it by time (the nearer one counts more; only one side: that one).
    For a standing ego O_s = O_r, so the frame simply takes the solved pose of the moment it stopped.

    frame_infos: the frames of ONE pass (any order); is_ref / is_target: predicates on a frame info.
    Rewrites ego2global, ego2global_translation, ego2global_rotation, lidar2global of the targets in place, marks
    them with 'ego2global_carried' and returns how many were rewritten."""
    fis = sorted(frame_infos, key=lambda f: int(f['timestamp']))
    ref = [i for i, f in enumerate(fis) if is_ref(f) and 'ego2global_original' in f]
    if not ref:
        return 0
    corr = {i: np.asarray(fis[i]['ego2global'], dtype=np.float64) @ np.linalg.inv(np.asarray(fis[i]['ego2global_original'], dtype=np.float64))
            for i in ref}
    ref_idx = np.asarray(ref)
    ts = np.array([int(f['timestamp']) for f in fis], dtype=np.float64)
    n = 0
    for i, f in enumerate(fis):
        if i in corr or not is_target(f):
            continue
        if 'ego2global_original' not in f:
            f['ego2global_original'] = np.array(f['ego2global'], dtype=np.float64)
        orig = np.asarray(f['ego2global_original'], dtype=np.float64)
        j = int(np.searchsorted(ref_idx, i))
        before = ref[j - 1] if j > 0 else None
        after = ref[j] if j < len(ref) else None
        if before is None or after is None:
            pose = corr[after if before is None else before] @ orig
        else:
            w = (ts[i] - ts[before]) / max(ts[after] - ts[before], 1.0)
            pa, pb = corr[before] @ orig, corr[after] @ orig
            pose = np.eye(4)
            pose[:3, 3] = (1.0 - w) * pa[:3, 3] + w * pb[:3, 3]
            pose[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([pa[:3, :3], pb[:3, :3]])))(w).as_matrix()
        f['ego2global'] = pose
        f['ego2global_translation'] = pose[:3, 3]
        f['ego2global_rotation'] = Quaternion(matrix=pose[:3, :3], atol=1e-6)
        if 'lidar2ego' in f:
            f['lidar2global'] = pose @ np.asarray(f['lidar2ego'], dtype=np.float64)
        f['ego2global_carried'] = True
        n += 1
    return n
