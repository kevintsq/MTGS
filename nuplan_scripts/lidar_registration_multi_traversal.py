#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
from nuplan_scripts.utils.console import say
from nuplan_scripts.utils.traj_step_qa import check_trajectory_steps, check_unrefined_passes
import os
import sys
import copy
import argparse

from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D

from nuplan_scripts.utils.config import load_config, RoadBlockConfig, FrameCentralConfig
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene
from nuplan_scripts.utils.constants import NUPLAN_MAP_VERSION, NUPLAN_MAPS_ROOT, CONSOLE


def _se2_constrain(pose_icp, pose_ins):
    from scipy.spatial.transform import Rotation as _R
    yaw = _R.from_matrix(pose_icp[:3, :3]).as_euler('zyx')[0]
    _, pitch, roll = _R.from_matrix(pose_ins[:3, :3]).as_euler('zyx')
    out = np.eye(4)
    out[:3, :3] = _R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    out[:3, 3] = [pose_icp[0, 3], pose_icp[1, 3], pose_ins[2, 3]]
    return out


def _relative_tilt_offsets(video_scene_dict, result_poses):
    """Per-traversal constant (pitch, roll, z) offset of ICP against INS, minus the
    part shared by all traversals.

    INS attitude differs between drives of the same road by up to ~1 deg (vehicle
    calibration), and that disagreement has to be removed or the joint model blurs.
    ICP measures it, but on 2 Hz lidar it also drifts along the route, and that drift
    is common to every traversal registered onto the same map (gapC_008: 0 -> -4.3 deg).
    A per-traversal constant keeps the first and cannot carry the second: anything
    shared is averaged out, and nothing varies along the route.
    """
    from scipy.spatial.transform import Rotation as _R
    med = {}
    for vt, v in video_scene_dict.items():
        d = []
        for info in v['frame_infos']:
            P = result_poses[info['token']]; I = np.asarray(info['ego2global'], float)
            _, pp, rp = _R.from_matrix(P[:3, :3]).as_euler('zyx')
            _, pi, ri = _R.from_matrix(I[:3, :3]).as_euler('zyx')
            d.append((pp - pi, rp - ri, P[2, 3] - I[2, 3]))
        med[vt] = np.median(np.asarray(d), axis=0)
    common = np.mean(np.stack(list(med.values())), axis=0)
    rel = {vt: m - common for vt, m in med.items()}
    for vt, r in rel.items():
        say(f"[level] {vt.split('-')[-1]}: pitch {np.degrees(r[0]):+.2f} deg, "
                    f"roll {np.degrees(r[1]):+.2f} deg, z {r[2]:+.2f} m "
                    f"(common removed: pitch {np.degrees(common[0]):+.2f} deg, z {common[2]:+.2f} m)")
    return rel


def _apply_relative_tilt(pose_icp, pose_ins, rel):
    """x, y, yaw from ICP; pitch, roll, z from INS plus this traversal's constant."""
    from scipy.spatial.transform import Rotation as _R
    yaw = _R.from_matrix(pose_icp[:3, :3]).as_euler('zyx')[0]
    _, pitch, roll = _R.from_matrix(pose_ins[:3, :3]).as_euler('zyx')
    out = np.eye(4)
    out[:3, :3] = _R.from_euler('zyx', [yaw, pitch + rel[0], roll + rel[1]]).as_matrix()
    out[:3, 3] = [pose_icp[0, 3], pose_icp[1, 3], pose_ins[2, 3] + rel[2]]
    return out


def _common_drift(video_scene_dict, result_poses, radius=20.0, min_neighbours=12):
    """Per frame: the (pitch, roll, z) disagreement between ICP and INS that is shared
    by every traversal at that place.

    INS errors of different drives are independent, so what all traversals agree on
    at one spot is the registration bending the shared map (gapC_008: ~5 deg over the
    route), while what one traversal does differently is ICP correcting that drive's
    INS -- which the constant-offset model showed is real and worth keeping (dropping
    it cost 1.5-3 dB and squashed fy). Estimate the shared part as the median over
    all frames, of all traversals, within `radius` metres.
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as _R
    tokens, xy, delta = [], [], []
    for v in video_scene_dict.values():
        for info in v['frame_infos']:
            P = result_poses[info['token']]; I = np.asarray(info['ego2global'], float)
            _, pp, rp = _R.from_matrix(P[:3, :3]).as_euler('zyx')
            _, pi, ri = _R.from_matrix(I[:3, :3]).as_euler('zyx')
            tokens.append(info['token']); xy.append(P[:2, 3])
            delta.append((pp - pi, rp - ri, P[2, 3] - I[2, 3]))
    xy = np.asarray(xy); delta = np.asarray(delta)
    tree = cKDTree(xy)
    common = {}
    for k, tok in enumerate(tokens):
        idx = tree.query_ball_point(xy[k], radius)
        if len(idx) < min_neighbours:
            idx = tree.query(xy[k], k=min(min_neighbours, len(tokens)))[1]
        common[tok] = np.median(delta[np.atleast_1d(idx)], axis=0)
    c = np.asarray(list(common.values()))
    say(f"[level2] common drift over the block: pitch {np.degrees(c[:,0].min()):+.2f}..{np.degrees(c[:,0].max()):+.2f} deg, "
                f"roll {np.degrees(c[:,1].min()):+.2f}..{np.degrees(c[:,1].max()):+.2f} deg, z {c[:,2].min():+.2f}..{c[:,2].max():+.2f} m")
    return common


def _remove_common_drift(pose_icp, common):
    """ICP pose with the shared pitch/roll/z drift taken out; x, y, yaw untouched."""
    from scipy.spatial.transform import Rotation as _R
    yaw, pitch, roll = _R.from_matrix(pose_icp[:3, :3]).as_euler('zyx')
    out = np.eye(4)
    out[:3, :3] = _R.from_euler('zyx', [yaw, pitch - common[0], roll - common[1]]).as_matrix()
    out[:3, 3] = [pose_icp[0, 3], pose_icp[1, 3], pose_icp[2, 3] - common[2]]
    return out


def calculate_errors(video_scene_dict: dict, result_poses_dict: dict):
    error_dict = {}
    total_errors = []
    total_heading_errors = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        errors = []
        heading_errors = []
        for idx, info in enumerate(frame_infos):
            token = info['token']
            pose_icp = result_poses_dict[token]

            T = pose_icp[:2, 3]
            T_orginal = info['ego2global'][:2, 3]
            error = np.linalg.norm(T - T_orginal)
            errors.append(error)

            heading_icp = Quaternion(matrix=pose_icp[:3, :3]).yaw_pitch_roll[0]
            heading_orginal = Quaternion(matrix=info['ego2global'][:3, :3]).yaw_pitch_roll[0]
            # wrap to (-pi, pi]: a raw abs() difference across the +/-pi seam
            # reports ~360 deg for two nearly identical headings.
            dh = (heading_icp - heading_orginal + np.pi) % (2 * np.pi) - np.pi
            heading_error = np.rad2deg(np.abs(dh))
            heading_errors.append(heading_error)

        total_errors.extend(errors)
        total_heading_errors.extend(heading_errors)
        if not errors:
            # degenerate video (no frames survived prefilter): flag as
            # unregistrable so the QA pass drops it instead of crashing
            error_dict[video_token] = {
                'max_end_point_error': float('inf'),
                'mean_error': float('inf'),
                'mean_heading_error': float('inf'),
            }
            continue
        max_end_point_error = max(errors[0], errors[-1])
        error_dict[video_token] = {
            'max_end_point_error': max_end_point_error,
            'mean_error': np.mean(errors),
            'mean_heading_error': np.mean(heading_errors)
        }

    table = PrettyTable(
        field_names = ["Video idx", "EPE", "ATE-BEV", "ARE-BEV"]
    )
    table.float_format = ".4"
    table.align = "r"
    for video_token, errors in error_dict.items():
        table.add_row([int(video_token.split('-')[-1]), errors['max_end_point_error'], errors['mean_error'], errors['mean_heading_error']])
    table.add_row(["Total", max(total_errors), np.mean(total_errors), np.mean(total_heading_errors)])
    say(table)
    return error_dict

_LAST_ALIGN_T = np.eye(4)

def _tilt_dev(R, R_ins):
    """World-frame tilt (deg, [x, y]) of the rotation R against the INS rotation R_ins: rotvec(R R_ins^T) without yaw."""
    from scipy.spatial.transform import Rotation as _R
    return _R.from_matrix(R @ R_ins.T).as_rotvec(degrees=True)[:2]


def _retilt(R, d_from, d_to):
    """R with its world-frame tilt moved from d_from to d_to (deg, [x, y]); yaw untouched."""
    from scipy.spatial.transform import Rotation as _R
    dd = np.asarray(d_to, dtype=float) - np.asarray(d_from, dtype=float)
    return _R.from_rotvec([dd[0], dd[1], 0.0], degrees=True).as_matrix() @ R


def _register_frame_gravity(self, frame, timestamps, gt_pose=None, traj_id=None):
    """KissICP.register_frame with the roll / pitch of the joint odometry held to gravity (2026-11-16).

    The stage trace of px_loc_0195 / ru_loc_0183 put the block tilts (up to 7.5 deg) into this step: (1) the first frame
    of every traversal starts at its INS pose and is ICP'd with the warm-up settings (6 m correspondences, 6-DoF) onto the
    shared map -- ru_loc_0183 pass 28 came out 4.7 deg tilted against a level map, px 26 5.4 deg, and the later
    traversals inherited it; (2) 2 Hz scan-to-map odometry drifts in roll / pitch along a pass (px 23: 0.4 -> 3.4 deg,
    26: 5.4 -> 10.7). Nothing after it corrects attitude (x/y alignment, z harmonisation, map ICP onto the majority).
    Here (gt_pose = the INS pose, rotation in the world frame):
    - traversal start: the tilt of the registered pose against the INS is clamped to MTGS_KISS_START_TILT_DEG (1.5 deg:
      the INS attitude of a log is off gravity by up to ~1.5 deg, facade check of INS-placed maps 0.1-1.6), so a real
      per-log offset survives and a snap does not; that clamped offset is the traversal's reference d0 (starting a
      snapped traversal on the INS attitude instead left pw_loc_0195's passes 0.5-0.8 m apart);
    - every other frame: the tilt deviation is pulled towards d0 by a complementary filter with time constant
      MTGS_KISS_GRAVITY_TAU_S (10 s at 2 Hz: ICP's short-term attitude is kept, drift of 0.04-0.07 deg/s is held
      to <= ~0.5 deg). Anchoring to d0 instead of to the INS keeps the log's own constant offset out of the correction
      (levelling each pass to its own INS attitude broke consistent passes apart, pw_loc_0195).
    MTGS_KISS_GRAVITY=0 restores the plain KissICP behaviour."""
    frame = self.compensator.deskew_scan(frame, timestamps, self.last_delta)
    frame = self.preprocess(frame)
    source, frame_downsample = self.voxelize(frame)
    sigma = self.adaptive_threshold.get_threshold()
    start = self.last_traj_id != traj_id
    if start:
        self.warmup = 1
        self.last_pose = gt_pose
    warm = self.warmup >= 0
    if warm:
        self.warmup -= 1
        initial_guess = gt_pose
        sigma = 2
    elif gt_pose is not None and getattr(self, '_last_gt', None) is not None and os.environ.get('MTGS_KISS_INS_PREDICT', '1') == '1':
        # motion prediction from the INS (2026-11-17): KissICP predicts with the last frame-to-frame motion (constant
        # velocity), which at 2 Hz is metres / degrees off whenever the ego brakes, accelerates or turns; the ICP then
        # locks onto a neighbouring minimum and the chain is lost for the rest of the pass (ml_pit01b: 6 of 9 passes,
        # 45-500 m, onsets where the speed changes; identical with plain KissICP). The INS relative motion over one
        # frame is good to centimetres, so the previous registered pose is advanced by it instead.
        initial_guess = self.last_pose @ (np.linalg.inv(self._last_gt) @ gt_pose)
    else:
        initial_guess = self.last_pose @ self.last_delta
    new_pose = self.registration.align_points_to_map(points=source, voxel_map=self.local_map, initial_guess=initial_guess,
                                                     max_correspondance_distance=3 * sigma, kernel=sigma / 3)
    if gt_pose is not None:
        dev = _tilt_dev(new_pose[:3, :3], gt_pose[:3, :3])
        if warm:
            lim = float(os.environ.get('MTGS_KISS_START_TILT_DEG', 1.5)); m = float(np.hypot(*dev))
            d0 = dev if m <= lim else dev * (lim / m)
            if start:
                self._grav_d0 = d0
            d0 = self._grav_d0
            new_pose = new_pose.copy(); new_pose[:3, :3] = _retilt(new_pose[:3, :3], dev, d0)
        else:
            tau = float(os.environ.get('MTGS_KISS_GRAVITY_TAU_S', 10.0)); a = 1.0 - np.exp(-0.5 / max(tau, 1e-3))
            d0 = getattr(self, '_grav_d0', np.zeros(2))
            new_pose = new_pose.copy(); new_pose[:3, :3] = _retilt(new_pose[:3, :3], dev, dev + a * (d0 - dev))
    model_deviation = np.linalg.inv(initial_guess) @ new_pose
    self.adaptive_threshold.update_model_deviation(model_deviation)
    self.local_map.update(frame_downsample, new_pose)
    self.last_delta = np.linalg.inv(self.last_pose) @ new_pose
    self.last_pose = new_pose
    self.last_traj_id = traj_id
    self._last_gt = None if gt_pose is None else np.array(gt_pose, dtype=float)
    return frame, source


def align_poses(video_scene_dict:dict, result_poses_dict: dict):
    original_poses = []
    icp_poses = []
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        for idx, info in enumerate(frame_infos):
            token = info['token']
            pose_icp = result_poses_dict[token]

            T = pose_icp[:2, 3]
            T_orginal = info['ego2global'][:2, 3]
            original_poses.append(T_orginal)
            icp_poses.append(T)
    original_poses = np.array(original_poses)
    icp_poses = np.array(icp_poses)

    # Center the points
    original_centered = original_poses - np.mean(original_poses, axis=0)
    icp_centered = icp_poses - np.mean(icp_poses, axis=0)

    H = icp_centered.T @ original_centered
    U, _, Vh = np.linalg.svd(H)
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.T @ U.T
    # Kabsch translation must be taken AFTER the rotation: t = mu_o - R @ mu_i.
    # Writing `mu_o - mu_i` rotates about the global ORIGIN rather than about the
    # centroid, which leaves a constant offset of (R - I) @ mu_i on every pose.
    # ego2global is UTM-scale (Boston northing ~4.69e6), so even a sub-arcsecond
    # residual rotation turns into a METRE-scale bias -- enough to push otherwise
    # good traversals past the QA gate (loc_0158's one clean traversal went from
    # 1.41/0.70 raw to 2.97/2.91 "aligned", and was then rejected as bad).
    t = np.mean(original_poses, axis=0) - R @ np.mean(icp_poses, axis=0)

    aligned_transform = np.eye(4)
    aligned_transform[:2, :2] = R[:2, :2]
    aligned_transform[:2, 3] = t[:2]
    global _LAST_ALIGN_T
    _LAST_ALIGN_T = aligned_transform   # for MTGS_PIN_KEEP_ICP: carry pre-dropped videos' ICP poses through the same fit

    # Update the result_poses_dict with aligned poses
    result_poses_dict = copy.deepcopy(result_poses_dict)
    for video_token in video_scene_dict:
        frame_infos = video_scene_dict[video_token]['frame_infos']
        for info in frame_infos:
            token = info['token']
            pose_icp = result_poses_dict[token]
            # Update the translation part of the pose
            pose_icp = aligned_transform @ pose_icp
            result_poses_dict[token] = pose_icp

    return result_poses_dict

def visualize_lidar_registration(video_scene: VideoScene, result_poses_dict: dict, filename="registration.pdf"):
    config = video_scene.config
    road_block = config.road_block
    buffer = config.expand_buffer

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.axis('equal')
    ax.set_xlim(road_block[0] - buffer, road_block[2] + buffer)
    ax.set_ylim(road_block[1] - buffer, road_block[3] + buffer)

    # visualize the map
    map_api = get_maps_api(NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, config.city)
    map_objects = map_api.get_proximal_map_objects(
        point=Point2D(x=video_scene.road_block_center[0], y=video_scene.road_block_center[1]), 
        radius=(max(video_scene.block_size) + video_scene.config.expand_buffer) / 2,
        layers=[
            SemanticMapLayer.ROADBLOCK, 
            SemanticMapLayer.INTERSECTION, 
            SemanticMapLayer.CARPARK_AREA
        ]
    )
    for geos in map_objects.values():
        for geo in geos:
            polygon_points = np.asarray(geo.polygon.exterior.coords)
            ax.plot(polygon_points[:, 0], polygon_points[:, 1], 'magenta')

    # visualize ego trajectory
    for video_token in video_scene_dict.keys():
        frame_infos = video_scene_dict[video_token]["frame_infos"]
        # A degenerate traversal (no frames survived the prefilter) has no
        # trajectory to draw; np.array([]) + centre[None] would raise a (0,)
        # vs (1,2) broadcast error and kill the whole stage BEFORE the QA pass
        # gets a chance to drop it. calculate_errors already tolerates this
        # case (flags it inf); the plot must skip it to match.
        if not frame_infos:
            continue
        raw_trajectory = np.array([info['ego2global'][:2, 3] for info in frame_infos]) + video_scene.road_block_center[:2][None]
        icp_trajectory = np.array([result_poses_dict[info['token']][:2, 3] for info in frame_infos]) + video_scene.road_block_center[:2][None]

        ax.plot(raw_trajectory[:, 0], raw_trajectory[:, 1], 'lime', linewidth=2, alpha=0.8)
        ax.plot(icp_trajectory[:, 0], icp_trajectory[:, 1], 'cyan', linewidth=2, alpha=0.8)

    # visualize the road block
    ax.add_patch(
        plt.Rectangle(
            (config.road_block[0] - video_scene.road_block_center[0],
             config.road_block[1] - video_scene.road_block_center[1]),
            config.road_block[2] - config.road_block[0],
            config.road_block[3] - config.road_block[1],
            edgecolor='red', 
            facecolor='none'
        )
    )
    plt.savefig(f"{video_scene.sub_data_root}/map_vis/{filename}", bbox_inches='tight')
    plt.close(fig)


def _veh_yaw_fix_table():
    out = {}
    for item in os.environ.get('MTGS_VEH_YAW_FIX', '').split(','):
        if ':' in item:
            veh, deg = item.split(':', 1)
            out[veh.strip()] = float(deg)
    return out

_VEH_FIX = _veh_yaw_fix_table()
_VEH_FIX_LOGGED = set()

def _apply_vehicle_yaw_fix(ego2global, lidar2ego, cams, log_name):
    """Rotate body and cameras by -delta about the lidar's z, lidar2global unchanged."""
    veh = next((v for v in _VEH_FIX if v in log_name), None)
    if veh is None:
        return ego2global, lidar2ego
    d = np.radians(_VEH_FIX[veh])
    c, s_ = np.cos(d), np.sin(d)
    Rz = np.eye(4); Rz[:2, :2] = [[c, -s_], [s_, c]]           # +delta
    Rzi = np.eye(4); Rzi[:2, :2] = [[c, s_], [-s_, c]]         # -delta
    E = np.asarray(ego2global, float); L = np.asarray(lidar2ego, float)
    E_new = E @ Rzi                      # body heading back onto INS
    L_new = Rz @ L                       # so that E_new @ L_new == E @ L (lidar fixed)
    for cam in cams.values():            # cameras: rotate with the body, i.e. by -delta about lidar z
        if 'sensor2lidar_rotation' in cam:
            Rc = np.asarray(cam['sensor2lidar_rotation'], float); tc = np.asarray(cam['sensor2lidar_translation'], float)
            cam['sensor2lidar_rotation'] = Rzi[:3, :3] @ Rc
            cam['sensor2lidar_translation'] = Rzi[:3, :3] @ tc
    if veh not in _VEH_FIX_LOGGED:
        _VEH_FIX_LOGGED.add(veh)
        say(f"[veh-yaw-fix] {veh}: body and cameras rotated {-_VEH_FIX[veh]:+.2f} deg about lidar z "
                    f"(lidar2global unchanged) for log {log_name}", style="yellow")
    return E_new, L_new

def _aligned_reference(video_scene_dict, result_poses_dict_aligned, exclude):
    """(xy [N,2], z_aligned - z_ins [N], block median) over the ICP-aligned frames.

    align_poses fits the ICP frame onto the INS in x/y only, so a pass put back on
    its GPS pose sits at INS altitude while the aligned passes sit at the ICP
    height: 1-7 m apart (fk_gapC_001 trav 2, fk_loc_0009 trav 259 ...), a second
    road layer in training and a camera under the road in HUGSIM (2026-10-20)."""
    xy, dz = [], []
    for vt, v in video_scene_dict.items():
        if vt in exclude:
            continue
        for info in v['frame_infos']:
            t = info['token']
            if t not in result_poses_dict_aligned:
                continue
            xy.append(result_poses_dict_aligned[t][:2, 3])
            dz.append(result_poses_dict_aligned[t][2, 3] - info['ego2global'][2, 3])
    xy = np.array(xy, dtype=float).reshape(-1, 2); dz = np.array(dz, dtype=float)
    return xy, dz, (float(np.median(dz)) if len(dz) else 0.0)


def _gps_poses_on_aligned_height(video, ref, radius=25.0):
    """{token: INS pose lifted by the local (aligned - INS) height offset}, for every frame of one pass.

    The offset of a frame is the median over the ICP-aligned frames of the other passes within `radius` m. Frames with
    no aligned frame that near take the offset interpolated along their own pass between the nearest frames that have
    one (constant beyond the ends); only a pass with no such frame at all takes the block median. Before 2026-11-16
    every such frame took the block median: ru_area0806 / pn_area0806 pass 357 (GPS-restored) drives a driveway no
    other pass comes within 25 m of, so 47 frames started at the block median (-0.78 m) next to frames at their local
    -3.1 m, a 2.4 m step the per-frame map ICP (1 m gate) could not bridge -> the road trained at several heights
    under them and the HUGSIM camera placed into it (white fog in scene e12142d7)."""
    ref_xy, ref_dz, ref_med = ref
    fis = sorted(video['frame_infos'], key=lambda i: int(i['timestamp']))
    if not fis:
        return {}
    P = np.array([np.asarray(i['ego2global'], dtype=float)[:3, 3] for i in fis])
    off = np.full(len(fis), np.nan)
    if len(ref_xy):
        from scipy.spatial import cKDTree
        for k, nb in enumerate(cKDTree(ref_xy).query_ball_point(P[:, :2], radius)):
            if len(nb) >= 3:
                off[k] = float(np.median(ref_dz[nb]))
    have = np.where(~np.isnan(off))[0]
    if len(have) == 0:
        off[:] = ref_med
    elif len(have) < len(fis):
        # along the driven path (+ a tiny index term: a standing ego repeats its position)
        s_ = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))] + 1e-6 * np.arange(len(fis))
        off = np.interp(s_, s_[have], off[have])
    out = {}
    for k, i in enumerate(fis):
        p = np.array(i['ego2global'], dtype=float).copy()
        p[2, 3] += off[k]
        out[i['token']] = p
    return out


def _z_inconsistent_videos(video_scene_dict, result_poses_dict_aligned, skip, gate, radius=15.0):
    """Videos whose registered height is more than `gate` m off the other videos'
    height along their own trajectory (ICP drift in Z, invisible to the BEV QA)."""
    P = {vt: np.array([result_poses_dict_aligned[i['token']][:3, 3] for i in v['frame_infos']
                       if i['token'] in result_poses_dict_aligned])
         for vt, v in video_scene_dict.items()}
    out = {}
    if len(P) < 3:          # with two passes the height disagreement has no majority
        return out
    allpts = np.concatenate([q for q in P.values() if len(q)])
    for vt, pts in P.items():
        if vt in skip or len(pts) == 0:
            continue
        res = []
        for p in pts:
            # median over every pass nearby, own frames included: with a consistent
            # majority the median sits on the majority height, so the odd pass
            # stands out and the good ones do not (leave-one-out flagged both sides)
            near = np.linalg.norm(allpts[:, :2] - p[:2], axis=1) < radius
            if near.sum() >= 3:
                res.append(p[2] - float(np.median(allpts[near, 2])))
        if len(res) >= 5 and abs(float(np.median(res))) > gate:
            out[vt] = float(np.median(res))
    return out


def _facade_tilt(video, poses, scans_by_token, nmax=80):
    """World tilt (deg, [x, y]) of one pass's lidar map against gravity from its building facades: wall normals
    (|n_z| < 0.3) of a level map are horizontal; a map rotated by the small horizontal rotation w has
    n_z = w_x n_y - w_y n_x (trimmed least squares). Independent of the IMU and of the INS altitude. None without scans."""
    import open3d as o3d
    fis = [i for i in video['frame_infos'] if i['token'] in poses and i['token'] in scans_by_token and not i.get('skipped')]
    fis = fis[::max(1, len(fis) // nmax)][:nmax]
    pts = []
    for i in fis:
        P = scans_by_token[i['token']][:, :3]; P = P[np.linalg.norm(P[:, :2], axis=1) < 40.0][::2]
        T = poses[i['token']] @ np.asarray(i['lidar2ego'], dtype=float)
        pts.append((T[:3, :3] @ P.T).T + T[:3, 3])
    if not pts:
        return None
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts))).voxel_down_sample(0.25)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=30)); n = np.asarray(pc.normals)
    m = np.abs(n[:, 2]) < 0.3
    if m.sum() < 2000:
        return None
    A = np.c_[n[m, 1], -n[m, 0]]; bb = n[m, 2]; keep = np.ones(len(bb), bool)
    for _ in range(3):
        x = np.linalg.lstsq(A[keep], bb[keep], rcond=None)[0]; r = bb - A @ x; keep = np.abs(r) < 2.5 * np.std(r[keep])
    return np.degrees(x)


def _level_passes_to_imu(video_scene_dict, result_poses_dict_aligned, scans_by_token, exclude=()):
    """Remove a rigid world-frame tilt of the registration against gravity, before the passes are harmonised.

    The joint KISS-ICP chain can leave the block, or single passes, rotated about a horizontal axis: px_loc_0195 passes
    26 / 57 / 58 / 61 by ~7.3 deg (2026-11-16) while 23 / 59 were level, so the steps after it took the tilted majority
    as the reference (the harmonisation "corrected" the level passes by -4 m, map ICP then pulled them onto the tilted
    ones); 131 served blocks hold passes > 1.5 deg off (ru_loc_0183: all four ~5 deg). Body-frame roll / pitch offsets
    swing with the heading for such a pass (pass 57: pitch -6.5 -> +1.5 deg through an 87 deg turn) while the
    world-frame tilt w = rotvec(R_reg R_ins^T)[x, y] stays constant.
    Two levels, both rigid rotations about ONE centre (the block's mean position):
    - the block: the median w over all frames, applied to every pass when > MTGS_REG_LEVEL_DEG (1 deg) and confirmed by
      the facades of the passes' lidar maps (_facade_tilt: median over passes within max(1.5 deg, 35 %) of it; the
      facade fit reads low, regression dilution: px 5.5 vs 7.3, ru_fq_loc_0188 1.0 vs 1.9 -> after levelling 0.2-0.4);
    - a pass whose own w departs from the block's by > MTGS_REG_LEVEL_PASS_DEG (3 deg) is levelled by the difference
      when its own facades confirm it. Smaller per-pass differences are left: every log's INS attitude carries its own
      bias of ~1 deg, and levelling each pass by its own w broke mutually consistent passes apart (pw_loc_0195: pass 5
      ended 3 m off the others).
    Returns ([(what, |w| deg, |facade| deg or None, applied, reason)], ...)."""
    from scipy.spatial.transform import Rotation as _R
    gate = float(os.environ.get('MTGS_REG_LEVEL_DEG', 1.0)); pgate = float(os.environ.get('MTGS_REG_LEVEL_PASS_DEG', 3.0))
    out = []
    if gate <= 0:
        return out
    P, Wp = {}, {}
    for vt, v in video_scene_dict.items():
        if vt in exclude:
            continue
        fis = [i for i in v['frame_infos'] if i['token'] in result_poses_dict_aligned]
        if len(fis) < 10:
            continue
        T = np.array([result_poses_dict_aligned[i['token']] for i in fis]); O = np.array([np.asarray(i['ego2global'], dtype=float) for i in fis])
        P[vt] = fis; Wp[vt] = np.array([_R.from_matrix(T[k, :3, :3] @ O[k, :3, :3].T).as_rotvec(degrees=True) for k in range(len(fis))])[:, :2]
    if not P:
        return out
    c = np.array([result_poses_dict_aligned[i['token']][:3, 3] for fis in P.values() for i in fis]).mean(axis=0)

    def _rotate(vts, w):
        Rc = _R.from_rotvec([-w[0], -w[1], 0.0], degrees=True).as_matrix()
        for vt in vts:
            for i in P[vt]:
                M = result_poses_dict_aligned[i['token']].copy()
                M[:3, :3] = Rc @ M[:3, :3]; M[:3, 3] = Rc @ (M[:3, 3] - c) + c
                result_poses_dict_aligned[i['token']] = M

    wb = np.median(np.concatenate(list(Wp.values())), axis=0); mb = float(np.hypot(*wb))
    if mb > gate:
        F = [f for f in (_facade_tilt(video_scene_dict[vt], result_poses_dict_aligned, scans_by_token) for vt in P) if f is not None]
        fb = np.median(np.array(F), axis=0) if F else None
        if fb is not None and np.hypot(*(fb - wb)) <= max(1.5, 0.35 * mb):
            _rotate(list(P), wb)
            for vt in P:
                Wp[vt] = Wp[vt] - wb
            out.append(("block", mb, float(np.hypot(*fb)), True, f"({wb[0]:+.1f}, {wb[1]:+.1f}) deg, facades ({fb[0]:+.1f}, {fb[1]:+.1f})"))
        else:
            out.append(("block", mb, None if fb is None else float(np.hypot(*fb)), False, "facades do not confirm it"))
    for vt in P:
        w = np.median(Wp[vt], axis=0); m = float(np.hypot(*w))
        if m <= pgate:
            continue
        f = _facade_tilt(video_scene_dict[vt], result_poses_dict_aligned, scans_by_token)
        if f is None or np.hypot(*(f - w)) > max(1.5, 0.35 * m):
            out.append((vt, m, None if f is None else float(np.hypot(*f)), False, "its facades do not confirm it")); continue
        _rotate([vt], w)
        out.append((vt, m, float(np.hypot(*f)), True, f"({w[0]:+.1f}, {w[1]:+.1f}) deg off the block, facades ({f[0]:+.1f}, {f[1]:+.1f})"))
    return out


def _harmonize_z(video_scene_dict, result_poses_dict_aligned, radius=15.0, min_nb=3, win=8, gate=0.3):
    """Put every pass on the height of the passes already registered around it.

    KISS-ICP runs all traversals as one sequence; each new traversal starts from
    its own GPS height, and when that is further off than the ICP correspondence
    threshold the flat road gives a stable local minimum one layer up or down
    (nb_05: six passes split into +1.1 m and -1.2 m groups, nb_04: -2.5..+2.8 m
    along one pass). x/y/yaw are fine; only z is wrong, and it can drift along the
    pass. Passes are processed from the longest: for each frame the offset is the
    median, over the own frames within `win` of it, of (own z - median z of the
    reference frames within `radius` m); the frame is lowered/raised by it when
    the pass's median |offset| exceeds `gate`. Frames without reference nearby
    take the offset of the nearest corrected frame of their pass. Returns
    {video: median offset applied}."""
    order = sorted(video_scene_dict, key=lambda vt: -len(video_scene_dict[vt]['frame_infos']))
    P = {vt: np.array([result_poses_dict_aligned[i['token']][:3, 3] for i in video_scene_dict[vt]['frame_infos']])
         for vt in order}
    ref = [P[order[0]]] if len(order) else []
    applied = {}
    for vt in order[1:]:
        pts = P[vt]
        if len(pts) == 0:
            continue
        R = np.concatenate(ref)
        raw = np.full(len(pts), np.nan)
        for i, p in enumerate(pts):
            near = np.linalg.norm(R[:, :2] - p[:2], axis=1) < radius
            if near.sum() >= min_nb:
                raw[i] = p[2] - float(np.median(R[near, 2]))
        ok = ~np.isnan(raw)
        if ok.sum() >= 5 and abs(float(np.median(raw[ok]))) > gate:
            off = np.full(len(pts), np.nan)
            for i in range(len(pts)):
                lo, hi = max(0, i - win), min(len(pts), i + win + 1)
                seg = raw[lo:hi]; seg = seg[~np.isnan(seg)]
                if len(seg):
                    off[i] = float(np.median(seg))
            idx_ok = np.where(~np.isnan(off))[0]
            for i in range(len(pts)):
                if np.isnan(off[i]):
                    off[i] = off[idx_ok[np.argmin(abs(idx_ok - i))]]
            for i, info in enumerate(video_scene_dict[vt]['frame_infos']):
                result_poses_dict_aligned[info['token']][2, 3] -= off[i]
            pts = pts.copy(); pts[:, 2] -= off
            applied[vt] = float(np.median(raw[ok]))
        ref.append(pts)
    return applied


def _map_icp_pass(video, scans_by_token, init_pose_sets, ref_cloud, label=""):
    """Rigidly register one pass onto the point map of the aligned passes.

    A pass put back on its GPS poses ghosts wherever the INS is biased against
    the ICP frame (fq_loc_0182, fq_gap_01: 1-3 m), and MTGS_PIN_KEEP_ICP ghosts
    where the ICP chain itself slipped. Neither frame is trusted blindly here:
    every candidate initialisation (ICP poses through the global fit, INS poses
    at the aligned height) is refined by a single rigid point-to-plane ICP of the
    pass's own lidar accumulation against the map of the good passes, and the
    best fit wins if it is good enough. Returns (token->pose dict, report) or
    (None, report)."""
    import open3d as o3d
    voxel = float(os.environ.get('MTGS_REG_MAP_ICP_VOXEL', 0.3))
    min_fit = float(os.environ.get('MTGS_REG_MAP_ICP_FITNESS', 0.4))
    max_rmse = float(os.environ.get('MTGS_REG_MAP_ICP_RMSE', 0.35))
    max_shift = float(os.environ.get('MTGS_REG_MAP_ICP_SHIFT', 6.0))

    def _cloud(points):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pc = pc.voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
        return pc

    def _accumulate(pose_of):
        pts = []
        for info in video['frame_infos'][::2]:
            t = info['token']
            if t not in scans_by_token or pose_of(t) is None:
                continue
            P = scans_by_token[t][:, :3]
            keep = np.linalg.norm(P[:, :2], axis=1) < 60.0
            P = P[keep][::3]
            pose = pose_of(t)
            pts.append((pose[:3, :3] @ P.T).T + pose[:3, 3])
        return np.concatenate(pts) if pts else None

    best, report = None, []
    ref = _cloud(ref_cloud)
    # Trajectory-shape gate (2026-10-30, w2_07 video 407): an ICP chain that slipped in a turn kept its straight
    # segment, so the rigid fit of the whole accumulation still passed (fitness/rmse are dominated by the good
    # frames) while the turn came out up to 60 m / 155 deg off -- floaters in training and a sim camera driving
    # into the geometry. A candidate whose poses, after a rigid fit onto the INS poses, deviate from the INS
    # trajectory by more than MTGS_REG_MAP_ICP_TRAJ_M metres or MTGS_REG_MAP_ICP_TRAJ_DEG degrees is rejected.
    ins_of = dict(init_pose_sets).get("ins")
    traj_m = float(os.environ.get('MTGS_REG_MAP_ICP_TRAJ_M', 3.0)); traj_deg = float(os.environ.get('MTGS_REG_MAP_ICP_TRAJ_DEG', 8.0))

    def _traj_dev(pose_of):
        if ins_of is None or pose_of is ins_of:
            return 0.0, 0.0
        A, B = [], []
        for info in video['frame_infos']:
            a, b = pose_of(info['token']), ins_of(info['token'])
            if a is not None and b is not None:
                A.append(a); B.append(b)
        if len(A) < 3:
            return 0.0, 0.0
        A = np.array(A); B = np.array(B); pa, pb = A[:, :3, 3], B[:, :3, 3]
        ca, cb = pa.mean(0), pb.mean(0); U, _, Vt = np.linalg.svd((pa - ca).T @ (pb - cb))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); Rk = Vt.T @ D @ U.T
        res = np.linalg.norm((Rk @ (pa - ca).T).T + cb - pb, axis=1)
        ya = np.degrees(np.arctan2([(Rk @ m[:3, :3])[1, 0] for m in A], [(Rk @ m[:3, :3])[0, 0] for m in A]))
        yb = np.degrees(np.arctan2(B[:, 1, 0], B[:, 0, 0])); dy = (ya - yb + 180) % 360 - 180; dy -= np.median(dy)
        return float(res.max()), float(np.abs(dy).max())

    for name, pose_of in init_pose_sets:
        dev_m, dev_deg = _traj_dev(pose_of)
        if dev_m > traj_m or dev_deg > traj_deg:
            report.append(f"{name}: trajectory disagrees with INS by {dev_m:.1f} m / {dev_deg:.1f} deg (slipped chain), skipped")
            continue
        src_pts = _accumulate(pose_of)
        if src_pts is None or len(src_pts) < 2000:
            report.append(f"{name}: no points"); continue
        src = _cloud(src_pts)
        T = np.eye(4)
        for corr in (2.0, 0.8, 0.4):
            res = o3d.pipelines.registration.registration_icp(
                src, ref, corr, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            T = res.transformation
        shift = float(np.linalg.norm(T[:3, 3])); yaw = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        report.append(f"{name}: fitness {res.fitness:.2f} rmse {res.inlier_rmse:.2f} m shift {shift:.2f} m yaw {yaw:+.2f} deg")
        ok = res.fitness >= min_fit and res.inlier_rmse <= max_rmse and shift <= max_shift and abs(yaw) <= 10.0
        if ok and (best is None or res.inlier_rmse < best[0]):
            best = (res.inlier_rmse, name, T, pose_of)
    if best is None:
        return None, "; ".join(report)
    _, name, T, pose_of = best
    out = {info['token']: T @ pose_of(info['token']) for info in video['frame_infos'] if pose_of(info['token']) is not None}
    # A rigid fit cannot absorb drift INSIDE the pass (fq_gap_01 video 19: INS
    # distorted along the drive, the rigid-corrected pass still ghosted). Refine
    # every frame's pose by scan-to-map ICP from the rigid result, small
    # correspondence, bounded correction, then smooth the corrections along the
    # pass so the trajectory stays continuous. MTGS_REG_MAP_ICP_FRAMES=0 disables.
    if os.environ.get('MTGS_REG_MAP_ICP_FRAMES', '1') == '1':
        out, frep = _map_icp_frames(video, scans_by_token, out, ref, voxel)
        report.append(frep)
    return out, f"took {name} (" + "; ".join(report) + ")"


def _scan_icp(src, ref, ref_pts, T0, cs, max_iter):
    """ICP chain of one scan (src, legacy cloud) onto the map `ref` from T0 at correspondence distances cs.

    registration_icp builds a KD-tree of its target on every call, and the target here was the whole block map
    (10^6 points, ~1 s per build, twice per frame, every frame of every pass and sweep): that was most of the
    map-ICP wall-clock. The map is cropped to a ball around the scan that holds every point the full-map search
    could ever pair with: the transformed scan lies within r_max (its farthest point) of the ICP's current origin,
    a correspondence within max(cs) of that, and the origin within `margin` of T0 -- the final shift is checked
    against margin / 2 and the chain redone on the full map if it ever exceeded it (a return from beyond the margin
    inside the 30 iterations is not covered, with sub-metre correspondences it does not happen). Same
    correspondences => same transforms, fitness and rmse (pn_w2_21: equal to 1e-14, the OpenMP reduction noise
    the full-map run has against itself). The open3d tensor ICP on CUDA was tried here (2026-11-07) and dropped:
    3.9 s per frame against 60 ms, with transforms 1.5e-5 m off."""
    import open3d as o3d
    margin = float(os.environ.get('MTGS_REG_MAP_ICP_CROP_MARGIN', 5.0))
    P = np.asarray(src.points)
    r = float(np.linalg.norm(P, axis=1).max()) + max(cs) + margin if len(P) else 0.0
    sel = np.flatnonzero(np.linalg.norm(ref_pts - T0[:3, 3], axis=1) < r)
    if len(sel) < 30:
        # No map within reach of the scan (ml_sg01 2026-11-16: a stretch of a pass far from every other pass in the
        # consistency sweep): nothing to register against. The empty crop has no normals and point-to-plane ICP raised
        # on it; this is a failed fit, not an error -- the caller tells it apart from a bad fit by its map coverage.
        return T0, 0.0, float('inf')
    tgt = ref.select_by_index(sel) if len(sel) < len(ref_pts) else ref
    for attempt in range(2):
        T = T0
        for c in cs:
            res = o3d.pipelines.registration.registration_icp(
                src, tgt, c, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
            T = res.transformation
        if tgt is ref or np.linalg.norm(T[:3, 3] - T0[:3, 3]) <= margin / 2:
            break
        tgt = ref                                 # drifted towards the crop's edge: redo on the full map
    return T, float(res.fitness), float(res.inlier_rmse)


def _map_icp_frames(video, scans_by_token, poses, ref, voxel, max_dt=1.0, max_dyaw_deg=3.0, win=5, tilt=None):
    """Per-frame scan-to-map refinement of `poses` (token -> 4x4) against the
    open3d map `ref`; returns (refined poses, report).

    tilt (default MTGS_REG_PASS_TILT=1): the INS attitude of a vehicle carries a constant roll/pitch bias (ir_w2_06,
    2026-10-30: veh-40 vs veh-28 scans differ by a 1.4 deg roll about the driving axis, visible as a tilted ground in
    a cross-section). That is a rotation about each frame's OWN axes, not a rigid motion of the pass, so the rigid fit
    cannot remove it and the per-frame step only fixes the frames that overlap the map (82/476 on pass 64). Here a
    first per-frame round measures the correction in the ego frame, its median roll/pitch is applied to EVERY frame
    of the pass, and the per-frame round is repeated from there."""
    import open3d as o3d
    from scipy.spatial.transform import Rotation as _R
    if tilt is None:
        tilt = os.environ.get('MTGS_REG_PASS_TILT', '0') == '1'
    tokens = [info['token'] for info in video['frame_infos'] if info['token'] in poses and info['token'] in scans_by_token]
    ref_pts = np.asarray(ref.points)                        # for the per-scan crop in _scan_icp
    # Map coverage of a frame whose fit failed (2026-11-16, ml_sg01): the share of its scan points, at its input pose,
    # with a map point within MTGS_REG_MAP_ICP_COVER_M (2 m: wider than any input-pose error the gates admit). Below
    # MTGS_REG_MAP_ICP_MIN_COVER the frame lies outside the map (a stretch no other pass drove): it is carried like any
    # unrefined frame, but it is no registration failure and does not count towards the unrefined-run stop below.
    cover_m = float(os.environ.get('MTGS_REG_MAP_ICP_COVER_M', 2.0))
    min_cover = float(os.environ.get('MTGS_REG_MAP_ICP_MIN_COVER', 0.10))
    _tree = []
    outside = set()

    def _cover(P, pose):
        if not _tree:
            from scipy.spatial import cKDTree
            _tree.append(cKDTree(ref_pts))
        if len(P) == 0:
            return 0.0
        d, _ = _tree[0].query((pose[:3, :3] @ P.T).T + pose[:3, 3], k=1, distance_upper_bound=cover_m)
        return float(np.isfinite(d).mean())
    tilt_rep = ""
    if tilt:
        ego_rv = []
        for t in tokens[::2]:
            P = scans_by_token[t][:, :3]
            P = P[np.linalg.norm(P[:, :2], axis=1) < 60.0][::2]
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(voxel)
            T, fitness, _ = _scan_icp(src, ref, ref_pts, poses[t], (0.8, 0.3), 30)
            D = T @ np.linalg.inv(poses[t])
            if fitness >= 0.3 and np.linalg.norm(T[:3, 3] - poses[t][:3, 3]) <= max_dt:   # scan-origin displacement, see below
                ego_rv.append(poses[t][:3, :3].T @ _R.from_matrix(D[:3, :3]).as_rotvec(degrees=True))   # roll (x), pitch (y), yaw (z) about the frame's own axes
        if len(ego_rv) >= 10:
            med = np.median(np.array(ego_rv), axis=0); roll, pitch = float(med[0]), float(med[1])
            if max(abs(roll), abs(pitch)) >= 0.1 and max(abs(roll), abs(pitch)) <= 3.0:
                Rt = _R.from_rotvec([roll, pitch, 0.0], degrees=True).as_matrix()
                poses = {t: (lambda M: np.block([[M[:3, :3] @ Rt, M[:3, 3:4]], [M[3:4, :]]]))(poses[t]) for t in poses}
                tilt_rep = f"pass tilt roll {roll:+.2f} / pitch {pitch:+.2f} deg from {len(ego_rv)} frames applied to all; "
            else:
                tilt_rep = f"pass tilt roll {roll:+.2f} / pitch {pitch:+.2f} deg ({len(ego_rv)} frames), not applied; "
        else:
            tilt_rep = f"pass tilt: only {len(ego_rv)} frames on the map, skipped; "
    # Corrections are gated and smoothed AT THE SCAN ORIGIN (2026-10-31, w2_06 rs_). The map-frame form
    # D = T @ inv(pose) carries a translation of theta*|p| for a rotation theta of a scan |p| metres from the
    # block origin (1.4 deg at 40 m is already 1 m), so the max_dt gate silently rejected every rotational
    # correction beyond ~0.3-1 deg once a pass was away from the origin, the window median mixed rotation-
    # induced translations of neighbouring frames, and the cross-vehicle roll of veh-28 vs veh-40 survived five
    # sweeps unchanged (pass QA 1.1-1.7 deg, "median correction 0.75 m, max 1.00 m" = theta*|p| at the cap).
    # The rotation is kept as the map-frame rotation vector of D (position independent), the translation as
    # the displacement of the scan origin; MTGS_REG_MAP_ICP_FRAME_ROT_DEG bounds the rotation per frame.
    max_rot_deg = float(os.environ.get('MTGS_REG_MAP_ICP_FRAME_ROT_DEG', 3.0))
    # A fit that matches the map closely may move a frame further than max_dt (2026-11-17, ml_bos02 pass 2 = veh-40
    # 00869_01319): its last 33 frames (185 m, the routes of 37083abd / 9684b440) sat 1.0-2.8 m off the map after the
    # rigid pass fit; their fits put the walls exactly onto the map (fitness 0.7 at 0.3 m correspondence) but max_dt
    # rejected every one, the run stayed on interpolated poses and the unrefined-run stop ended the stage. Fitness >=
    # MTGS_REG_MAP_ICP_HIFIT (0.6) admits an origin shift up to MTGS_REG_MAP_ICP_HIFIT_DT (3 m); rotation gates unchanged,
    # the window median still needs >= 2 refined frames, and a step left at the end of such a stretch is the step QA's job.
    hi_fit = float(os.environ.get('MTGS_REG_MAP_ICP_HIFIT', 0.6))
    hi_dt = float(os.environ.get('MTGS_REG_MAP_ICP_HIFIT_DT', 3.0))
    n_hifit = [0]

    def _measure(idx, base):
        """Scan-to-map correction (map-frame rotation vector, scan-origin displacement) of the frames idx from the
        poses base, or None where the fit fails a gate."""
        res = []
        for i in idx:
            t = tokens[i]
            P = scans_by_token[t][:, :3]
            P = P[np.linalg.norm(P[:, :2], axis=1) < 60.0][::2]
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(voxel)
            T, fitness, _ = _scan_icp(src, ref, ref_pts, base[t], (0.8, 0.3), 30)
            D = T @ np.linalg.inv(base[t])                      # correction in the map frame (rotation part used)
            rvec = _R.from_matrix(D[:3, :3]).as_rotvec()
            dp = T[:3, 3] - base[t][:3, 3]                      # displacement of the scan origin
            dt = float(np.linalg.norm(dp)); drot = float(np.degrees(np.linalg.norm(rvec)))
            dyaw = abs(float(_R.from_matrix(D[:3, :3]).as_euler('zyx', degrees=True)[0]))
            rot_ok = drot <= max_rot_deg and dyaw <= max_dyaw_deg
            ok = fitness >= 0.3 and dt <= max_dt and rot_ok
            if not ok and rot_ok and fitness >= hi_fit and dt <= hi_dt:
                ok = True; n_hifit[0] += 1
            res.append((rvec, dp) if ok else None)
            if ok:
                outside.discard(i)
            elif _cover(np.asarray(src.points), base[t]) < min_cover:
                outside.add(i)
            else:
                outside.discard(i)
        return res

    def _smooth(c):
        """Component-wise median of each frame's window (rotation vector and origin displacement), so a single bad
        ICP frame cannot kink the pass; None where fewer than 2 frames of the window were refined."""
        sm = [None] * len(c)
        for i in range(len(c)):
            nb = [c[j] for j in range(max(0, i - win // 2), min(len(c), i + win // 2 + 1)) if c[j] is not None]
            if len(nb) >= 2:
                sm[i] = (np.median(np.array([x[0] for x in nb]), axis=0), np.median(np.array([x[1] for x in nb]), axis=0))
        return sm

    def _apply(c, P):
        M = np.eye(4); M[:3, :3] = _R.from_rotvec(c[0]).as_matrix() @ P[:3, :3]; M[:3, 3] = P[:3, 3] + c[1]
        return M

    corr = _measure(range(len(tokens)), poses)
    n_ok = sum(c is not None for c in corr)
    sm = _smooth(corr)
    # Frames without a refined frame near them must not keep the input pose: next to refined frames that sit up to
    # max_dt away from it, the pass then holds two frames of reference with a step between them (ru_area0806 pass 357,
    # 2026-11-16: 47 frames left on the INS shape, 2.2 m / 0.85 m z steps to the refined frames on either side, a road
    # smeared over 2 m under them, the camera placed into it -> white fog in HUGSIM). They take the correction of the
    # refined frames before and after them, blended by frame index (one side only: that one), like ego_pose_carry ...
    ok_idx = [i for i in range(len(tokens)) if sm[i] is not None]
    carried = [i for i in range(len(tokens)) if sm[i] is None] if ok_idx else []
    if ok_idx:
        oi = np.asarray(ok_idx)
        for i in carried:
            k = int(np.searchsorted(oi, i)); a = ok_idx[k - 1] if k > 0 else None; b = ok_idx[k] if k < len(ok_idx) else None
            if a is None or b is None:
                sm[i] = sm[b if a is None else a]
            else:
                w = (i - a) / (b - a)
                ra, rb = _R.from_rotvec(sm[a][0]), _R.from_rotvec(sm[b][0])
                sm[i] = ((ra * _R.from_rotvec(w * (ra.inv() * rb).as_rotvec())).as_rotvec(), (1.0 - w) * sm[a][1] + w * sm[b][1])
    out = dict(poses)
    for i, t in enumerate(tokens):
        if sm[i] is not None:
            out[t] = _apply(sm[i], poses[t])
    # ... and that blend is only the starting point: a second scan-to-map round from it refines the frames whose first
    # fit failed because the input pose was further off than max_dt (MTGS_REG_MAP_ICP_ROUND2=0 disables it).
    left, n2 = list(carried), 0
    if carried and os.environ.get('MTGS_REG_MAP_ICP_ROUND2', '1') == '1':
        full = [None] * len(tokens)
        for i, c in zip(carried, _measure(carried, out)):
            full[i] = c
        sm2 = _smooth(full)
        for i in carried:
            if sm2[i] is not None:
                out[tokens[i]] = _apply(sm2[i], out[tokens[i]]); n2 += 1
        left = [i for i in carried if sm2[i] is None]
    mags = [float(np.linalg.norm(x[1])) for x in corr if x is not None]
    rots = [float(np.degrees(np.linalg.norm(x[0]))) for x in corr if x is not None]
    rep = (tilt_rep + f"per-frame map ICP: {n_ok}/{len(tokens)} frames refined, median correction "
           f"{np.median(mags) if mags else 0:.2f} m / {np.median(rots) if rots else 0:.2f} deg, "
           f"max {max(mags) if mags else 0:.2f} m / {max(rots) if rots else 0:.2f} deg"
           + (f" ({n_hifit[0]} fits beyond {max_dt:g} m admitted by fitness >= {hi_fit:g})" if n_hifit[0] else ""))
    if carried:
        rep += f"; {len(carried)} unrefined frames carried from their refined neighbours, {n2} of them refined in a second round"
    out_left = [i for i in left if i in outside]
    if out_left:
        rep += f"; {len(out_left)} of them outside the map's coverage (no map within {cover_m:g} m of their scans), kept on the carried poses"
    left = [i for i in left if i not in outside]
    if left:
        # what is still unrefined holds interpolated poses only: report every run, stop when one is too long to trust
        runs, s0 = [], left[0]
        for x, y in zip(left, left[1:] + [None]):
            if y != x + 1:
                runs.append((s0, x)); s0 = y
        def _len_m(r):
            P = np.array([out[tokens[i]][:3, 3] for i in range(max(0, r[0] - 1), min(len(tokens), r[1] + 2))])
            return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
        spans = [(r, _len_m(r)) for r in runs]
        (r_w, m_w) = max(spans, key=lambda x: x[1])
        rep += (f"; {len(left)} frames left on interpolated poses in {len(runs)} run(s), longest frames {r_w[0]}-{r_w[1]} "
                f"({r_w[1] - r_w[0] + 1} frames, {m_w:.0f} m)")
        # 100 m: the longest runs of good registrations are 5-73 m (pn_area0806 pass 357 after the fix: 73 m between refined
        # frames, smooth; px_loc_0195: edge runs up to 70 m); a step at a run's ends is the step QA's job
        lim = float(os.environ.get('MTGS_REG_MAP_ICP_MAX_UNREFINED_M', 100.0))
        if lim > 0 and m_w > lim:
            vt = str(video['frame_infos'][0].get('video_token', '?')) if video['frame_infos'] else '?'
            msg = (f"per-frame map ICP of pass {vt.rsplit('-', 1)[-1]}: frames {r_w[0]}-{r_w[1]} ({r_w[1] - r_w[0] + 1} frames, "
                   f"{m_w:.0f} m) could not be refined against the map even from their neighbours' correction; their poses "
                   f"are interpolated only (limit MTGS_REG_MAP_ICP_MAX_UNREFINED_M={lim:g} m). Stopping instead of training "
                   f"on them. Report so far: {rep}")
            say(msg, style="bold red")
            raise SystemExit(msg)
    return out, rep


def _pass_accumulation(video, scans_by_token, poses):
    """World-frame accumulation of a pass (every 2nd frame, r < 60 m, every 3rd point) at `poses`."""
    pts = []
    for info in video['frame_infos'][::2]:
        t = info['token']
        if t not in scans_by_token or t not in poses:
            continue
        P = scans_by_token[t][:, :3]
        P = P[np.linalg.norm(P[:, :2], axis=1) < 60.0][::3]
        pose = poses[t]
        pts.append((pose[:3, :3] @ P.T).T + pose[:3, 3])
    return np.concatenate(pts) if pts else None


def _refine_on_own_lidar(video, result_poses_dict_aligned, scans_by_token):
    """Two rounds of per-frame map ICP of a pass onto the accumulation of its OWN scans at its current poses: the pass
    gets lidar-consistent frame poses while its absolute frame (GPS / INS) stays. For passes no other pass can anchor
    (see the GPS-pass consolidation). Returns (poses, report)."""
    import open3d as o3d
    vox = float(os.environ.get('MTGS_REG_MAP_ICP_VOXEL', 0.3))
    pr = {i['token']: result_poses_dict_aligned[i['token']] for i in video['frame_infos'] if i['token'] in result_poses_dict_aligned}
    rep = "no scans"
    for _ in range(2):
        pts = _pass_accumulation(video, scans_by_token, pr)
        if pts is None or len(pts) < 5000:
            break
        ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(vox)
        ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 4, max_nn=30))
        pr, rep = _map_icp_frames(video, scans_by_token, pr, ref, vox)
    return pr, rep


def _inconsistent_passes(vt, video_scene_dict, poses, scans_by_token, clouds):
    """Passes that overlap pass `vt` but are NOT already registered with it. Sampled scans of vt are ICP'd one by one
    onto the other pass's accumulation; the median correction (translation, and roll/pitch/yaw about the scan's own
    axes) over the overlapping scans must stay within MTGS_REG_SWEEP_TOL_M / _DEG (0.15 m / 0.3 deg). A rigid fit of
    the whole pass was tried first (2026-10-30) and missed the per-frame 1.4 deg roll of ir_w2_06 entirely. Same-day
    passes of one vehicle agree to a few cm, so with the plain all-others map they anchor each other and a cross-
    vehicle offset survives the sweeps. Returns (inconsistent passes, report, {other pass: median shift in m})."""
    import open3d as o3d
    from scipy.spatial.transform import Rotation
    voxel = float(os.environ.get('MTGS_REG_MAP_ICP_VOXEL', 0.3))
    tol_m = float(os.environ.get('MTGS_REG_SWEEP_TOL_M', 0.15)); tol_deg = float(os.environ.get('MTGS_REG_SWEEP_TOL_DEG', 0.3))
    nsamp = int(os.environ.get('MTGS_REG_SWEEP_SAMPLES', 16))
    tokens = [i['token'] for i in video_scene_dict[vt]['frame_infos'] if i['token'] in poses and i['token'] in scans_by_token]
    if len(tokens) < 5:
        return [], "no frames", {}
    bad, rep, shifts = [], [], {}
    for u in video_scene_dict:
        if u == vt:
            continue
        if u not in clouds:
            pts = _pass_accumulation(video_scene_dict[u], scans_by_token, poses)
            if pts is None or len(pts) < 2000:
                clouds[u] = None
            else:
                tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(voxel)
                tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
                clouds[u] = (tgt, np.asarray(tgt.points)[:, :2])
        if clouds[u] is None:
            continue
        tgt, txy = clouds[u]
        # scans of vt that lie within 30 m of pass u's points
        from scipy.spatial import cKDTree
        tree = cKDTree(txy)
        cand = [t for t in tokens if tree.query(poses[t][:2, 3])[0] < 30.0]
        if len(cand) < 5:
            continue
        cand = cand[::max(1, len(cand) // nsamp)][:nsamp]
        dts, rvs = [], []
        for t in cand:
            P = scans_by_token[t][:, :3]
            P = P[np.linalg.norm(P[:, :2], axis=1) < 60.0][::2]
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(voxel)
            T = poses[t]; res = None
            for c in (1.0, 0.5, 0.3):
                res = o3d.pipelines.registration.registration_icp(
                    src, tgt, c, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
                T = res.transformation
            if res.fitness < 0.3:
                continue
            D = T @ np.linalg.inv(poses[t])
            dts.append(float(np.linalg.norm(D[:3, 3] + (D[:3, :3] - np.eye(3)) @ poses[t][:3, 3])))
            rvs.append(poses[t][:3, :3].T @ Rotation.from_matrix(D[:3, :3]).as_rotvec(degrees=True))
        if len(dts) < 5:
            continue   # no real overlap
        shift = float(np.median(dts)); rv = np.median(np.array(rvs), axis=0); shifts[u] = shift
        rep.append(f"{u.split('-')[-1]}: {shift:.2f} m, roll {rv[0]:+.2f} pitch {rv[1]:+.2f} yaw {rv[2]:+.2f} deg ({len(dts)} scans)")
        if shift > tol_m or np.abs(rv).max() > tol_deg:
            bad.append(u)
    return bad, ", ".join(rep), shifts


def _map_icp_inputs(video_scene_dict, result_poses_dict_aligned, exclude, scans_by_token, cache=None):
    """Point map of the aligned (trusted) passes for _map_icp_pass.

    cache (dict, optional): per-pass world-frame points keyed by the pass's current poses, so a sweep over N passes
    re-transforms only the pass that moved instead of every scan of every other pass N times. The concatenation is
    the same array as before (same passes, same order), so the voxelised map and the ICP are unchanged; the voxel
    grid + normals of the merged map cannot be cached per pass (a voxel's point is the mean over ALL passes)."""
    pts = []
    for vt, v in video_scene_dict.items():
        if vt in exclude:
            continue
        toks = [info['token'] for info in v['frame_infos'][::2]
                if info['token'] in scans_by_token and info['token'] in result_poses_dict_aligned]
        if not toks:
            continue
        key = (vt, np.stack([result_poses_dict_aligned[t] for t in toks]).tobytes())
        if cache is None or cache.get(vt, (None,))[0] != key:
            part = []
            for t in toks:
                P = scans_by_token[t][:, :3]
                P = P[np.linalg.norm(P[:, :2], axis=1) < 60.0][::3]
                pose = result_poses_dict_aligned[t]
                part.append((pose[:3, :3] @ P.T).T + pose[:3, 3])
            part = np.concatenate(part)
            if cache is not None:
                cache[vt] = (key, part)
        else:
            part = cache[vt][1]
        pts.append(part)
    return np.concatenate(pts) if pts else None


def _is_pinned(video):
    # default: the required logs (the served scenes' own drives, MTGS_REQUIRE_LOGS from the prep env) -- without it the
    # pre-alignment gate dropped own passes for good (2026-11-03: pn_nb_17 / pn_w2_07 lost them, scenes fell back)
    pins = [x for x in os.environ.get('MTGS_PIN_LOGS', os.environ.get('MTGS_REQUIRE_LOGS', '')).split(',') if x]
    infos = video.get('frame_infos', [])
    ln = infos[0].get('log_name', '') if infos else ''
    return any(pin in ln for pin in pins)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config: RoadBlockConfig = load_config(args.config)
    video_scene = VideoScene(config)
    video_scene_dict = video_scene.load_pickle(video_scene.pickle_path_raw)

    from kiss_icp.datasets.mtgs import MTGSDataset
    from kiss_icp.pipeline import OdometryPipeline

    kiss_icp_dataset = MTGSDataset(
        data_dir = Path(os.path.dirname(video_scene.pickle_path)),
        video_scene = video_scene
    )

    kiss_icp_pipeline = OdometryPipeline(
        dataset=kiss_icp_dataset,
        config=None,
        max_range=max(video_scene.block_size) / 2 + config.expand_buffer + 50,
        deskew=False,
        visualize=False,
    )
    kiss_icp_pipeline.config.out_dir = video_scene.registration_path
    if os.environ.get('MTGS_KISS_GRAVITY', '1') == '1':
        import types
        kiss_icp_pipeline.odometry.register_frame = types.MethodType(_register_frame_gravity, kiss_icp_pipeline.odometry)
    kiss_icp_pipeline.run()

    # The registritaion works on the first frames coordinate.
    # We translate the poses back to the global coordinate here.
    result_poses = kiss_icp_dataset.apply_calibration(kiss_icp_pipeline.poses)
    tokens = kiss_icp_dataset.tokens
    result_poses_dict = {token: pose for token, pose in zip(tokens, result_poses)}
    # raw scans (lidar frame) for the map ICP of passes that fail the gates
    _scans_by_token = {token: scan for token, scan in zip(tokens, kiss_icp_dataset.scans)} \
        if os.environ.get('MTGS_REG_MAP_ICP', '1') == '1' else {}

    if os.environ.get('MTGS_SKIP_ICP') == '1':
        # KISS-ICP is scan-to-map odometry meant for ~10Hz sequences. nuPlan
        # sensor data is 2Hz (4-6m between scans) and the Hazelwood blocks are
        # straight roads with grass verges: nothing constrains motion ALONG the
        # road, so the estimate slides (86m end-point error). KISS-Matcher, a
        # global method, calls the same pairs either INVALID or a 0.57m
        # correction -- i.e. the GPS/IMU poses were already right and there is
        # no geometry here to refine against. Keep them.
        say("MTGS_SKIP_ICP=1: keeping GPS/IMU poses, no ICP refinement",
                    style="bold yellow")
        for video_token in video_scene_dict:
            for info in video_scene_dict[video_token]['frame_infos']:
                info['ego2global_original'] = info['ego2global'].copy()
        video_scene.video_scene_dict = video_scene_dict
        video_scene_dict = video_scene.video_scene_dict_process(
            ['filter_low_velocity', 'inject_trajectory'], inline=True)
        check_unrefined_passes(video_scene, 'regis', video_scene.pickle_path_registered.replace('.pkl', '_unrefined_qa_failed.pkl'), warn_only=True)
        check_trajectory_steps(video_scene, 'regis', video_scene.pickle_path_registered.replace('.pkl', '_step_qa_failed.pkl'))
        video_scene.dump_pickle(video_scene.pickle_path_registered)
        video_scene.update_pickle_link(video_scene.pickle_path_registered)
        sys.exit(0)

    say("ICP errors", style="bold green")
    error_dict = calculate_errors(video_scene_dict, result_poses_dict)
    visualize_lidar_registration(video_scene, result_poses_dict)

    # Robust pre-alignment rejection: align_poses is one global Kabsch fit over
    # ALL traversals, so a single badly-registered video (raw ATE > 1 m) skews
    # the alignment and inflates every video's post-alignment error -- the QA
    # gate then removes ALL of them. Drop gross raw outliers first; they could
    # never pass the final (1.0 / 0.5) thresholds anyway.
    PRE_EPE = float(os.environ.get('MTGS_REG_PRE_EPE', 2.0))
    PRE_MEAN = float(os.environ.get('MTGS_REG_PRE_MEAN', 1.0))
    keep_first = list(video_scene_dict.keys())[:1] if isinstance(config, FrameCentralConfig) else []
    # The raw errors are measured against the INS, which is itself off by 1-3 m in
    # some blocks (Boston nb_03/05/06, Vegas nb_13: every pass raw EPE 2-3 m, the
    # survivor 0.2 m after alignment). An absolute gate then threw 4-5 of 6 good
    # passes away and put them back on GPS poses, while a block whose passes ALL
    # exceeded it (nb_20: 432 m and 117 m garbage next to 1-5 m) dropped nothing
    # and the garbage skewed the fit for everyone. So an outlier is a pass far
    # from the block's typical raw error (MTGS_REG_PRE_REL x median), and only
    # above the absolute floor.
    PRE_REL = float(os.environ.get('MTGS_REG_PRE_REL', 2.5))
    _epe = np.array([error_dict[vt]['max_end_point_error'] for vt in video_scene_dict])
    _mean = np.array([error_dict[vt]['mean_error'] for vt in video_scene_dict])
    epe_gate = max(PRE_EPE, PRE_REL * float(np.median(_epe)))
    mean_gate = max(PRE_MEAN, PRE_REL * float(np.median(_mean)))
    say(f"Pre-alignment outlier gates: EPE > {epe_gate:.2f} m, mean > {mean_gate:.2f} m "
                f"(block medians {np.median(_epe):.2f} / {np.median(_mean):.2f})")
    # Absolute ceiling on top of the relative gate (2026-10-30, w2_02: half the passes diverged to 145-189 m, the
    # block median became 73 m and the relative gate let 157 m and 145 m through). A raw EPE of tens of metres is a
    # lost odometry chain, never an INS bias.
    PRE_ABS = float(os.environ.get('MTGS_REG_PRE_ABS_EPE', 20.0))
    outliers = [vt for vt in list(video_scene_dict.keys())
                if vt not in keep_first
                and (error_dict[vt]['max_end_point_error'] > epe_gate
                     or error_dict[vt]['mean_error'] > mean_gate
                     or error_dict[vt]['max_end_point_error'] > PRE_ABS)]
    pre_dropped = {}
    if outliers and len(outliers) < len(video_scene_dict):
        for vt in outliers:
            say(f"Drop video {vt.split('-')[-1]} before alignment (raw ICP outlier).",
                        style="yellow")
            pre_dropped[vt] = video_scene_dict.pop(vt)

    say("ICP errors after alignment", style="bold green")
    result_poses_dict_aligned = align_poses(video_scene_dict, result_poses_dict)
    error_dict_aligned = calculate_errors(video_scene_dict, result_poses_dict_aligned)

    # Sliders (large translation, tiny rotation) are never refinement, so revert
    # them to GPS poses whenever the fallback is enabled; the count-based rescue
    # below then decides whether any OTHER reject also has to be kept.
    fb_on = os.environ.get('MTGS_ICP_FALLBACK_ENABLE', '1') == '1'
    fb_max = float(os.environ.get('MTGS_ICP_FALLBACK_MAX', 2.0)) if fb_on else 0.0
    gps_videos = set()   # passes carried on (height-corrected) GPS poses
    if fb_max > 0:
        touched = False
        sliders = [vt for vt in video_scene_dict
                   if error_dict_aligned[vt]['mean_error'] > fb_max and error_dict_aligned[vt]['mean_heading_error'] < 2.0]
        ref = _aligned_reference(video_scene_dict, result_poses_dict_aligned, set(sliders))
        for vt in list(video_scene_dict.keys()):
            e = error_dict_aligned[vt]
            if e['mean_error'] > fb_max and e['mean_heading_error'] < 2.0:
                say(f"Traversal {vt.split('-')[-1]}: ICP moved it "
                            f"{e['mean_error']:.1f}m with only "
                            f"{e['mean_heading_error']:.1f} deg of rotation change "
                            f"(slide, not refinement) -- keeping the GPS pose.",
                            style="yellow")
                result_poses_dict_aligned.update(_gps_poses_on_aligned_height(video_scene_dict[vt], ref))
                gps_videos.add(vt)
                touched = True
        if touched:
            error_dict_aligned = calculate_errors(video_scene_dict, result_poses_dict_aligned)
    visualize_lidar_registration(video_scene, result_poses_dict_aligned, filename="registration_aligned.pdf")

    # filter out bad video_info
    bad_videos = []
    # The post-alignment translation gates compare the ICP pose with the INS, so a
    # block whose INS is itself off by half a metre on some logs (Boston nb_03:
    # three passes at mean 0.53-0.63 m, all consistent with the map) fails them
    # for no registration reason and gets those passes put back on GPS poses.
    # Like the pre-alignment gate, scale them with the block's typical residual.
    QA_REL = float(os.environ.get('MTGS_REG_QA_REL', 2.5))
    _m = np.array([error_dict_aligned[vt]['mean_error'] for vt in video_scene_dict if vt not in gps_videos])
    _e = np.array([error_dict_aligned[vt]['max_end_point_error'] for vt in video_scene_dict if vt not in gps_videos])
    qa_mean_gate = max(float(os.environ.get('MTGS_REG_QA_MEAN', 0.5)), QA_REL * float(np.median(_m)) if len(_m) else 0.0)
    qa_epe_gate = max(float(os.environ.get('MTGS_REG_QA_EPE', 1.0)), QA_REL * float(np.median(_e)) if len(_e) else 0.0)
    # Ceilings on the relative gates (2026-10-30): when most passes are broken the median is broken too.
    qa_epe_gate = min(qa_epe_gate, max(float(os.environ.get('MTGS_REG_QA_EPE', 1.0)), float(os.environ.get('MTGS_REG_QA_ABS_EPE', 5.0))))
    qa_mean_gate = min(qa_mean_gate, max(float(os.environ.get('MTGS_REG_QA_MEAN', 0.5)), float(os.environ.get('MTGS_REG_QA_ABS_MEAN', 2.0))))
    say(f"Post-alignment QA gates: EPE > {qa_epe_gate:.2f} m, mean > {qa_mean_gate:.2f} m "
                f"(block medians {np.median(_e) if len(_e) else 0:.2f} / {np.median(_m) if len(_m) else 0:.2f})")
    for video_token in list(video_scene_dict.keys()):
        max_end_point_error = error_dict_aligned[video_token]['max_end_point_error']
        mean_error = error_dict_aligned[video_token]['mean_error']
        # Translation-only QA misses traversals that land in the right place
        # but face the wrong way. Across the 24 rebuilt blocks the 95th pct of
        # post-alignment mean_heading_error is 1.7 deg while the visually
        # broken ones sit at 3.8-5.6 deg, so gate at 2.5 deg.
        mean_heading_error = error_dict_aligned[video_token]['mean_heading_error']
        # A pinned pass (the scene's own drive) is what the simulated ego follows,
        # so its heading must agree with the INS to within MTGS_REG_QA_ARE_PINNED
        # (1 deg): at 1.9 deg (fl_loc_0241 / veh-48) the render from its registered
        # pose no longer matched the photo while the INS pose did. Over the gate it
        # is kept on GPS poses by the pinned rescue below.
        are_gate = float(os.environ.get('MTGS_REG_QA_ARE', 2.5))
        if _is_pinned(video_scene_dict[video_token]):
            # Off by default (2.5 = the general gate): the fn_ rebuild that held pinned
            # passes to 1 deg and put them on GPS poses rendered WORSE (ghosting, uneven
            # ground) than the ICP poses in every one of the 7 blocks (user, 2026-10-16).
            are_gate = min(are_gate, float(os.environ.get('MTGS_REG_QA_ARE_PINNED', 2.5)))
        if (max_end_point_error > qa_epe_gate
                or mean_error > qa_mean_gate
                or mean_heading_error > are_gate):
            bad_videos.append(video_token)

    # Height harmonisation: bring every pass onto the height of the passes registered
    # around it before the height QA.
    # rigid world tilts of whole passes against gravity first: the harmonisation below and every map ICP after it take
    # the other passes as their reference
    for what, m, mf, ok, why in (_level_passes_to_imu(video_scene_dict, result_poses_dict_aligned, _scans_by_token, set(bad_videos)) if _scans_by_token else []):
        say(("Block" if what == "block" else f"Traversal {what.split('-')[-1]}") + f": tilted {m:.1f} deg against the IMU in the world "
            f"frame (facades " + ("n/a" if mf is None else f"{mf:.1f} deg") + f") {why} -> " + ("levelled" if ok else "left as is"),
            style="yellow" if ok else None)
    _applied = _harmonize_z(video_scene_dict, result_poses_dict_aligned)
    for vt, off in _applied.items():
        say(f"Traversal {vt.split('-')[-1]}: height corrected by {-off:+.2f} m onto the other passes.", style="yellow")
    # Height QA: the BEV metrics above cannot see a pass that ICP settled 1-3 m
    # above or below the others (fk_loc_0185 trav 20 +3.3 m, fp_gapC_010 trav 18
    # +1.0 m passed EPE/ATE/ARE and trained a second road layer).
    dz_gate = float(os.environ.get('MTGS_REG_QA_DZ', 0.75))
    if dz_gate > 0:
        for vt, dz in _z_inconsistent_videos(video_scene_dict, result_poses_dict_aligned,
                                             set(bad_videos) | gps_videos, dz_gate).items():
            say(f"Traversal {vt.split('-')[-1]}: registered {dz:+.2f} m off the other "
                        f"traversals' height (gate {dz_gate} m) -> bad registration.", style="yellow")
            bad_videos.append(vt)

    if isinstance(config, FrameCentralConfig) and list(video_scene_dict.keys())[0] in bad_videos:
        say("WARNING: The central log has bad registration result. Use log pose and disable multi-traversal.", style="bold red")
        first_key = list(video_scene_dict.keys())[0]
        video_scene_dict = {first_key: video_scene_dict[first_key]}
    else:
        # Bring back EVERY pre-alignment casualty. They skip alignment entirely, so the fit the survivors got is
        # untouched. Until 2026-11-17 only the pinned ones and as many as MTGS_MIN_TRAVERSALS was short of came back;
        # any other selected pass disappeared without a line in the log (ml_pit01b video 22, 172 frames: dropped as a
        # raw ICP outlier, never mentioned again). A selected pass is restored by map ICP from its INS poses; if that
        # fails it goes on GPS poses into the GPS-pass consolidation (map ICP onto the passes trusted by then, else its
        # own lidar), and the pass QA / unrefined QA gates stop the stage if it still does not fit.
        if pre_dropped:
            chosen = sorted(pre_dropped.items(),
                            key=lambda kv: (not _is_pinned(kv[1]),
                                            -len(kv[1]['frame_infos'])))
            # MTGS_PIN_KEEP_ICP=1: a pinned pass (the scene's own drive) comes back on its
            # ICP poses carried through the global fit, not on GPS. Experiment for the
            # rainy own passes of fl_loc_0252 / fl_loc_0261, which on GPS poses left a
            # grey blob / smeared route (ICP had moved them 1-4.6 m; GPS on a rainy day).
            keep_icp = os.environ.get('MTGS_PIN_KEEP_ICP', '0') == '1'
            ref = _aligned_reference(video_scene_dict, result_poses_dict_aligned, set(bad_videos) | gps_videos)
            _map = _map_icp_inputs(video_scene_dict, result_poses_dict_aligned, set(bad_videos) | gps_videos, _scans_by_token) if _scans_by_token else None
            for vt, v in chosen:
                video_scene_dict[vt] = v
                if _map is not None:
                    # try to put the pass back on the map, not on the INS
                    _ins = _gps_poses_on_aligned_height(v, ref).get
                    _icp = lambda t: (_LAST_ALIGN_T @ result_poses_dict[t]) if t in result_poses_dict else None
                    fixed, rep = _map_icp_pass(v, _scans_by_token, [("icp+global", _icp), ("ins", _ins)], _map, vt)
                    if fixed is not None:
                        say(f"Restoring video {vt.split('-')[-1]} by map ICP ({len(fixed)} frames): {rep}", style="yellow")
                        result_poses_dict_aligned.update(fixed)
                        continue
                    say(f"Map ICP failed for video {vt.split('-')[-1]} ({rep}); falling back to GPS poses", style="bold red")
                if keep_icp and _is_pinned(v):
                    say(f"Restoring video {vt.split('-')[-1]} on its ICP poses through the "
                                f"global alignment ({len(v['frame_infos'])} frames, MTGS_PIN_KEEP_ICP).",
                                style="yellow")
                    for info in v['frame_infos']:
                        result_poses_dict_aligned[info['token']] = _LAST_ALIGN_T @ result_poses_dict[info['token']]
                    continue
                say(f"Restoring video {vt.split('-')[-1]} on GPS poses "
                            f"({len(v['frame_infos'])} frames): dropped before alignment; "
                            f"the GPS-pass consolidation registers it.", style="yellow")
                result_poses_dict_aligned.update(_gps_poses_on_aligned_height(v, ref))
                gps_videos.add(vt)

        if config.exclude_bad_registration:
            # Rescue by outcome: dropping every reject is only acceptable while
            # enough traversals remain. Below the target, keep the least-bad
            # rejects on their GPS/IMU poses instead of shipping a thin block.
            target = int(os.environ.get('MTGS_MIN_TRAVERSALS', 0)) if fb_on else 0
            keep_n = max(0, target - (len(video_scene_dict) - len(bad_videos)))
            ranked = sorted(bad_videos,
                            key=lambda v: (not _is_pinned(video_scene_dict[v]),
                                           error_dict_aligned[v]['mean_error'],
                                           error_dict_aligned[v]['mean_heading_error']))
            rescued = ranked[:keep_n] + [v for v in ranked[keep_n:]
                                         if _is_pinned(video_scene_dict[v])]
            ref = _aligned_reference(video_scene_dict, result_poses_dict_aligned, set(bad_videos) | gps_videos)
            _map = _map_icp_inputs(video_scene_dict, result_poses_dict_aligned, set(bad_videos) | gps_videos, _scans_by_token) if (_scans_by_token and rescued) else None
            for video_token in rescued:
                e = error_dict_aligned[video_token]
                if _map is not None:
                    _v = video_scene_dict[video_token]
                    _ins = _gps_poses_on_aligned_height(_v, ref).get
                    _icp = lambda t: result_poses_dict_aligned.get(t)
                    fixed, rep = _map_icp_pass(_v, _scans_by_token, [("aligned icp", _icp), ("ins", _ins)], _map, video_token)
                    if fixed is not None:
                        say(f"Keeping video {video_token.split('-')[-1]} by map ICP (QA said mean {e['mean_error']:.2f} m / {e['mean_heading_error']:.2f} deg): {rep}", style="yellow")
                        result_poses_dict_aligned.update(fixed)
                        continue
                    say(f"Map ICP failed for video {video_token.split('-')[-1]} ({rep}); falling back to GPS poses", style="bold red")
                if os.environ.get('MTGS_PIN_KEEP_ICP', '0') == '1' and _is_pinned(video_scene_dict[video_token]):
                    say(f"Keeping video {video_token.split('-')[-1]} on its aligned ICP poses "
                                f"(QA said mean {e['mean_error']:.2f}m / {e['mean_heading_error']:.2f} deg; "
                                f"MTGS_PIN_KEEP_ICP).", style="yellow")
                    continue
                say(f"Keeping video {video_token.split('-')[-1]} on GPS poses "
                            f"(QA said mean {e['mean_error']:.2f}m / "
                            f"{e['mean_heading_error']:.2f} deg, but the block would "
                            f"otherwise be short).", style="yellow")
                result_poses_dict_aligned.update(_gps_poses_on_aligned_height(video_scene_dict[video_token], ref))
                gps_videos.add(video_token)
            for video_token in bad_videos:
                if video_token in rescued:
                    continue
                say(f"Remove video {video_token.split('-')[-1]} due to bad registration result.", style="yellow")
                video_scene_dict.pop(video_token)
            if len(video_scene_dict) == 0:
                raise ValueError("All videos have bad registration result.")

        # GPS-pass consolidation (2026-10-30, w2_02 / w2_06 / v2_07 / nb_18: 4-6
        # passes left on raw GPS poses, offset from each other by the GPS error -> visible doubling). Every pass
        # that is still on GPS poses is registered by map ICP (INS start, trajectory gate, per-frame refinement)
        # onto the passes already trusted, largest first; a newly registered pass joins the reference. When NO pass
        # is trusted, the pinned (own) pass, else the longest, is taken as the reference on its GPS poses.
        if _scans_by_token and gps_videos:
            _live = [vt for vt in video_scene_dict if vt in gps_videos]
            _trusted = {vt for vt in video_scene_dict if vt not in gps_videos}
            if not _trusted and _live:
                _root = sorted(_live, key=lambda vt: (not _is_pinned(video_scene_dict[vt]), -len(video_scene_dict[vt]['frame_infos'])))[0]
                _trusted.add(_root); _live.remove(_root)
                # The reference (a single-pass block's only pass included) must not stay on raw INS poses: 9 served passes
                # in 8 blocks (all five single-pass st_ blocks, fp_bx_08, ...) did, an INS altitude glitch became a 0.96 m
                # bump of the camera against the model (st_e756), and a single GPS pass never even reached the old
                # 'len(_live) > 1' branch (no map to consolidate onto -> left as is). It is refined on its own lidar: two
                # rounds of per-frame map ICP onto the accumulation of its own scans; the INS keeps the absolute frame.
                _v = video_scene_dict[_root]
                _pr, _rep = _refine_on_own_lidar(_v, result_poses_dict_aligned, _scans_by_token)
                _d = [float(np.linalg.norm(_pr[t][:3, 3] - result_poses_dict_aligned[t][:3, 3])) for t in _pr]
                result_poses_dict_aligned.update(_pr)
                say(f"No ICP-registered pass to anchor to: video {_root.split('-')[-1]} (GPS poses) is the reference, refined "
                    f"on its own lidar (moved median {np.median(_d) if _d else 0:.2f} m, max {max(_d) if _d else 0:.2f} m; {_rep})", style="yellow")
            for vt in sorted(_live, key=lambda vt: -len(video_scene_dict[vt]['frame_infos'])):
                _map = _map_icp_inputs(video_scene_dict, result_poses_dict_aligned, set(video_scene_dict) - _trusted, _scans_by_token)
                if _map is None or len(_map) < 5000:
                    continue
                # start from the INS lifted on the heights of the passes trusted NOW: the GPS poses the pass carries were
                # lifted before the height harmonisation, on a reference that could still hold passes metres apart in
                # height (px_loc_0195 pass 61: lifted next to passes 23 / 59 while they sat 4 m off the others -> a 4.1 m
                # step at the edge of what map ICP could refine). The poses it carries stay a second candidate.
                _ref_now = _aligned_reference(video_scene_dict, result_poses_dict_aligned, set(video_scene_dict) - _trusted)
                _ins = _gps_poses_on_aligned_height(video_scene_dict[vt], _ref_now).get
                _cur = {i['token']: result_poses_dict_aligned[i['token']].copy() for i in video_scene_dict[vt]['frame_infos']
                        if i['token'] in result_poses_dict_aligned}.get
                fixed, rep = _map_icp_pass(video_scene_dict[vt], _scans_by_token, [("ins", _ins), ("carried", _cur)], _map, vt)
                if fixed is not None:
                    _d = [np.linalg.norm(fixed[t][:3, 3] - result_poses_dict_aligned[t][:3, 3]) for t in fixed if t in result_poses_dict_aligned]
                    say(f"GPS pass {vt.split('-')[-1]} registered onto the trusted passes by map ICP "
                                f"(moved median {np.median(_d) if _d else 0:.2f} m, max {max(_d) if _d else 0:.2f} m): {rep}", style="yellow")
                    result_poses_dict_aligned.update(fixed); gps_videos.discard(vt); _trusted.add(vt)
                else:
                    # no overlap with the trusted passes (nb_20 pass 22: fitness 0, a 17 m "fit"): it cannot be anchored,
                    # but it must not stay on raw INS either -- refine it on its own lidar, the INS keeps its frame
                    _pr, _rep = _refine_on_own_lidar(video_scene_dict[vt], result_poses_dict_aligned, _scans_by_token)
                    _d = [float(np.linalg.norm(_pr[t][:3, 3] - result_poses_dict_aligned[t][:3, 3])) for t in _pr]
                    result_poses_dict_aligned.update(_pr)
                    say(f"GPS pass {vt.split('-')[-1]} cannot be placed on the trusted passes (map ICP: {rep}); refined on its "
                        f"own lidar instead (moved median {np.median(_d) if _d else 0:.2f} m, max {max(_d) if _d else 0:.2f} m; {_rep})",
                        style="bold yellow")

        # MTGS_REG_MAP_ICP_ALL=1: inter-pass consistency sweep. The passes only agree
        # as well as the odometry chain and the x/y Kabsch let them (fq_loc_0182,
        # loc_0188, loc_0211: every pass 1-4 m from the INS and from each other, the
        # model a blur of ghost copies). Register each pass, worst QA first, by rigid +
        # per-frame scan-to-map ICP against the map of all the OTHER passes, and accept
        # the same way as a restored pass. Two sweeps; the second sees the corrected map.
        if _scans_by_token and os.environ.get('MTGS_REG_MAP_ICP_ALL', '0') == '1':
            _sweeps = int(os.environ.get('MTGS_REG_MAP_ICP_SWEEPS', 2))
            _order = sorted(video_scene_dict.keys(),
                            key=lambda vt: -error_dict_aligned.get(vt, {}).get('mean_error', 0.0))
            _pairwise = os.environ.get('MTGS_REG_SWEEP_PAIRWISE', '0') == '1'
            _parts = {}                    # _map_icp_inputs cache: a pass is re-transformed only after it moved
            for _sw in range(_sweeps):
                _clouds = {}
                for vt in _order:
                    v = video_scene_dict[vt]
                    _excl = {vt}
                    if _pairwise:
                        # register only against the passes vt does NOT already agree with (see _inconsistent_passes)
                        _bad, _prep, _ = _inconsistent_passes(vt, video_scene_dict, result_poses_dict_aligned, _scans_by_token, _clouds)
                        if not _bad:
                            say(f"Sweep {_sw + 1}: video {vt.split('-')[-1]} agrees with every overlapping pass ({_prep}), left as is", style="yellow")
                            continue
                        say(f"Sweep {_sw + 1}: video {vt.split('-')[-1]} disagrees with {[u.split('-')[-1] for u in _bad]} ({_prep}); registering onto them", style="yellow")
                        _excl = set(video_scene_dict) - set(_bad)
                    _map = _map_icp_inputs(video_scene_dict, result_poses_dict_aligned, _excl, _scans_by_token, _parts)
                    if _map is None or len(_map) < 5000:
                        continue
                    _cur = lambda t: result_poses_dict_aligned.get(t)
                    fixed, rep = _map_icp_pass(v, _scans_by_token, [("aligned", _cur)], _map, vt)
                    if fixed is not None:
                        _d = [np.linalg.norm(fixed[t][:3, 3] - result_poses_dict_aligned[t][:3, 3]) for t in fixed if t in result_poses_dict_aligned]
                        say(f"Sweep {_sw + 1}: video {vt.split('-')[-1]} moved onto the other passes "
                                    f"(median {np.median(_d) if _d else 0:.2f} m, max {max(_d) if _d else 0:.2f} m): {rep}", style="yellow")
                        result_poses_dict_aligned.update(fixed); _clouds.pop(vt, None)
                    else:
                        say(f"Sweep {_sw + 1}: video {vt.split('-')[-1]} left as is ({rep})", style="yellow")
            # the sweeps move passes onto each other, which can carry the whole block
            # off the INS by a common offset (synthetic: 1.4 m); re-anchor with the same
            # x/y Kabsch the block was aligned with, so fallback placement stays valid
            result_poses_dict_aligned = align_poses(video_scene_dict, result_poses_dict_aligned)
            _e = calculate_errors(video_scene_dict, result_poses_dict_aligned)
            say("After the consistency sweeps (vs INS): " + ", ".join(
                f"{vt.split('-')[-1]}: {_e[vt]['mean_error']:.2f} m" for vt in video_scene_dict), style="yellow")

        # Cross-pass consistency gate (2026-11-16, px_loc_0195: two groups of passes 3.1-3.4 m apart, each consistent
        # inside, passed every gate: the BEV / vs-INS gates scale with the block's median error, the sweeps measure each
        # pass against a map that holds its partner). Every overlapping pair of passes is measured by scan-to-map ICP of
        # one pass's scans onto the other's accumulation (_inconsistent_passes). A pair more than MTGS_QA_PASS_PAIR_STOP_M
        # (0.5 m; good blocks agree to 0.02-0.07 m) apart is repaired once: every pass that disagrees with most of the
        # passes it overlaps is registered by map ICP onto the others; what still disagrees stops the stage.
        _pair_stop = float(os.environ.get('MTGS_QA_PASS_PAIR_STOP_M', 0.5))
        if _scans_by_token and _pair_stop > 0 and len(video_scene_dict) > 1:
            def _pair_table():
                _cl, _tab = {}, {}
                for _vt in video_scene_dict:
                    for _u, _sh in _inconsistent_passes(_vt, video_scene_dict, result_poses_dict_aligned, _scans_by_token, _cl)[2].items():
                        _tab[(_vt, _u)] = _sh
                return _tab
            _id = lambda vt: vt.split('-')[-1]
            _fmt = lambda tab: ", ".join(f"{_id(a)}->{_id(b)} {sh:.2f} m" for (a, b), sh in sorted(tab.items(), key=lambda kv: -kv[1]) if sh > _pair_stop)
            _tab = _pair_table()
            if any(sh > _pair_stop for sh in _tab.values()):
                say(f"[pass QA] passes disagree by more than {_pair_stop:g} m: {_fmt(_tab)}", style="bold red")
                _mv = []
                for vt in video_scene_dict:
                    _nb = {u: sh for (a, u), sh in _tab.items() if a == vt}
                    if _nb and 2 * sum(sh > _pair_stop for sh in _nb.values()) > len(_nb):
                        _mv.append(vt)
                _ref = _aligned_reference(video_scene_dict, result_poses_dict_aligned, set(_mv))
                _map = _map_icp_inputs(video_scene_dict, result_poses_dict_aligned, set(_mv), _scans_by_token) if _mv else None
                for vt in _mv:
                    v = video_scene_dict[vt]
                    if _map is None or len(_map) < 5000:
                        break
                    _cur = {i['token']: result_poses_dict_aligned[i['token']].copy() for i in v['frame_infos']}.get
                    fixed, rep = _map_icp_pass(v, _scans_by_token, [("aligned", _cur), ("ins", _gps_poses_on_aligned_height(v, _ref).get)], _map, vt)
                    if fixed is not None:
                        _d = [float(np.linalg.norm(fixed[t][:3, 3] - result_poses_dict_aligned[t][:3, 3])) for t in fixed]
                        say(f"[pass QA] video {_id(vt)} disagrees with most passes it overlaps: registered onto the others "
                            f"(moved median {np.median(_d):.2f} m, max {max(_d):.2f} m): {rep}", style="yellow")
                        result_poses_dict_aligned.update(fixed)
                    else:
                        say(f"[pass QA] video {_id(vt)}: map ICP onto the other passes failed ({rep})", style="bold red")
                _tab = _pair_table()
                if any(sh > _pair_stop for sh in _tab.values()):
                    msg = (f"[pass QA] STOP: passes still disagree by more than MTGS_QA_PASS_PAIR_STOP_M={_pair_stop:g} m after the "
                           f"repair: {_fmt(_tab)}. The block would train ghost copies of the scene (two frames of reference).")
                    say(msg, style="bold red")
                    raise SystemExit(msg)
            say(f"[pass QA] {len(_tab)} overlapping pass pairs, largest disagreement "
                f"{max(_tab.values()) if _tab else 0:.2f} m (stop above {_pair_stop:g} m)")

        # MTGS_REG_LEVEL: 0 = full ICP, 1 = INS attitude + one constant per traversal
        # (measured worse: -1.5..-3 dB), 2 = full ICP minus the drift all traversals share
        level_mode = os.environ.get('MTGS_REG_LEVEL', '0')
        level = level_mode == '1'
        rel_tilt = _relative_tilt_offsets(video_scene_dict, result_poses_dict_aligned) if level else None
        common_drift = _common_drift(video_scene_dict, result_poses_dict_aligned) if level_mode == '2' else None
        for video_token in video_scene_dict:
            frame_infos = video_scene_dict[video_token]['frame_infos']
            for info in frame_infos:
                token = info['token']
                pose_icp = result_poses_dict_aligned[token]
                info['ego2global_original'] = info['ego2global'].copy()
                # Off by default: pinning roll/pitch to INS leaves the ~1 deg pitch
                # disagreement between traversals in place, which blurs the joint model
                # and drives the triangulator's fy down to absorb it (loc_0078: fy/fx
                # 0.86, -1.6 dB). It was a tilt experiment, never the adopted recipe.
                if os.environ.get('MTGS_REG_SE2', '0') == '1':
                    # Scan-to-map on 2 Hz lidar drifts in pitch/roll (gapC_008: 0 -> -4.3 deg
                    # pitch along every traversal, all six inheriting the same bent map), and
                    # the QA gate never looked at either. The INS is trustworthy there, so keep
                    # only what ICP is good at -- x, y, yaw -- and take z, roll, pitch from it.
                    pose_icp = _se2_constrain(pose_icp, info['ego2global_original'])
                if level:
                    pose_icp = _apply_relative_tilt(pose_icp, info['ego2global_original'],
                                                    rel_tilt[video_token])
                if common_drift is not None:
                    pose_icp = _remove_common_drift(pose_icp, common_drift[token])
                # MTGS_VEH_YAW_FIX="veh-48:1.71,veh-45:-1.34": per-vehicle lidar/camera
                # extrinsic yaw error (deg, = the constant ICP-INS heading offset those
                # vehicles show in every log and block, sd 0.1 deg). ICP aligns the lidar
                # correctly; the body and the cameras chained through lidar2ego/cam2lidar
                # inherit the extrinsic error. Keep lidar2global as ICP found it, turn the
                # body back onto the INS heading and the cameras with it.
                pose_icp, info['lidar2ego'] = _apply_vehicle_yaw_fix(
                    pose_icp, info['lidar2ego'], info.get('cams', {}), info.get('log_name', ''))
                info['ego2global'] = pose_icp
                info['ego2global_translation'] = pose_icp[:3, 3]
                info['ego2global_rotation'] = Quaternion(matrix=pose_icp[:3, :3])
                info['lidar2global'] = pose_icp @ info['lidar2ego']

    video_scene.video_scene_dict = video_scene_dict
    # after registration, we filter out low velocity frames
    video_scene_dict = video_scene.video_scene_dict_process([
        # skip frames with low velocity
        'filter_low_velocity',
        'inject_trajectory'
    ], inline=True)

    check_unrefined_passes(video_scene, 'regis', video_scene.pickle_path_registered.replace('.pkl', '_unrefined_qa_failed.pkl'))
    check_trajectory_steps(video_scene, 'regis', video_scene.pickle_path_registered.replace('.pkl', '_step_qa_failed.pkl'))
    video_scene.dump_pickle(video_scene.pickle_path_registered)
    video_scene.update_pickle_link(video_scene.pickle_path_registered)

