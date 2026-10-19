#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
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
        CONSOLE.log(f"[level] {vt.split('-')[-1]}: pitch {np.degrees(r[0]):+.2f} deg, "
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
    CONSOLE.log(f"[level2] common drift over the block: pitch {np.degrees(c[:,0].min()):+.2f}..{np.degrees(c[:,0].max()):+.2f} deg, "
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
    print(table)
    return error_dict

_LAST_ALIGN_T = np.eye(4)

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
        CONSOLE.log(f"[veh-yaw-fix] {veh}: body and cameras rotated {-_VEH_FIX[veh]:+.2f} deg about lidar z "
                    f"(lidar2global unchanged) for log {log_name}", style="yellow")
    return E_new, L_new

def _is_pinned(video):
    pins = [x for x in os.environ.get('MTGS_PIN_LOGS', '').split(',') if x]
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
    kiss_icp_pipeline.run()

    # The registritaion works on the first frames coordinate.
    # We translate the poses back to the global coordinate here.
    result_poses = kiss_icp_dataset.apply_calibration(kiss_icp_pipeline.poses)
    tokens = kiss_icp_dataset.tokens
    result_poses_dict = {token: pose for token, pose in zip(tokens, result_poses)}

    if os.environ.get('MTGS_SKIP_ICP') == '1':
        # KISS-ICP is scan-to-map odometry meant for ~10Hz sequences. nuPlan
        # sensor data is 2Hz (4-6m between scans) and the Hazelwood blocks are
        # straight roads with grass verges: nothing constrains motion ALONG the
        # road, so the estimate slides (86m end-point error). KISS-Matcher, a
        # global method, calls the same pairs either INVALID or a 0.57m
        # correction -- i.e. the GPS/IMU poses were already right and there is
        # no geometry here to refine against. Keep them.
        CONSOLE.log("MTGS_SKIP_ICP=1: keeping GPS/IMU poses, no ICP refinement",
                    style="bold yellow")
        for video_token in video_scene_dict:
            for info in video_scene_dict[video_token]['frame_infos']:
                info['ego2global_original'] = info['ego2global'].copy()
        video_scene.video_scene_dict = video_scene_dict
        video_scene_dict = video_scene.video_scene_dict_process(
            ['filter_low_velocity', 'inject_trajectory'], inline=True)
        video_scene.dump_pickle(video_scene.pickle_path_registered)
        video_scene.update_pickle_link(video_scene.pickle_path_registered)
        sys.exit(0)

    CONSOLE.log("ICP errors", style="bold green")
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
    outliers = [vt for vt in list(video_scene_dict.keys())
                if vt not in keep_first
                and (error_dict[vt]['max_end_point_error'] > PRE_EPE
                     or error_dict[vt]['mean_error'] > PRE_MEAN)]
    pre_dropped = {}
    if outliers and len(outliers) < len(video_scene_dict):
        for vt in outliers:
            CONSOLE.log(f"Drop video {vt.split('-')[-1]} before alignment (raw ICP outlier).",
                        style="yellow")
            pre_dropped[vt] = video_scene_dict.pop(vt)

    CONSOLE.log("ICP errors after alignment", style="bold green")
    result_poses_dict_aligned = align_poses(video_scene_dict, result_poses_dict)
    error_dict_aligned = calculate_errors(video_scene_dict, result_poses_dict_aligned)

    # Sliders (large translation, tiny rotation) are never refinement, so revert
    # them to GPS poses whenever the fallback is enabled; the count-based rescue
    # below then decides whether any OTHER reject also has to be kept.
    fb_on = os.environ.get('MTGS_ICP_FALLBACK_ENABLE', '1') == '1'
    fb_max = float(os.environ.get('MTGS_ICP_FALLBACK_MAX', 2.0)) if fb_on else 0.0
    if fb_max > 0:
        touched = False
        for vt in list(video_scene_dict.keys()):
            e = error_dict_aligned[vt]
            if e['mean_error'] > fb_max and e['mean_heading_error'] < 2.0:
                CONSOLE.log(f"Traversal {vt.split('-')[-1]}: ICP moved it "
                            f"{e['mean_error']:.1f}m with only "
                            f"{e['mean_heading_error']:.1f} deg of rotation change "
                            f"(slide, not refinement) -- keeping the GPS pose.",
                            style="yellow")
                for info in video_scene_dict[vt]['frame_infos']:
                    result_poses_dict_aligned[info['token']] = info['ego2global'].copy()
                touched = True
        if touched:
            error_dict_aligned = calculate_errors(video_scene_dict, result_poses_dict_aligned)
    visualize_lidar_registration(video_scene, result_poses_dict_aligned, filename="registration_aligned.pdf")

    # filter out bad video_info
    bad_videos = []
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
        if (max_end_point_error > float(os.environ.get('MTGS_REG_QA_EPE', 1.0))
                or mean_error > float(os.environ.get('MTGS_REG_QA_MEAN', 0.5))
                or mean_heading_error > are_gate):
            bad_videos.append(video_token)

    if isinstance(config, FrameCentralConfig) and list(video_scene_dict.keys())[0] in bad_videos:
        CONSOLE.log("WARNING: The central log has bad registration result. Use log pose and disable multi-traversal.", style="bold red")
        first_key = list(video_scene_dict.keys())[0]
        video_scene_dict = {first_key: video_scene_dict[first_key]}
    else:
        # Bring back pre-alignment casualties on GPS poses if the block needs
        # them. They skip alignment entirely, so the fit the survivors got is
        # untouched -- this only adds coverage that would otherwise be lost.
        if pre_dropped:
            target = int(os.environ.get('MTGS_MIN_TRAVERSALS', 0))
            short = target - (len(video_scene_dict) - len(bad_videos))
            picks = sorted(pre_dropped.items(),
                           key=lambda kv: (not _is_pinned(kv[1]),
                                           -len(kv[1]['frame_infos'])))
            n_keep = max(0, short)
            chosen = picks[:n_keep] + [kv for kv in picks[n_keep:] if _is_pinned(kv[1])]
            # MTGS_PIN_KEEP_ICP=1: a pinned pass (the scene's own drive) comes back on its
            # ICP poses carried through the global fit, not on GPS. Experiment for the
            # rainy own passes of fl_loc_0252 / fl_loc_0261, which on GPS poses left a
            # grey blob / smeared route (ICP had moved them 1-4.6 m; GPS on a rainy day).
            keep_icp = os.environ.get('MTGS_PIN_KEEP_ICP', '0') == '1'
            for vt, v in chosen:
                video_scene_dict[vt] = v
                if keep_icp and _is_pinned(v):
                    CONSOLE.log(f"Restoring video {vt.split('-')[-1]} on its ICP poses through the "
                                f"global alignment ({len(v['frame_infos'])} frames, MTGS_PIN_KEEP_ICP).",
                                style="yellow")
                    for info in v['frame_infos']:
                        result_poses_dict_aligned[info['token']] = _LAST_ALIGN_T @ result_poses_dict[info['token']]
                    continue
                CONSOLE.log(f"Restoring video {vt.split('-')[-1]} on GPS poses "
                            f"({len(v['frame_infos'])} frames): dropped before alignment, "
                            f"but the block would be short without it.", style="yellow")
                for info in v['frame_infos']:
                    result_poses_dict_aligned[info['token']] = info['ego2global'].copy()

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
            for video_token in rescued:
                e = error_dict_aligned[video_token]
                if os.environ.get('MTGS_PIN_KEEP_ICP', '0') == '1' and _is_pinned(video_scene_dict[video_token]):
                    CONSOLE.log(f"Keeping video {video_token.split('-')[-1]} on its aligned ICP poses "
                                f"(QA said mean {e['mean_error']:.2f}m / {e['mean_heading_error']:.2f} deg; "
                                f"MTGS_PIN_KEEP_ICP).", style="yellow")
                    continue
                CONSOLE.log(f"Keeping video {video_token.split('-')[-1]} on GPS poses "
                            f"(QA said mean {e['mean_error']:.2f}m / "
                            f"{e['mean_heading_error']:.2f} deg, but the block would "
                            f"otherwise be short).", style="yellow")
                for info in video_scene_dict[video_token]['frame_infos']:
                    result_poses_dict_aligned[info['token']] = info['ego2global'].copy()
            for video_token in bad_videos:
                if video_token in rescued:
                    continue
                CONSOLE.log(f"Remove video {video_token.split('-')[-1]} due to bad registration result.", style="yellow")
                video_scene_dict.pop(video_token)
            if len(video_scene_dict) == 0:
                raise ValueError("All videos have bad registration result.")

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

    video_scene.dump_pickle(video_scene.pickle_path_registered)
    video_scene.update_pickle_link(video_scene.pickle_path_registered)

