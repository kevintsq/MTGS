"""Height steps in an ego trajectory that the vehicle's own pitch does not explain (2026-11-16).

Between two consecutive frames (< 0.7 s apart) the ego climbs by about (horizontal distance) x (forward-axis slope);
whatever the stored trajectory climbs beyond that is a step of the pose solution, not of the road. The BEV / vs-INS
gates of the registration do not see it. Cases: ru_area0806 pass 357 (47 frames the per-frame map ICP left on the INS
shape: 2.2 m / 0.85 m steps, the road trained at several heights, the HUGSIM camera placed into it -> white fog in
scene e12142d7), st_fx02 pass 41 (an INS altitude jump of 5.8 m in 0.5 s kept as the pose).
Limits from the 2,567 passes of the 467 served blocks: the worst step per pass is <= 0.10 m for 90 %, <= 0.31 m for
99.5 %; 14 passes exceed 0.3 m and 3 exceed 1.0 m (st_fx02 41: 5.8, nb_20 32: 1.54, gapC_035 12: 1.03).
MTGS_QA_STEP_WARN_M (0.3) logs a warning, MTGS_QA_STEP_STOP_M (1.0) stops the stage; MTGS_QA_STEP_STOP_M=0 disables
the stop (the warning stays)."""
import os

import numpy as np

from nuplan_scripts.utils.console import say


def pass_steps(frame_infos, max_dt=0.7):
    """(unexplained height step per consecutive frame pair [N-1], frames sorted by time)."""
    fis = sorted(frame_infos, key=lambda f: int(f['timestamp']))
    if len(fis) < 2:
        return np.zeros(0), fis
    T = np.array([np.asarray(f['ego2global'], dtype=np.float64) for f in fis])
    ts = np.array([int(f['timestamp']) for f in fis], dtype=np.float64) / 1e6
    fw = T[:, :3, 0]
    slope = fw[:, 2] / np.maximum(np.linalg.norm(fw[:, :2], axis=1), 1e-6)
    d = np.diff(T[:, :3, 3], axis=0)
    ez = d[:, 2] - np.linalg.norm(d[:, :2], axis=1) * 0.5 * (slope[1:] + slope[:-1])
    ez[np.diff(ts) >= max_dt] = 0.0
    return ez, fis


def check_trajectory_steps(video_scene, stage, failed_path=None):
    """Log every pass with a step above the warning limit; above the stop limit write the vsd to failed_path (for
    inspection, never linked) and stop the stage."""
    warn = float(os.environ.get('MTGS_QA_STEP_WARN_M', 0.3))
    stop = float(os.environ.get('MTGS_QA_STEP_STOP_M', 1.0))
    bad, n_warn = [], 0
    for vt, v in video_scene.video_scene_dict.items():
        ez, fis = pass_steps(v['frame_infos'])
        idx = np.where(np.abs(ez) > warn)[0]
        if len(idx) == 0:
            continue
        n_warn += 1
        worst = float(np.abs(ez[idx]).max())
        lst = ", ".join(f"frames {i}->{i + 1} {ez[i]:+.2f} m" for i in idx[np.argsort(-np.abs(ez[idx]))][:6])
        say(f"[step QA {stage}] pass {vt.rsplit('-', 1)[-1]}: {len(idx)} height step(s) > {warn:g} m that the vehicle "
            f"pitch does not explain: {lst}", style="bold red" if stop > 0 and worst > stop else "yellow")
        if stop > 0 and worst > stop:
            bad.append(f"{vt.rsplit('-', 1)[-1]} ({worst:.2f} m)")
    if bad:
        if failed_path:
            video_scene.dump_pickle(failed_path)
        msg = (f"[step QA {stage}] STOP: height steps above MTGS_QA_STEP_STOP_M={stop:g} m in pass(es) {', '.join(bad)}: "
               f"the poses hold two frames of reference (see the lines above); the block would train the ground at "
               f"several heights. Fix the pose solution (or set MTGS_QA_STEP_STOP_M=0 knowingly)."
               + (f" vsd written to {failed_path} for inspection." if failed_path else ""))
        say(msg, style="bold red")
        raise SystemExit(msg)
    say(f"[step QA {stage}] {len(video_scene.video_scene_dict)} passes, {n_warn} with a step > {warn:g} m, none > {stop:g} m")
