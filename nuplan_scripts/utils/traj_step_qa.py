"""Height steps in an ego trajectory that the vehicle's own pitch does not explain (2026-11-16).

Between two consecutive frames (< 0.7 s apart) the ego climbs by about (horizontal distance) x (forward-axis slope);
whatever the stored trajectory climbs beyond that is a step of the pose solution, not of the road. The BEV / vs-INS
gates of the registration do not see it. Cases: ru_area0806 pass 357 (47 frames the per-frame map ICP left on the INS
shape: 2.2 m / 0.85 m steps, the road trained at several heights, the HUGSIM camera placed into it -> white fog in
scene e12142d7), st_fx02 pass 41 (an INS altitude jump of 5.8 m in 0.5 s kept as the pose).
Limits from the 2,567 passes of the 467 served blocks: the worst step per pass is <= 0.10 m for 90 %, <= 0.31 m for
99.5 %; 14 passes exceed 0.3 m and 3 exceed 1.0 m (st_fx02 41: 5.8, nb_20 32: 1.54, gapC_035 12: 1.03).
MTGS_QA_STEP_WARN_M (0.3) logs a warning, MTGS_QA_STEP_STOP_M (1.0) stops the stage; MTGS_QA_STEP_STOP_M=0 disables
the stop (the warning stays).
Horizontal jumps (2026-11-17, pass_hjumps): MTGS_QA_HSTEP_WARN_M (0.5) / MTGS_QA_HSTEP_STOP_M (1.0). Served blocks: worst
jump per pass p99.5 0.46 m; > 1.0 m in 4 of 2,449 final passes (gapC_011 1: 4.24, fj_gapC_037 58: 2.70, nb_20 6: 2.35,
pn_w2_06 8: 1.24)."""
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


def pass_hjumps(frame_infos, max_dt=0.7):
    """(horizontal jump of the pose solution per consecutive frame pair [N-1], frames sorted by time), 2026-11-17.

    A jump at i -> i+1 needs BOTH: the stored trajectory kinks there (horizontal second difference at frame i or i+1)
    and its displacement i -> i+1 differs from the INS displacement (ego2global_original). min() of the two: hard
    braking kinks both tracks alike (difference ~0), an INS glitch the registration removed leaves the stored track
    smooth (second difference ~0). Cases: ml_bos02 pass 2 (3.49 m at frame 0 -> 1 and 1-2 m inside the pass), ml_sg03
    pass 6 (2.6 m along the road at its end) - the per-frame map ICP of MTGS 7c4a6d8, invisible to the height check.
    Zeros where the INS pose is missing or a gap is >= max_dt."""
    fis = sorted(frame_infos, key=lambda f: int(f['timestamp']))
    n = len(fis)
    if n < 2:
        return np.zeros(0), fis
    if any('ego2global_original' not in f for f in fis):
        return np.zeros(n - 1), fis
    p = np.array([np.asarray(f['ego2global'], dtype=np.float64)[:2, 3] for f in fis])
    q = np.array([np.asarray(f['ego2global_original'], dtype=np.float64)[:2, 3] for f in fis])
    ts = np.array([int(f['timestamp']) for f in fis], dtype=np.float64) / 1e6
    ok = np.diff(ts) < max_dt
    e = np.linalg.norm(np.diff(p, axis=0) - np.diff(q, axis=0), axis=1)          # vs the INS motion
    a = np.zeros(n)                                                                   # kink of the stored track
    if n >= 3:
        a[1:-1] = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) * (ok[1:] & ok[:-1])
    kink = np.maximum(a[:-1], a[1:])
    if n == 2:
        kink = e.copy()
    h = np.minimum(e, kink)
    h[~ok] = 0.0
    return h, fis


def repair_short_hjumps(video_scene, stage):
    """Make the registered track continuous where it jumps against the INS motion (2026-11-17), before the step QA.

    A pass is split into runs at every frame pair whose horizontal jump (pass_hjumps) exceeds MTGS_QA_HSTEP_REPAIR_M
    (0.5 m). The correction of a frame is its pose against its INS pose; the INS moves the frames of a pass relative to
    each other to centimetres, so that correction varies slowly along a correct track.
    - a run of at most MTGS_QA_HSTEP_REPAIR_MAX (8) frames that is not the longest run is a stray piece of the pose
      solution (ml_bos02 pass 2 frame 0: the KISS start frame 3.5 m off frames 1-3, left by every later stage; ml_sg03
      pass 6: its last frames slid 2.6 m along the road): its correction is replaced by the neighbours' correction,
      interpolated along the path (one side at a pass end), as for unrefined frames;
    - a jump between two longer runs is a switch of the correction field (ml_bos02 pass 2 frames 57-59: the per-frame
      map ICP followed a 3.5 m change of the INS error over 50 m in a few 1-2 m switches): the corrections within
      MTGS_QA_HSTEP_RAMP_M (15 m) of it are replaced by the ramp between the corrections at the window's ends.
    An INS glitch the registration removed leaves the registered track smooth and is not touched (pass_hjumps needs a
    kink of the registered track). MTGS_QA_HSTEP_REPAIR_M=0 disables."""
    from pyquaternion import Quaternion
    from scipy.spatial.transform import Rotation as _R
    thr = float(os.environ.get('MTGS_QA_HSTEP_REPAIR_M', 0.5))
    mx = int(os.environ.get('MTGS_QA_HSTEP_REPAIR_MAX', 8))
    ramp = float(os.environ.get('MTGS_QA_HSTEP_RAMP_M', 15.0))
    if thr <= 0:
        return
    def _state(fis):
        P = np.array([np.asarray(f['ego2global'], dtype=np.float64) for f in fis])
        Q = np.array([np.asarray(f['ego2global_original'], dtype=np.float64) for f in fis])
        D = np.einsum('nij,njk->nik', P, np.linalg.inv(Q))                    # correction against the INS pose
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q[:, :2, 3], axis=0), axis=1))]
        return Q, _R.from_matrix(D[:, :3, :3]).as_rotvec(), D[:, :3, 3].copy(), s

    def _write(fis, Q, rv, tr, idx):
        for i in idx:
            M = np.eye(4); M[:3, :3] = _R.from_rotvec(rv[i]).as_matrix(); M[:3, 3] = tr[i]
            pose = M @ Q[i]; f = fis[i]
            f['ego2global'] = pose; f['ego2global_translation'] = pose[:3, 3]
            f['ego2global_rotation'] = Quaternion(matrix=pose[:3, :3], atol=1e-6)
            if 'lidar2ego' in f:
                f['lidar2global'] = pose @ np.asarray(f['lidar2ego'], dtype=np.float64)

    for vt, v in video_scene.video_scene_dict.items():
        msgs = []
        # (1) short stray runs, once
        h, fis = pass_hjumps(v['frame_infos'])
        cut = [int(c) for c in np.where(h > thr)[0]]
        if not cut or len(fis) < 3:
            continue
        runs, s0 = [], 0
        for c in cut:
            runs.append((s0, c)); s0 = c + 1
        runs.append((s0, len(fis) - 1))
        k = max(range(len(runs)), key=lambda r: runs[r][1] - runs[r][0])
        short = [r for j, r in enumerate(runs) if j != k and r[1] - r[0] + 1 <= mx]
        if short:
            Q, rv, tr, s = _state(fis)
            bad = np.zeros(len(fis), bool)
            for a, b in short:
                bad[a:b + 1] = True
            good = np.where(~bad)[0]
            if len(good):
                idx = np.where(bad)[0]
                for i in idx:
                    lo = good[good < i]; hi = good[good > i]
                    if len(lo) and len(hi):
                        l, u = lo[-1], hi[0]; w = (s[i] - s[l]) / max(s[u] - s[l], 1e-6)
                    else:
                        l = u = (lo[-1] if len(lo) else hi[0]); w = 0.0
                    rv[i] = (1 - w) * rv[l] + w * rv[u]; tr[i] = (1 - w) * tr[l] + w * tr[u]
                _write(fis, Q, rv, tr, idx)
                msgs += [f"frames {a}-{b} re-carried from their neighbours" for a, b in short]
        # (2) whatever still jumps is a switch of the correction field: ramp it
        for _ in range(3):
            if ramp <= 0:
                break
            h, fis = pass_hjumps(v['frame_infos'])
            cut = [int(c) for c in np.where(h > thr)[0]]
            if not cut:
                break
            Q, rv, tr, s = _state(fis)
            rv0, tr0 = rv.copy(), tr.copy(); touched = set()
            for c in cut:
                sc = 0.5 * (s[c] + s[c + 1])
                win = np.where(np.abs(s - sc) <= ramp)[0]
                lo = np.where(s < sc - ramp)[0]; hi = np.where(s > sc + ramp)[0]
                l = int(lo[-1]) if len(lo) else int(win[0]); u = int(hi[0]) if len(hi) else int(win[-1])
                for i in win:
                    w = (s[i] - s[l]) / max(s[u] - s[l], 1e-6)
                    rv[i] = (1 - w) * rv0[l] + w * rv0[u]; tr[i] = (1 - w) * tr0[l] + w * tr0[u]
                touched |= set(int(i) for i in win)
                msgs.append(f"switch at frames {c}->{c + 1} ({h[c]:.2f} m) ramped over frames {win[0]}-{win[-1]}")
            _write(fis, Q, rv, tr, sorted(touched))
        if msgs:
            say(f"[step QA {stage}] pass {vt.rsplit('-', 1)[-1]}: track jumps against the INS motion repaired: "
                + "; ".join(msgs), style="yellow")


def check_trajectory_steps(video_scene, stage, failed_path=None):
    """Log every pass with a step above the warning limit; above the stop limit write the vsd to failed_path (for
    inspection, never linked) and stop the stage."""
    warn = float(os.environ.get('MTGS_QA_STEP_WARN_M', 0.3))
    stop = float(os.environ.get('MTGS_QA_STEP_STOP_M', 1.0))
    hwarn = float(os.environ.get('MTGS_QA_HSTEP_WARN_M', 0.5))
    hstop = float(os.environ.get('MTGS_QA_HSTEP_STOP_M', 1.0))
    bad, n_warn, hbad, n_hwarn = [], 0, [], 0
    for vt, v in video_scene.video_scene_dict.items():
        hj, _ = pass_hjumps(v['frame_infos'])
        hidx = np.where(hj > hwarn)[0]
        if len(hidx):
            n_hwarn += 1
            hworst = float(hj[hidx].max())
            hlst = ", ".join(f"frames {i}->{i + 1} {hj[i]:.2f} m" for i in hidx[np.argsort(-hj[hidx])][:6])
            say(f"[step QA {stage}] pass {vt.rsplit('-', 1)[-1]}: {len(hidx)} horizontal jump(s) > {hwarn:g} m of the pose "
                f"solution (the track kinks where the INS motion does not): {hlst}",
                style="bold red" if hstop > 0 and hworst > hstop else "yellow")
            if hstop > 0 and hworst > hstop:
                hbad.append(f"{vt.rsplit('-', 1)[-1]} ({hworst:.2f} m)")
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
    if bad or hbad:
        if failed_path:
            video_scene.dump_pickle(failed_path)
        parts = []
        if bad:
            parts.append(f"height steps above MTGS_QA_STEP_STOP_M={stop:g} m in pass(es) {', '.join(bad)} (the block would "
                         f"train the ground at several heights)")
        if hbad:
            parts.append(f"horizontal jumps above MTGS_QA_HSTEP_STOP_M={hstop:g} m in pass(es) {', '.join(hbad)} (ghost "
                         f"copies along the jump)")
        msg = (f"[step QA {stage}] STOP: " + "; ".join(parts) + ": the poses hold two frames of reference (see the lines "
               "above). Fix the pose solution (or set MTGS_QA_STEP_STOP_M / MTGS_QA_HSTEP_STOP_M=0 knowingly)."
               + (f" vsd written to {failed_path} for inspection." if failed_path else ""))
        say(msg, style="bold red")
        raise SystemExit(msg)
    say(f"[step QA {stage}] {len(video_scene.video_scene_dict)} passes, {n_warn} with a step > {warn:g} m, none > {stop:g} m; "
        f"{n_hwarn} with a horizontal jump > {hwarn:g} m, none > {hstop:g} m")


def check_unrefined_passes(video_scene, stage, failed_path=None, warn_only=False):
    """Passes the registration left on their INS poses (2026-11-16): x / y equal to the INS ('ego2global_original') on
    more than MTGS_QA_UNREFINED_FRAC (0.9) of the frames. The height harmonisation moves z only, so such a pass looks
    registered while no lidar ever touched it: 9 served passes were exactly INS (single-pass st_ blocks, fp_bx_08, ...)
    and two more INS in x / y with a harmonised z (nb_20 pass 32, fq_loc_0002 pass 72, e12142d7's own pass, 0.33-0.69 m
    off the block's other passes). x / y only: the vehicle-yaw fix may rotate an otherwise untouched pass.
    Stops the stage (vsd written to failed_path), or only warns when keeping the INS was asked for (MTGS_SKIP_ICP=1)."""
    frac_lim = float(os.environ.get('MTGS_QA_UNREFINED_FRAC', 0.9))
    bad = []
    for vt, v in video_scene.video_scene_dict.items():
        fis = [f for f in v['frame_infos'] if 'ego2global_original' in f]
        if not fis:
            continue
        d = np.array([np.linalg.norm(np.asarray(f['ego2global'], dtype=np.float64)[:2, 3]
                                     - np.asarray(f['ego2global_original'], dtype=np.float64)[:2, 3]) for f in fis])
        frac = float(np.mean(d < 1e-4))
        if frac > frac_lim:
            bad.append(f"{vt.rsplit('-', 1)[-1]} ({frac * 100:.0f} % of {len(fis)} frames)")
    if not bad:
        say(f"[unrefined QA {stage}] every pass was moved off its INS poses by the registration")
        return
    msg = (f"[unrefined QA {stage}] pass(es) {', '.join(bad)} still on their INS poses in x / y: no lidar registration "
           f"reached them (a z-only harmonisation does not count).")
    if warn_only:
        say(msg + " Kept: MTGS_SKIP_ICP=1 asks for the INS poses.", style="bold yellow")
        return
    if failed_path:
        video_scene.dump_pickle(failed_path)
    msg += (" Fix the registration of that pass (or set MTGS_QA_UNREFINED_FRAC=1 knowingly)."
            + (f" vsd written to {failed_path} for inspection." if failed_path else ""))
    say(msg, style="bold red")
    raise SystemExit(msg)
