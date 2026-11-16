"""Look-ahead coverage of served scenes (2026-11-16, user rule).

The route-coverage rule (every route point within 12-15 m of trained frames) says nothing about what the cameras SEE
beyond the route: ml_pit01 / 442ce328 ended where its own log ended and nothing in the block had been trained on the
road ahead -> the last frames rendered a smeared, empty street. A served scene also needs trained frames on

  ahead   the road 20-50 m past its route end,
  behind  the road 20-50 m before its route start,
  side    side streets / cross roads within 8-30 m of its route,

measured ALONG THE ROAD: the regions are points of driven trajectories (any pass of the block's logs). "ahead" walks
each trajectory that passes the route end (within END_R m) in the route's direction of travel and keeps the stretch
20-50 m of path length further on; "behind" walks against it from the route start; "side" keeps trajectory points
8-30 m from the route polyline.

regions(route, trajs)            -> {"ahead": [k,2], "behind": [k,2], "side": [k,2]} (global xy)
coverage(region, frames_xy, r)   -> frames within r m per region point
CLI audit on a trained block:  python -m nuplan_scripts.auto_reconstruction.lookahead <vsd.pkl> <served_routes.json>
    [--min_frames 5] [--radius 15]  -> per scene: share of each region's points with >= min_frames trained frames
    within radius, and the share seen by another day's pass.
"""
import argparse
import json
import pickle

import numpy as np

AHEAD = (20.0, 50.0)       # path length past the route end / before the route start, m
SIDE = (8.0, 30.0)         # distance from the route polyline, m
END_R = 8.0                # a trajectory takes part in "ahead"/"behind" when it passes within END_R m of the route end/start


def _dist_to_polyline(p, line):
    """Distance of points p [n,2] to the polyline line [m,2]."""
    if len(line) == 1:
        return np.linalg.norm(p - line[0], axis=1)
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    q = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(p[:, None, :] - q, axis=2).min(1)


def _heading(route, at_end, span=5.0):
    """Unit direction of travel at the route end (or start), over the last (first) `span` m."""
    r = route[::-1] if at_end else route
    d = np.cumsum(np.r_[0, np.linalg.norm(np.diff(r, axis=0), axis=1)])
    j = int(np.searchsorted(d, span)) if d[-1] >= span else len(r) - 1
    v = (r[0] - r[j]) if at_end else (r[j] - r[0])
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _walk(traj, i0, step, lo, hi):
    """Points of traj from index i0 in direction step whose path length from i0 lies in [lo, hi]."""
    out, acc, i = [], 0.0, i0
    while 0 <= i + step < len(traj):
        acc += float(np.linalg.norm(traj[i + step] - traj[i]))
        i += step
        if acc > hi:
            break
        if acc >= lo:
            out.append(traj[i])
    return out


def regions(route, trajs):
    """Look-ahead regions of one served route (global xy [n,2]) from driven trajectories trajs ([m_i,2] each)."""
    route = np.asarray(route, float)[:, :2]
    out = {"ahead": [], "behind": [], "side": []}
    if len(route) < 2:
        return {k: np.zeros((0, 2)) for k in out}
    for name, at_end in (("ahead", True), ("behind", False)):
        h = _heading(route, at_end)
        if h is None:
            continue
        anchor = route[-1] if at_end else route[0]
        want = h if at_end else -h                     # walk forward past the end, backward before the start
        for tr in trajs:
            tr = np.asarray(tr, float)[:, :2]
            if len(tr) < 2:
                continue
            d = np.linalg.norm(tr - anchor, axis=1)
            j = int(d.argmin())
            if d[j] > END_R:
                continue
            nb = tr[min(j + 1, len(tr) - 1)] - tr[max(j - 1, 0)]
            step = 1 if float(nb @ want) >= 0 else -1
            out[name] += _walk(tr, j, step, *AHEAD)
    allp = np.concatenate([np.asarray(t, float)[:, :2] for t in trajs if len(t)]) if trajs else np.zeros((0, 2))
    if len(allp):
        near = np.linalg.norm(allp - route.mean(0), axis=1) < np.linalg.norm(route - route.mean(0), axis=1).max() + SIDE[1] + 1
        cand = allp[near]
        if len(cand):
            dd = _dist_to_polyline(cand, route)
            out["side"] = list(cand[(dd > SIDE[0]) & (dd <= SIDE[1])])
    return {k: (np.unique(np.round(np.asarray(v, float), 1), axis=0) if len(v) else np.zeros((0, 2))) for k, v in out.items()}


def coverage(region, frames_xy, radius=15.0):
    """Trained frames within `radius` m of each region point."""
    if len(region) == 0 or len(frames_xy) == 0:
        return np.zeros(len(region), dtype=int)
    from scipy.spatial import cKDTree
    return np.array([len(x) for x in cKDTree(frames_xy).query_ball_point(region, radius)], dtype=int)


def audit(vsd, routes, min_frames=5, radius=15.0, scene_logs=None, tracks=None):
    """Per scene and region: points, share with >= min_frames trained frames within radius, share covered by a pass of
    another log (another day) than the scene's own (scene_logs: {scene: log name}, optional). tracks: driven
    trajectories of ALL logs around the block ({log: [n,2]} global xy, e.g. TTA vis_compare/log_xy_cache.npz) to build
    the regions from -- with only the block's own frames a stretch nobody in the block drove has no region at all."""
    xy, logs, trajs = [], [], []
    for v in vsd.values():
        g = np.asarray(v["global2world_translation"], float)[:2]
        p = np.array([np.asarray(fi["ego2global"], float)[:2, 3] + g for fi in v["frame_infos"]]) if v["frame_infos"] else np.zeros((0, 2))
        xy.append(p); logs += [v["log_name"]] * len(p); trajs.append(p)
    xy = np.concatenate(xy) if xy else np.zeros((0, 2)); logs = np.array(logs)
    res = {}
    if tracks:
        lo, hi = xy.min(0) - 200, xy.max(0) + 200
        trajs = [np.asarray(t, float)[:, :2] for t in tracks.values() if len(t) and ((np.asarray(t)[:, :2] > lo) & (np.asarray(t)[:, :2] < hi)).all(1).any()]
    for scene, r in routes.items():
        reg = regions(np.asarray(r, float), trajs)
        row = {}
        for name, pts in reg.items():
            c = coverage(pts, xy, radius)
            ent = dict(points=int(len(pts)), covered=float((c >= min_frames).mean()) if len(pts) else None,
                       median_frames=float(np.median(c)) if len(pts) else None)
            own = (scene_logs or {}).get(scene)
            if own is not None and len(pts):
                c2 = coverage(pts, xy[logs != own], radius)
                ent["other_log_covered"] = float((c2 >= min_frames).mean())
            row[name] = ent
        res[scene] = row
    return res


def main():
    ap = argparse.ArgumentParser(description="look-ahead coverage audit of a trained block")
    ap.add_argument("vsd"); ap.add_argument("served_routes")
    ap.add_argument("--min_frames", type=int, default=5); ap.add_argument("--radius", type=float, default=15.0)
    ap.add_argument("--scene_logs", default="", help="json {scene: own log name} for the other-day share")
    ap.add_argument("--tracks", default="", help="npz with d = {log: [n,2] global xy} of all logs (regions along every driven road)")
    a = ap.parse_args()
    vsd = pickle.load(open(a.vsd, "rb"))
    routes = {k: np.asarray(v, float) for k, v in json.load(open(a.served_routes)).items()}
    sl = json.load(open(a.scene_logs)) if a.scene_logs else None
    tr = np.load(a.tracks, allow_pickle=True)["d"].item() if a.tracks else None
    res = audit(vsd, routes, a.min_frames, a.radius, sl, tracks=tr)
    bad = 0
    for scene, row in res.items():
        flags = [f"{k} {v['covered']:.2f}" + (f" (other log {v['other_log_covered']:.2f})" if "other_log_covered" in v else "")
                 for k, v in row.items() if v["covered"] is not None]
        low = any(v["covered"] is not None and v["covered"] < 0.5 for v in row.values())
        bad += low
        print(f"{scene}: " + ", ".join(flags) + ("   <-- look-ahead gap" if low else ""))
    print(f"{bad} of {len(res)} scenes with a region < 50 % covered (>= {a.min_frames} trained frames within {a.radius:g} m)")


if __name__ == "__main__":
    main()
