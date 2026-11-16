#-------------------------------------------------------------------------------#
# Automated traversal selection for batch multi-traversal reconstruction.       #
#                                                                               #
# Replaces the manual `preview.py` step of the official MTGS flow: the paper    #
# experiments hand-pick 2-6 traversals per road block; at 380 blocks that is    #
# not feasible, so we (1) run nuplan_video_processing WITHOUT prefilter to      #
# discover every in-block traversal (db-only, no GPU), then (2) pick up to      #
# --max_traversals per block and write them back into the config yaml, so the   #
# subsequent preprocess.sh --prefilter run only processes the selection.        #
#                                                                               #
# Selection policy:                                                             #
#   1. traversals from the location's NAVSIM navtest logs first (their BA-      #
#      refined poses + reconstructed actors are needed for HUGSIM scenarios),   #
#      preferring more in-block frames;                                         #
#   2. remaining slots filled with other traversals maximising start-time       #
#      spread (date/lighting diversity), preferring more frames on ties;       #
#   3. blocks with fewer traversals than the cap keep everything.               #
#-------------------------------------------------------------------------------#
from nuplan_scripts.utils.console import say
import os
import json
import argparse

import numpy as np
import jsonlines

from nuplan_scripts.utils.config import load_config

MIN_FRAMES = 50  # 10Hz lidar_pcs (~5s in block); drops clips that barely touch it


def discover_video_stats(config, num_workers):
    """All in-block traversal segments, db-only (no per-frame sensor-file work).

    This is the cheap half of nuplan_video_processing: split each log's
    lidar_pc stream into in-block videos. The expensive ProduceVideoSceneDict
    (per-frame camera association + NFS file stats) is deferred to the real
    preprocessing, which then only sees the <=K selected traversals.
    """
    from nuplan_scripts.nuplan_video_processing import SplitVideoFromNuPlanDB, sort_video_infos
    from nuplan_scripts.utils.constants import NUPLAN_DB_FILES

    road_block = np.asarray(config.road_block)
    with jsonlines.open(os.path.join(config.data_root, "nuplan_log_infos.jsonl")) as reader:
        log_infos = {item["log_name"]: item for item in reader}

    db_files = []
    for log_name, item in log_infos.items():
        if config.split != "all" and item["split"] != config.split:
            continue
        traj = np.asarray(item["trajectory"])
        in_region = np.all(traj > road_block[:2], axis=1) & np.all(traj < road_block[2:], axis=1)
        if in_region.any():
            db_files.append(os.path.join(NUPLAN_DB_FILES, f"{log_name}.db"))

    _, video_infos = SplitVideoFromNuPlanDB.run(
        config=config, db_filenames=db_files, num_workers=num_workers)
    video_infos = sort_video_infos(config, video_infos)
    return [{"idx": v["video_idx"], "log": v["log_name"],
             "n": len(v["lidar_pcs"]), "ts": int(v["start_ts"]),
             "ts_end": int(v["lidar_pcs"][-1].timestamp),
             "traj": np.asarray(v["trajectory"], dtype=float)[:, :2]}
            for v in video_infos]


def select(stats, test_logs, k, require_logs=()):
    # MTGS_APPEARANCE_PER_LOG=1 (block-env flag, 2026-11-16, user: "按 log 数算"): k counts distinct LOGS, not passes --
    # a log crossing the block twice gives two passes that share one appearance row in training (same day, minutes
    # apart), so its second pass costs images, not a day. MTGS_MAX_PASSES (default 9 = 6 logs x 1.5) bounds the passes,
    # i.e. the training images, so the coverage fill cannot keep adding passes of logs it already holds. Flag off: k
    # counts passes exactly as before.
    # "Log" here is the DRIVE (user 2026-11-16): the NAVSIM splits of one recorded drive count as one (drive_key).
    from nuplan_scripts.utils.log_names import drive_key
    per_log = os.environ.get('MTGS_APPEARANCE_PER_LOG', '0') == '1'
    max_passes = int(os.environ.get('MTGS_MAX_PASSES', 9)) if per_log else k
    def _n_logs(ss):
        return len({drive_key(s_["log"]) for s_ in ss})
    def _can_add(chosen_, s_):
        if not per_log:
            return len(chosen_) < k
        return len(chosen_) < max_passes and (drive_key(s_["log"]) in {drive_key(c["log"]) for c in chosen_} or _n_logs(chosen_) < k)
    def _full(chosen_):
        if not per_log:
            return len(chosen_) >= k
        return len(chosen_) >= max_passes or (_n_logs(chosen_) >= k and not any(
            drive_key(s_["log"]) in {drive_key(c["log"]) for c in chosen_} and s_["idx"] not in {c["idx"] for c in chosen_} for s_ in stats))
    # Bad-weather logs poison a reconstruction (wet-road spray bakes into the
    # background; a clear scene renders rainy). Exclusion is a hard filter,
    # overriding even require_logs -- prefer clear multi-traversal over a
    # faithful-but-rainy own log.
    windows = {}
    for item in os.environ.get('MTGS_REQUIRE_WINDOWS', '').split(';'):
        if '=' in item:
            lg, span = item.split('=', 1); t0, t1 = span.split('-')
            windows.setdefault(lg, []).append((int(t0), int(t1)))
    def _required(s):
        if s["log"] not in set(require_logs):
            return False
        if s["log"] not in windows:
            return True
        return any(s["ts"] <= t1 and s.get("ts_end", s["ts"]) >= t0 for t0, t1 in windows[s["log"]])
    # MTGS_EXCLUDE_EXEMPT_REQUIRED=1: the one pass that holds a served scene stays
    # even when its log is bad-weather. Needed where the clear traversals leave
    # the scene's route thin (<20 frames within 15 m: bx_03b/218dd135, loc_0261,
    # gap_13 ...): the own pass is then the only observation of the route end.
    exempt = os.environ.get('MTGS_EXCLUDE_EXEMPT_REQUIRED', '0') == '1'
    excl = tuple(x for x in os.environ.get('MTGS_EXCLUDE_LOGS', '').split(',') if x)
    if excl:
        dropped = [s for s in stats if s["log"].startswith(excl) and not (exempt and _required(s))]
        kept = [s for s in stats if s["log"].startswith(excl) and exempt and _required(s)]
        if dropped:
            say(f"[auto_select] excluding {len(dropped)} bad-weather traversals "
                  f"({len(set(s['log'] for s in dropped))} logs)", flush=True)
        if kept:
            say(f"[auto_select] keeping {len(kept)} bad-weather pass(es) that hold a served "
                  f"scene (MTGS_EXCLUDE_EXEMPT_REQUIRED): {sorted(s['idx'] for s in kept)}", flush=True)
        drop_idx = {s["idx"] for s in dropped}
        stats = [s for s in stats if s["idx"] not in drop_idx]
        if not exempt:
            gone = [r for r in require_logs if r.startswith(excl)]
            if gone:
                # the documented rule (user 2026-08-25: exclusion overrides require_logs unless
                # MTGS_EXCLUDE_EXEMPT_REQUIRED=1) -- applied, but never silently (user rule 2026-11-16)
                say(f"[auto_select] WARNING: {len(gone)} required log(s) excluded as bad weather / lens "
                    f"(MTGS_EXCLUDE_LOGS; set MTGS_EXCLUDE_EXEMPT_REQUIRED=1 to keep a served scene's pass): {gone}",
                    style="bold yellow")
            require_logs = tuple(r for r in require_logs if not r.startswith(excl))
    # A required pass is a served scene's own pass: it is kept even when it is shorter than MIN_FRAMES (the cut is
    # for clips that barely touch the block; before 2026-11-16 a short required pass fell out here without a word).
    short_req = [s for s in stats if s["n"] < MIN_FRAMES and _required(s)]
    if short_req:
        say(f"[auto_select] keeping {len(short_req)} required pass(es) shorter than {MIN_FRAMES} frames: "
            f"{[(s['log'], s['n']) for s in short_req]}", style="bold yellow")
    usable = [s for s in stats if s["n"] >= MIN_FRAMES or _required(s)] or stats
    # A block serving a specific scene must reconstruct the road that scene
    # drives, so its own log takes a slot before the by-length ranking runs;
    # otherwise a location with 20 test logs keeps only the six longest and the
    # target route can end up unobserved.
    # A log can pass through the block several times; only the pass that holds
    # the served scene is required (MTGS_REQUIRE_WINDOWS: "log=t0-t1;...", us).
    # Without windows every pass of a required log takes a slot, and a block with
    # four served logs had no slot left for covering the routes (loc_0093).
    req = sorted([s for s in usable if _required(s)], key=lambda s: -s["n"])
    # A required log WITHOUT a window keeps only its longest pass as required (2026-11-03): with every pass required,
    # gap3_01's three remaining logs (2 passes each) filled all six slots -> three logs' worth of viewpoints, blurry
    # far buildings. Its other passes still compete for the free slots below. MTGS_REQUIRE_ALL_PASSES=1: old rule.
    if os.environ.get('MTGS_REQUIRE_ALL_PASSES', '0') != '1':
        seen, one = set(), []
        for s_ in req:
            if s_["log"] in windows or s_["log"] not in seen:
                one.append(s_); seen.add(s_["log"])
        req = one
    # Every required log must be in the block (2026-11-16, user rule: no silent fallbacks): a required log with no
    # in-block pass (outside the box, not in the window, missing db) or more required passes than slots stopped
    # nobody -- ml_pit02 lost required 2021.09.16.19.27.01_veh-45_01749_03230 to `req[:k]` because another
    # required log crosses the block twice (7 required passes, k = 6), visible only as "required logs present: 5/6".
    missing = sorted(set(require_logs) - {s_["log"] for s_ in req})
    if missing:
        msg = (f"[auto_select] required log(s) without a usable in-block pass "
               f"(windows: {[windows.get(m) for m in missing]}): {missing}")
        if os.environ.get('MTGS_REQUIRE_MISSING', '') == 'warn':
            say(msg + " -- continuing (MTGS_REQUIRE_MISSING=warn)", style="bold yellow")
        else:
            raise SystemExit(msg + "\nFix the block box / windows, or set MTGS_REQUIRE_MISSING=warn to build without them.")
    over = (_n_logs(req) > k or len(req) > max_passes) if per_log else len(req) > k
    if over:
        rows = "\n  ".join(f"{s_['log']} pass {s_['idx']}: {s_['n']} frames, "
                           f"ts {s_['ts']}-{s_.get('ts_end', s_['ts'])}" for s_ in req)
        what = (f"{_n_logs(req)} required drives / {len(req)} passes for {k} drives / {max_passes} passes "
                f"(MTGS_APPEARANCE_PER_LOG=1)" if per_log else
                f"{len(req)} required passes for {k} slots (a log crossing the block twice counts twice)")
        if os.environ.get('MTGS_REQUIRE_OVERFLOW', '') == 'drop_shortest':
            say(f"[auto_select] WARNING: {what}, dropping the shortest (MTGS_REQUIRE_OVERFLOW=drop_shortest):\n  {rows}",
                style="bold yellow")
        else:
            raise SystemExit(f"[auto_select] {what}:\n  {rows}\nNarrow MTGS_REQUIRE_WINDOWS, raise --max_traversals"
                             f"{' / MTGS_MAX_PASSES' if per_log else ''}, or set MTGS_REQUIRE_OVERFLOW=drop_shortest "
                             f"to drop the shortest ones.")
    if per_log:
        kept_req = []
        for s_ in req:                       # longest first: the same cut as req[:k], counted in logs and passes
            if _can_add(kept_req, s_):
                kept_req.append(s_)
        req = kept_req
    else:
        req = req[:k]
    req_idx = {s["idx"] for s in req}
    test = sorted([s for s in usable
                   if s["log"] in test_logs and s["idx"] not in req_idx],
                  key=lambda s: -s["n"])
    rest = [s for s in usable if s["log"] not in test_logs and s["idx"] not in req_idx]

    chosen = list(req)
    # The remaining slots go to whatever best covers the served routes (the frames
    # of the required logs' own passes): a route end seen by its own pass alone
    # reconstructs poorly (loc_0093 / d1db26f8). Greedy: each pick maximises the
    # number of route points that gain a frame within COV_R metres while still
    # under COV_K frames; ties prefer navtest logs, then length. Only once the
    # routes are covered does the old timestamp-diversity fill take over.
    COV_R = float(os.environ.get('MTGS_COV_RADIUS', 15.0))
    COV_K = int(os.environ.get('MTGS_COV_FRAMES', 30))   # trajectories here are ~5 Hz; 30 ~ 12 frames of the 2 Hz vsd
    targets = np.concatenate([s["traj"] for s in req]) if req else np.zeros((0, 2))
    # MTGS_SERVED_ROUTES=<json {scene: [[x, y], ...]}> (global xy): the routes of the scenes this block will serve.
    # Without them coverage only targets the required logs' own passes; when those are rainy (dropped above) or the
    # scenes' logs are not in the block at all, nothing pulled in passes along the served routes (gap3_01,
    # 2026-11-03: 3/5 required logs left, far buildings and a turn unobserved).
    served = os.environ.get('MTGS_SERVED_ROUTES', '')
    routes = []
    if served and os.path.exists(served):
        import json
        routes = [np.asarray(r, dtype=float)[:, :2] for r in json.load(open(served)).values() if len(r)]
        if routes:
            targets = np.concatenate([targets] + routes)
            say(f"[auto_select] coverage targets include {len(routes)} served scene routes ({served})", flush=True)
    if len(targets) > 400:
        targets = targets[:: len(targets) // 400 + 1]
    # Look-ahead (2026-11-16, user rule): the road 20-50 m past each served route's end / before its start and the side
    # streets within 30 m of it are what the cameras see beyond the route; without trained frames there the last
    # frames render an empty, smeared street (ml_pit01 / 442ce328). Regions along the driven trajectories of the
    # candidate passes (lookahead.py); a second coverage target with its own quota, after the route itself.
    # MTGS_LOOKAHEAD=0 switches it off (A/B).
    ahead = np.zeros((0, 2))
    if routes and os.environ.get('MTGS_LOOKAHEAD', '1') != '0':
        from nuplan_scripts.auto_reconstruction.lookahead import regions as _la_regions
        _trajs = [s_["traj"] for s_ in usable]
        _reg = [_la_regions(r, _trajs) for r in routes]
        parts = [x[k] for x in _reg for k in ("ahead", "behind", "side") if len(x[k])]
        if parts:
            ahead = np.concatenate(parts)
            if len(ahead) > 400:
                ahead = ahead[:: len(ahead) // 400 + 1]
            say(f"[auto_select] look-ahead targets: {len(ahead)} points (ahead / behind / side of {len(routes)} routes)", flush=True)
    COV_KA = int(os.environ.get('MTGS_COV_FRAMES_AHEAD', 12))   # ~5 frames of the 2 Hz vsd within COV_R
    W_AHEAD = float(os.environ.get('MTGS_LOOKAHEAD_WEIGHT', 0.5))
    def frames_near(pts, traj):
        if len(pts) == 0 or len(traj) == 0:
            return np.zeros(len(pts), dtype=int)
        d = np.linalg.norm(pts[:, None, :] - traj[None, :, :], axis=2)
        return (d < COV_R).sum(axis=1)
    cover = np.zeros(len(targets), dtype=int)
    cover_a = np.zeros(len(ahead), dtype=int)
    for s_ in chosen:
        cover += frames_near(targets, s_["traj"])
        cover_a += frames_near(ahead, s_["traj"])
    pool = [s_ for s_ in usable if s_["idx"] not in {c["idx"] for c in chosen}]
    while not _full(chosen) and pool and (len(targets) or len(ahead)) and \
            ((cover < COV_K).any() or (cover_a < COV_KA).any()):
        def gain(s_):
            g = np.minimum(frames_near(targets, s_["traj"]), np.maximum(COV_K - cover, 0)).sum()
            ga = np.minimum(frames_near(ahead, s_["traj"]), np.maximum(COV_KA - cover_a, 0)).sum()
            return (g + W_AHEAD * ga, s_["log"] in test_logs, s_["n"])
        cand = [s_ for s_ in pool if _can_add(chosen, s_)]
        if not cand:
            break
        best = max(cand, key=gain)
        if gain(best)[0] <= 0:
            break
        pool.remove(best); chosen.append(best)
        cover += frames_near(targets, best["traj"]); cover_a += frames_near(ahead, best["traj"])
    if targets.size:
        say(f"[auto_select] route coverage after selection: min {int(cover.min())} / median "
              f"{int(np.median(cover))} frames within {COV_R:.0f} m (target {COV_K})", flush=True)
    if len(ahead):
        say(f"[auto_select] look-ahead coverage after selection: {float((cover_a >= COV_KA).mean()):.0%} of "
            f"{len(ahead)} points with >= {COV_KA} frames within {COV_R:.0f} m (median {int(np.median(cover_a))})", flush=True)
    test = [s_ for s_ in test if s_["idx"] not in {c["idx"] for c in chosen}]
    if per_log:
        for s_ in test:
            if _can_add(chosen, s_):
                chosen.append(s_)
    else:
        chosen += test[:max(0, k - len(chosen))]
    rest = [s_ for s_ in rest if s_["idx"] not in {c["idx"] for c in chosen}]
    # fill remaining slots by greedy max-min timestamp spread
    while not _full(chosen) and rest:
        if not chosen:
            rest.sort(key=lambda s: -s["n"])
            chosen.append(rest.pop(0))
            continue
        cand = [s_ for s_ in rest if _can_add(chosen, s_)]
        if not cand:
            break
        best = max(cand, key=lambda s: (min(abs(s["ts"] - c["ts"]) for c in chosen), s["n"]))
        rest.remove(best)
        chosen.append(best)
    if per_log:
        say(f"[auto_select] MTGS_APPEARANCE_PER_LOG: {len(chosen)} passes from {_n_logs(chosen)} drives "
            f"(caps {k} drives / {max_passes} passes)", flush=True)
    return sorted(s["idx"] for s in chosen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--groups", default=os.environ.get("MTGS_GROUPS_JSON", ""),
                   help="navtest_location_groups.json; enables test-log priority")
    p.add_argument("--max_traversals", type=int, default=6,
                   help="cap per block (MTGS paper uses up to 6)")
    p.add_argument("--require_logs", default="",
                   help="comma-separated log names that must be selected")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--force", action="store_true",
                   help="re-select even if the config already has selected_videos")
    p.add_argument("--manifest_out", default=None,
                   help="write the selected traversals' log names here (one per "
                        "line), for per-job sensor-blob stage-in on tar-based storage")
    args = p.parse_args()

    config = load_config(args.config)
    if config.selected_videos and not args.force:
        say(f"[auto_select] {config.road_block_name}: selected_videos already set "
              f"({list(config.selected_videos)}), skipping. Use --force to redo.")
        if args.manifest_out and not os.path.exists(args.manifest_out):
            stats = discover_video_stats(config, args.num_workers)
            logs = sorted({s["log"] for s in stats if s["idx"] in set(config.selected_videos)})
            os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)), exist_ok=True)
            with open(args.manifest_out, "w") as f:
                f.write("\n".join(logs) + "\n")
            say(f"[auto_select] manifest -> {args.manifest_out} ({len(logs)} logs)")
        return

    # 1. discovery pass: all in-block traversals, db-only
    stats = discover_video_stats(config, args.num_workers)
    say(f"[auto_select] {config.road_block_name}: {len(stats)} traversals discovered")

    test_logs = set()
    if args.groups and os.path.exists(args.groups):
        groups = json.load(open(args.groups))["groups"]
        for g in groups:
            # Experiment prefixes (je_, fi_, ...) are prepended to the location id;
            # the groups json knows only the bare id. Without this the lookup silently
            # failed for 111/114 je_ blocks and no navtest log was pinned.
            _name = config.road_block_name
            _cands = {_name, _name.split("_", 1)[1] if "_" in _name else _name}
            if g["location_id"] in _cands:
                test_logs = set(g["test_logs"])
                break
    if not test_logs:
        say("[auto_select] WARNING: no groups json / location match; "
              "selecting purely by diversity")

    require = [x for x in args.require_logs.split(",") if x]
    picked = select(stats, test_logs, args.max_traversals, require)
    if require:
        got = {s["log"] for s in stats if s["idx"] in set(picked)} & set(require)
        say(f"[auto_select] required logs present: {len(got)}/{len(require)}")
    n_test = sum(1 for s in stats if s["idx"] in picked and s["log"] in test_logs)
    say(f"[auto_select] picked {picked} ({n_test} from navtest logs)")

    if args.manifest_out:
        logs = sorted({s["log"] for s in stats if s["idx"] in set(picked)})
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)), exist_ok=True)
        with open(args.manifest_out, "w") as f:
            f.write("\n".join(logs) + "\n")
        say(f"[auto_select] manifest -> {args.manifest_out} ({len(logs)} logs)")

    # 2. write the selection back into the config yaml
    config.selected_videos = tuple(picked)
    config.save_config(args.config)
    say(f"[auto_select] updated {args.config}")


if __name__ == "__main__":
    main()
