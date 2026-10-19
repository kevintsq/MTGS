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
            print(f"[auto_select] excluding {len(dropped)} bad-weather traversals "
                  f"({len(set(s['log'] for s in dropped))} logs)", flush=True)
        if kept:
            print(f"[auto_select] keeping {len(kept)} bad-weather pass(es) that hold a served "
                  f"scene (MTGS_EXCLUDE_EXEMPT_REQUIRED): {sorted(s['idx'] for s in kept)}", flush=True)
        drop_idx = {s["idx"] for s in dropped}
        stats = [s for s in stats if s["idx"] not in drop_idx]
        if not exempt:
            require_logs = tuple(r for r in require_logs if not r.startswith(excl))
    usable = [s for s in stats if s["n"] >= MIN_FRAMES] or stats
    # A block serving a specific scene must reconstruct the road that scene
    # drives, so its own log takes a slot before the by-length ranking runs;
    # otherwise a location with 20 test logs keeps only the six longest and the
    # target route can end up unobserved.
    # A log can pass through the block several times; only the pass that holds
    # the served scene is required (MTGS_REQUIRE_WINDOWS: "log=t0-t1;...", us).
    # Without windows every pass of a required log takes a slot, and a block with
    # four served logs had no slot left for covering the routes (loc_0093).
    req = sorted([s for s in usable if _required(s)], key=lambda s: -s["n"])[:k]
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
    if len(targets) > 400:
        targets = targets[:: len(targets) // 400 + 1]
    def frames_near(pts, traj):
        if len(pts) == 0 or len(traj) == 0:
            return np.zeros(len(pts), dtype=int)
        d = np.linalg.norm(pts[:, None, :] - traj[None, :, :], axis=2)
        return (d < COV_R).sum(axis=1)
    cover = np.zeros(len(targets), dtype=int)
    for s_ in chosen:
        cover += frames_near(targets, s_["traj"])
    pool = [s_ for s_ in usable if s_["idx"] not in {c["idx"] for c in chosen}]
    while len(chosen) < k and pool and len(targets) and (cover < COV_K).any():
        def gain(s_):
            g = np.minimum(frames_near(targets, s_["traj"]), np.maximum(COV_K - cover, 0)).sum()
            return (g, s_["log"] in test_logs, s_["n"])
        best = max(pool, key=gain)
        if gain(best)[0] <= 0:
            break
        pool.remove(best); chosen.append(best); cover += frames_near(targets, best["traj"])
    if targets.size:
        print(f"[auto_select] route coverage after selection: min {int(cover.min())} / median "
              f"{int(np.median(cover))} frames within {COV_R:.0f} m (target {COV_K})", flush=True)
    test = [s_ for s_ in test if s_["idx"] not in {c["idx"] for c in chosen}]
    chosen += test[:max(0, k - len(chosen))]
    rest = [s_ for s_ in rest if s_["idx"] not in {c["idx"] for c in chosen}]
    # fill remaining slots by greedy max-min timestamp spread
    while len(chosen) < k and rest:
        if not chosen:
            rest.sort(key=lambda s: -s["n"])
            chosen.append(rest.pop(0))
            continue
        best = max(rest, key=lambda s: (min(abs(s["ts"] - c["ts"]) for c in chosen), s["n"]))
        rest.remove(best)
        chosen.append(best)
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
        print(f"[auto_select] {config.road_block_name}: selected_videos already set "
              f"({list(config.selected_videos)}), skipping. Use --force to redo.")
        if args.manifest_out and not os.path.exists(args.manifest_out):
            stats = discover_video_stats(config, args.num_workers)
            logs = sorted({s["log"] for s in stats if s["idx"] in set(config.selected_videos)})
            os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)), exist_ok=True)
            with open(args.manifest_out, "w") as f:
                f.write("\n".join(logs) + "\n")
            print(f"[auto_select] manifest -> {args.manifest_out} ({len(logs)} logs)")
        return

    # 1. discovery pass: all in-block traversals, db-only
    stats = discover_video_stats(config, args.num_workers)
    print(f"[auto_select] {config.road_block_name}: {len(stats)} traversals discovered")

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
        print("[auto_select] WARNING: no groups json / location match; "
              "selecting purely by diversity")

    require = [x for x in args.require_logs.split(",") if x]
    picked = select(stats, test_logs, args.max_traversals, require)
    if require:
        got = {s["log"] for s in stats if s["idx"] in set(picked)} & set(require)
        print(f"[auto_select] required logs present: {len(got)}/{len(require)}")
    n_test = sum(1 for s in stats if s["idx"] in picked and s["log"] in test_logs)
    print(f"[auto_select] picked {picked} ({n_test} from navtest logs)")

    if args.manifest_out:
        logs = sorted({s["log"] for s in stats if s["idx"] in set(picked)})
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)), exist_ok=True)
        with open(args.manifest_out, "w") as f:
            f.write("\n".join(logs) + "\n")
        print(f"[auto_select] manifest -> {args.manifest_out} ({len(logs)} logs)")

    # 2. write the selection back into the config yaml
    config.selected_videos = tuple(picked)
    config.save_config(args.config)
    print(f"[auto_select] updated {args.config}")


if __name__ == "__main__":
    main()
