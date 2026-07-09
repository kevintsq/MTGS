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
             "n": len(v["lidar_pcs"]), "ts": int(v["start_ts"])}
            for v in video_infos]


def select(stats, test_logs, k):
    usable = [s for s in stats if s["n"] >= MIN_FRAMES] or stats
    test = sorted([s for s in usable if s["log"] in test_logs],
                  key=lambda s: -s["n"])
    rest = [s for s in usable if s["log"] not in test_logs]

    chosen = test[:k]
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
            if g["location_id"] == config.road_block_name:
                test_logs = set(g["test_logs"])
                break
    if not test_logs:
        print("[auto_select] WARNING: no groups json / location match; "
              "selecting purely by diversity")

    picked = select(stats, test_logs, args.max_traversals)
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
