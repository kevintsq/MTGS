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
import pickle
import argparse
import subprocess
import sys

from nuplan_scripts.utils.config import load_config
from nuplan_scripts.utils.video_scene_dict_tools import VideoScene

MIN_FRAMES = 10  # drop degenerate clips that barely touch the block


def video_stats(vsd):
    stats = []
    for token, v in vsd.items():
        idx = int(token.rsplit("-", 1)[1])
        frames = v.get("frame_infos") or v.get("lidar_pcs") or []
        ts = None
        if v.get("frame_infos"):
            ts = int(v["frame_infos"][0]["timestamp"])
        elif "start_ts" in v:
            ts = int(v["start_ts"])
        stats.append({"idx": idx, "log": v.get("log_name", ""), "n": len(frames), "ts": ts or 0})
    return sorted(stats, key=lambda s: s["idx"])


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
    args = p.parse_args()

    config = load_config(args.config)
    if config.selected_videos and not args.force:
        print(f"[auto_select] {config.road_block_name}: selected_videos already set "
              f"({list(config.selected_videos)}), skipping. Use --force to redo.")
        return

    # 1. discovery pass: all in-block traversals, db-only
    subprocess.run([sys.executable, "-m", "nuplan_scripts.nuplan_video_processing",
                    "--config", args.config, "--num_workers", str(args.num_workers)],
                   check=True)

    video_scene = VideoScene(config)
    with open(video_scene.pickle_path_raw, "rb") as f:
        vsd = pickle.load(f)
    stats = video_stats(vsd)
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

    # 2. write the selection back into the config yaml
    config.selected_videos = tuple(picked)
    config.save_config(args.config)
    print(f"[auto_select] updated {args.config}")


if __name__ == "__main__":
    main()
