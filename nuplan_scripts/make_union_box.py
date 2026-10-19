"""Rewrite a block config with box = existing box UNION (all served routes + margin).

Two failure modes this avoids: a route whose start/end falls outside the block (no
coverage -> smeared render, gapC_008/rg_0140b), and a box narrowed to the routes
alone, which drops the generous coverage the original box had (loc_0218 lost 25%
of its frames that way).

  make_union_box.py <old_block> <new_block> [--margin 40]
"""
import argparse, json, os, pickle, re
import numpy as np

import os
W = os.environ.get("MTGS_WORK", "/home/kevin/MTGS/data/MTGS_navtest")
HA = os.environ.get("HUGSIM_ASSETS", "/home/kevin/TTA/HUGSIM_ASSETS")

ap = argparse.ArgumentParser()
ap.add_argument("old"); ap.add_argument("new")
ap.add_argument("--margin", type=float, default=40.0)
a = ap.parse_args()

raw = open(f"{W}/configs_v1final/{a.old}.yml").read()
seg = raw.split("road_block:")[1].split("road_block_name")[0]
ob = [float(x) for x in re.findall(r"- (-?[\d.]+)", seg)[:4]]

q = json.load(open(f"{HA}/convert_state/scene_block_override.json"))["quality"]
idx = json.load(open(f"{HA}/convert_state/scene_log_index.json"))
toks = [t for t, v in q.items() if v["loc"] == a.old]
pts, bylog = [], {}
for t in toks:
    p = idx.get(t)
    if not p:
        continue
    if p not in bylog:
        bylog[p] = pickle.load(open(p, "rb"))
    pts += [np.asarray(f["ego2global"], float)[:2, 3]
            for f in bylog[p] if f["scene_token"] == t]

box = list(ob)
if pts:
    pts = np.array(pts); m = a.margin
    box = [min(ob[0], pts[:, 0].min() - m), min(ob[1], pts[:, 1].min() - m),
           max(ob[2], pts[:, 0].max() + m), max(ob[3], pts[:, 1].max() + m)]

out = raw.replace(a.old, a.new).replace("use_colmap_ba: false", "use_colmap_ba: true")
head, rest = out.split("road_block:", 1)
lines = rest.split("\n"); j = 0
for i, l in enumerate(lines):
    mm = re.match(r"^(\s*)- -?[\d.]+\s*$", l)
    if mm and j < 4:
        lines[i] = f"{mm.group(1)}- {box[j]:.1f}"; j += 1
    if j == 4:
        break
assert j == 4, f"could not rewrite box for {a.old}"
open(f"{W}/configs_v1final/{a.new}.yml", "w").write(head + "road_block:" + "\n".join(lines))
h = open(f"{HA}/mtgs_navtest_configs_v1final/{a.old}.yml").read().replace(a.old, a.new)
open(f"{HA}/mtgs_navtest_configs_v1final/{a.new}.yml", "w").write(h)
print(f"{a.new}: {ob[2]-ob[0]:.0f}x{ob[3]-ob[1]:.0f} -> {box[2]-box[0]:.0f}x{box[3]-box[1]:.0f} m "
      f"({len(toks)} scenes)")
