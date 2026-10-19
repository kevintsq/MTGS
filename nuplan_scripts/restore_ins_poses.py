"""Revert a block vsd to its INS ego poses.

Joint BA writes its own ego poses into the vsd (MTGS_JOINT_WRITE_EGO), keeping the
INS ones in ego2global_original. When the BA solution is not tight enough to trust
(Sim3 residual above the gate), the near-field road is better served by INS, so put
those back before training.

Only the poses revert. colmap_param stays, because it also carries the refined
lens parameters, and the gate says nothing about those -- dropping the whole dict
is how the je batch ended up training on nominal intrinsics while v1 used the
refined ones. Whether the BA camera poses inside it are consumed is decided by
the dataparser's use_colmap_pose, which the caller sets from the same gate.

  restore_ins_poses.py <video_scene_dict.pkl> [--drop-colmap-param]
"""
import pickle, sys
import numpy as np
from pyquaternion import Quaternion

args = [a for a in sys.argv[1:] if not a.startswith("-")]
drop = "--drop-colmap-param" in sys.argv
path = args[0]
vsd = pickle.load(open(path, "rb"))
n = 0
for v in vsd.values():
    for info in v["frame_infos"]:
        if "ego2global_original" not in info:
            continue
        pose = np.asarray(info["ego2global_original"], float)
        info["ego2global"] = pose.copy()
        info["ego2global_translation"] = pose[:3, 3]
        info["ego2global_rotation"] = Quaternion(matrix=pose[:3, :3], atol=1e-6)
        info["lidar2global"] = pose @ info["lidar2ego"]
        if drop:
            for cam in info["cams"].values():
                cam.pop("colmap_param", None)
        n += 1
pickle.dump(vsd, open(path, "wb"))
print(f"restored INS poses for {n} frames"
      + (" (colmap_param dropped)" if drop else " (colmap_param kept)"))
