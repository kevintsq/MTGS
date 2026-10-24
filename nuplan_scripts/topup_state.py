"""Survivor count and log names after registration, for the producer's top-up loop.

  topup_state.py <video_scene_dict.pkl> count   -> "<n>"
  topup_state.py <video_scene_dict.pkl> logs    -> comma-separated log names
"""
import pickle, sys
try:
    vsd = pickle.load(open(sys.argv[1], "rb"))
except Exception:
    print(0 if sys.argv[2] == "count" else "")
    sys.exit(0)
if sys.argv[2] == "count":
    print(len(vsd))
else:
    print(",".join(sorted({v.get("log_name", "") for v in vsd.values()} - {""})))
