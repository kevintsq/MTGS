"""How many traversals the config currently selects (the top-up loop's growth test)."""
import sys, yaml
sys.path.insert(0, "/home/kevin/MTGS")
from nuplan_scripts.utils.config import RoadBlockConfig
try:
    print(len(yaml.unsafe_load(open(sys.argv[1])).selected_videos or ()))
except Exception:
    print(0)
