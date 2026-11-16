"""NAVSIM / nuPlan log names.

A NAVSIM log name is a split of one recorded drive: <date>.<time>_veh-<n>_<first frame>_<last frame>, e.g.
2021.10.06.07.26.10_veh-52_01245_02064. The splits of one drive are minutes apart, same day, same weather and light,
and disjoint in time. "Same log" for appearance sharing and for the per-block log cap means the same DRIVE (user
2026-11-16): drive_key() is the one place that defines it.
"""
import re

_DRIVE = re.compile(r"^(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)(?:_\d+_\d+)?$")


def drive_key(log_name):
    """The drive a (split) log belongs to: '2021.10.06.07.26.10_veh-52_01245_02064' -> '2021.10.06.07.26.10_veh-52'.
    A name in another format is its own drive."""
    m = _DRIVE.match(str(log_name))
    return m.group(1) if m else str(log_name)
