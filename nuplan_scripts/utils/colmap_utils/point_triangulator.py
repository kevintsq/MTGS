#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import sys
import shutil
import subprocess
import argparse
import platform


def _gpu_flags(command):
    """Tell colmap to stay off the GPU when this job was not granted one.

    The CUDA build calls GetNumCudaDevices() while constructing the feature
    extractor / matcher, and aborts with SIGABRT rather than degrading when no
    device is usable. A scheduler that co-schedules CPU stages onto GPU boxes
    leaves /dev/nvidiactl present with no allocation, so gate on the actual grant
    (CUDA_VISIBLE_DEVICES); COLMAP_USE_GPU overrides either way.
    """
    import os
    use = os.environ.get("COLMAP_USE_GPU")
    if use is None:
        use = "1" if os.environ.get("CUDA_VISIBLE_DEVICES") else "0"
    if use == "1":
        return []
    return {
        "feature_extractor": ["--FeatureExtraction.use_gpu", "0"],
        "exhaustive_matcher": ["--FeatureMatching.use_gpu", "0"],
        "sequential_matcher": ["--FeatureMatching.use_gpu", "0"],
        "matches_importer": ["--FeatureMatching.use_gpu", "0"],
    }.get(command, [])


def _run_colmap(args, colmap_path, **kwargs):
    """Run a colmap command from inside its own model directory.

    /proc/<pid>/cmdline is world-readable and colmap is a C++ binary that cannot
    rename its own process, so every path on its command line is visible to every
    account on the machine -- including the block name. Setting the working directory
    to the model folder and passing the paths relative to it keeps the listing down to
    bare filenames without changing what colmap is told to do.
    """
    prefix = colmap_path.rstrip("/") + "/"
    rel = [a[len(prefix):] if isinstance(a, str) and a.startswith(prefix) else a
           for a in args]
    if len(rel) > 1:
        rel = rel + _gpu_flags(rel[1])
    return subprocess.run(rel, cwd=colmap_path, **kwargs)


def run_point_triangulator(
    colmap_path, 
    input_model='sparse_model', 
    output_model='sfm_model',
    num_workers=8
):

    colmap_exe = "colmap.bat" if platform.system() == "Windows" else "colmap"

    print("extracting features...")
    colmap_feature_extractor_args = [
        colmap_exe, "feature_extractor",
        "--database_path", f"{colmap_path}/database.db",
        "--image_path", f"{colmap_path}/images",
    ]
    if os.path.exists(f"{colmap_path}/masks"):
        colmap_feature_extractor_args += ["--ImageReader.mask_path", f"{colmap_path}/masks"]
    # COLMAP 4.x renamed SiftExtraction.num_threads -> FeatureExtraction.num_threads
    colmap_feature_extractor_args += ["--FeatureExtraction.num_threads", f"{num_workers}"]

    try:
        _run_colmap(colmap_feature_extractor_args, colmap_path, check=True)
        # The July reconstructions ran without this rewrite (colmap's own rigs were
        # left in place); MTGS_NORMALIZE_RIGS=0 reproduces that for comparison.
        if os.environ.get("MTGS_NORMALIZE_RIGS", "1") == "1":
            from nuplan_scripts.utils.colmap_utils.bundle_adjustment import _normalize_rigs
            _normalize_rigs(f"{colmap_path}/database.db")
        print(f"[rigs] normalize={os.environ.get('MTGS_NORMALIZE_RIGS', '1')}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap feature_extractor: {e}")
        sys.exit(1)


    print("feature matching...")
    colmap_matches_importer_args = [
        colmap_exe, "matches_importer",
        "--database_path", f"{colmap_path}/database.db",
        "--match_list_path", f"{colmap_path}/image_pairs.txt",
        "--FeatureMatching.num_threads", f"{num_workers}"
    ]

    try:
        _run_colmap(colmap_matches_importer_args, colmap_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap matches_importer: {e}")
        sys.exit(1)

    if os.path.exists(os.path.join(colmap_path, output_model)):
        shutil.rmtree(os.path.join(colmap_path, output_model))

    os.makedirs(os.path.join(colmap_path, output_model, "0"), exist_ok=True)

    print("dropping rigs from database to prevent crash...")
    try:
        import sqlite3
        conn = sqlite3.connect(f"{colmap_path}/database.db")
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS rigs;")
        cursor.execute("DROP TABLE IF EXISTS rig_sensors;")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to drop rigs: {e}")

    colmap_point_triangulator_args = [
        colmap_exe, "point_triangulator",
        "--refine_intrinsics", os.environ.get("COLMAP_REFINE_INTRINSICS", "1"),
        "--Mapper.num_threads", f"{num_workers}",
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.ba_global_max_num_iterations", "30",
        "--Mapper.ba_global_max_refinements", "3",
        "--Mapper.ba_refine_sensor_from_rig", "0",
    ]

    # point triangulation
    try:
        _run_colmap(colmap_point_triangulator_args + [
            "--database_path", f"{colmap_path}/database.db",
            "--image_path", f"{colmap_path}/images",
            "--input_path", f"{colmap_path}/{input_model}",
            "--output_path", f"{colmap_path}/{output_model}/0",
            ], colmap_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap_point_triangulator_args: {e}")
        sys.exit(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path', type=str, help='Input colmap dir', required=True)
    parser.add_argument('--input_model', type=str, default='sparse_model')
    parser.add_argument('--output_model', type=str, default='sfm_model')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    colmap_path = args.colmap_path

    run_point_triangulator(colmap_path, args.input_model, args.output_model, args.num_workers)
