#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import os
import sys
import shutil
import platform
import subprocess
import argparse


def _match_threads(num_workers):
    """Threads for matching, capped well below the core count.

    Matching is two phases with opposite appetites: geometric verification scales
    with cores, but importing the results goes through one SQLite writer. Handing
    the stage every core on a 152-core node put 85 threads in futex_wait and drove
    the process to 83% system time -- the lock convoy costs more than the extra
    parallelism buys. COLMAP_MATCH_THREADS overrides.
    """
    import os
    override = os.environ.get("COLMAP_MATCH_THREADS")
    if override:
        return max(1, int(override))
    return max(1, min(int(num_workers), 32))   # measured peak on EPYC 9634


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
    r = _run_colmap_raw(args, colmap_path, **kwargs)
    out = None
    if "--output_path" in args:
        out = args[args.index("--output_path") + 1]
    listing = sorted(os.listdir(out)) if out and os.path.isdir(out) else None
    print(f"[colmap] {args[1] if len(args) > 1 else '?'} rc={r.returncode} "
          f"output={os.path.basename(out) if out else '-'} files={listing}", flush=True)
    return r


def _run_colmap_raw(args, colmap_path, **kwargs):
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


def _normalize_rigs(database_path):
    """COLMAP 4.x feature_extractor auto-creates one rig per camera folder with
    arbitrary ref sensors. Loading a rig-less prior model then synthesizes rigs as
    rig_id == ref_sensor_id == camera_id, and Reconstruction::Load dies on
    'existing_rig.RefSensorId() == rig.RefSensorId()'. Rewrite the table to the
    trivial mapping so both sides agree (frames table is empty at this point)."""
    import sqlite3
    db = sqlite3.connect(database_path)
    c = db.cursor()
    try:
        c.execute("DELETE FROM rigs")
        c.execute("DELETE FROM rig_sensors")
        for (cid,) in c.execute("SELECT camera_id FROM cameras").fetchall():
            c.execute("INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) VALUES (?,?,0)", (cid, cid))
        db.commit()
    except sqlite3.OperationalError:
        pass  # colmap 3.x database: no rigs table, nothing to fix
    db.close()


def run_bundle_adjustment(
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
        _normalize_rigs(f"{colmap_path}/database.db")
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap feature_extractor: {e}")
        sys.exit(1)


    print("feature matching...")
    colmap_matches_importer_args = [
        colmap_exe, "matches_importer",
        "--database_path", f"{colmap_path}/database.db",
        "--match_list_path", f"{colmap_path}/image_pairs.txt",
        "--FeatureMatching.num_threads", f"{_match_threads(num_workers)}"
    ]

    try:
        _run_colmap(colmap_matches_importer_args, colmap_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap matches_importer: {e}")
        sys.exit(1)

    if os.path.exists(os.path.join(colmap_path, output_model)):
        shutil.rmtree(os.path.join(colmap_path, output_model))

    os.makedirs(os.path.join(colmap_path, output_model, "t"), exist_ok=True)
    os.makedirs(os.path.join(colmap_path, output_model, "b"), exist_ok=True)
    os.makedirs(os.path.join(colmap_path, output_model, "t2"), exist_ok=True)
    os.makedirs(os.path.join(colmap_path, output_model, "0"), exist_ok=True)

    colmap_point_triangulator_args = [
        colmap_exe, "point_triangulator",
        "--refine_intrinsics", os.environ.get("COLMAP_REFINE_INTRINSICS", "1"),
        "--Mapper.num_threads", f"{num_workers}",
        "--Mapper.ba_global_function_tolerance", "0.000001",
        "--Mapper.ba_global_max_num_iterations", "30",
        "--Mapper.ba_global_max_refinements", "3",
    ]

    # COLMAP_BA_MODE picks how the standalone bundle_adjuster is driven:
    #   full    -- current options (colmap 4.x spelling, free focal + distortion)
    #   legacy  -- the exact July argv (colmap 3.x spelling), to reproduce what the
    #              adopted reconstructions were actually built with
    #   nointr  -- poses only; intrinsics are left to the triangulator's refinement
    _ba_mode = os.environ.get("COLMAP_BA_MODE", "full")
    if _ba_mode == "legacy":
        colmap_bundle_adjuster_args = [
            colmap_exe, "bundle_adjuster",
            "--BundleAdjustment.function_tolerance", "0.000001",
            "--BundleAdjustment.max_linear_solver_iterations", "200",
            "--BundleAdjustment.max_num_iterations", "100",
            "--BundleAdjustment.refine_focal_length", "1",
            "--BundleAdjustment.refine_principal_point", "0",
            "--BundleAdjustment.refine_extra_params", "1"
        ]
    else:
        colmap_bundle_adjuster_args = [
            colmap_exe, "bundle_adjuster",
            "--BundleAdjustmentCeres.function_tolerance", "0.000001",
            "--BundleAdjustmentCeres.max_linear_solver_iterations", "200",
            "--BundleAdjustmentCeres.max_num_iterations", "100",
            "--BundleAdjustment.refine_focal_length", os.environ.get("COLMAP_REFINE_INTRINSICS", "1"),
            "--BundleAdjustment.refine_principal_point", "0",
            "--BundleAdjustment.refine_extra_params", os.environ.get("COLMAP_REFINE_INTRINSICS", "1")
        ]
        if _ba_mode == "nointr":
            for flag in ("--BundleAdjustment.refine_focal_length",
                         "--BundleAdjustment.refine_extra_params"):
                colmap_bundle_adjuster_args[colmap_bundle_adjuster_args.index(flag) + 1] = "0"
    print(f"[ba] COLMAP_BA_MODE={_ba_mode}: {' '.join(colmap_bundle_adjuster_args[1:])}", flush=True)
    # MTGS_JOINT_SFM=1: pure structure-from-motion -- the mapper reconstructs from the matches
    # alone (no pose priors), intrinsics fixed to the calibration; the largest model becomes
    # <output_model>/0 and align_model() then puts it into the metric world frame with one
    # Sim3 against the lidar-registered camera centres (COLMAP has no scale of its own).
    # User 2026-10-25: multi-traversal reconstructions were best with plain COLMAP; the
    # prior-seeded triangulate+BA below can stay near the (inconsistent) prior poses.
    if os.environ.get("MTGS_JOINT_SFM") == "1":
        mode = os.environ.get("COLMAP_SFM_MAPPER", "incremental")
        raw = os.path.join(colmap_path, output_model + "_raw"); shutil.rmtree(raw, ignore_errors=True); os.makedirs(raw)
        fix = os.environ.get("COLMAP_REFINE_INTRINSICS", "0") != "1"
        if mode == "global":
            mapper_args = [colmap_exe, "global_mapper", "--database_path", f"{colmap_path}/database.db",
                           "--image_path", f"{colmap_path}/images", "--output_path", raw]
        else:
            mapper_args = [colmap_exe, "mapper", "--database_path", f"{colmap_path}/database.db",
                           "--image_path", f"{colmap_path}/images", "--output_path", raw,
                           "--Mapper.num_threads", f"{num_workers}",
                           "--Mapper.ba_refine_focal_length", "0" if fix else "1",
                           "--Mapper.ba_refine_principal_point", "0",
                           "--Mapper.ba_refine_extra_params", "0" if fix else "1",
                           "--Mapper.multiple_models", "1", "--Mapper.min_model_size", "50"]
        print(f"[sfm] {mode} mapper: {' '.join(mapper_args[1:])}", flush=True)
        try:
            _run_colmap(mapper_args, colmap_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing colmap {mode} mapper: {e}")
            sys.exit(1)
        models = [d for d in sorted(os.listdir(raw)) if os.path.exists(os.path.join(raw, d, "images.bin"))]
        if not models:
            print("[sfm] mapper produced no model"); sys.exit(1)
        sizes = {}
        for d in models:
            with open(os.path.join(raw, d, "images.bin"), "rb") as f:
                sizes[d] = int.from_bytes(f.read(8), "little")      # images.bin starts with the image count
        best = max(sizes, key=sizes.get)
        print(f"[sfm] models (images): {sizes}; using {best}", flush=True)
        shutil.rmtree(os.path.join(colmap_path, output_model, "0"), ignore_errors=True)
        shutil.copytree(os.path.join(raw, best), os.path.join(colmap_path, output_model, "0"))
        sys.exit(0)

    # 2 rounds of triangulation + bundle adjustment
    try:
        _run_colmap(colmap_point_triangulator_args + [
            "--database_path", f"{colmap_path}/database.db",
            "--image_path", f"{colmap_path}/images",
            "--input_path", f"{colmap_path}/{input_model}",
            "--output_path", f"{colmap_path}/{output_model}/t",
            ], colmap_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap_point_triangulator_args: {e}")
        sys.exit(1)

    try:
        _run_colmap(colmap_bundle_adjuster_args + [
            "--input_path", f"{colmap_path}/{output_model}/t",
            "--output_path", f"{colmap_path}/{output_model}/b",
        ], colmap_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap_bundle_adjuster_args: {e}")
        sys.exit(1)

    try:
        _run_colmap(colmap_point_triangulator_args + [
            "--database_path", f"{colmap_path}/database.db",
            "--image_path", f"{colmap_path}/images",
            "--input_path", f"{colmap_path}/{output_model}/b",
            "--output_path", f"{colmap_path}/{output_model}/t2",
            ], colmap_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap_point_triangulator_args: {e}")
        sys.exit(1)

    try:
        _run_colmap(colmap_bundle_adjuster_args + [
            "--input_path", f"{colmap_path}/{output_model}/t2",
            "--output_path", f"{colmap_path}/{output_model}/0",
        ], colmap_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap_bundle_adjuster_args: {e}")
        sys.exit(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap_path', type=str, help='Input colmap dir', required=True)
    parser.add_argument('--input_model', type=str, default='sparse_model')
    parser.add_argument('--output_model', type=str, default='sfm_model')
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    colmap_path = args.colmap_path

    run_bundle_adjustment(colmap_path, args.input_model, args.output_model, args.num_workers)
