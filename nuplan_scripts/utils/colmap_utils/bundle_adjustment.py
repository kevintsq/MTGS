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
import numpy as np


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
    if len(rel) > 1 and rel[1] in _ONE_BLAS_THREAD and "env" not in kwargs:
        # Feature extraction and matching are parallel over N worker threads already; the BLAS/OpenMP
        # runtime inside each worker (the colmap/colmap image links MKL + libomp, Eigen sees OpenMP) opens
        # another team of OMP_NUM_THREADS, so with the stage-wide OMP_NUM_THREADS=24 one matches_importer
        # ran 606 threads (577 runnable) on 24 cores. Measured 2026-11-04 (civo-h200-2, 14 cores, 3000
        # pairs, CPU matching): OMP=14 235 s, OMP=1 109 s. COLMAP_BLAS_THREADS overrides.
        n = os.environ.get("COLMAP_BLAS_THREADS", "1")
        kwargs["env"] = {**os.environ, "OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n}
    return subprocess.run(rel, cwd=colmap_path, **kwargs)


def _write_pose_priors(colmap_path):
    """Camera centres of sparse_model/images.txt (the registered poses) -> pose_priors (CARTESIAN) for every
    image the feature extractor registered in database.db."""
    import sqlite3
    # no pyquaternion here: this module runs under whatever `python` the stage shell resolves (civo's lacked it,
    # 2026-11-05, two pose_prior attempts of fk_loc_0113 died on the import); COLMAP images.txt is (qw qx qy qz tx ty tz)
    def _rot(qw, qx, qy, qz):
        n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                         [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                         [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    centres = {}
    with open(f"{colmap_path}/sparse_model/images.txt") as f:
        for line in f:
            t = line.split()
            if len(t) < 10 or t[0].startswith("#"):
                continue
            R = _rot(*[float(x) for x in t[1:5]]); tv = np.array([float(x) for x in t[5:8]])
            centres[t[9]] = -(R.T @ tv)
    dbp = f"{colmap_path}/database.db"
    db = sqlite3.connect(dbp)
    cols = [r[1] for r in db.execute("pragma table_info(pose_priors)")]
    rows = db.execute("select image_id, name, camera_id from images").fetchall()
    n = 0
    if "corr_sensor_id" in cols:
        # COLMAP 4.3 schema (manco / kaitain / workstation builds, and the civo snapshots): priors keyed by
        # (sensor type 0 = camera, sensor id = camera_id, data id = image_id); gravity unset = three NaNs,
        # coordinate_system 1 = CARTESIAN -- written with sqlite directly, exactly as pycolmap 4.3 writes them
        # (2026-11-05: three fk_loc_0113 node runs on civo lost 55 min of matching each at `import pycolmap`,
        # the stage python there has no pycolmap, and a pycolmap of another version would write another schema)
        db.execute("delete from pose_priors")   # idempotent: a stage restart on an imported database must not duplicate
        cov = np.eye(3, dtype=np.float64).tobytes()
        grav = np.full(3, np.nan, dtype=np.float64).tobytes()
        for image_id, name, camera_id in rows:
            if name not in centres:
                continue
            db.execute("insert into pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type, position, position_covariance, gravity, coordinate_system) values (?, ?, 0, ?, ?, ?, 1)",
                       (int(image_id), int(camera_id), centres[name].astype(np.float64).tobytes(), cov, grav))
            n += 1
        db.commit(); db.close()
    else:   # COLMAP 4.2 (the docker image on torch / civo / ARC): image_id keyed
        db.execute("delete from pose_priors")
        cov = (np.eye(3) * 1.0).astype(np.float64).tobytes()
        for image_id, name, _cid in rows:
            if name in centres:
                db.execute("insert or replace into pose_priors(image_id, position, coordinate_system, position_covariance) values (?, ?, 1, ?)",
                           (image_id, centres[name].astype(np.float64).tobytes(), cov))
                n += 1
        db.commit(); db.close()
    print(f"[sfm] pose priors written for {n}/{len(centres)} images", flush=True)


def _colmap_scale(colmap_path):
    """Metres -> COLMAP units. create_colmap_model scales the world to ~10 units per block (scale_factor = 10 / block
    size) and exports it as MTGS_COLMAP_SCALE / sparse_model/scale_factor.txt; every metre knob of the SfM stage goes
    through this (2026-11-05: pn_w2_21 ran with std 0.1 *units* = 1.8 m, max err 27 m, pair prune never firing)."""
    v = os.environ.get("MTGS_COLMAP_SCALE")
    if not v:
        f = os.path.join(colmap_path, "sparse_model", "scale_factor.txt")
        v = open(f).read().strip() if os.path.exists(f) else None
    if not v:
        print("[sfm] WARNING: no MTGS_COLMAP_SCALE / sparse_model/scale_factor.txt; metre thresholds used as raw units", flush=True)
        return 1.0
    return float(v)


def _prune_db_to_pair_list(db_path, pairs_txt):
    """An imported database is a cache of features and matches; the match graph is defined by this run's
    image_pairs.txt. Matches / two-view geometries of pairs outside it are deleted (2026-11-05: pn_w2_21 resumed
    from an unpruned snapshot and MTGS_MAX_PAIR_DIST had no effect, the mapper saw every far pair again)."""
    import sqlite3
    if not os.path.exists(pairs_txt):
        return
    db = sqlite3.connect(db_path)
    ids = {n: i for n, i in db.execute("select name, image_id from images")}
    allowed = set()
    if not ids:
        db.close(); return
    with open(pairs_txt) as f:
        for line in f:
            t = line.split()
            if len(t) < 2 or t[0] not in ids or t[1] not in ids:
                continue
            a, b = ids[t[0]], ids[t[1]]
            if a > b:
                a, b = b, a
            allowed.add(a * 2147483647 + b)
    if not allowed:   # a mismatch between the pair list and the database must never wipe the cache (2026-11-05: inverted name map deleted 367k matches)
        print(f"[sfm] WARNING: no pair of image_pairs.txt resolves against the imported database; cache left untouched", flush=True)
        db.close(); return
    db.execute("create temp table keep (pair_id integer primary key)")
    db.executemany("insert into keep values (?)", ((x,) for x in allowed))
    n_m = db.execute("select count(*) from matches where pair_id not in (select pair_id from keep)").fetchone()[0]
    n_g = db.execute("select count(*) from two_view_geometries where pair_id not in (select pair_id from keep)").fetchone()[0]
    db.execute("delete from matches where pair_id not in (select pair_id from keep)")
    db.execute("delete from two_view_geometries where pair_id not in (select pair_id from keep)")
    db.commit(); db.close()
    print(f"[sfm] imported database pruned to the pair list: {len(allowed)} pairs kept, {n_m} matches / {n_g} two-view geometries outside it deleted", flush=True)


def _drop_far_from_prior(colmap_path, model_dir, max_err_m, colmap_exe, scale=1.0):
    """Delete images whose reconstructed centre is more than max_err_m metres from their prior (pose_prior mode)."""
    # no pycolmap here: its GPU build needs libcudart at import, which the civo stage environment lacks -- pn_w2_16 died
    # at this line after every 30 h mapping (2026-11-09/10); the centres come from images.bin directly
    import sqlite3
    from nuplan_scripts.utils.colmap_utils.read_write_model import read_images_binary, qvec2rotmat
    centres = {i: -qvec2rotmat(im.qvec).T @ im.tvec for i, im in read_images_binary(os.path.join(model_dir, "images.bin")).items()}
    db = sqlite3.connect(f"{colmap_path}/database.db")
    cols = [c[1] for c in db.execute("pragma table_info(pose_priors)")]
    key = "image_id" if "image_id" in cols else "corr_data_id"      # COLMAP 4.2 vs 4.3 schema
    pri = {i: np.frombuffer(p, dtype=np.float64) for i, p in db.execute(f"select {key}, position from pose_priors")}
    db.close()
    max_err = max_err_m * scale
    err = {i: np.linalg.norm(c - pri[i]) for i, c in centres.items() if i in pri}
    bad = [i for i, e in err.items() if e > max_err]
    e_m = np.array(list(err.values())) / scale if err else np.zeros(1)
    print(f"[sfm] {len(bad)} of {len(centres)} images > {max_err_m} m from their prior "
          f"(median {np.median(e_m):.3f} m, p90 {np.percentile(e_m, 90):.3f} m)", flush=True)
    if not bad:
        return
    ids = os.path.join(colmap_path, "prior_outliers.txt")
    open(ids, "w").write("\n".join(map(str, bad)) + "\n")
    tmp = model_dir + "_filtered"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    _run_colmap([colmap_exe, "image_deleter", "--input_path", model_dir, "--output_path", tmp, "--image_ids_path", ids],
                colmap_path, check=True)
    shutil.rmtree(model_dir); shutil.move(tmp, model_dir)


_ONE_BLAS_THREAD = {"feature_extractor", "matches_importer", "exhaustive_matcher", "sequential_matcher",
                    "spatial_matcher", "vocab_tree_matcher", "transitive_matcher"}


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

    # MTGS_SFM_IMPORT=<model dir>: take a finished SfM model (e.g. an incremental mapper run on
    # a faster single-core machine from this block's own database, 2026-10-25) instead of
    # extracting, matching and mapping here. Its image names must follow this run's naming
    # (rig mode: <veh>_<cam>/<frame token>.jpg), which name_to_token.json maps back.
    imp = os.environ.get("MTGS_SFM_IMPORT")
    if os.environ.get("MTGS_JOINT_SFM") == "1" and imp:
        if not os.path.exists(os.path.join(imp, "images.bin")):
            print(f"[sfm] MTGS_SFM_IMPORT={imp} has no images.bin"); sys.exit(1)
        dst = os.path.join(colmap_path, output_model, "0")
        shutil.rmtree(dst, ignore_errors=True); shutil.copytree(imp, dst)
        print(f"[sfm] imported model {imp} -> {dst}", flush=True)
        sys.exit(0)

    # MTGS_SFM_DB_IMPORT=<database.db>: resume from a snapshot of this block's COLMAP database (civo node
    # reboot 2026-11-05: joint matching takes ~15 h per block and the stage restarts from scratch otherwise).
    # Features are kept as they are; matches_importer skips pairs that are already matched, so a snapshot
    # taken mid-matching resumes where it stopped; one taken after matching goes straight to the mapper.
    dbimp = os.environ.get("MTGS_SFM_DB_IMPORT")
    if os.environ.get("MTGS_JOINT_SFM") == "1" and dbimp and os.path.exists(dbimp):
        shutil.copy2(dbimp, f"{colmap_path}/database.db")
        print(f"[sfm] imported database {dbimp} ({os.path.getsize(dbimp)/1e9:.2f} GB): extraction skipped, matching resumes", flush=True)
        _prune_db_to_pair_list(f"{colmap_path}/database.db", f"{colmap_path}/image_pairs.txt")
    else:
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
            rig_cfg = f"{colmap_path}/rig_config.json"
            if os.environ.get("MTGS_JOINT_SFM") == "1" and os.path.exists(rig_cfg):
                # rigs + frames from the calibration (ba_multi_traversal._write_rig_config); the
                # mapper then refines cam_from_rig (Mapper.ba_refine_sensor_from_rig, default on)
                print("[sfm rig] rig_configurator", flush=True)
                _run_colmap([colmap_exe, "rig_configurator", "--database_path", f"{colmap_path}/database.db",
                             "--rig_config_path", rig_cfg], colmap_path, check=True)
            else:
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
    if _ba_mode == "distonly":
        # COLMAP_BA_MODE=distonly (user 2026-10-31: "让colmap优化但限制同车同相机内参相同"): in the joint model every pass
        # of a vehicle already shares one camera (create_cameras_per_vehicle), so refining here gives ONE lens per
        # vehicle-camera; the calibrated focal length and principal point are kept (a forward-driving rig constrains
        # fy weakly: per-video refinement drifted fy to 0.90-0.96 of the calibration in ir_w2_06) and only the
        # OPENCV distortion is refined -- by the bundle_adjuster only: COLMAP's point_triangulator copies
        # --refine_intrinsics into ba_refine_focal_length and ba_refine_extra_params, overriding --Mapper.ba_refine_*
        # (is_/ij_/if_w2_06 refined fy to 1300-1420 in the triangulation), so the triangulation keeps the calibration.
        colmap_point_triangulator_args[colmap_point_triangulator_args.index("--refine_intrinsics") + 1] = "0"
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
        elif _ba_mode == "distonly":
            colmap_bundle_adjuster_args[colmap_bundle_adjuster_args.index("--BundleAdjustment.refine_focal_length") + 1] = "0"
            colmap_bundle_adjuster_args[colmap_bundle_adjuster_args.index("--BundleAdjustment.refine_extra_params") + 1] = "1"
    print(f"[ba] COLMAP_BA_MODE={_ba_mode}: {' '.join(colmap_bundle_adjuster_args[1:])}", flush=True)
    # MTGS_JOINT_SFM=1: pure structure-from-motion -- the mapper reconstructs from the matches
    # alone (no pose priors), intrinsics fixed to the calibration; the largest model becomes
    # <output_model>/0 and align_model() then puts it into the metric world frame with one
    # Sim3 against the lidar-registered camera centres (COLMAP has no scale of its own).
    # User 2026-10-25: multi-traversal reconstructions were best with plain COLMAP; the
    # prior-seeded triangulate+BA below can stay near the (inconsistent) prior poses.
    if os.environ.get("MTGS_JOINT_SFM") == "1":
        mode = os.environ.get("COLMAP_SFM_MAPPER", "incremental")
        raw = os.path.join(colmap_path, output_model + "_raw"); shutil.rmtree(raw, ignore_errors=True); os.makedirs(raw, exist_ok=True)
        # colmap may run inside a container (manco: ~/bin/colmap -> podman devbox) whose user
        # owns any directory IT creates, so the host could no longer delete sfm_model_raw/<n>
        # (PermissionError on images.bin, 2026-10-25). Pre-create the model dirs host-side.
        for _i in range(32):
            os.makedirs(os.path.join(raw, str(_i)), exist_ok=True)
        fix = os.environ.get("COLMAP_REFINE_INTRINSICS", "0") != "1"
        if mode == "global":
            # 2026-11-01: the global mapper (GLOMAP in COLMAP >= 4.3) refines focal/extra params by default;
            # honour COLMAP_REFINE_INTRINSICS like the incremental branch (calibrated lens fixed when 0).
            mapper_args = [colmap_exe, "global_mapper", "--database_path", f"{colmap_path}/database.db",
                           "--image_path", f"{colmap_path}/images", "--output_path", raw,
                           "--GlobalMapper.num_threads", f"{num_workers}",
                           # Ceres on the CPU: COLMAP 4.3 defaults gp_use_gpu/ba_ceres_use_gpu to 1 and the GPU
                           # solver gives worse models (user finding, 2026-11-01)
                           "--GlobalMapper.gp_use_gpu", "0", "--GlobalMapper.ba_ceres_use_gpu", "0",
                           "--GlobalMapper.ba_refine_focal_length", "0" if fix else "1",
                           "--GlobalMapper.ba_refine_principal_point", "0",
                           "--GlobalMapper.ba_refine_extra_params", "0" if fix else "1"]
        elif mode == "pose_prior":
            # 2026-11-04 (pn_w2_21): incremental mapper with hard position priors from the registered poses
            # (sparse_model/images.txt -> pose_priors table), std COLMAP_PRIOR_STD m. With 0.1 m and
            # MTGS_MAX_PAIR_DIST=25 the block that fragmented under the plain mapper registered 3240/3240 images
            # at 0.07 m median / 0.14 m p90 from the priors, 0.81 px reprojection (33 outliers > 1.5 m).
            scale = _colmap_scale(colmap_path)
            std_m = float(os.environ.get("COLMAP_PRIOR_STD", "0.1")); std = f"{std_m * scale:.6g}"
            print(f"[sfm] prior std {std_m} m = {std} colmap units (scale {scale:.5g})", flush=True)
            _write_pose_priors(colmap_path)
            mapper_args = [colmap_exe, "pose_prior_mapper", "--database_path", f"{colmap_path}/database.db",
                           "--image_path", f"{colmap_path}/images", "--output_path", raw,
                           "--Mapper.num_threads", f"{num_workers}",
                           "--Mapper.ba_refine_focal_length", "0" if fix else "1",
                           "--Mapper.ba_refine_principal_point", "0",
                           "--Mapper.ba_refine_extra_params", "0" if fix else "1",
                           "--Mapper.multiple_models", "1", "--Mapper.min_model_size", "50",
                           "--overwrite_priors_covariance", "1",
                           "--prior_position_std_x", std, "--prior_position_std_y", std, "--prior_position_std_z", std,
                           "--use_robust_loss_on_prior_position", os.environ.get("COLMAP_PRIOR_ROBUST", "0")]
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
        models = [d for d in sorted(os.listdir(raw), key=lambda x: int(x) if x.isdigit() else 1e9)
                  if os.path.exists(os.path.join(raw, d, "images.bin"))]
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
        if mode == "pose_prior" and float(os.environ.get("MTGS_SFM_PRIOR_MAX_ERR", "0")) > 0:
            _drop_far_from_prior(colmap_path, os.path.join(colmap_path, output_model, "0"),
                                 float(os.environ["MTGS_SFM_PRIOR_MAX_ERR"]), colmap_exe, scale=_colmap_scale(colmap_path))
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
