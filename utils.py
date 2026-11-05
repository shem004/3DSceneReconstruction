import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import numpy as np
import open3d as o3d
import os
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

DEPTH_SCALE  = 1000.0     # uint16 depth PNGs store millimetres
# Image axes (+X right, +Y down, +Z forward) to Habitat's sensor axes
# (+X right, +Y up, +Z backward), used by the scorer.
CAM_TO_GT_AXES = np.diag([1.0, -1.0, -1.0])



def depth_image_to_point_cloud(rgb, depth_m, width, height, hfov):
    """Back-project one RGB-D frame into a colored 3-D point cloud.

    `rgb` is HxWx3 uint8 in BGR order and `depth_m` the same frame's HxW
    depths in metres; `width`/`height`/`hfov` are that capture's own camera
    parameters (read from its `intrinsics.json` by the caller — never
    hardcoded, so one capture is never unprojected through another's
    camera). Geometry is a pinhole with square pixels, image-centre
    principal point, and focal length derived from `width` and `hfov`.

    Returns points in metres in the frame's camera frame (+X right, +Y
    down, +Z forward) with colors as RGB floats in [0, 1]. A pixel yields
    a point iff `depth_m > 0` (zero means no return and is dropped, never
    placed at the origin).
    Points come out in row-major pixel order; an all-invalid frame yields
    an empty cloud. Inputs are not modified.
    """
    # TODO: Convert `rgb` and `depth_m` to NumPy-compatible arrays and
    # validate that the depth shape agrees with `(height, width)`.
    rgb = np.asarray(rgb)
    # `depth_m` is already in metres don't rescaled again.
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if depth_m.shape != (height, width):
        raise ValueError(f"Expected depth shape {(height, width)}, got {depth_m.shape}")
    if rgb.shape != (height, width, 3):
        raise ValueError(f"Expected rgb shape {(height, width, 3)}, got {rgb.shape}")

    # TODO: Convert depth values to metres when the input representation
    # requires it; follow the capture convention used by the caller and do
    # not hardcode camera parameters from another capture.
    cx, cy = width/2.0, height/2.0
    fx = width / (2.0 * np.tan(np.deg2rad(hfov)/2.0))
    fy = fx #assume square pixels

    # TODO: Create a row-major pixel grid `(u, v)` covering the image.
    v, u = np.indices((height, width))

    # TODO: Build a validity mask and discard invalid depth values
    # (`depth <= 0`).
    valid = np.isfinite(depth_m) & (depth_m > 0)
    z = depth_m[valid]
    u_val = u[valid]
    v_val = v[valid]

    # TODO: Derive focal lengths and the principal point from `width`,
    # `height`, and `hfov`.

    # TODO: Implement pinhole unprojection:
    x = (u_val - cx) * z / fx
    y = (v_val - cy) * z / fy
    #z = depth

    points = np.stack((x,y,z), axis = -1)
    # Use the coordinate sign convention documented by the function. In an
    # assignment variant that uses a camera looking toward `-Z`, the final
    # line is `z = -depth`; reconcile that choice with the existing
    # docstring and `CAM_TO_GT_AXES` convention before implementing.
    # TODO: Construct and return an Open3D `PointCloud`, including RGB
    # colors normalized to `[0, 1]` and preserving row-major point order.
    colors = rgb[valid][:, ::-1]/255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    #Can is use Open3D here this last part?
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    """Downsample a cloud and prepare it for feature matching and ICP.

    Voxel-downsamples at `voxel_size`, estimates normals (radius
    `voxel_size * 2`, up to 30 neighbours) for point-to-plane ICP, and
    computes FPFH descriptors (radius `voxel_size * 5`, up to 100
    neighbours) for RANSAC matching. Returns the downsampled cloud and
    its FPFH feature set. Pairwise registration is performed by reconstruct().
    """
    # TODO: Voxel-downsample `pcd` with `voxel_size`.
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # TODO: Estimate normals for `pcd_down`, because Point-to-Plane ICP
    # requires them. Use a hybrid KD-tree search radius of
    # `voxel_size * 2` and at most 30 neighbours.
    radius_normal = voxel_size * 2 
    pcd_down.estimate_normals((o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)))

    # TODO: Compute FPFH features for `pcd_down` using radius
    # `voxel_size * 5` and at most 100 neighbours.
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    # TODO: Return the downsampled cloud and its FPFH feature object as
    # `(pcd_down, fpfh)`.
    return pcd_down, fpfh


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """Refine an initial alignment with point-to-plane ICP.

    Estimates normals on either cloud if missing (radius `threshold * 2`,
    up to 30 neighbours), then runs up to 100 ICP iterations at the single
    correspondence `threshold`. Returns the RegistrationResult; its
    `.transformation` is the refined 4x4.
    """
    # TODO: Ensure both source and target clouds have normals, estimating
    # them if necessary.
    if not source_down.has_normals():
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30)
        )
    if not target_down.has_normals():
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30)
            )
        
    # TODO: Call `o3d.pipelines.registration.registration_icp` with
    # `source_down`, `target_down`, `threshold`, and `trans_init`.
    # TODO: Use `TransformationEstimationPointToPlane()` and an explicit
    # convergence criterion.
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    # TODO: Return the registration result unchanged so its transformation
    # and metrics remain available to `reconstruct` and diagnostics
    # callbacks.
    return result
    


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """From-scratch point-to-plane ICP (the optional Bonus Track backend).

    Each iteration transforms the source by the current estimate, finds
    nearest neighbours in the target via cKDTree, keeps pairs closer than
    `voxel_size * 1.5` (stops below 10 survivors), and solves the linearised
    point-to-plane update (small-angle rotation + translation) using the
    target normals. The delta is composed onto the running transform for up
    to 60 iterations, stopping when the mean inlier error changes by less
    than 1e-6. Returns a small object exposing the final 4x4 as
    `.transformation`.
    """
    # TODO: Initialize the running transform from `trans_init`.
    src_pts = np.asarray(source_down.points, dtype=np.float64)
    tgt_pts = np.asarray(target_down.points, dtype=np.float64)
    T = np.array(trans_init, dtype=np.float64, copy=True)
    threshold = float(voxel_size) * 1.5

    # Target normals are required for point-to-plane. Estimate them on a copy
    # if missing so the caller's cloud is not modified.
    if target_down.has_normals():
        tgt_nrm = np.asarray(target_down.normals, dtype=np.float64)
    else:
        tgt_tmp = o3d.geometry.PointCloud(target_down)
        tgt_tmp.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
        tgt_nrm = np.asarray(tgt_tmp.normals, dtype=np.float64)

    class _ICPResult:
        def __init__(self, transformation, fitness=0.0, inlier_rmse=0.0, correspondences=None):
            self.transformation = transformation
            self.fitness = fitness
            self.inlier_rmse = inlier_rmse
            self.correspondence_set = [] if correspondences is None else correspondences

    if len(src_pts) < 10 or len(tgt_pts) < 10:
        return _ICPResult(T)

    tree = cKDTree(tgt_pts)
    prev_error = float("inf")

    for _ in range(60):  # max 60 iterations
        # TODO: Transform source points with the current estimate.
        src_trans = src_pts @ T[:3, :3].T + T[:3, 3]

        # TODO: Establish nearest-neighbour correspondences against the target,
        # then reject pairs outside the distance threshold.
        dists, indices = tree.query(src_trans)
        mask = dists < threshold
        if np.count_nonzero(mask) < 10:
            break

        p = src_trans[mask]            # current source points  (N,3)
        q = tgt_pts[indices[mask]]     # matched target points  (N,3)
        n = tgt_nrm[indices[mask]]     # target normals at q    (N,3)

        # TODO: Point-to-plane linearised update. For a small rotation w and
        # translation t, the residual is
        #     n . ((I + [w]x) p + t - q) = n.(p - q) + w.(p x n) + n.t
        # so each pair gives one row of  A x = b  with x = [w, t]:
        #     A_i = [ p x n , n ],   b_i = n . (q - p)
        A = np.hstack((np.cross(p, n), n))              # (N,6)
        b = np.einsum("ij,ij->i", n, q - p)             # (N,)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)       # least-squares, safe if degenerate
        if not np.all(np.isfinite(x)):
            break

        # Rebuild a true rotation from the small-angle vector (Rodrigues),
        # so the update stays rigid rather than I + [w]x.
        T_delta = np.eye(4, dtype=np.float64)
        T_delta[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
        T_delta[:3, 3] = x[3:]

        # TODO: Compose the update into the running 4x4 transform.
        T = T_delta @ T

        # TODO: Stop on convergence (< 1e-6 change in mean inlier error).
        mean_error = float(np.mean(dists[mask]))
        if abs(prev_error - mean_error) < 1e-6:
            break
        prev_error = mean_error

    # Final evaluation at the FINAL transform, so fitness / RMSE / correspondence
    # count describe the returned T (these feed the ICPInlier* factors).
    src_trans = src_pts @ T[:3, :3].T + T[:3, 3]
    dists, indices = tree.query(src_trans)
    mask = dists < threshold
    n_in = int(np.count_nonzero(mask))
    fitness = n_in / len(src_pts)
    inlier_rmse = float(np.sqrt(np.mean(dists[mask] ** 2))) if n_in > 0 else 0.0
    correspondences = np.column_stack((np.flatnonzero(mask), indices[mask]))

    return _ICPResult(T, fitness=float(fitness), inlier_rmse=inlier_rmse,
                      correspondences=correspondences)


def remove_ceiling_points(point_cloud, margin=1.0):
    """Return the map without the ceiling band, in frame-0 camera coordinates.

    Camera-frame +Y points down, so ceiling returns occupy the map's minimum
    Y values. Drop the same 1 m band that the viewer historically hid. This
    operates only on the merged output map; registration and trajectory scores
    continue to use the original per-frame depth clouds.
    """
    points = np.asarray(point_cloud.points)
    if len(points) == 0:
        return point_cloud
    keep = points[:, 1] > (points[:, 1].min() + float(margin))
    return point_cloud.select_by_index(np.flatnonzero(keep))


# ══════════════════════════════════════════════════════════════════════════════
#  Reconstruction (headless) + metric
# ══════════════════════════════════════════════════════════════════════════════
def reconstruct(data_root, version="open3d", voxel_size=0.05, verbose=True,
                build_cloud=True, frames=None,
                down_voxel=None, registration_seed=0,
                pipeline_callback=None,
                diagnostics_callback=None):
    """Reconstruct a camera trajectory and optionally merge its point clouds.

    Steps for each selected frame: build and preprocess an RGB-D cloud; register
    it to the previous frame with RANSAC initialization and ICP refinement; chain
    the transform and, when requested, add the cloud to the map.

    Frame selection:
      frames=None -> whole batch: every stem present under BOTH rgb/ and
        depth/, ascending numeric order.
      frames=[...] -> only those integer stems, IN THE GIVEN ORDER;
        "consecutive" means consecutive in the subset.

    Returns (global_pcd, pred_cam_pos):
      global_pcd   : every frame's cloud in the FRAME-0 CAMERA frame
                     (+X right, +Y down, +Z forward). Empty when
                     build_cloud=False (trajectory identical either way).
      pred_cam_pos : (N,3) float64 camera centres, frame-0 anchored.
                     GT stays on disk; visualize_and_evaluate loads and
                     subsets it by the same stems when scoring.
    """
    # TODO Stage 1: Load capture-specific intrinsics (`width`, `height`,
    # `hfov`) and select RGB/depth frame pairs in the requested order.
    # `frames=None` uses numeric batch order; an explicit `frames` list
    # keeps its given order.
    intrinsics_path = os.path.join(data_root, "intrinsics.json")
    with open(intrinsics_path, "r", encoding="utf-8") as f:
        intr = json.load(f)
    width = int(intr["width"])
    height = int(intr["height"])
    hfov = float(intr["hfov"])
    depth_scale = float(intr.get("depth_scale", DEPTH_SCALE))


    rgb_dir = os.path.join(data_root, "rgb")
    depth_dir = os.path.join(data_root, "depth")
    rgb_map = {
        int(os.path.splitext(fn)[0]): os.path.join(rgb_dir, fn)
        for fn in os.listdir(rgb_dir)
        if fn.endswith(".png") and os.path.splitext(fn)[0].isdigit()
    }
    depth_map = {
        int(os.path.splitext(fn)[0]): os.path.join(depth_dir, fn)
        for fn in os.listdir(depth_dir)
        if fn.endswith(".png") and os.path.splitext(fn)[0].isdigit()
    }
    common_stems = set(rgb_map.keys()) & set(depth_map.keys())

    if frames is None:
        selected_stems = sorted(common_stems)
    else:
        selected_stems = [int(s) for s in frames if int(s) in common_stems]

    pairs = [(rgb_map[s], depth_map[s]) for s in selected_stems]

    T_global = np.eye(4)          # frame i -> frame-0 camera frame
    pred_cam_pos = []
    global_pcd = o3d.geometry.PointCloud()
    prev_down = prev_fpfh = None
    prev_depth_path = None
    prev_stem = None
    prev_pair_index = None
    icp_thr = voxel_size * 1.5

    for i, (rgb_path, depth_path) in enumerate(pairs):
        # Stage 2: Extract stem, read RGB & Depth frame, convert to meters
        # skip unreadable frames using
        # the existing warning and control-flow behavior, 
        stem = int(os.path.splitext(os.path.basename(rgb_path))[0])
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_m = None if depth_raw is None else depth_raw.astype(np.float64) / depth_scale

        if rgb is None or depth_m is None:
            if verbose:
                print(f"  Warning: could not load frame {i}, skipping.")
            continue


        # TODO Stage 3: Convert the frame to a point cloud with
        # `depth_image_to_point_cloud`. DONE
        pcd = depth_image_to_point_cloud(rgb, depth_m, width, height, hfov)
        # TODO Stage 4: Preprocess it with `preprocess_point_cloud` for
        # voxel downsampling, normals, and FPFH. DONE
        cur_down, cur_fpfh = preprocess_point_cloud(pcd, voxel_size)
        if pipeline_callback is not None:
            pipeline_callback("frame", {
                "stem": stem,
                "raw_point_count": len(pcd.points),
                "point_yield": len(cur_down.points),
            })

        if prev_down is not None:
            # TODO Stage 6: For every frame after the first processed frame,
            # perform global FPFH feature matching and RANSAC registration
            # to obtain `trans_init`.

            #Global RANSAC Regis.
            if registration_seed is not None and hasattr(o3d.utility, "random"):
                o3d.utility.random.seed(int(registration_seed))
            distance_threshold = voxel_size * 1.5
            coarse_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                cur_down, prev_down,               # source cloud, target cloud
                cur_fpfh, prev_fpfh,               # source FPFH, target FPFH
                True,                     # mutual_filter
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
            trans_init = coarse_result.transformation

            # TODO Stage 7: Run the selected local backend: Open3D
            # Point-to-Plane ICP for `version == "open3d"`, otherwise the
            # custom ICP backend.
            if version == 'open3d':
                result_icp = local_icp_algorithm(
                    cur_down, prev_down, trans_init, icp_thr)
            else:   # my_icp
                result_icp = my_local_icp_algorithm(
                    cur_down, prev_down, trans_init, voxel_size)
                
            # TODO Stage 8: Accumulate the returned transform into the
            # frame-0 coordinate system (`T_global` maps each processed
            # frame into the frame-0 camera coordinate system;
            # registration is always current frame to the previous
            # processed frame).

            #So transforms into camera's frame-0
            T_global = T_global @ result_icp.transformation

            if pipeline_callback is not None and prev_pair_index == i - 1:
                pipeline_callback("pair", {
                    "prev_stem": prev_stem,
                    "stem": stem,
                    "coarse_fitness": float(coarse_result.fitness),
                    "coarse_rmse": float(coarse_result.inlier_rmse),
                    "coarse_correspondences": len(coarse_result.correspondence_set),
                    "icp_fitness": float(getattr(result_icp, "fitness", float("nan"))),
                    "icp_rmse": float(getattr(result_icp, "inlier_rmse", float("nan"))),
                    "icp_correspondences": len(getattr(result_icp, "correspondence_set", [])),
                })
            if diagnostics_callback is not None and prev_pair_index == i - 1:
                diagnostics_callback({
                    "frame_index": i,
                    "prev_stem": prev_stem,
                    "stem": stem,
                    "raw_point_count": len(pcd.points),
                    "point_yield": len(cur_down.points),
                    "coarse_fitness": float(coarse_result.fitness),
                    "coarse_rmse": float(coarse_result.inlier_rmse),
                    "coarse_correspondences": len(coarse_result.correspondence_set),
                    "icp_fitness": float(result_icp.fitness),
                    "icp_rmse": float(result_icp.inlier_rmse),
                    "icp_correspondences": len(getattr(result_icp, "correspondence_set", [])),
                    "trans_init": trans_init,
                    "icp_transformation": result_icp.transformation,
                })

        # TODO Stage 10: Append the current camera position to
        # `pred_cam_pos` (one entry per successfully processed frame).
        pred_cam_pos.append(T_global[:3, 3].copy())

        if build_cloud:
            # TODO Stage 11: If `build_cloud` is enabled, optionally
            # downsample only the merged map, transform the current cloud
            # by the accumulated transform, and add it to `global_pcd`.
            # This must not alter trajectory estimation.
            if down_voxel is not None and down_voxel > 0:
                frame_cloud = pcd.voxel_down_sample(float(down_voxel))
            else:
                frame_cloud = o3d.geometry.PointCloud(pcd)
            frame_cloud.transform(T_global)
            global_pcd += frame_cloud

        # TODO Stage 12: Update previous-frame state and preserve progress
        # reporting.
        prev_down, prev_fpfh = cur_down, cur_fpfh
        prev_depth_path = depth_path
        prev_stem = stem
        prev_pair_index = i

        if verbose and (i % 25 == 0 or i == len(pairs) - 1):
            print(f"  frame {i:>4d}/{len(pairs)-1}")

    if not pred_cam_pos:
        return o3d.geometry.PointCloud(), np.zeros((0, 3))
    if build_cloud:
        global_pcd = remove_ceiling_points(global_pcd)
    return global_pcd, np.array(pred_cam_pos, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation + visualisation (used by the thin reconstruct.py CLI)
# ══════════════════════════════════════════════════════════════════════════════
def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, data_root,
                           frames=None, title="reconstruction", show=True,
                           save_path=None):
    """Score the trajectory and visualise map + trajectories.

    GT is loaded here from data_root/GT_pose.npy as (M,7)
    [x,y,z,qw,qx,qy,qz], or None when absent. GT rows are in capture order
    while image stems are identifiers, so when `frames` is given each stem
    is mapped to its ordinal in the sorted common RGB/depth frame list
    before indexing (stems absent from the capture or beyond GT are skipped,
    requested order preserved) — the score stays meaningful on subsets.

    1. Mean L2 over camera centres: reconcile frames first (pred is frame-0
       camera on image axes → flip Y and Z; GT world poses → R0^T (t_i - t_0)
       via pose 0), then mean over i of ||pred[i] - gt[i]||_2 with
       L2 = sqrt(dx^2 + dy^2 + dz^2). No scale/rotation/offset fitting TO the
       GT. inf when either side is missing/empty. Prints it and returns it.
    2. Estimated trajectory LineSet (red), GT trajectory LineSet (black).
    3. Open3D window with the ceiling-cropped map plus both trajectories
       (skipped when show=False, e.g. headless scoring). Ceiling points have
       already been removed from the merged map by `reconstruct`.
    """
    # TODO: Load `GT_pose.npy` when present and map requested `frames` to
    # the correct ground-truth rows while preserving requested order.
    gt_poses = None
    if data_root is not None:
        gt_path = os.path.join(data_root, "GT_pose.npy")
        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            if frames is None:
                gt_poses = gt
            else:
                rgb_dir = os.path.join(data_root, "rgb")
                depth_dir = os.path.join(data_root, "depth")
                if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir):
                    rgb_stems = {int(os.path.splitext(f)[0]) for f in os.listdir(rgb_dir) if f.endswith(".png") and os.path.splitext(f)[0].isdigit()}
                    depth_stems = {int(os.path.splitext(f)[0]) for f in os.listdir(depth_dir) if f.endswith(".png") and os.path.splitext(f)[0].isdigit()}
                    ordered_stems = sorted(rgb_stems & depth_stems)
                    stem_to_idx = {s: idx for idx, s in enumerate(ordered_stems)}
                    rows = [stem_to_idx[int(s)] for s in frames if int(s) in stem_to_idx and stem_to_idx[int(s)] < len(gt)]
                    gt_poses = gt[rows]
                else:
                    gt_poses = gt[:len(frames)]
    # TODO: Handle missing or empty ground truth and empty predictions
    # without raising; return an infinite mean error when no score can be
    # computed.
    pred_disp = None
    gt_disp = None
    if predicted_cam_poses is not None and len(predicted_cam_poses) > 0:
        pred_disp = np.asarray(predicted_cam_poses, dtype=np.float64) @ CAM_TO_GT_AXES

    if gt_poses is not None and len(gt_poses) > 0:
        gt = np.asarray(gt_poses, dtype=np.float64)
        t = gt[:, :3]
        qw, qx, qy, qz = gt[0, 3:7]
        R0 = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        gt_disp = (t - t[0]) @ R0
    # TODO: Reconcile predicted and ground-truth coordinate conventions
    # using the existing camera-axis and frame-0 alignment conventions.
    # TODO: Compute mean camera-centre L2 distance:
    #   dx = predicted_x - gt_x
    #   dy = predicted_y - gt_y
    #   dz = predicted_z - gt_z
    #   L2 = sqrt(dx**2 + dy**2 + dz**2)
    #   mean_l2 = mean(L2)
    if pred_disp is None or gt_disp is None or len(pred_disp) == 0 or len(gt_disp) == 0:
        n = 0
        mean_l2_error = float("inf")
    else:
        n = min(len(pred_disp), len(gt_disp))
        dx = pred_disp[:n, 0] - gt_disp[:n, 0]
        dy = pred_disp[:n, 1] - gt_disp[:n, 1]
        dz = pred_disp[:n, 2] - gt_disp[:n, 2]
        l2 = np.sqrt(dx**2 + dy**2 + dz**2)
        mean_l2_error = float(np.mean(l2))


    print(f"Mean L2 distance: {mean_l2_error:.6f} meters  (over {n} frames, "
          f"{0 if predicted_cam_poses is None else len(predicted_cam_poses)} "
          f"reconstructed)")
    if save_path is not None:
        save_reconstruction_image(
            reconstructed_pcd, predicted_cam_poses, data_root,
            save_path, frames=frames, title=title)
    if not show:
        return mean_l2_error
    # TODO: Create a `LineSet` for the estimated trajectory colored red.
    # TODO: Create a `LineSet` for the ground-truth trajectory colored black.
    # TODO: Add the available map and trajectories to the Open3D
    # visualization when `show=True`; keep headless scoring functional when
    # `show=False`.
    geoms = []
    if reconstructed_pcd is not None and len(reconstructed_pcd.points) > 0:
        scene_vis = o3d.geometry.PointCloud(reconstructed_pcd)
        to_gt_axes = np.eye(4)
        to_gt_axes[:3, :3] = CAM_TO_GT_AXES
        scene_vis.transform(to_gt_axes)
        geoms.append(scene_vis)

    if pred_disp is not None and len(pred_disp) > 1:
        pred_pts = pred_disp[:n] if n > 1 else pred_disp
        edges = [[i, i + 1] for i in range(len(pred_pts) - 1)]
        pred_lines = o3d.geometry.LineSet()
        pred_lines.points = o3d.utility.Vector3dVector(pred_pts)
        pred_lines.lines = o3d.utility.Vector2iVector(edges)
        pred_lines.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * len(edges)) # Red
        geoms.append(pred_lines)

    if gt_disp is not None and len(gt_disp) > 1:
        gt_pts = gt_disp[:n] if n > 1 else gt_disp
        edges = [[i, i + 1] for i in range(len(gt_pts) - 1)]
        gt_lines = o3d.geometry.LineSet()
        gt_lines.points = o3d.utility.Vector3dVector(gt_pts)
        gt_lines.lines = o3d.utility.Vector2iVector(edges)
        gt_lines.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]] * len(edges))   # Black
        geoms.append(gt_lines)

    print(f"[visualize] opening visualiser for {title} …  (press Q to close)")
    o3d.visualization.draw_geometries(
        geoms, window_name=f'HW1 Reconstruction - {title}',
        zoom=0.5, front=[0, -1, 0], lookat=[0, 0, 0], up=[0, 0, 1])
    return mean_l2_error

def save_reconstruction_image(reconstructed_pcd, predicted_cam_poses, data_root,
                              save_path, frames=None, title="reconstruction"):
    """Render reconstructed point cloud and camera trajectories to a PNG file."""
    if save_path is None:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    gt_poses = None
    if data_root is not None:
        gt_path = os.path.join(data_root, 'GT_pose.npy')
        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            if frames is None:
                gt_poses = gt
            else:
                rgb_dir = os.path.join(data_root, 'rgb')
                if os.path.exists(rgb_dir):
                    ordered_stems = sorted(
                        int(os.path.splitext(f)[0])
                        for f in os.listdir(rgb_dir)
                        if f.endswith('.png'))
                    rows = [ordered_stems.index(int(stem)) for stem in frames
                            if int(stem) in ordered_stems
                            and ordered_stems.index(int(stem)) < len(gt)]
                    gt_poses = gt[rows]
                else:
                    gt_poses = gt[:len(frames)]

    pred_disp = None
    gt_disp = None
    if predicted_cam_poses is not None and len(predicted_cam_poses) > 0:
        pred_disp = np.asarray(predicted_cam_poses, dtype=np.float64) @ CAM_TO_GT_AXES
    if gt_poses is not None and len(gt_poses) > 0:
        gt = np.asarray(gt_poses, dtype=np.float64)
        t = gt[:, :3]
        qw, qx, qy, qz = gt[0, 3:7]
        R0 = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        gt_disp = (t - t[0]) @ R0

    geoms = []
    if reconstructed_pcd is not None and len(reconstructed_pcd.points) > 0:
        scene_no_ceil = o3d.geometry.PointCloud()
        scene_no_ceil.points = o3d.utility.Vector3dVector(np.asarray(reconstructed_pcd.points).copy())
        if reconstructed_pcd.has_colors():
            scene_no_ceil.colors = o3d.utility.Vector3dVector(
                np.asarray(reconstructed_pcd.colors).copy())
        to_gt_axes = np.eye(4)
        to_gt_axes[:3, :3] = CAM_TO_GT_AXES
        scene_no_ceil.transform(to_gt_axes)
        geoms.append(scene_no_ceil)

    if pred_disp is not None and gt_disp is not None:
        n_min = min(len(pred_disp), len(gt_disp))
        if n_min > 1:
            for traj, color in ((pred_disp[:n_min], [1.0, 0.0, 0.0]),
                                (gt_disp[:n_min], [0.0, 0.0, 0.0])):
                edges = [[i, i + 1] for i in range(len(traj) - 1)]
                lineset = o3d.geometry.LineSet()
                lineset.points = o3d.utility.Vector3dVector(np.asarray(traj))
                lineset.lines = o3d.utility.Vector2iVector(edges)
                lineset.colors = o3d.utility.Vector3dVector([color] * len(edges))
                geoms.append(lineset)
    elif pred_disp is not None and len(pred_disp) > 1:
        edges = [[i, i + 1] for i in range(len(pred_disp) - 1)]
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.asarray(pred_disp))
        lineset.lines = o3d.utility.Vector2iVector(edges)
        lineset.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * len(edges))
        geoms.append(lineset)

    if not os.environ.get("DISPLAY"):
        if os.path.exists("/tmp/.X11-unix/X1"):
            os.environ["DISPLAY"] = ":1"
        elif os.path.exists("/tmp/.X11-unix/X0"):
            os.environ["DISPLAY"] = ":0"

    rendered = False
    if geoms and os.environ.get("DISPLAY"):
        try:
            vis = o3d.visualization.Visualizer()
            if vis.create_window(visible=False, width=1920, height=1080):
                for g in geoms:
                    vis.add_geometry(g)
                opt = vis.get_render_option()
                if opt is not None:
                    opt.background_color = np.asarray([1.0, 1.0, 1.0])
                    if hasattr(opt, "line_width"):
                        opt.line_width = 3.0
                ctr = vis.get_view_control()
                if ctr is not None:
                    ctr.set_zoom(0.5)
                    ctr.set_front([0, -1, 0])
                    ctr.set_lookat([0, 0, 0])
                    ctr.set_up([0, 0, 1])
                vis.poll_events()
                vis.update_renderer()
                vis.capture_screen_image(save_path, do_render=True)
                vis.destroy_window()
                rendered = os.path.isfile(save_path) and os.path.getsize(save_path) > 0
        except Exception:
            rendered = False

    if not rendered:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            if reconstructed_pcd is not None and len(reconstructed_pcd.points) > 0:
                pts = np.asarray(reconstructed_pcd.points)
                if len(pts) > 5000:
                    pts = pts[::len(pts) // 5000]
                pts_gt = pts @ CAM_TO_GT_AXES
                ax.scatter(pts_gt[:, 0], pts_gt[:, 1], pts_gt[:, 2], s=0.5, c="gray", alpha=0.3)
            if pred_disp is not None and len(pred_disp) > 0:
                ax.plot(pred_disp[:, 0], pred_disp[:, 1], pred_disp[:, 2], "r-", label="Predicted")
            if gt_disp is not None and len(gt_disp) > 0:
                ax.plot(gt_disp[:, 0], gt_disp[:, 1], gt_disp[:, 2], "k-", label="Ground Truth")
            ax.set_title(title)
            ax.legend()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            rendered = os.path.isfile(save_path) and os.path.getsize(save_path) > 0
        except Exception as exc:
            print(f"[reconstruct] warning: failed to save reconstruction image {save_path}: {exc}")
    return save_path if rendered else None