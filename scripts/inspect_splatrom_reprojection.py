from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Matplotlib is required for the 3D outlier plot.") from exc

SCENE = Path(
    r"D:\zra\PROJECTS\Github\4C4D\data\SplatROM\ISH822_aria_inspecting_3ab0"
)
SPARSE = SCENE / "sparse" / "0"
IMAGE_DIR = SCENE / "images"
OUT_DIR = SCENE / "diagnostics" / "anchor0_reprojection_v1"

NAMES = [
    "cam00_0000.png",
    "cam01_0000.png",
    "cam02_0000.png",
    "cam03_0000.png",
]
CAM_SHORT = ["cam00", "cam01", "cam02", "cam03"]

SIFT_FEATURES = 6000
RATIO_TEST = 0.75
RANSAC_THRESHOLD_PX = 1.5
CALIBRATED_SAMPSON_THRESHOLD_PX = 2.0
MAX_REPROJECTION_PX = 4.0
MEDIAN_REPROJECTION_PX = 2.0
MIN_PARALLAX_DEG = 1.0
KNN_K = 8


def read_cameras(path: Path) -> dict:
    cameras = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if not p or p[0].startswith("#"):
            continue
        camera_id = int(p[0])
        model = p[1]
        width, height = int(p[2]), int(p[3])
        params = np.asarray([float(v) for v in p[4:]], dtype=np.float64)
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(f"Unsupported camera model: {model}")
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        cameras[camera_id] = {
            "K": K, "width": width, "height": height, "model": model
        }
    return cameras


def qvec_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_images(path: Path) -> dict:
    images = {}
    wanted = set(NAMES)
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) < 10 or p[0].startswith("#"):
            continue
        name = " ".join(p[9:])
        if name not in wanted:
            continue
        q = np.asarray([float(v) for v in p[1:5]], dtype=np.float64)
        t = np.asarray([float(v) for v in p[5:8]], dtype=np.float64)
        camera_id = int(p[8])
        R = qvec_to_rotation(q)
        C = -R.T @ t
        images[name] = {"R": R, "t": t, "C": C, "camera_id": camera_id}
    return images


def skew(t: np.ndarray) -> np.ndarray:
    x, y, z = t
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )


def sampson_errors(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([p1, np.ones(len(p1), dtype=np.float64)])
    x2 = np.column_stack([p2, np.ones(len(p2), dtype=np.float64)])
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numerator = np.sum(x2 * Fx1, axis=1) ** 2
    denominator = (
        Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2
        + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    )
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        if value not in self.parent:
            self.parent[value] = value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        grouped = {}
        for value in self.parent:
            grouped.setdefault(self.find(value), []).append(value)
        return list(grouped.values())


def triangulate_track(observations, projection_matrices):
    rows = []
    for camera_index, xy, _keypoint_index in observations:
        P = projection_matrices[camera_index]
        x, y = xy
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.asarray(rows, dtype=np.float64)
    _u, _s, vt = np.linalg.svd(A)
    Xh = vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    X = Xh[:3] / Xh[3]
    return X if np.isfinite(X).all() else None


def project_point(P: np.ndarray, X: np.ndarray):
    q = P @ np.append(X, 1.0)
    if not np.isfinite(q).all() or q[2] <= 1e-12:
        return None, float("inf")
    xy = q[:2] / q[2]
    return xy, float(q[2])


def robust_outlier_flags(points: np.ndarray):
    n = len(points)
    if n < 10:
        return np.zeros(n, dtype=bool), np.zeros(n), {"method": "insufficient_points"}

    # Pairwise distance is inexpensive for this 366-point diagnostic cloud.
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    k = min(KNN_K, n - 1)
    knn = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    knn_mean = knn.mean(axis=1)

    med = float(np.median(knn_mean))
    mad = float(np.median(np.abs(knn_mean - med)))
    robust_scale = 1.4826 * mad
    if robust_scale > 1e-12:
        threshold = med + 3.0 * robust_scale
        method = "mean_8nn > median + 3 * (1.4826 * MAD)"
    else:
        threshold = float(np.percentile(knn_mean, 97.5))
        method = "mean_8nn > 97.5th_percentile (MAD near zero)"

    flags = knn_mean > threshold
    details = {
        "method": method,
        "k_neighbors": int(k),
        "median_mean_knn_distance_m": med,
        "mad_mean_knn_distance_m": mad,
        "threshold_m": float(threshold),
        "flagged_count": int(flags.sum()),
        "warning": "Flags are review suggestions only; they are not automatic deletions or proof of bad geometry.",
    }
    return flags, knn_mean, details


def annotate_overlay(image_bgr, camera_index, camera_stats, observed_rows):
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    for row in observed_rows:
        ox, oy = row["observed_xy"]
        px, py = row["projected_xy"]
        err = row["error_px"]
        obs = (int(round(ox)), int(round(oy)))
        proj = (int(round(px)), int(round(py)))
        if not (0 <= obs[0] < w and 0 <= obs[1] < h):
            continue
        if not (0 <= proj[0] < w and 0 <= proj[1] < h):
            continue
        line_color = (0, 220, 255) if err <= 2.0 else (0, 0, 255)
        cv2.line(canvas, obs, proj, line_color, 1, cv2.LINE_AA)
        cv2.drawMarker(
            canvas, obs, (0, 255, 0), markerType=cv2.MARKER_CROSS,
            markerSize=8, thickness=1, line_type=cv2.LINE_AA
        )
        cv2.circle(canvas, proj, 3, (255, 80, 20), 1, cv2.LINE_AA)

    median = camera_stats["median_error_px"]
    p90 = camera_stats["p90_error_px"]
    title = f"{CAM_SHORT[camera_index]} | tracks={camera_stats['tracks']} | median={median:.3f}px | p90={p90:.3f}px"
    cv2.rectangle(canvas, (0, 0), (w, 34), (15, 15, 15), -1)
    cv2.putText(canvas, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 1, cv2.LINE_AA)
    legend = "GREEN cross: SIFT observation | BLUE circle: 3D reprojection | YELLOW line <=2px / RED line >2px"
    cv2.rectangle(canvas, (0, h - 28), (w, h), (15, 15, 15), -1)
    cv2.putText(canvas, legend, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    required = [SPARSE / "cameras.txt", SPARSE / "images.txt"] + [IMAGE_DIR / n for n in NAMES]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))

    cameras = read_cameras(SPARSE / "cameras.txt")
    images = read_images(SPARSE / "images.txt")
    if len(images) != 4:
        raise RuntimeError(f"Expected 4 camera anchor records, found {len(images)}: {list(images)}")

    feature_data = {}
    source_images = {}
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    print("=== EXTRACTING ANCHOR FEATURES ===")
    for ci, name in enumerate(NAMES):
        bgr = cv2.imread(str(IMAGE_DIR / name), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {IMAGE_DIR / name}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            raise RuntimeError(f"No SIFT descriptors for {name}")
        feature_data[ci] = (keypoints, descriptors)
        source_images[ci] = bgr
        print(f"{name}: {len(keypoints)} keypoints")

    projection_matrices = []
    camera_data = []
    for name in NAMES:
        info = images[name]
        cam = cameras[info["camera_id"]]
        P = cam["K"] @ np.column_stack([info["R"], info["t"]])
        projection_matrices.append(P)
        camera_data.append({
            "R": info["R"], "t": info["t"], "C": info["C"],
            "K": cam["K"], "width": cam["width"], "height": cam["height"],
        })

    dsu = DisjointSet()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pair_audit = []
    print("\n=== REBUILDING CALIBRATED TRACKS ===")
    for i in range(4):
        for j in range(i + 1, 4):
            kp1, d1 = feature_data[i]
            kp2, d2 = feature_data[j]
            knn = matcher.knnMatch(d1, d2, k=2)
            good = [m for pair in knn if len(pair) == 2
                    for m, n in [pair] if m.distance < RATIO_TEST * n.distance]
            if len(good) < 8:
                pair_audit.append({"pair": [NAMES[i], NAMES[j]], "raw": len(good), "ransac": 0, "calibrated": 0})
                continue
            p1 = np.asarray([kp1[m.queryIdx].pt for m in good], dtype=np.float64)
            p2 = np.asarray([kp2[m.trainIdx].pt for m in good], dtype=np.float64)
            a, b = images[NAMES[i]], images[NAMES[j]]
            R21 = b["R"] @ a["R"].T
            t21 = b["t"] - R21 @ a["t"]
            E = skew(t21) @ R21
            K1, K2 = camera_data[i]["K"], camera_data[j]["K"]
            F_cal = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
            _F_est, ransac_mask = cv2.findFundamentalMat(
                p1, p2, cv2.FM_RANSAC, RANSAC_THRESHOLD_PX, 0.999, 10000
            )
            if ransac_mask is None:
                pair_audit.append({"pair": [NAMES[i], NAMES[j]], "raw": len(good), "ransac": 0, "calibrated": 0})
                continue
            ransac_mask = ransac_mask.ravel().astype(bool)
            calibrated = sampson_errors(F_cal, p1, p2) <= CALIBRATED_SAMPSON_THRESHOLD_PX
            accepted = ransac_mask & calibrated
            for match, keep in zip(good, accepted):
                if keep:
                    dsu.union((i, match.queryIdx), (j, match.trainIdx))
            pair_audit.append({
                "pair": [NAMES[i], NAMES[j]], "raw": int(len(good)),
                "ransac": int(ransac_mask.sum()), "calibrated": int(accepted.sum())
            })
            print(f"{CAM_SHORT[i]}-{CAM_SHORT[j]}: raw={len(good)}, RANSAC={int(ransac_mask.sum())}, calibrated={int(accepted.sum())}")

    tracks = []
    ambiguous = 0
    for group in dsu.groups():
        per_camera = {}
        for ci, ki in group:
            per_camera.setdefault(ci, []).append(ki)
        if len(per_camera) < 2:
            continue
        if any(len(v) != 1 for v in per_camera.values()):
            ambiguous += 1
            continue
        obs = []
        for ci, indices in sorted(per_camera.items()):
            ki = indices[0]
            xy = np.asarray(feature_data[ci][0][ki].pt, dtype=np.float64)
            obs.append((ci, xy, ki))
        tracks.append(obs)

    print(f"Usable tracks: {len(tracks)}; ambiguous rejected: {ambiguous}")

    point_records = []
    rejected = Counter()
    for obs in tracks:
        X = triangulate_track(obs, projection_matrices)
        if X is None:
            rejected["invalid_homogeneous"] += 1
            continue

        errors = []
        projected_by_camera = {}
        positive_depth = True
        ray_vectors = []
        for ci, observed_xy, ki in obs:
            cam = camera_data[ci]
            camera_xyz = cam["R"] @ (X - cam["C"])
            if not np.isfinite(camera_xyz).all() or camera_xyz[2] <= 0:
                positive_depth = False
                break
            projected_xy, depth = project_point(projection_matrices[ci], X)
            if projected_xy is None:
                positive_depth = False
                break
            err = float(np.linalg.norm(projected_xy - observed_xy))
            errors.append(err)
            projected_by_camera[ci] = projected_xy
            ray = X - cam["C"]
            ray_norm = np.linalg.norm(ray)
            if ray_norm <= 1e-12:
                positive_depth = False
                break
            ray_vectors.append(ray / ray_norm)

        if not positive_depth:
            rejected["nonpositive_depth"] += 1
            continue
        if not errors or max(errors) > MAX_REPROJECTION_PX or np.median(errors) > MEDIAN_REPROJECTION_PX:
            rejected["reprojection"] += 1
            continue

        pair_angles = []
        for a in range(len(ray_vectors)):
            for b in range(a + 1, len(ray_vectors)):
                dot = float(np.clip(np.dot(ray_vectors[a], ray_vectors[b]), -1.0, 1.0))
                pair_angles.append(float(np.degrees(np.arccos(dot))))
        max_parallax = max(pair_angles) if pair_angles else 0.0
        if max_parallax < MIN_PARALLAX_DEG:
            rejected["low_parallax"] += 1
            continue

        point_records.append({
            "xyz": X,
            "observations": obs,
            "projected_by_camera": projected_by_camera,
            "errors": errors,
            "max_parallax_deg": max_parallax,
        })

    if not point_records:
        raise RuntimeError("No valid multi-view triangulated tracks were produced.")

    points = np.asarray([r["xyz"] for r in point_records], dtype=np.float64)
    outlier_flags, knn_scores, outlier_info = robust_outlier_flags(points)
    print(f"Valid 3D tracks: {len(points)}")
    print(f"Spatial review flags (not removed): {int(outlier_flags.sum())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_camera_rows = {i: [] for i in range(4)}
    all_errors = []
    csv_rows = []

    for index, record in enumerate(point_records, start=1):
        X = record["xyz"]
        obs_map = {ci: (xy, ki) for ci, xy, ki in record["observations"]}
        row = {
            "track_id": index,
            "x_m": float(X[0]), "y_m": float(X[1]), "z_m": float(X[2]),
            "num_views": len(obs_map),
            "camera_names": ";".join(CAM_SHORT[ci] for ci in sorted(obs_map)),
            "max_parallax_deg": float(record["max_parallax_deg"]),
            "knn8_mean_distance_m": float(knn_scores[index - 1]),
            "spatial_review_flag": bool(outlier_flags[index - 1]),
        }
        for ci in range(4):
            prefix = CAM_SHORT[ci]
            uv_proj, depth = project_point(projection_matrices[ci], X)
            row[f"{prefix}_proj_x_px"] = float(uv_proj[0]) if uv_proj is not None else ""
            row[f"{prefix}_proj_y_px"] = float(uv_proj[1]) if uv_proj is not None else ""
            row[f"{prefix}_depth_m"] = float(depth) if uv_proj is not None else ""
            if ci in obs_map:
                observed_xy, _ki = obs_map[ci]
                err = float(np.linalg.norm(uv_proj - observed_xy))
                per_camera_rows[ci].append({
                    "track_id": index, "observed_xy": observed_xy,
                    "projected_xy": uv_proj, "error_px": err,
                })
                all_errors.append(err)
                row[f"{prefix}_obs_x_px"] = float(observed_xy[0])
                row[f"{prefix}_obs_y_px"] = float(observed_xy[1])
                row[f"{prefix}_reprojection_error_px"] = err
            else:
                row[f"{prefix}_obs_x_px"] = ""
                row[f"{prefix}_obs_y_px"] = ""
                row[f"{prefix}_reprojection_error_px"] = ""
        csv_rows.append(row)

    camera_stats = {}
    overlay_paths = []
    montage_cells = []
    for ci in range(4):
        rows = per_camera_rows[ci]
        errs = np.asarray([r["error_px"] for r in rows], dtype=np.float64)
        stats = {
            "tracks": int(len(rows)),
            "median_error_px": float(np.median(errs)) if len(errs) else None,
            "mean_error_px": float(np.mean(errs)) if len(errs) else None,
            "p90_error_px": float(np.percentile(errs, 90)) if len(errs) else None,
            "max_error_px": float(np.max(errs)) if len(errs) else None,
            "fraction_le_1px": float(np.mean(errs <= 1.0)) if len(errs) else None,
            "fraction_le_2px": float(np.mean(errs <= 2.0)) if len(errs) else None,
        }
        camera_stats[CAM_SHORT[ci]] = stats
        overlay = annotate_overlay(source_images[ci], ci, stats, rows)
        out_path = OUT_DIR / f"{CAM_SHORT[ci]}_reprojection_overlay.png"
        if not cv2.imwrite(str(out_path), overlay):
            raise RuntimeError(f"Could not write overlay: {out_path}")
        overlay_paths.append(str(out_path))
        scale = min(640 / overlay.shape[1], 512 / overlay.shape[0])
        resized = cv2.resize(overlay, (int(overlay.shape[1] * scale), int(overlay.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        montage_cells.append(resized)
        print(
            f"{CAM_SHORT[ci]}: tracks={stats['tracks']}, "
            f"median={stats['median_error_px']:.3f}px, "
            f"p90={stats['p90_error_px']:.3f}px, max={stats['max_error_px']:.3f}px"
        )

    # 2x2 montage (the portrait camera retains its portrait aspect ratio).
    cell_w, cell_h = 640, 512
    montage = np.full((cell_h * 2 + 48, cell_w * 2, 3), 245, dtype=np.uint8)
    labels = ["cam00", "cam01", "cam02", "cam03"]
    for ci, img in enumerate(montage_cells):
        r, c = divmod(ci, 2)
        x0, y0 = c * cell_w, r * (cell_h + 24) + 24
        h, w = img.shape[:2]
        montage[y0:y0 + h, x0:x0 + w] = img
        cv2.putText(montage, labels[ci], (x0 + 8, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (20, 20, 20), 1, cv2.LINE_AA)
    montage_path = OUT_DIR / "anchor0_reprojection_montage.png"
    cv2.imwrite(str(montage_path), montage)

    csv_path = OUT_DIR / "anchor0_reprojection_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    outlier_csv = OUT_DIR / "suspected_spatial_outliers.csv"
    with outlier_csv.open("w", encoding="utf-8-sig", newline="") as f:
        flagged_rows = [r for r in csv_rows if r["spatial_review_flag"]]
        fields = list(csv_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flagged_rows)

    # 3D plot: flagged spatial points are highlighted, but retained in the cloud.
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    keep = ~outlier_flags
    ax.scatter(points[keep, 0], points[keep, 1], points[keep, 2], s=16, alpha=0.75, label="Other accepted tracks")
    if outlier_flags.any():
        ax.scatter(points[outlier_flags, 0], points[outlier_flags, 1], points[outlier_flags, 2],
                   s=42, marker="x", label="Spatial review flag")
    centers = np.asarray([camera_data[i]["C"] for i in range(4)])
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=110, marker="^", label="Camera centers")
    for i, c in enumerate(centers):
        ax.text(c[0], c[1], c[2], CAM_SHORT[i])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("SplatROM anchor 0 — spatial outlier review (flags are not deletions)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    outlier_plot_path = OUT_DIR / "anchor0_spatial_outlier_review.png"
    fig.tight_layout()
    fig.savefig(outlier_plot_path, dpi=180)
    plt.close(fig)

    overall = np.asarray(all_errors, dtype=np.float64)
    report = {
        "scene": "ISH822__aria_inspecting_3ab0",
        "timestamp": "anchor frame 0000 in each rig camera",
        "units": "meters",
        "input_calibration": "4C4D sparse/0 cameras.txt and images.txt exported from SHOW3D",
        "reconstruction": {
            "valid_3d_tracks": int(len(points)),
            "track_distribution_by_views": dict(Counter(int(r["num_views"]) for r in csv_rows)),
            "ambiguous_tracks_rejected": int(ambiguous),
            "rejected_during_triangulation": dict(rejected),
        },
        "reprojection_all_observations": {
            "observation_count": int(len(overall)),
            "median_px": float(np.median(overall)),
            "p90_px": float(np.percentile(overall, 90)),
            "max_px": float(np.max(overall)),
            "fraction_le_1px": float(np.mean(overall <= 1.0)),
            "fraction_le_2px": float(np.mean(overall <= 2.0)),
        },
        "per_camera": camera_stats,
        "spatial_outlier_review": outlier_info,
        "xyz_min_m": points.min(axis=0).tolist(),
        "xyz_max_m": points.max(axis=0).tolist(),
        "outputs": {
            "montage": str(montage_path),
            "overlays": overlay_paths,
            "track_audit_csv": str(csv_path),
            "suspected_spatial_outliers_csv": str(outlier_csv),
            "spatial_outlier_plot": str(outlier_plot_path),
        },
        "interpretation_warning": (
            "Reprojection checks use the same feature observations that formed each triangulated track; "
            "they verify internal consistency, not independent ground-truth accuracy. Spatial flags are review-only. "
            "This remains a sparse single-timestamp diagnostic, not a dense reconstruction or validated 4C4D initialization."
        ),
        "pair_audit": pair_audit,
    }
    report_path = OUT_DIR / "reprojection_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== REPROJECTION / OUTLIER AUDIT COMPLETE ===")
    print(f"Valid tracks: {len(points)}")
    print(f"All observed reprojection errors: median={np.median(overall):.3f}px, p90={np.percentile(overall,90):.3f}px, max={np.max(overall):.3f}px")
    print(f"Spatial review flags (not removed): {int(outlier_flags.sum())}")
    print(f"Montage: {montage_path}")
    print(f"Report: {report_path}")
    print(f"Track CSV: {csv_path}")
    print(f"Outlier CSV: {outlier_csv}")
    print("No source images or sparse/0 files were modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
