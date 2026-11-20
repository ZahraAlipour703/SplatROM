from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

SCENE = Path(
    r"D:\zra\PROJECTS\Github\4C4D\data\SplatROM\ISH822_aria_inspecting_3ab0"
)
SPARSE = SCENE / "sparse" / "0"
IMAGE_DIR = SCENE / "images"
OUT_DIR = SCENE / "diagnostics" / "anchor0_tracks_v1"

NAMES = [
    "cam00_0000.png",
    "cam01_0000.png",
    "cam02_0000.png",
    "cam03_0000.png",
]

SIFT_FEATURES = 6000
RATIO_TEST = 0.75
RANSAC_THRESHOLD_PX = 1.5
CALIBRATED_SAMPSON_THRESHOLD_PX = 2.0
MAX_REPROJECTION_PX = 4.0
MEDIAN_REPROJECTION_PX = 2.0
MIN_PARALLAX_DEG = 1.0


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
            if len(params) < 4:
                raise ValueError(f"Invalid PINHOLE parameters for camera {camera_id}")
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            if len(params) < 3:
                raise ValueError(f"Invalid SIMPLE_PINHOLE parameters for camera {camera_id}")
            fx = fy = params[0]
            cx, cy = params[1:3]
        else:
            raise ValueError(
                f"Unsupported camera model {model!r}; this script requires PINHOLE/SIMPLE_PINHOLE"
            )

        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        cameras[camera_id] = {
            "K": K,
            "width": width,
            "height": height,
            "model": model,
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
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) < 10 or p[0].startswith("#"):
            continue
        try:
            q = np.asarray([float(v) for v in p[1:5]], dtype=np.float64)
            t = np.asarray([float(v) for v in p[5:8]], dtype=np.float64)
            camera_id = int(p[8])
            name = " ".join(p[9:])
        except ValueError:
            continue
        if name in NAMES:
            R = qvec_to_rotation(q)
            C = -R.T @ t
            images[name] = {"R": R, "t": t, "C": C, "camera_id": camera_id}
    return images


def skew(t: np.ndarray) -> np.ndarray:
    x, y, z = t
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def sampson_errors(F: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([p1, np.ones(len(p1), dtype=np.float64)])
    x2 = np.column_stack([p2, np.ones(len(p2), dtype=np.float64)])
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numerator = np.sum(x2 * Fx1, axis=1) ** 2
    denominator = (
        Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
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
    if not np.isfinite(X).all():
        return None
    return X


def reprojection_error(P: np.ndarray, X: np.ndarray, xy: np.ndarray) -> float:
    q = P @ np.append(X, 1.0)
    if q[2] <= 1e-12:
        return float("inf")
    projected = q[:2] / q[2]
    return float(np.linalg.norm(projected - xy))


def sample_rgb(image_bgr: np.ndarray, xy: np.ndarray, radius: int = 2):
    x, y = np.rint(xy).astype(int)
    h, w = image_bgr.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([180, 180, 180], dtype=np.uint8)
    bgr = np.mean(patch.reshape(-1, 3), axis=0)
    return np.clip(np.rint(bgr[::-1]), 0, 255).astype(np.uint8)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray):
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(points, colors):
            f.write(
                f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def main():
    for required in (SPARSE / "cameras.txt", SPARSE / "images.txt"):
        if not required.exists():
            raise FileNotFoundError(required)
    for name in NAMES:
        if not (IMAGE_DIR / name).exists():
            raise FileNotFoundError(IMAGE_DIR / name)

    cameras = read_cameras(SPARSE / "cameras.txt")
    images = read_images(SPARSE / "images.txt")
    if len(images) != 4:
        raise RuntimeError(f"Expected 4 anchor images in images.txt; parsed {len(images)}: {list(images)}")

    feature_data = {}
    color_images = {}
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    print("=== FEATURE EXTRACTION ===")
    for ci, name in enumerate(NAMES):
        color = cv2.imread(str(IMAGE_DIR / name), cv2.IMREAD_COLOR)
        if color is None:
            raise RuntimeError(f"Could not read image: {IMAGE_DIR / name}")
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            raise RuntimeError(f"No SIFT descriptors for {name}")
        feature_data[ci] = (keypoints, descriptors)
        color_images[ci] = color
        print(f"{name}: {len(keypoints)} keypoints")

    projection_matrices = []
    camera_data = []
    for name in NAMES:
        info = images[name]
        if info["camera_id"] not in cameras:
            raise RuntimeError(f"Camera ID {info['camera_id']} missing for {name}")
        cam = cameras[info["camera_id"]]
        P = cam["K"] @ np.column_stack([info["R"], info["t"]])
        projection_matrices.append(P)
        camera_data.append({
            "R": info["R"],
            "C": info["C"],
            "K": cam["K"],
            "width": cam["width"],
            "height": cam["height"],
        })

    dsu = DisjointSet()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pair_audit = []
    print("\n=== CALIBRATED FEATURE MATCHING ===")

    for i in range(4):
        for j in range(i + 1, 4):
            kp1, d1 = feature_data[i]
            kp2, d2 = feature_data[j]
            knn = matcher.knnMatch(d1, d2, k=2)
            good = [
                m for pair in knn if len(pair) == 2
                for m, n in [pair]
                if m.distance < RATIO_TEST * n.distance
            ]
            if len(good) < 8:
                pair_audit.append({"pair": [NAMES[i], NAMES[j]], "raw": len(good), "ransac": 0, "calibrated": 0})
                print(f"{NAMES[i]} - {NAMES[j]}: insufficient matches ({len(good)})")
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
                print(f"{NAMES[i]} - {NAMES[j]}: RANSAC failed")
                continue

            ransac_mask = ransac_mask.ravel().astype(bool)
            cal_error = sampson_errors(F_cal, p1, p2)
            accepted = ransac_mask & (cal_error <= CALIBRATED_SAMPSON_THRESHOLD_PX)
            for match, keep in zip(good, accepted):
                if keep:
                    dsu.union((i, match.queryIdx), (j, match.trainIdx))

            counts = {
                "pair": [NAMES[i], NAMES[j]],
                "raw": int(len(good)),
                "ransac": int(ransac_mask.sum()),
                "calibrated": int(accepted.sum()),
            }
            pair_audit.append(counts)
            print(
                f"{NAMES[i]} - {NAMES[j]}: raw={counts['raw']}, "
                f"RANSAC={counts['ransac']}, calibrated={counts['calibrated']}"
            )

    print("\n=== TRACK BUILDING ===")
    tracks = []
    ambiguous_tracks = 0
    for group in dsu.groups():
        per_camera = {}
        for camera_index, keypoint_index in group:
            per_camera.setdefault(camera_index, []).append(keypoint_index)
        if len(per_camera) < 2:
            continue
        if any(len(indices) != 1 for indices in per_camera.values()):
            ambiguous_tracks += 1
            continue
        observations = []
        for ci, indices in sorted(per_camera.items()):
            ki = indices[0]
            xy = np.asarray(feature_data[ci][0][ki].pt, dtype=np.float64)
            observations.append((ci, xy, ki))
        tracks.append(observations)

    track_distribution = Counter(len(track) for track in tracks)
    print(f"Usable tracks: {len(tracks)}")
    print(f"Track distribution: {dict(sorted(track_distribution.items()))}")
    print(f"Ambiguous tracks rejected: {ambiguous_tracks}")

    print("\n=== TRIANGULATION AND VALIDATION ===")
    accepted_points = []
    accepted_colors = []
    rows = []
    rejected = Counter()

    for observations in tracks:
        X = triangulate_track(observations, projection_matrices)
        if X is None:
            rejected["invalid_homogeneous_point"] += 1
            continue

        errors = []
        rays = []
        positive_depth = True
        for ci, xy, _ki in observations:
            cam = camera_data[ci]
            camera_xyz = cam["R"] @ (X - cam["C"])
            if not np.isfinite(camera_xyz).all() or camera_xyz[2] <= 0:
                positive_depth = False
                break
            errors.append(reprojection_error(projection_matrices[ci], X, xy))
            ray = X - cam["C"]
            norm = np.linalg.norm(ray)
            if norm <= 1e-12:
                positive_depth = False
                break
            rays.append(ray / norm)

        if not positive_depth:
            rejected["nonpositive_depth"] += 1
            continue
        if not errors or max(errors) > MAX_REPROJECTION_PX or np.median(errors) > MEDIAN_REPROJECTION_PX:
            rejected["reprojection_error"] += 1
            continue

        angles = []
        for a in range(len(rays)):
            for b in range(a + 1, len(rays)):
                dot = float(np.clip(np.dot(rays[a], rays[b]), -1.0, 1.0))
                angles.append(float(np.degrees(np.arccos(dot))))
        max_parallax = max(angles) if angles else 0.0
        if max_parallax < MIN_PARALLAX_DEG:
            rejected["low_parallax"] += 1
            continue

        observation_colors = [sample_rgb(color_images[ci], xy) for ci, xy, _ki in observations]
        rgb = np.clip(np.rint(np.mean(observation_colors, axis=0)), 0, 255).astype(np.uint8)

        track_id = len(accepted_points) + 1
        accepted_points.append(X)
        accepted_colors.append(rgb)
        rows.append({
            "track_id": track_id,
            "x_m": float(X[0]),
            "y_m": float(X[1]),
            "z_m": float(X[2]),
            "num_views": len(observations),
            "camera_names": ";".join(NAMES[ci] for ci, _xy, _ki in observations),
            "median_reprojection_px": float(np.median(errors)),
            "max_reprojection_px": float(max(errors)),
            "max_parallax_deg": float(max_parallax),
            "red": int(rgb[0]),
            "green": int(rgb[1]),
            "blue": int(rgb[2]),
        })

    points = np.asarray(accepted_points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(accepted_colors, dtype=np.uint8).reshape(-1, 3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points_path = OUT_DIR / "anchor0_tracks_preview.ply"
    csv_path = OUT_DIR / "anchor0_tracks.csv"
    cameras_ply_path = OUT_DIR / "camera_centers.ply"
    report_path = OUT_DIR / "report.json"

    write_ply(points_path, points, colors)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "track_id", "x_m", "y_m", "z_m", "num_views", "camera_names",
            "median_reprojection_px", "max_reprojection_px", "max_parallax_deg",
            "red", "green", "blue"
        ])
        writer.writeheader()
        writer.writerows(rows)

    camera_centers = np.asarray([camera_data[i]["C"] for i in range(4)], dtype=np.float64)
    camera_colors = np.asarray([[255, 60, 60], [60, 255, 60], [60, 60, 255], [255, 220, 40]], dtype=np.uint8)
    write_ply(cameras_ply_path, camera_centers, camera_colors)

    report = {
        "dataset_scene": "ISH822__aria_inspecting_3ab0",
        "source_images": NAMES,
        "coordinate_units": "meters",
        "method": "SIFT + pairwise ratio test + RANSAC + calibrated Sampson filtering + DSU tracks + linear multi-view triangulation",
        "thresholds": {
            "ratio_test": RATIO_TEST,
            "ransac_px": RANSAC_THRESHOLD_PX,
            "calibrated_sampson_px": CALIBRATED_SAMPSON_THRESHOLD_PX,
            "max_reprojection_px": MAX_REPROJECTION_PX,
            "median_reprojection_px": MEDIAN_REPROJECTION_PX,
            "minimum_max_parallax_deg": MIN_PARALLAX_DEG,
        },
        "pair_audit": pair_audit,
        "usable_tracks_before_triangulation": len(tracks),
        "track_distribution": {str(k): int(v) for k, v in sorted(track_distribution.items())},
        "ambiguous_tracks_rejected": ambiguous_tracks,
        "valid_3d_points": int(len(points)),
        "rejected_track_counts": dict(rejected),
        "median_reprojection_px": float(np.median([r["median_reprojection_px"] for r in rows])) if rows else None,
        "p90_max_reprojection_px": float(np.percentile([r["max_reprojection_px"] for r in rows], 90)) if rows else None,
        "median_max_parallax_deg": float(np.median([r["max_parallax_deg"] for r in rows])) if rows else None,
        "xyz_min_m": points.min(axis=0).tolist() if len(points) else None,
        "xyz_max_m": points.max(axis=0).tolist() if len(points) else None,
        "outputs": {
            "point_cloud_ply": str(points_path),
            "tracks_csv": str(csv_path),
            "camera_centers_ply": str(cameras_ply_path),
        },
        "warning": "Sparse diagnostic cloud from one timestamp only; not a dense reconstruction and not yet a validated 4C4D points3D.bin.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== EXPORT COMPLETE ===")
    print(f"Valid 3D points: {len(points)}")
    print(f"PLY: {points_path}")
    print(f"CSV: {csv_path}")
    print(f"Camera centers PLY: {cameras_ply_path}")
    print(f"Report: {report_path}")
    print("No files in sparse/0 were modified. No points3D.bin was created.")


if __name__ == "__main__":
    main()
