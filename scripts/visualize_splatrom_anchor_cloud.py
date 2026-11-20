from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

SCENE = Path(
    r"D:\zra\PROJECTS\Github\4C4D\data\SplatROM\ISH822_aria_inspecting_3ab0"
)
CLOUD_DIR = SCENE / "diagnostics" / "anchor0_tracks_v1"
POINTS_PLY = CLOUD_DIR / "anchor0_tracks_preview.ply"
CAMERAS_PLY = CLOUD_DIR / "camera_centers.ply"
OUTPUT_PNG = CLOUD_DIR / "anchor0_tracks_visualization.png"


def read_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    with path.open("r", encoding="ascii") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        if f.readline().strip() != "format ascii 1.0":
            raise ValueError(f"Expected ASCII PLY format: {path}")

        vertex_count = None
        properties: list[str] = []
        in_vertex_element = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected end of PLY header: {path}")
            line = line.strip()
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element":
                in_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex_element:
                properties.append(parts[-1])
            elif parts[0] == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"No vertex element in PLY: {path}")

        required = ["x", "y", "z"]
        if not all(p in properties for p in required):
            raise ValueError(f"PLY is missing x/y/z properties: {path}")

        rows = []
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                raise ValueError(f"PLY ended before {vertex_count} vertices: {path}")
            values = [float(v) for v in line.split()]
            if len(values) != len(properties):
                raise ValueError(f"Unexpected vertex row in {path}: {line.strip()}")
            rows.append(values)

    data = np.asarray(rows, dtype=np.float64)
    xyz = data[:, [properties.index(k) for k in required]]

    rgb_names = ["red", "green", "blue"]
    if all(k in properties for k in rgb_names):
        rgb = data[:, [properties.index(k) for k in rgb_names]] / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.tile(np.array([[0.25, 0.55, 0.85]]), (len(xyz), 1))

    return xyz, rgb


def set_equal_3d_axes(ax, arrays: list[np.ndarray]) -> None:
    combined = np.vstack([a for a in arrays if len(a)])
    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 0.05) * 1.08

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main() -> int:
    points, point_rgb = read_ascii_ply(POINTS_PLY)
    camera_centers, camera_rgb = read_ascii_ply(CAMERAS_PLY)

    if len(points) == 0:
        raise RuntimeError("The point cloud is empty.")
    if len(camera_centers) != 4:
        raise RuntimeError(
            f"Expected 4 camera centers, found {len(camera_centers)}."
        )

    camera_names = ["cam00", "cam01", "cam02", "cam03"]

    print("=== SPLATROM ANCHOR CLOUD ===")
    print(f"3D points: {len(points)}")
    print(f"Camera centers: {len(camera_centers)}")
    print("Point XYZ min (m):", np.round(points.min(axis=0), 4))
    print("Point XYZ max (m):", np.round(points.max(axis=0), 4))
    print("Camera centers (m):")
    for name, center in zip(camera_names, camera_centers):
        print(f"  {name}: {np.round(center, 4)}")

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    # 3D overview: colors are sampled from the source images by the exporter.
    ax3d.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=point_rgb, s=19, alpha=0.9, depthshade=False,
        label=f"3D tracks ({len(points)})"
    )
    ax3d.scatter(
        camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2],
        c=camera_rgb, s=150, marker="^", edgecolors="black",
        linewidths=0.7, depthshade=False, label="Camera centers"
    )
    for name, center in zip(camera_names, camera_centers):
        ax3d.text(*center, f" {name}", fontsize=9)
    ax3d.set_title("3D point cloud and camera centers")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(loc="best")
    set_equal_3d_axes(ax3d, [points, camera_centers])
    ax3d.view_init(elev=22, azim=-58)

    # Orthographic projections make spatial concentration and outliers easier to see.
    projections = [
        (ax_xy, 0, 1, "X (m)", "Y (m)", "XY projection"),
        (ax_xz, 0, 2, "X (m)", "Z (m)", "XZ projection"),
        (ax_yz, 1, 2, "Y (m)", "Z (m)", "YZ projection"),
    ]
    for ax, u, v, xlabel, ylabel, title in projections:
        ax.scatter(points[:, u], points[:, v], c=point_rgb, s=18, alpha=0.85)
        ax.scatter(
            camera_centers[:, u], camera_centers[:, v],
            c=camera_rgb, s=115, marker="^", edgecolors="black", linewidths=0.7
        )
        for name, center in zip(camera_names, camera_centers):
            ax.annotate(name, (center[u], center[v]), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        "SplatROM — single-timestamp sparse triangulation diagnostic\n"
        "Not a dense reconstruction or a validated 4C4D initialization",
        fontsize=13
    )
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    print(f"\nSaved visualization: {OUTPUT_PNG}")
    print("Opening plot window. Close it to finish.")
    plt.show()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
