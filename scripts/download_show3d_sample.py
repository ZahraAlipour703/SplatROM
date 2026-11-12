#!/usr/bin/env python3
"""
Download one public SHOW3D sample scene from Hugging Face and run a
CPU-only hand-kinematics export.

No Kaggle/ETH account is required. The Hugging Face dataset is public.
The download is selective: one recording, not the full dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_BASE = "https://huggingface.co/datasets/facebook/show3d-dataset/resolve/main/"
DEFAULT_SUBJECT = "ISH822"
DEFAULT_SCENE = "aria_inspecting_3ab0"
DEFAULT_VIEWS = ("rig0", "rig1", "rig2", "rig3", "headset0")

def build_opener():
    """Use the Windows HTTP proxy for both HTTP and HTTPS CONNECT.

    This avoids inheriting a malformed https:// proxy scheme from Windows
    registry settings. No proxy credentials are printed.
    """
    proxies = urllib.request.getproxies()
    raw = proxies.get("http") or proxies.get("https")
    if raw:
        parsed = urllib.parse.urlsplit(raw)
        if parsed.hostname and parsed.port:
            host = parsed.hostname
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            proxy_url = f"http://{host}:{parsed.port}"
            return urllib.request.build_opener(
                urllib.request.ProxyHandler({
                    "http": proxy_url,
                    "https": proxy_url,
                })
            )
    # An explicit empty proxy map prevents urllib from silently re-reading
    # a malformed Windows registry HTTPS proxy when no usable HTTP proxy exists.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))

def download_file(opener, repo_path: str, target: Path, optional=False):
    url = REPO_BASE + urllib.parse.quote(repo_path, safe="/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[SKIP] {repo_path} (already downloaded: {target.stat().st_size:,} bytes)")
        return True

    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "SplatROM/0.3 (research data downloader)"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        response = opener.open(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and optional:
            print(f"[OPTIONAL-MISSING] {repo_path}")
            return False
        raise RuntimeError(f"Download failed for {repo_path}: HTTP {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not reach Hugging Face for {repo_path}: {exc}") from exc

    with response:
        status = getattr(response, "status", response.getcode())
        append = (offset > 0 and status == 206)
        if offset and not append:
            print(f"[RESUME-RESET] server did not honor byte-range for {repo_path}")
        mode = "ab" if append else "wb"
        downloaded = offset if append else 0
        total_header = response.headers.get("Content-Length")
        total = (downloaded + int(total_header)) if total_header and total_header.isdigit() else None
        last_report = time.time()
        with open(partial, mode) as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if time.time() - last_report >= 2:
                    if total:
                        pct = downloaded * 100.0 / total
                        print(f"  {repo_path}: {downloaded:,}/{total:,} bytes ({pct:.1f}%)")
                    else:
                        print(f"  {repo_path}: {downloaded:,} bytes")
                    last_report = time.time()

    if not partial.exists() or partial.stat().st_size == 0:
        raise RuntimeError(f"Downloaded an empty file: {repo_path}")
    partial.replace(target)
    print(f"[OK] {repo_path} -> {target} ({target.stat().st_size:,} bytes)")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download one SHOW3D sample and export hand kinematics.")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS),
                        help="Camera videos to fetch. Default: rig0 rig1 rig2 rig3 headset0")
    parser.add_argument("--skip-videos", action="store_true",
                        help="Download annotations/calibration only.")
    parser.add_argument("--root", default="data/raw/show3d",
                        help="Dataset root (default: data/raw/show3d).")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Do not run the kinematics exporter after download.")
    args = parser.parse_args()

    allowed = {f"rig{i}" for i in range(8)} | {"headset0", "headset1"}
    bad = sorted(set(args.views) - allowed)
    if bad:
        parser.error(f"Unknown views: {bad}. Allowed: rig0..rig7, headset0, headset1")
    if any(x in ("", ".", "..") for x in (args.subject, args.scene)):
        parser.error("Invalid subject/scene.")

    root = Path(args.root).resolve()
    subject, scene = args.subject, args.scene
    scene_root = root / "scenes" / subject / scene
    annotation_root = root / "hand_pose" / "v3" / "scenes" / subject / scene

    paths = [
        (f"scenes/{subject}/{scene}/metadata/recording_info.json", scene_root / "metadata/recording_info.json", False),
        (f"scenes/{subject}/{scene}/metadata/frame_info.json", scene_root / "metadata/frame_info.json", False),
        (f"hand_pose/v3/scenes/{subject}/{scene}/hand_pose_umetrack.json",
         annotation_root / "hand_pose_umetrack.json", False),
        (f"hand_pose/v3/scenes/{subject}/{scene}/hand_pose_mano.json",
         annotation_root / "hand_pose_mano.json", True),
        (f"hand_pose/hand_profiles/{subject}/profile_umetrack.json",
         root / "hand_pose/hand_profiles" / subject / "profile_umetrack.json", True),
        (f"hand_pose/hand_profiles/{subject}/profile_mano.json",
         root / "hand_pose/hand_profiles" / subject / "profile_mano.json", True),
    ]

    for view in args.views:
        paths.append((
            f"scenes/{subject}/{scene}/camera_calibration/{view}.json",
            scene_root / "camera_calibration" / f"{view}.json",
            view.startswith("headset")  # still attempt, but optional
        ))
        if not args.skip_videos:
            paths.append((
                f"scenes/{subject}/{scene}/{view}.mp4",
                scene_root / f"{view}.mp4",
                True
            ))

    opener = build_opener()
    print(f"SHOW3D sample: {subject}/{scene}")
    print(f"Dataset root: {root}")
    print("Selected views:", ", ".join(args.views))
    print("Downloading only one scene; the full SHOW3D dataset is NOT downloaded.")

    failures = []
    for repo_path, target, optional in paths:
        try:
            download_file(opener, repo_path, target, optional=optional)
        except Exception as exc:
            failures.append((repo_path, str(exc)))
            print(f"[ERROR] {repo_path}: {exc}")
            # Required files stop the run; optional views can be missing.
            if not optional:
                break

    if failures:
        print("\nDownload errors:")
        for path, error in failures:
            print(f" - {path}: {error}")
        raise SystemExit(2)

    if not (annotation_root / "hand_pose_umetrack.json").exists():
        raise SystemExit("Required hand-pose annotation is missing.")

    print("\nDownload completed.")
    if not args.no_analysis:
        exporter = Path(__file__).with_name("extract_show3d_kinematics.py")
        cmd = [
            sys.executable, str(exporter),
            "--annotation", str(annotation_root / "hand_pose_umetrack.json"),
            "--out", str(Path("data/processed/show3d") / f"{subject}__{scene}"),
        ]
        print("\nRunning CPU-only kinematics export:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
