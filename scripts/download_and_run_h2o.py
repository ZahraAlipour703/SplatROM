"""Download a single H2O subject, extract it, then run the kinematic baseline.

Requires a valid academic-use account from https://h2odataset.ethz.ch/.
This script uses only Python's standard library plus the project's NumPy dependency.
"""
from __future__ import annotations

import argparse
import base64
import getpass
import re
import ssl
import sys
import tarfile
import zipfile
from itertools import chain
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, HTTPSHandler, build_opener, getproxies

from splatom.data_h2o import run_h2o_pose_baseline

BASE_URL = "https://h2odataset.ethz.ch/data/dataset/"
FILES_BY_MODE = {
    "pose": ["subject1_pose_v1_1.tar.gz"],
    "ego": ["subject1_ego_v1_1.tar.gz"],
    "all": ["subject1_v1_1.tar.gz"],
}
COMMON_FILES = ["object.zip", "label_split.zip"]
MAGIC = {
    ".gz": bytes.fromhex("1f8b"),
    ".zip": b"PK\x03\x04",
}


def build_network_opener():
    """Build a verified HTTPS opener and normalize Windows HTTPS-proxy schemes.

    On some Windows setups, urllib reads a `https://proxy:port` registry entry
    for an HTTP CONNECT proxy. That makes TLS try to start against the proxy
    itself. The user's earlier pip fix established that the local proxy listens
    as HTTP on the HTTP proxy port; normalize both URL schemes to that endpoint.
    """
    system_proxies = getproxies()
    proxy_raw = system_proxies.get("http") or system_proxies.get("https")
    proxy_map: dict[str, str] = {}
    if proxy_raw:
        parsed = urlsplit(proxy_raw if "://" in proxy_raw else f"http://{proxy_raw}")
        if not parsed.hostname:
            raise RuntimeError("A system proxy was detected but its host could not be parsed.")
        hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        port = parsed.port or 80
        normalized = f"http://{hostname}:{port}"
        proxy_map = {"http": normalized, "https": normalized}
        print(f"Network route: system HTTP proxy on port {port}")
    else:
        print("Network route: direct connection (no system proxy detected)")

    return build_opener(
        ProxyHandler(proxy_map),
        HTTPSHandler(context=ssl.create_default_context()),
    )


def safe_name(filename: str) -> str:
    return Path(filename).name


def download_file(opener, filename: str, authorization: str, archive_dir: Path, force: bool = False) -> Path:
    filename = safe_name(filename)
    final_path = archive_dir / filename
    partial_path = archive_dir / f"{filename}.part"
    expected_magic = MAGIC[final_path.suffix]

    if final_path.exists() and final_path.stat().st_size > 0 and not force:
        with final_path.open("rb") as stream:
            if stream.read(len(expected_magic)) == expected_magic:
                print(f"Already downloaded; keeping {filename}")
                return final_path
        print(f"Existing {filename} has an invalid file signature; downloading again.")
        final_path.unlink()

    offset = partial_path.stat().st_size if partial_path.exists() and not force else 0
    headers = {"Authorization": authorization, "User-Agent": "SplatROM-H2O-downloader/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = Request(BASE_URL + filename, headers=headers, method="GET")
    try:
        response = opener.open(request, timeout=90)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "H2O rejected the credentials (HTTP 401/403). Re-register or use the "
                "currently valid username/password emailed by ETH Zurich."
            ) from exc
        raise RuntimeError(f"H2O download failed for {filename}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error downloading {filename}: {exc.reason}") from exc

    status = getattr(response, "status", response.getcode())
    append = bool(offset and status == 206)
    if offset and status != 206:
        print("Server did not resume the partial file; restarting this download.")
        offset = 0
    mode = "ab" if append else "wb"

    content_range = response.headers.get("Content-Range", "")
    total_match = re.search(r"/([0-9]+)$", content_range)
    total_size = int(total_match.group(1)) if total_match else None
    content_length = response.headers.get("Content-Length")
    if total_size is None and content_length and content_length.isdigit():
        total_size = int(content_length) + (offset if append else 0)

    first_chunk = True
    downloaded = offset if append else 0
    with response, partial_path.open(mode) as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            if first_chunk:
                first_chunk = False
                if not chunk.startswith(expected_magic) and downloaded == 0:
                    raise RuntimeError(
                        f"The server response for {filename} was not a valid archive. "
                        "Check the H2O credentials and dataset server response."
                    )
            output.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = min(100.0, downloaded * 100.0 / total_size)
                print(
                    f"\r{filename}: {downloaded / (1024**2):.1f} / "
                    f"{total_size / (1024**2):.1f} MiB ({percent:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r{filename}: {downloaded / (1024**2):.1f} MiB", end="", flush=True)
    print()

    if not partial_path.exists() or partial_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {filename}")
    with partial_path.open("rb") as stream:
        if stream.read(len(expected_magic)) != expected_magic:
            raise RuntimeError(f"Archive signature check failed for {filename}")
    partial_path.replace(final_path)
    return final_path


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if "\\" in info.filename:
                raise RuntimeError(f"Unsafe path separator in ZIP archive: {info.filename}")
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or (name.parts and name.parts[0].endswith(":")):
                raise RuntimeError(f"Unsafe path in ZIP archive: {info.filename}")
            target = (destination / Path(*name.parts)).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe path in ZIP archive: {info.filename}")
        zf.extractall(destination)


def _safe_extract_tar(archive: Path, extracted_root: Path) -> Path:
    """Safely stream-extract a tar archive without loading all headers in RAM."""
    extracted_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        iterator = iter(tf)
        prefix_members = []
        first_top = None
        for member in iterator:
            prefix_members.append(member)
            parts = PurePosixPath(member.name).parts
            if parts:
                first_top = parts[0]
                break
        # Official subject archives generally contain h1/h2/k1/... at their root.
        # If the archive already has a subject1/ prefix, preserve it at extracted/.
        target = extracted_root if first_top == "subject1" else extracted_root / "subject1"
        target.mkdir(parents=True, exist_ok=True)
        root = target.resolve()

        for member in chain(prefix_members, iterator):
            if "\\" in member.name:
                raise RuntimeError(f"Unsafe path separator in TAR archive: {member.name}")
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or (name.parts and name.parts[0].endswith(":")):
                raise RuntimeError(f"Unsafe path in TAR archive: {member.name}")
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                # Dataset archives do not need links, devices, or FIFOs.
                continue
            dest = (target / Path(*name.parts)).resolve()
            if root != dest and root not in dest.parents:
                raise RuntimeError(f"Unsafe path in TAR archive: {member.name}")
            tf.extract(member, path=target)
    return target


def find_subject_root(extracted_root: Path) -> Path:
    direct = extracted_root / "subject1"
    if direct.exists() and any(direct.rglob("hand_pose")):
        return direct
    if any(extracted_root.rglob("cam4")):
        return extracted_root
    raise FileNotFoundError(
        "Extraction finished, but no subject1/cam4 folder was found. "
        "Inspect data/raw/h2o/extracted and verify the selected H2O archive."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download H2O Subject 1 and run SplatROM's first CPU-side pose baseline."
    )
    parser.add_argument(
        "--mode", choices=("pose", "ego", "all"), default="pose",
        help=("pose = lightweight annotations only (default); ego = egocentric view; "
              "all = all five camera views, required for the multi-view 4DGS stage"),
    )
    parser.add_argument("--force", action="store_true", help="redownload even if archive files exist")
    parser.add_argument("--skip-download", action="store_true", help="extract local archives and run baseline only")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw" / "h2o"
    archive_dir = raw_root / "archives"
    extracted_root = raw_root / "extracted" / args.mode
    common_root = raw_root / "extracted" / "common"
    processed_dir = project_root / "data" / "processed" / "h2o" / args.mode
    archive_dir.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    common_root.mkdir(parents=True, exist_ok=True)

    archive_names = [*COMMON_FILES, *FILES_BY_MODE[args.mode]]
    if not args.skip_download:
        print("H2O credentials are required by ETH Zurich; this script cannot bypass registration.")
        username = input("H2O username: ").strip()
        password = getpass.getpass("H2O password (hidden): ")
        if not username or not password:
            print("Username and password are required.", file=sys.stderr)
            return 2
        authorization = "Basic " + base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        opener = build_network_opener()
        for filename in archive_names:
            print(f"\nDownloading {filename}")
            download_file(opener, filename, authorization, archive_dir, force=args.force)
        # Drop references as soon as downloads are complete.
        del password, authorization
    else:
        missing = [name for name in archive_names if not (archive_dir / name).is_file()]
        if missing:
            print("Missing local archive(s): " + ", ".join(missing), file=sys.stderr)
            return 2

    # Extract metadata ZIPs and the selected subject archive.
    for filename in COMMON_FILES:
        archive = archive_dir / filename
        marker = common_root / (".object_zip_extracted" if filename == "object.zip" else ".label_split_zip_extracted")
        if args.force and marker.exists():
            marker.unlink()
        if marker.exists():
            print(f"Already extracted: {filename}")
        else:
            print(f"Extracting {filename}")
            _safe_extract_zip(archive, common_root)
            marker.write_text("ok\n", encoding="utf-8")

    subject_archive = archive_dir / FILES_BY_MODE[args.mode][0]
    if args.mode == "all":
        data_has_camera_images = True
    else:
        data_has_camera_images = args.mode == "ego"
    # Extract once. If files already exist, do not unpack the archive again.
    extraction_marker = extracted_root / ".subject1_archive_extracted"
    if args.force and extraction_marker.exists():
        extraction_marker.unlink()
    if extraction_marker.exists():
        print("Subject archive already extracted for this mode; skipping extraction.")
    else:
        print(f"Extracting {subject_archive.name}; this may take a few minutes.")
        _safe_extract_tar(subject_archive, extracted_root)
        extraction_marker.write_text("ok\n", encoding="utf-8")

    subject_root = find_subject_root(extracted_root)
    summary = run_h2o_pose_baseline(subject_root, processed_dir)

    print("\n=== SPLATROM H2O BASELINE COMPLETE ===")
    print(f"Mode: {args.mode}")
    print(f"Subject root: {subject_root}")
    print(f"Pose files: {summary['pose_files_found']}")
    print(f"Rows written: {summary['rows_written']}")
    print(f"Sequences/takes: {summary['sequences_and_takes']}")
    print(f"Hand annotations: {summary['hand_annotations']}")
    print(f"CSV: {processed_dir / 'subject1_cam4_angles.csv'}")
    print(f"Summary: {processed_dir / 'summary.json'}")
    if args.mode == "pose":
        print("NOTE: pose mode contains annotations, not RGB-D frames; it is for the kinematic baseline.")
        print("For multi-view 4DGS inputs later, rerun with --mode all using the same valid H2O credentials.")
    elif data_has_camera_images:
        print("Image data is present for this mode; check extracted cam*/rgb folders before 4DGS preprocessing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
