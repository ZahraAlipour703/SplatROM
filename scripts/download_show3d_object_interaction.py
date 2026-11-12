#!/usr/bin/env python3
"""Download the current SHOW3D object-pose v2 and interaction-field v1 annotations.

This is an optional supplement to the sample already downloaded. It fetches only
two annotation files for one recording; it does not re-download videos or hands.
"""
from __future__ import annotations

import argparse
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/facebook/show3d-dataset/resolve/main/"

def make_opener():
    # Python on this Windows installation reads the proxy from Registry.
    # If an HTTP proxy is detected, use it as an HTTP CONNECT proxy for HTTPS.
    proxies = urllib.request.getproxies()
    raw = proxies.get("http") or proxies.get("https")
    if raw:
        parsed = urllib.parse.urlsplit(raw)
        if parsed.hostname and parsed.port:
            host = parsed.hostname
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            proxy = f"http://{host}:{parsed.port}"
            return urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            )
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))

def fetch(opener, remote: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"[SKIP] {target} already exists ({target.stat().st_size:,} bytes)")
        return True
    url = BASE + urllib.parse.quote(remote, safe="/")
    req = urllib.request.Request(url, headers={"User-Agent": "SplatROM/0.4"})
    try:
        with opener.open(req, timeout=120) as response, open(target, "wb") as out:
            total = response.headers.get("Content-Length")
            total_n = int(total) if total and total.isdigit() else None
            done = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total_n:
                    print(f"\r  {done:,}/{total_n:,} bytes ({100*done/total_n:.1f}%)", end="")
                else:
                    print(f"\r  {done:,} bytes", end="")
        print()
        if not target.exists() or target.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty.")
        print(f"[OK] {remote} -> {target} ({target.stat().st_size:,} bytes)")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"[NOT AVAILABLE FOR THIS SCENE] {remote}")
            if target.exists():
                target.unlink()
            return False
        raise RuntimeError(f"HTTP {exc.code} for {remote}") from exc

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="ISH822")
    p.add_argument("--scene", default="aria_inspecting_3ab0")
    p.add_argument("--root", default="data/raw/show3d")
    args = p.parse_args()

    root = Path(args.root)
    subject, scene = args.subject, args.scene
    opener = make_opener()

    tasks = [
        (
            f"object_pose/v2/scenes/{subject}/{scene}/object_pose.json",
            root / "object_pose" / "v2" / "scenes" / subject / scene / "object_pose.json",
        ),
        (
            f"interaction_field/v1/scenes/{subject}/{scene}/interaction_field.jsonl",
            root / "interaction_field" / "v1" / "scenes" / subject / scene / "interaction_field.jsonl",
        ),
    ]

    print(f"Supplemental annotations for {subject}/{scene}")
    found = {}
    for remote, local in tasks:
        found[remote.split("/")[0]] = fetch(opener, remote, local)

    print("\nSummary:")
    print(f"  object_pose/v2: {'available' if found.get('object_pose') else 'not present'}")
    print(f"  interaction_field/v1: {'available' if found.get('interaction_field') else 'not present'}")
    print("\nNo videos or hand-pose files were downloaded again.")

if __name__ == "__main__":
    main()
