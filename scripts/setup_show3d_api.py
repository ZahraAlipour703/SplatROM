#!/usr/bin/env python3
"""Clone official SHOW3D API and install its core OpenCV dependency."""
from __future__ import annotations
import argparse, os, subprocess, sys, urllib.request, urllib.parse
from pathlib import Path

def proxy_env():
    env = os.environ.copy(); proxies = urllib.request.getproxies()
    raw = proxies.get("http") or proxies.get("https")
    if raw:
        p = urllib.parse.urlsplit(raw)
        if p.hostname and p.port and not p.username and not p.password:
            host = f"[{p.hostname}]" if ":" in p.hostname and not p.hostname.startswith("[") else p.hostname
            proxy = f"http://{host}:{p.port}"
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"): env[k] = proxy
    return env

def run(cmd, env):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), env=env, check=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--api-dir", default="../show3d-api")
    ap.add_argument("--with-viz", action="store_true", help="also install matplotlib for optional visualization")
    args = ap.parse_args(); api = Path(args.api_dir).resolve(); env = proxy_env()
    if not (api / "show3d" / "dataset.py").exists():
        api.parent.mkdir(parents=True, exist_ok=True)
        if api.exists() and any(api.iterdir()): raise SystemExit(f"Directory exists but is not a SHOW3D checkout: {api}")
        run(["git", "clone", "https://github.com/facebookresearch/show3d.git", api], env)
    packages = ["opencv-python<4.12"] + (["matplotlib"] if args.with_viz else [])
    run([sys.executable, "-m", "pip", "install", *packages], env)
    env["PYTHONPATH"] = str(api) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    run([sys.executable, "-c", "from show3d.dataset import Show3DPaths; from show3d.camera import project_to_image; print('SHOW3D API import: OK')"], env)
    print("\nSetup complete. In this PowerShell session run:")
    print(f'$env:PYTHONPATH = "{api}"')

if __name__ == "__main__": main()
