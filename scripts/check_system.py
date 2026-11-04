from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    print("=== SplatROM local system check ===")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"CPU: {platform.processor() or 'unknown'}")

    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        print(f"RAM: {vm.total / (1024**3):.1f} GB total; {vm.available / (1024**3):.1f} GB available")
    except Exception:
        print("RAM: not detected (optional package psutil is not installed)")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                check=False, capture_output=True, text=True, timeout=10,
            )
            print("NVIDIA:", result.stdout.strip() or result.stderr.strip() or f"exit={result.returncode}")
        except Exception as exc:
            print(f"NVIDIA query failed: {exc}")
    else:
        print("NVIDIA: nvidia-smi not found on PATH (GPU may still be present)")

    try:
        import torch  # optional; do not install locally just for this check
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available to PyTorch: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        print("PyTorch: not installed (expected on the lightweight local setup)")

    root = Path(__file__).resolve().parents[1]
    try:
        usage = shutil.disk_usage(root.anchor or os.getcwd())
        print(f"Disk free ({root.anchor or os.getcwd()}): {usage.free / (1024**3):.1f} GB")
    except Exception as exc:
        print(f"Disk free: unable to determine ({exc})")

    print("\nRecommendation: use this machine for CPU-side data/kinematics/evaluation only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
