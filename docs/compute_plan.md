# Compute plan

## Local laptop

Known hardware: Intel i7-6500U, 8 GB RAM, NVIDIA GeForce 940M with 2 GB reported VRAM, Intel HD 520. `nvidia-smi` was not found on the reported system.

Use the laptop for:
- Python 3.10 environment and Git operations.
- Reading pose annotations and small metadata files.
- Kinematics and metric computation.
- Small figures and experiment manifests.
- Low-resolution frame inspection (one frame at a time).

Do not use it for:
- Training 3DGS/4DGS.
- Building CUDA rasterizer extensions.
- Decoding many full-resolution camera streams simultaneously.
- Loading the complete SHOW3D release.

The GeForce 940M is too constrained for a dependable modern 4DGS training setup. Absence of `nvidia-smi` only means the NVIDIA command-line utility is unavailable/not on PATH; it does not prove the GPU is physically absent.

## Cloud GPU

Use a hosted notebook (e.g. Colab) for one selected 4DGS baseline. Verify GPU model, VRAM, CUDA, PyTorch and available disk at the start of every session. Free-tier GPU availability and limits vary and are not guaranteed. Save checkpoints and logs to persistent storage frequently.

Do not assume the latest PyTorch/CUDA will work with an older 4DGS repository: the original CVPR 2024 4D-GS code documents an older Python/PyTorch/CUDA-extension stack. Create a separate pinned environment for the chosen baseline rather than mixing it with the local data-analysis environment.

## Storage policy

- H2O: request/download pose-only first; avoid RGB-D archives at project start.
- SHOW3D: do not download the full ~227 GB release. Select a small subset only if required.
- Store raw data outside Git. Track manifests, checksums, subject/scene IDs and preprocessing scripts in Git.
