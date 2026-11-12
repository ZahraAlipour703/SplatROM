# SplatROM v0.5 — Geometry validation and multi-view preparation

This package advances the downloaded sample `ISH822/aria_inspecting_3ab0`.

## What the v0.4 audit established

- 1,786 frames; strictly increasing timestamps; median delta approximately 1/60 s.
- Both hands have valid 21-point world landmarks in all 1,786 frames at confidence > 0.1.
- Every hand frame contains 22 native UmeTrack angle parameters.
- `object_pose/v2` and `interaction_field/v1` exist locally.
- The previous local-to-world residual (~1e-14 mm) is algebraic consistency, not independent camera validation.

## 1. Install the official SHOW3D API

From the SplatROM project root:

```powershell
python scripts\setup_show3d_api.py --api-dir ..\show3d-api
```

It clones `https://github.com/facebookresearch/show3d` into a sibling directory and installs OpenCV. Then set the package path in the current PowerShell session:

```powershell
$env:PYTHONPATH = (Resolve-Path ..\show3d-api).Path
```

No PyTorch, CUDA, MANO asset, or physical device is needed for this stage.

## 2. Run the preparation pipeline

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_multiview_preparation.ps1
```

The script safely corrects the thumb landmark order, regenerates the exploratory kinematics CSV, performs a headset0 2D reprojection check and rig camera audit, then extracts rig0–rig3 at 10 FPS. `--posed-only` uses the object-pose v2 frame grid and retains frames with object-pose confidence above 0.5.

## Outputs

- `results/show3d_geometry_audit/geometry_audit.json`
- `results/show3d_geometry_audit/headset0_reprojection.csv`
- `data/processed/show3d/ISH822__aria_inspecting_3ab0/frames_10fps/index.jsonl`
- `data/processed/show3d/ISH822__aria_inspecting_3ab0/frames_10fps/extract_info.json`

Approximately 300 frames per view are expected; the exact count depends on object-pose confidence and its frame grid.

## Interpretation

- Headset0 reprojection error compares projected 3D landmarks with the dataset's 2D headset0 keypoints. It is a stronger image-space consistency test than local-to-world algebra, but it is not external motion-capture ground truth.
- Rig rotations and translations are read through the official API, which validates rigid 4x4 transforms.
- Interaction-field vector norms are distances to the nearest object-mesh vertex. Fractions below 5/10 mm are exploratory proximity summaries, not binary contact labels.
- `ISH822` is a training/development subject. Do not use it as the held-out test subject in the final paper.

## After v0.5

Next deliverable: export per-image camera intrinsics/extrinsics and timestamps in a format supported by a selected 4DGS implementation. Do not start 4DGS training until the reprojection and camera-rig audit have been reviewed.

## Attribution

Official API: https://github.com/facebookresearch/show3d
Dataset: https://huggingface.co/datasets/facebook/show3d-dataset
SHOW3D dataset and API are CC BY-NC 4.0; preserve attribution and non-commercial restrictions.
