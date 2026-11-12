# SplatROM — SHOW3D sample pipeline (v0.3)

This replaces the blocked H2O/ETH download path with a public, selectively downloadable
multi-view hand-object dataset hosted on Hugging Face:

- Dataset: `facebook/show3d-dataset`
- Sample scene: subject `ISH822`, scene `aria_inspecting_3ab0`
- License: CC BY-NC 4.0 (attribution required; non-commercial use only)
- Dataset overview: https://huggingface.co/datasets/facebook/show3d-dataset
- Dataset API/code: https://github.com/facebookresearch/show3d

## Why this dataset

SHOW3D contains synchronized 60-fps videos from 8–10 calibrated cameras per recording,
plus per-frame 3D hand poses. The scene README documents:
- videos: `headset0.mp4`, `headset1.mp4`, `rig0.mp4` ... `rig7.mp4`
- intrinsics and camera-to-rig transforms
- per-frame timing/missing-camera metadata
- hand pose annotations with 21 3D landmarks and confidence

Only one recording and selected views are downloaded here. The full dataset is not downloaded.

Important coordinate note: SHOW3D's "world" is the back-mounted camera-rig frame. It is
consistent across cameras at one timestamp, but moves with the participant across time.
This is suitable for synchronized hand reconstruction relative to the rig; it is not a
static room/world coordinate system unless the rig motion is additionally estimated.

## Hardware plan

The downloader and landmark/kinematics export use Python's standard library plus NumPy.
No PyTorch, CUDA, or local GPU is needed for this first stage. 4DGS training/rendering
will be moved to a cloud GPU after the data pipeline and calibration are validated.

## Install / run

From the existing SplatROM root, with its `.venv` active:

```powershell
python scripts\download_show3d_sample.py
```

The script downloads:
- four calibrated exocentric videos (`rig0`–`rig3`) and one headset view (`headset0`)
- matching camera calibration files
- frame and recording metadata
- UmeTrack hand pose JSON (plus optional MANO and subject profiles)

It then runs the CPU-only kinematics exporter automatically.

To fetch fewer files while testing the annotation pipeline:

```powershell
python scripts\download_show3d_sample.py --skip-videos
```

To fetch only three exocentric cameras:

```powershell
python scripts\download_show3d_sample.py --views rig0 rig1 rig2
```

To use a different recording later:

```powershell
python scripts\download_show3d_sample.py --subject SUBJECT_ID --scene SCENE_ID
```

## Output

```text
data/
  raw/show3d/
    scenes/ISH822/aria_inspecting_3ab0/
      rig0.mp4 ... rig3.mp4
      headset0.mp4
      camera_calibration/
      metadata/
    hand_pose/v3/scenes/ISH822/aria_inspecting_3ab0/
      hand_pose_umetrack.json
      hand_pose_mano.json (if available)
    hand_pose/hand_profiles/ISH822/
  processed/show3d/ISH822__aria_inspecting_3ab0/
    hand_kinematics.csv
    landmarks_3d.csv
    summary.json
```

## Research cautions

- This first export is a data/annotation sanity check and geometric baseline, not the proposed
  reliability-aware 4DGS method.
- The MCP values are explicitly named `*_mcp_proxy_deg` because they use palm-centre anchoring.
  Do not report them as clinical-grade anatomical angles.
- Preserve the dataset's train/test protocol in any paper. Do not tune on test recordings.
- Do not upload raw videos or annotations to GitHub. Keep only scripts, configuration, and
  derived results allowed by the license.
- Cite the SHOW3D CVPR 2026 paper and attribute the dataset. The license is CC BY-NC 4.0.

## Citation

Rim, P. et al. "SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild."
CVPR 2026. https://arxiv.org/abs/2603.28760
