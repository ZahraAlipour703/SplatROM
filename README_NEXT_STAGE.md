# SplatROM v0.4 — SHOW3D next-stage audit

This supplements the already-downloaded sample:
`ISH822/aria_inspecting_3ab0`.

## New information available in SHOW3D (released 19 Oct 2026)

The dataset now includes:
- `object_pose/v2`: per-frame 6DoF object pose.
- `interaction_field/v1`: for each hand landmark, a vector in mm to the nearest vertex of the object's surface.

These two annotations are valuable for SplatROM's hand-object interaction direction. They are not part of the first downloader's bundle.

## Run from the project root

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\download_show3d_object_interaction.py
python scripts\audit_show3d_sample.py
```

The first command tries to fetch only two optional annotation files for this one scene. If a file is absent for the selected scene, it reports that rather than treating it as a fatal error.

The audit:
- verifies the official 21-landmark UmeTrack schema;
- counts accepted hand frames using confidence > 0.1 and world geometry;
- records the native 22-value `joint_angles` vector in radians and degrees without inventing joint names;
- compares `R_world_from_wrist * landmarks_local + t_world_from_wrist` against world landmarks;
- reports timestamp cadence and missing-camera counts;
- writes three CSVs and one JSON report.

## Why this audit comes before 4DGS

We need to verify coordinate frames and annotation consistency before using them as supervision. The current parser's MCP angles are palm-centre anchored proxies, not clinical joint-angle ground truth. The native UmeTrack 22-angle vector must be decoded using the subject's model profile before per-joint anatomical claims.

SHOW3D positions are in the moving back-mounted rig coordinate frame. For multi-view 4D reconstruction, this is suitable as a participant-centric coordinate frame, but not as a static room/world frame.

## Research use

A strong path is to evaluate a hand-object reconstruction against:
1. multi-view RGB reprojection;
2. 3D hand landmarks;
3. 6DoF object pose when present;
4. interaction-field distances/contact geometry;
5. temporal consistency under view removal and occlusion.

Do not call the project "uncertainty calibrated" until uncertainty is empirically calibrated against a defined error target and split.
