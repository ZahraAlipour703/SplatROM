$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Location).Path
$ApiDir = (Resolve-Path (Join-Path $ProjectRoot "..\show3d-api")).Path
$env:PYTHONPATH = $ApiDir
$RawRoot = Join-Path $ProjectRoot "data\raw\show3d"
$ManifestDir = Join-Path $ProjectRoot "data\manifests"
$Manifest = Join-Path $ManifestDir "show3d_ish822_scene.jsonl"
$OutDir = Join-Path $ProjectRoot "data\processed\show3d\ISH822__aria_inspecting_3ab0\frames_10fps"
if (!(Test-Path (Join-Path $ApiDir "show3d\extract_images.py"))) { throw "Official SHOW3D API not found. Run setup_show3d_api.py first." }
New-Item -ItemType Directory -Force -Path $ManifestDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
'{"subject_id":"ISH822","scene_id":"aria_inspecting_3ab0"}' | Set-Content -Path $Manifest -Encoding Ascii
Write-Host "`n=== Correct thumb proxy ordering ===" -ForegroundColor Cyan
python scripts\fix_thumb_proxy.py
if ($LASTEXITCODE -ne 0) { throw "Thumb proxy patch failed." }
Write-Host "`n=== Re-run kinematics export ===" -ForegroundColor Cyan
$HandPose = Join-Path $RawRoot "hand_pose\v3\scenes\ISH822\aria_inspecting_3ab0\hand_pose_umetrack.json"
python scripts\extract_show3d_kinematics.py --annotation $HandPose --out "data\processed\show3d\ISH822__aria_inspecting_3ab0"
if ($LASTEXITCODE -ne 0) { throw "Kinematics export failed." }
Write-Host "`n=== Headset reprojection and rig calibration audit ===" -ForegroundColor Cyan
python scripts\validate_show3d_geometry.py --root $RawRoot --out "results\show3d_geometry_audit"
if ($LASTEXITCODE -ne 0) { throw "Geometry audit failed." }
Write-Host "`n=== Extract four synchronized rig views at 10 FPS ===" -ForegroundColor Cyan
python -m show3d.extract_images --root $RawRoot --out $OutDir --manifest $Manifest --fps 10 --views rig0 rig1 rig2 rig3 --posed-only --object-pose-version v2 --workers 1
if ($LASTEXITCODE -ne 0) { throw "Multi-view frame extraction failed." }
Write-Host "`nPREPARATION COMPLETE" -ForegroundColor Green
Write-Host "Audit: results\show3d_geometry_audit\geometry_audit.json"
Write-Host "Image index: $OutDir\index.jsonl"
Write-Host "Extraction info: $OutDir\extract_info.json"
