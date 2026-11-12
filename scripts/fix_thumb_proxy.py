#!/usr/bin/env python3
"""Safely fix the thumb landmark order in the existing kinematics exporter."""
from pathlib import Path
p = Path("scripts/extract_show3d_kinematics.py")
if not p.exists(): raise SystemExit(f"Run from SplatROM root; missing {p}")
s = p.read_text(encoding="utf-8")
old = '''    out["thumb_base_proxy_deg"] = flexion_deg(palm, points[7], points[6])
    out["thumb_ip_deg"] = flexion_deg(points[7], points[6], points[0])'''
new = '''    # Landmark order: 5=wrist, 6=thumb intermediate, 7=thumb distal, 0=thumb tip.
    # These remain geometric proxies, not clinical anatomical joint ground truth.
    out["thumb_mcp_proxy_deg"] = flexion_deg(points[5], points[6], points[7])
    out["thumb_ip_proxy_deg"] = flexion_deg(points[6], points[7], points[0])'''
if old in s:
    p.write_text(s.replace(old, new), encoding="utf-8"); print("[OK] Thumb proxy order corrected.")
elif new in s: print("[SKIP] Thumb proxy order already corrected.")
else: raise SystemExit("Expected old code not found; no file changed. Inspect compute_angles() manually.")
