# SplatROM

**Measurement reliability for biomechanical hand-motion analysis from dynamic 3D representations.**

> Research status: exploratory. The novelty claim is provisional and must be updated after a focused literature review and baseline reproduction.

## Research question

When a dynamic hand reconstruction is incomplete or degraded (occlusion, missing frames, blur, or fewer camera views), can we estimate *when its biomechanical measurements are trustworthy* and selectively abstain from reporting unreliable joint angles?

The project is not a new 4D Gaussian Splatting renderer at this stage. We first evaluate the measurement layer on top of existing reconstruction/pose outputs. This avoids spending compute on a new renderer before establishing a measurable research gap.

## Compute strategy

- Local laptop (8 GB RAM, GeForce 940M 2 GB): data inspection, pose parsing, kinematics, metrics, plots, experiment bookkeeping, and lightweight preprocessing only.
- Cloud GPU notebook: 4DGS training/inference and any GPU-heavy baseline. GPU availability is not guaranteed on free notebook tiers.
- Do not download full multi-camera datasets initially. Start with H2O pose-only files and a short, selected RGB sequence only after the pose pipeline passes its tests.

## Repository layout

```text
configs/       experiment configuration
 data/         dataset mount points; never commit dataset files
 docs/         research and compute plans
 scripts/      command-line utilities
 src/splatom/  reusable Python code
 tests/        unit tests
 experiments/  per-experiment manifests and notes
 results/      generated metrics and figures
```

## Local setup (Windows PowerShell)

Use Python 3.10. Do not install CUDA or PyTorch on the laptop for this project.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-local.txt
pip install -e .
python scripts/check_system.py
python -m pytest -q
```

If PowerShell blocks activation, run the scripts using `.venv\Scripts\python.exe` directly.

## First data target: H2O pose-only

H2O requires registration and acceptance of ETH Zurich's terms. Download only the pose package for one subject initially (not the RGB-D archives). See `docs/research_plan.md` and the official dataset site: <https://h2odataset.ethz.ch/>.

After extraction, point `dataset_root` in `configs/project.yaml` to the directory containing `subject1`, `subject2`, etc. Never commit the downloaded data.

## First milestone

1. Confirm the local environment with `scripts/check_system.py`.
2. Obtain one H2O pose-only subject archive under the dataset terms.
3. Inspect its MANO/3D pose files and establish the exact joint convention.
4. Validate angle computation against known MANO/hand-pose ground truth.
5. Only then select a short RGB sequence and test a 4DGS baseline on a cloud GPU.

## Important research safeguards

- Separate subjects across train/validation/test; do not randomly split adjacent frames.
- Distinguish pose-estimation error, reconstruction error, and biomechanical measurement error.
- A confidence score is not calibrated merely because it lies in [0, 1]. Evaluate calibration and risk–coverage.
- Report failures and compute cost, not just best-case renderings.
- The final contribution must be adjusted if the literature review finds an equivalent method.

## H2O first-run pipeline

See [`docs/h2o_pipeline.md`](docs/h2o_pipeline.md). After accepting ETH Zurich's dataset terms and receiving credentials, run:

```powershell
python scripts/download_and_run_h2o.py
```

This downloads only Subject 1's annotation-only H2O archive by default, extracts it, and generates the first 3D landmark-angle baseline. Use `--mode all` later for all five camera views needed by multi-view 4D Gaussian Splatting. Credentials are prompted at runtime and are not saved. Raw dataset files must not be committed to GitHub.
