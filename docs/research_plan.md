# Research plan (working version)

## Working title

**SplatROM: Reliability-Calibrated Biomechanical Measurements from Dynamic 3D Hand Reconstructions**

## Why narrow the original idea?

By late 2026, hand-specific 4D Gaussian reconstruction, interaction-aware hand-object 4DGS, and uncertainty-aware dynamic Gaussian reconstruction have all appeared. Therefore, “add a hand skeleton and Gaussian uncertainty” is not a sufficient novelty claim.

The potentially defensible gap is measurement-centric: quantify and calibrate the reliability of *biomechanical outputs* (e.g., MCP/PIP/DIP angles and range of motion) produced from dynamic reconstructions when observations are degraded. A renderer's photometric quality or per-Gaussian uncertainty does not automatically establish that joint-angle measurements are accurate.

This is a hypothesis to test, not a confirmed literature gap.

## Research question

Can a measurement-level reliability estimator predict the error of joint-angle and range-of-motion measurements from dynamic hand reconstructions under controlled visibility and temporal degradation, enabling selective reporting of measurements that meet a target error tolerance?

## Candidate contributions (keep provisional)

1. **Measurement-centred evaluation protocol:** evaluate biomechanical joint measurements—not only rendering quality—under controlled frame loss, view sparsity, occlusion, and blur.
2. **Measurement-level reliability model:** estimate expected joint-angle error using available signals such as pose visibility, reprojection residuals, temporal inconsistency, and kinematic violations. Compare against simple confidence baselines.
3. **Selective measurement and calibration:** calibrate predicted reliability against actual angle error; report risk–coverage curves and the proportion of measurements that meet a predeclared tolerance.

An additional method contribution (e.g. kinematic-temporal recovery) should only be added if baseline results identify a clear failure mode and the literature review shows it is not already solved.

## Dataset plan

### H2O first

H2O supplies synchronized multi-view RGB-D, camera poses, MANO hand annotations, object poses, and meshes. Its official download supports a pose-only option, making it a lower-storage first step than downloading all RGB-D data. Access requires registration and acceptance of ETH Zurich terms. Start with one subject's pose package only.

Official sources:
- Dataset and terms: https://h2odataset.ethz.ch/
- Dataset paper: https://openaccess.thecvf.com/content/ICCV2021/html/Kwon_H2O_Two_Hands_Manipulating_Objects_for_First_Person_Interaction_Recognition_ICCV_2021_paper.html
- Official repository/data layout: https://github.com/taeinkwon/h2odataset

### SHOW3D later

SHOW3D (CVPR 2026) provides in-the-wild multi-view hand-object recordings and 3D annotations, but the full release is about 227 GB and is CC BY-NC 4.0. Do not download it at project start. Use a carefully selected subset only if cross-dataset generalization is necessary and access/terms are acceptable.

Sources:
- https://show3d-dataset.github.io/
- https://huggingface.co/datasets/facebook/show3d-dataset

## Splits and leakage prevention

Use subject-independent splits. Never randomly split frames from the same recording across train and test. First reproduce the dataset's official split definitions, then document any deviations. Fit calibration/reliability models using training/validation subjects only; reserve held-out subjects for final evaluation.

## Baselines

- Ground-truth hand pose/MANO angles (measurement reference and implementation check).
- Available hand-pose estimates without 4D reconstruction, where applicable.
- One selected public hand/dynamic 4DGS reconstruction baseline.
- Simple reliability baselines: visibility-only, confidence-only, temporal-residual-only, and a combined regressor.

Do not compare methods on different frames or different camera subsets without explicitly reporting the difference.

## Evaluation

### Measurement accuracy
- Per-joint angle MAE and RMSE in degrees.
- Range-of-motion error over a sequence.
- Temporal jitter on nominally smooth trajectories.

### Reliability quality
- Calibration curve / expected calibration error (with a clearly defined binning scheme).
- Spearman correlation between predicted error and observed error.
- Risk–coverage curve and area under risk–coverage curve.
- Fraction of reported measurements within prespecified angle-error tolerances (choose clinically/biomechanically justified tolerances only after consulting the measurement literature).

### Stress tests
- Frame dropping at declared rates.
- Reduced camera/view count.
- Synthetic occlusion with documented masks.
- Motion blur at controlled strengths.
- Lighting changes only if the selected dataset supports a defensible controlled protocol.

### Rendering metrics (secondary)
PSNR, SSIM and LPIPS can characterize reconstruction appearance, but they are not substitutes for biomechanical measurement metrics.

## Stop/go criteria

Do not begin method development until:
1. The H2O pose parser is verified against its documented format.
2. Joint-angle conventions and units are documented.
3. The measurement baseline runs on multiple sequences and subjects.
4. We can identify a repeatable failure mode under at least one controlled degradation.
5. A focused literature matrix confirms that the proposed reliability-to-measurement contribution is not already covered.
