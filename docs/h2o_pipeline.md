# H2O acquisition and first SplatROM baseline

## Official access requirements

H2O is controlled by ETH Zurich. Register at <https://h2odataset.ethz.ch/> and accept the dataset terms. ETH emails HTTP Basic Authentication credentials. The site says these credentials are valid for 7 days. The downloader cannot bypass this process.

Official source code: <https://github.com/taeinkwon/h2odataset> (`download_script.py`). Official H2OPlayer parser/visualization: <https://github.com/taeinkwon/h2oplayer>.

The script in this repository is a small, standard-library-only downloader for **Subject 1**. It downloads the matching archive for one selected mode plus `object.zip` and `label_split.zip`:

- `pose` (default): `subject1_pose_v1_1.tar.gz`; lightweight annotations only, for the first kinematics baseline.
- `ego`: `subject1_ego_v1_1.tar.gz`; the egocentric camera stream and annotations.
- `all`: `subject1_v1_1.tar.gz`; all five camera views and annotations, the intended H2O input for the later multi-view 4DGS stage.

Only Subject 1 is downloaded to avoid retrieving the full dataset. The official H2O README describes `cam0`–`cam3` as fixed views and `cam4` as the head-mounted view. For the 4DGS paper, use `all` rather than `pose` once the baseline has been validated.

## Run

From the SplatROM project root with the `.venv` activated:

```powershell
python scripts/download_and_run_h2o.py
```

The script prompts for the H2O username and password. Password input is hidden. It normalizes the Windows HTTP proxy mapping used on the current workstation and retains normal TLS certificate verification.

Select a larger mode explicitly:

```powershell
python scripts/download_and_run_h2o.py --mode ego
python scripts/download_and_run_h2o.py --mode all
```

To process archives already in `data/raw/h2o/archives` without downloading again:

```powershell
python scripts/download_and_run_h2o.py --skip-download
```

To force archive redownload:

```powershell
python scripts/download_and_run_h2o.py --force
```

## Outputs

Archives are saved to `data/raw/h2o/archives/`; extracted content is under `data/raw/h2o/extracted/`. The first baseline writes:

- `data/processed/h2o/subject1_cam4_angles.csv`
- `data/processed/h2o/summary.json`

The parser reads H2O `hand_pose` records as 128 values: each hand has one annotation flag followed by 21 XYZ points. This layout and the wrist-to-five-chain skeleton topology are checked against the H2O dataset README and H2OPlayer's `read_text`/hand-projection code.

## Scientific limitation

The initial CSV contains geometric 3D internal angles along the five landmark chains. The base angle uses wrist landmark 0 as the preceding point. These are **not yet validated clinical MCP/PIP/DIP goniometry measurements**. Before a paper claim, the joint semantics, coordinate units, annotation validity and reference-angle protocol must be verified. Subject 1 is only a pipeline smoke-test subject, not a generalization evaluation split.

The pose-only mode has no RGB-D images and cannot train 4DGS. The `all` mode is the multi-view data path for the later Gaussian reconstruction experiments. Observe ETH Zurich's academic/non-commercial/no-redistribution terms; never commit raw data or credentials to GitHub.
