# Exo-SNN
<<<<<<< HEAD
AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves 


# Exo-SNN
=======
>>>>>>> 3c3a5c3 (vetting report generator modified)

**AI-enabled detection and vetting of exoplanet transit candidates in noisy TESS light curves.**

Exo-SNN takes a TESS light curve (downloaded live from MAST, or supplied as arrays), cleans and flattens it, searches it for periodic transit-like dips with Box Least Squares (BLS), phase-folds it into a *global* and a *local* view, scores those views with a three-branch neural network that includes a **spiking neural network (SNN)** branch, and cross-checks the candidate with independent physical vetting (transit count, odd/even depth, secondary eclipse). Results are shown in an interactive Streamlit dashboard or written out as a one-page PNG vetting sheet.

> **Important:** the AI score is a *model score*, not confirmation of a planet. The final status shown to the user comes from BLS significance and physical vetting (see [Overall status codes](#overall-status-codes)), not from the network alone.

---

## Table of contents

1. [Highlights](#highlights)
2. [How it works](#how-it-works)
3. [Project layout](#project-layout)
4. [Installation](#installation)
5. [Quick start: the dashboard](#quick-start-the-dashboard)
6. [Command-line usage](#command-line-usage)
7. [Using the pipeline from Python](#using-the-pipeline-from-python)
8. [Pipeline stages in detail](#pipeline-stages-in-detail)
9. [Model architecture](#model-architecture)
10. [Overall status codes](#overall-status-codes)
11. [Training](#training)
12. [Evaluation and benchmarks](#evaluation-and-benchmarks)
13. [Weights and checkpoints](#weights-and-checkpoints)
14. [Bundled data and caches](#bundled-data-and-caches)
15. [Tests](#tests)
16. [Troubleshooting](#troubleshooting)
17. [Limitations and honest caveats](#limitations-and-honest-caveats)
18. [Data sources and references](#data-sources-and-references)

---

## Highlights

- **End-to-end pipeline:** MAST download, cleaning, BLS search, dual-view extraction, neural classification, Grad-CAM explanation, physical vetting, and report generation behind one `Pipeline` class.
- **Tri-branch classifier:** a global-view CNN, a local-view CNN and a differentiable spiking (LIF) branch fused into one classifier.
- **Conservative status logic:** a high AI score cannot override a weak BLS peak, a boundary-pinned period, or too few distinct transit events. All status logic lives in one file (`root/status.py`), so the dashboard, CLI and PNG report can never disagree.
- **Explainable:** Grad-CAM saliency is overlaid on the local transit view.
- **Fails loudly:** untrained or legacy weights are flagged everywhere (CLI, dashboard, report) instead of silently producing meaningless scores. Too-short light curves raise an error rather than returning a number.
- **Leakage-safe evaluation:** benchmark and unseen-set scripts exclude every TIC used for training/validation and freeze their manifests.
- **Offline-friendly pieces:** synthetic light-curve simulator for the dashboard and for pretraining; bundled QLP FITS files and cached real light curves for tests and training.

---

## How it works

```
 TIC ID / sector  (or raw time+flux arrays)
        │
        ▼
 1. Ingest         root/data/ingest.py          MAST via lightkurve; SPOC first, falls back to other authors (QLP / TARS HLSP)
        │
        ▼
 2. Clean+flatten  root/preprocessing/clean.py  NaN/dup removal, sigma clipping, 48 h Savitzky-Golay detrend
        │
        ▼
 3. Detect         root/detection/bls_detector.py   BLS period search (0.5 to 15 d), SNR, boundary check
        │
        ├──► 4. Dual views   root/detection/dual_stream.py   global (201 pts) + local (61 pts), z-scored
        │            │
        │            ▼
        │    5. Classify     root/classification/tri_branch_ensemble.py   CNN + CNN + SNN  →  planet score
        │            │
        │            ▼
        │    6. Explain      root/classification/xai.py   Grad-CAM saliency on the local view
        │
        └──► 7. Vet          root/classification/physical_verification.py   transit count, odd/even, secondary
                     │
                     ▼
 8. Status + report   root/status.py, root/reporting/   overall status code, PNG vetting sheet, dashboard
```

The AI branch and the physical-vetting branch are **independent evidence streams**: the v9 network is morphology-only and does not consume the vetting measurements.

---

## Project layout

Run everything from the **project root**, the directory that *contains* the `root/` folder (the `astro/` directory in the distributed zip).

```
astro/
├── README.md
├── .gitignore
├── tests/
│   └── test_pipeline_fixes.py        pytest regression suite
└── root/
    ├── __init__.py
    ├── Requirements.txt              Python dependencies
    ├── main_pipeline.py              Pipeline class + single-target CLI
    ├── status.py                     single source of truth for overall status
    ├── views.py                      phase-fold / normalise helpers (shared train + inference)
    ├── checkpoint_io.py              safe, version-tolerant checkpoint loader
    │
    ├── data/
    │   ├── ingest.py                 MAST download, sector listing, generic HLSP FITS reader
    │   ├── synthetic_data.py         LightCurveSimulator (planets, EBs, variables, artifacts)
    │   └── dataset_builder.py        compatibility dataset builder / cache
    ├── preprocessing/
    │   └── clean.py                  clean_and_flatten()
    ├── detection/
    │   ├── bls_detector.py           run_bls()
    │   └── dual_stream.py            extract_dual_views()
    ├── classification/
    │   ├── tri_branch_ensemble.py    TriBranchTESSNet
    │   ├── snn_branch.py             SNNTemporalBranch
    │   ├── snn_core.py               LIF neuron + surrogate gradient
    │   ├── delta_modulation.py       delta-modulation spike encoding utilities
    │   ├── feature_extraction.py     summary-feature helpers
    │   ├── physical_verification.py  verify_astrophysics()
    │   └── xai.py                    compute_1d_gradcam()
    ├── reporting/
    │   └── vetting_report.py         generate_vetting_report() → PNG sheet
    ├── ux/
    │   └── dashboard.py              Streamlit dashboard
    │
    ├── train_all.py                  stage 1: synthetic pretraining
    ├── train_real.py                 stage 2: real-TESS fine-tuning (NASA TOI labels)
    ├── evaluate_real.py              evaluate on user-supplied labelled cases
    ├── benchmark_real.py             frozen 40-object leakage-safe benchmark
    ├── evaluate_unseen.py            larger 200-object unseen evaluation
    │
    ├── weights/                      checkpoints (+ JSON metadata sidecar)
    ├── real_data_cache/              cached real light curves (.npz) for training
    ├── benchmark_results/            benchmark manifest, CSV, metrics
    ├── unseen_evaluation_200/        unseen-set manifest, CSV, metrics
    └── reports/                      generated PNG vetting sheets
```

The bundled `mastDownload/HLSP/` folder (QLP FFI light curves in FITS format) sits next to `root/` and is used by the tests and as a local data sample.

---

## Installation

**Requirements:** Python 3.9 or newer is recommended, plus internet access to `mast.stsci.edu` for real-target runs (the dashboard's synthetic modes work offline).

```bash
# 1. Go to the project root (the folder that contains root/)
cd astro

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r root/Requirements.txt
```

Dependencies (minimum versions, not a lock file):

| Package | Min version | Used for |
|---|---|---|
| `torch` | 2.0.0 | the neural network |
| `numpy` | 1.24.0 | array maths |
| `scipy` | 1.10.0 | Savitzky-Golay detrending |
| `astropy` | 5.3.0 | BLS, FITS reading |
| `lightkurve` | 2.4.0 | TESS light-curve search/download |
| `astroquery` | 0.4.7 | MAST queries |
| `matplotlib` | 3.7.0 | plots and PNG report |
| `pandas` | 2.0.0 | tabular handling |
| `scikit-learn` | 1.3.0 | metrics (AUC, F1, ...) |
| `streamlit` | 1.30.0 | dashboard |
| `requests` | 2.31.0 | NASA Exoplanet Archive TAP queries |
| `pytest` | 7.0 | tests |

For fully reproducible deployments, run `pip freeze > requirements.lock.txt` in your working environment and keep that file alongside the project.

---

## Quick start: the dashboard

From the project root (the folder containing `root/`):

```bash
python -m streamlit run root/ux/dashboard.py
```

Streamlit prints a local URL (normally `http://localhost:8501`) and opens it in your browser.

### Using the dashboard

1. In the sidebar choose a **Data source**:
   - **Real TESS target (MAST)**: enter a TIC ID (default `383715654`), optionally a sector, and click **List available sectors** to see what MAST holds for that star. Requires network access and can take a minute or more.
   - **Synthetic planet sample** / **Synthetic false-positive sample**: generates a fresh simulated light curve with known ground truth (shown as a caption under the overview). Works offline.
2. Optionally set the stellar parameters (radius in R☉, effective temperature, log g).
3. Click **▶ Run Pipeline**. The model is loaded once and cached for the session.
4. Read the results top to bottom:
   - **Overall status** banner (green / amber / red): the headline verdict.
   - **Pipeline status** tiles and **Quick overview** metrics (BLS SNR and AI confidence).
   - **Light curve**, raw vs cleaned.
   - **BLS periodogram** with the best period marked.
   - **Global view** (phase-folded, full period) and **local view** with Grad-CAM saliency shading.
   - **Estimated parameters** (period, duration, depth, SNR) and **physical vetting** (odd/even depths, mismatch %, secondary depth, EB flag).
   - **Generate Vetting Report Image** to render, preview and download the PNG sheet.

Nothing in the dashboard is hard-coded: if a field is missing from the pipeline's result, it is shown as *N/A*.

> **Warnings you may see:** a red *UNTRAINED WEIGHTS* box means no checkpoint was found; an amber *LEGACY WEIGHTS: UNVALIDATED* box means an old-format checkpoint was loaded. In both cases the AI confidence should be ignored.

---

## Command-line usage

All commands are run from the project root and always as modules (`python -m root.<name>`).

### Analyse one real target

```bash
python -m root.main_pipeline --tic 172518755 --sector 21
```

| Flag | Default | Meaning |
|---|---|---|
| `--tic` | `383715654` | TIC ID of the target |
| `--sector` | first available | TESS sector number |
| `--weights` | `root/weights/tri_branch_tess_net.pt` | checkpoint to load |
| `--min-snr` | `7.0` | BLS SNR below which the target is reported as `NO_SIGNIFICANT_DETECTION` |
| `--list-sectors` | off | only query MAST for available sectors/authors, then exit |

```bash
# See which sectors exist before choosing one
python -m root.main_pipeline --tic 172518755 --list-sectors
```

The run prints the best period, SNR, period quality, transit counts, overall status and AI score, and saves a PNG vetting sheet to `root/reports/TIC_<id>_report.png`.

### Train

```bash
# Stage 1: synthetic pretraining
python -m root.train_all

# Stage 2: real-TESS fine-tuning (needs network: NASA Exoplanet Archive + MAST)
python -m root.train_real
```

See [Training](#training) for the options.

### Evaluate

```bash
# Labelled cases you supply: TIC:SECTOR:LABEL[:REFERENCE_PERIOD]
python -m root.evaluate_real \
  --case 172518755:21:1:3.2087 \
  --case 224245334:29:1:3.7853 \
  --case 331484419:55:0:6.2566

# Frozen 40-object leakage-safe benchmark
python -m root.benchmark_real --n-per-class 20

# Larger 200-object unseen evaluation
python -m root.evaluate_unseen --n-per-class 100
```

### Run the tests

```bash
python -m pytest tests -v
```

---

## Using the pipeline from Python

Build **one** `Pipeline` and reuse it; constructing it loads the weights.

```python
from root.main_pipeline import Pipeline

pipe = Pipeline()                       # loads root/weights/tri_branch_tess_net.pt

# Real target from MAST
result = pipe.run(172518755, sector=21, r_star=1.0, teff=5778.0, logg=4.44,
                  generate_report=True)

# Your own light curve (e.g. a synthetic or locally loaded one)
result = pipe.run_from_arrays(time, flux, target_id="my_target")

print(result["overall_status"])         # e.g. PERIODICITY_SUPPORTED
print(result["bls"]["period"], result["bls"]["snr"])
print(result["ai"]["planet_probability"])
print(result["weights_loaded"])         # False → ignore the AI score
```

### Result dictionary

| Key | Contents |
|---|---|
| `target_id` | e.g. `TIC_172518755` |
| `weights_loaded` / `weights_present` | whether a *validated-format* checkpoint was loaded / whether any file existed |
| `light_curve` | `time_raw`, `flux_raw`, `time_clean`, `flux_clean` |
| `bls` | `period`, `t0`, `duration`, `depth`, `depth_err`, `snr`, `periods`, `powers`, `period_quality`, boundary flags, `n_trial_periods`, ... |
| `views` | `global_view`, `local_view`, `saliency` (Grad-CAM, may be `None`), plus coverage metadata |
| `ai` | `planet_probability`, `planet_score` (temperature-scaled softmax probability) |
| `verification` | `odd_depth`, `even_depth`, `odd_even_mismatch_pct`, `secondary_depth`, `is_eclipsing_binary_flag`, `observed_transits`, `expected_transits`, `period_validation_status`, status fields |
| `overall_status` | one of the [status codes](#overall-status-codes) |
| `stellar` | `r_star`, `teff`, `logg` as supplied |
| `report_file` | path to the PNG (only when `generate_report=True`) |

---

## Pipeline stages in detail

### 1. Ingest (`root/data/ingest.py`)

- Searches MAST through `lightkurve` for TESS light curves, **SPOC first** (PDCSAP), then falls back to all authors if SPOC has nothing.
- Generic HLSP products (QLP, TARS) are read directly from their FITS files, bypassing Lightkurve's quality-bitmask reader. The QLP quality mask is opt-in (`qlp_quality_mask=True`).
- If no sector is given, the first available sector is used. Multi-sector stitching is available explicitly with `all_sectors=True`.
- Network calls use a 45 s timeout and 3 retries with back-off. A failure raises a descriptive `ConnectionError` pointing at firewall/VPN/proxy causes.
- `list_available_sectors(tic)` returns sector, author and exposure time for each product.

### 2. Clean and flatten (`root/preprocessing/clean.py`)

- Drops non-finite and non-positive samples, sorts by time, removes duplicate timestamps, and sanitises bad flux errors (a robust noise estimate is substituted when none are available).
- Asymmetric sigma clipping (default 5σ upper / 15σ lower) so deep transits are kept while flares and spikes are removed.
- Savitzky-Golay detrending with a window specified in **hours** (default 48 h), deliberately much longer than the 1 to 4 h transit durations searched, so transits are not fitted away.
- A light curve with fewer than **100** valid cadences after cleaning raises a `ValueError`.

### 3. BLS detection (`root/detection/bls_detector.py`)

- Astropy `BoxLeastSquares` over periods of **0.5 to 15 days** with trial durations in the 1 to 4 hour range.
- Very long (multi-sector) series are binned to at most 20,000 points to keep runtime bounded; the trial-period grid is capped.
- Reports a transit SNR and a `period_quality` of `INTERIOR_PEAK` or `BOUNDARY_PEAK`. A peak at either edge of the search range is not trusted.

### 4. Dual views (`root/detection/dual_stream.py`)

- **Global view:** 201 phase bins across the full period.
- **Local view:** 61 points in a window of roughly two transit durations either side of mid-transit.
- Both are per-sample z-scored with the **same normalisation used in training**. This train/inference parity matters: skipping it makes the network collapse to chance.

### 5. Classification and explanation

See [Model architecture](#model-architecture). The raw logits are divided by the checkpoint's calibration **temperature** before the softmax. Grad-CAM (`xai.py`) highlights which parts of the local view drove the "planet" score; if it fails the run continues without a saliency map.

### 6. Physical vetting (`root/classification/physical_verification.py`)

Conservative, light-curve-only checks. A test that lacks data is reported as **indeterminate**, never as a zero or a pass.

- **Transit counting:** counts *distinct* transit events actually observed against those expected. At least **3** distinct events are needed for `period_validation_status = SUPPORTED`; otherwise `INSUFFICIENT_TRANSITS`, `NO_TRANSITS` or `INVALID_EPHEMERIS`.
- **Odd/even depth test:** a mismatch above 35% flags a possible eclipsing binary.
- **Secondary-eclipse test** at phase 0.5.
- A minimum of 6 in-transit points is required for the depth measurements.

> Not included: pixel-level Target Pixel File centroid analysis. Background eclipsing binaries that contaminate the aperture cannot be ruled out by this pipeline.

### 7. Report (`root/reporting/vetting_report.py`)

A 12 × 8 inch PNG with the BLS periodogram, summary and vetting text, and the phase-folded views, titled with the TIC number and showing the same overall status as the dashboard.

---

## Model architecture

`TriBranchTESSNet` (`root/classification/tri_branch_ensemble.py`) is a **morphology-only** classifier (the v9 contract):

| Branch | Input | Structure | Output |
|---|---|---|---|
| Global CNN | 201-pt global view | 2 × (Conv1D + GroupNorm + ReLU + MaxPool), channels 16 → 32, kernel 5, adaptive pool to 4 | 128 features |
| Local CNN | 61-pt local view | 2 × (Conv1D + GroupNorm + ReLU + MaxPool), channels 16 → 32, kernel 3, adaptive pool to 4 | 128 features |
| SNN temporal branch | 61-pt local view | per-sample linear projection (32), **Leaky-Integrate-and-Fire** neuron evolving across all 61 positions, spike-rate readout, dense to 16 | 16 features |

The three outputs (128 + 128 + 16 = 272) are concatenated and passed through an MLP head `272 → 96 → 32 → 2` with dropout (0.25, 0.10). Output is two logits (false positive / planet).

SNN details (`snn_core.py`): LIF decay 0.85, threshold 1.0, reset-by-subtraction, with an arctan-style **surrogate gradient** so the spiking neuron can be trained by backpropagation. GroupNorm (rather than BatchNorm) keeps inference stable on single samples.

---

## Overall status codes

Defined in `root/status.py`. Evaluated in this order:

| Code | Meaning |
|---|---|
| `NO_SIGNIFICANT_DETECTION` | BLS SNR is below the threshold (default 7.0, or NaN). White-noise curves can reach SNR ≈ 5, so such a peak is never promoted. |
| `UNCONFIRMED_BOUNDARY_PEAK_AND_INSUFFICIENT_TRANSITS` | best period is pinned to the search edge **and** fewer than 3 distinct transits support it |
| `UNCONFIRMED_BOUNDARY_PEAK` | best period is pinned to the search edge |
| `UNCONFIRMED_INSUFFICIENT_TRANSITS` | fewer than 3 distinct transit events observed |
| `EB_FLAGGED` | period is supported but odd/even or secondary tests suggest an eclipsing binary |
| `PERIODICITY_SUPPORTED` | significant BLS peak, interior period, ≥ 3 transits, no EB flag |

`PERIODICITY_SUPPORTED` means a *periodic transit-like signal is supported by the data*. It is a candidate, not a confirmed planet.

---

## Training

Training is two-stage. Both stages write checkpoints to `root/weights/`.

### Stage 1: synthetic pretraining (`train_all.py`)

Trains on simulated light curves from `LightCurveSimulator`: planets with trapezoidal ingress/egress, dilution, and variable depth/duration/period, versus hard negatives (eclipsing binaries, periodic stellar variability, repeated instrumental dips, aperiodic artifacts). Selects the checkpoint by validation ROC-AUC rather than last epoch, with a deterministic stratified split.

```bash
python -m root.train_all --n-samples 1200 --epochs 30 --batch-size 32
```

| Flag | Default |
|---|---|
| `--n-samples` | 1200 |
| `--epochs` | 30 |
| `--batch-size` | 32 |
| `--lr` | 5e-4 |
| `--weight-decay` | 1e-4 |
| `--val-frac` | 0.2 |
| `--seed` | 42 |
| `--points-per-day` | 360 |
| `--days` | 27 |
| `--workers` | CPU count |
| `--weights-out` | `root/weights/tri_branch_tess_net_pretrained.pt` |

### Stage 2: real-TESS fine-tuning (`train_real.py`)

Queries the NASA Exoplanet Archive TOI table: **CP/KP → planet (1)**, **FP/FA → false positive (0)**. Any TIC with a CP/KP disposition is excluded from the negative pool. Splits are made **by TIC before augmentation**, so a host star never appears in both train and validation. Light curves are cached in `root/real_data_cache/`.

```bash
python -m root.train_real --max-per-class 40 --epochs 25
```

| Flag | Default | Meaning |
|---|---|---|
| `--max-per-class` | 40 | planets and false positives per class |
| `--val-frac` | 0.2 | validation fraction (by TIC) |
| `--epochs` | 25 | |
| `--batch-size` | 8 | |
| `--lr` | 2e-5 | backbone learning rate |
| `--head-lr` | 5e-5 | classifier-head learning rate |
| `--augment` | 2 | augmentation factor |
| `--pretrained` | `...pretrained.pt` | stage-1 checkpoint to start from |
| `--cache-dir` | `root/real_data_cache` | |
| `--weights-out` | `root/weights/tri_branch_tess_net.pt` | |
| `--holdout` | `172518755,224245334,331484419` | TICs excluded from training |

A JSON sidecar (`tri_branch_tess_net.pt.json`) records the model contract, calibration temperature, validation metrics, and the train/validation TIC lists, which the evaluation scripts use to avoid leakage.

---

## Evaluation and benchmarks

Three increasingly rigorous options, all excluding the TICs used in training/validation:

| Script | What it does | Outputs |
|---|---|---|
| `evaluate_real.py` | Runs the pipeline on cases you label yourself and reports score, label and (optionally) period error | console |
| `benchmark_real.py` | Deterministic, frozen sample (default 20 planets + 20 FP/FA) drawn from the TOI table | `root/benchmark_results/` |
| `evaluate_unseen.py` | Larger deterministic sample (default 100 + 100), also excluding every benchmark TIC | `root/unseen_evaluation_200/` |

Both use `--select-only` to freeze a manifest without downloading light curves. **Do not retrain or tune after running them** if you intend to report the numbers as a blind test.

### Results shipped with this version (threshold 0.5)

| Set | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|---|---|
| Real-TESS validation (checkpoint metadata) | 14 | 0.643 | n/a | n/a | 0.615 | 0.771 | 0.744 |
| Frozen benchmark | 40 (20 / 20) | 0.725 | 0.800 | 0.600 | 0.686 | 0.775 | 0.832 |
| Unseen evaluation | 200 (100 / 100) | 0.635 | 0.652 | 0.580 | 0.614 | 0.730 | 0.746 |

Unseen-set confusion matrix (rows = true FP, planet; columns = predicted FP, planet): `[[69, 31], [42, 58]]`.

**Read these honestly:** on 200 genuinely unseen real targets the network ranks planets above false positives better than chance (ROC-AUC ≈ 0.73), but it is *not* a high-accuracy classifier. This is why the pipeline pairs it with BLS-significance and physical-vetting gates, and why the AI score should be treated as supporting evidence only.

---

## Weights and checkpoints

| File | Purpose |
|---|---|
| `root/weights/tri_branch_tess_net.pt` | **production** checkpoint used by the dashboard and CLI (real-TESS fine-tuned) |
| `root/weights/tri_branch_tess_net.pt.json` | metadata sidecar: model contract, temperature, validation metrics, train/val TICs |
| `root/weights/tri_branch_tess_net_pretrained.pt` | synthetic-pretrained starting point for stage 2 |

How a checkpoint is treated when loaded:

- **`format_version` ≥ 5**: the v9 real-TESS morphology checkpoint; accepted and marked validated.
- **`format_version` 2 to 4**: accepted as a validated-format checkpoint.
- **Older / raw `state_dict`**: loaded for inspection but flagged `weights_loaded = False` (*LEGACY, UNVALIDATED*) because the preprocessing contract has changed.
- **Missing keys** raise an error; unexpected (obsolete) keys are ignored with a note.
- **No file**: the network runs with random weights and every output is flagged as untrained.

`load_checkpoint` prefers `torch.load(..., weights_only=True)`, and falls back to `weights_only=False` only when a checkpoint carries non-tensor metadata. **Only load checkpoints you produced or trust.**

---

## Bundled data and caches

- `mastDownload/HLSP/` holds QLP full-frame-image light curves (FITS) used by the tests and as a local sample (e.g. TIC 172518755, sector 21).
- `root/real_data_cache/*.npz` holds cached real light curves, named `TIC_<id>_S<sector>.npz`, reused by `train_real.py` so training does not re-download.
- `root/reports/` holds generated PNG vetting sheets. Its contents are git-ignored (only `.gitkeep` is tracked).
- `root/weights/dataset_cache/` is a git-ignored cache for the synthetic dataset builder.

---

## Tests

```bash
python -m pytest tests -v
```

`tests/test_pipeline_fixes.py` covers, among other things:

- cleaning and views: normalisation, sorting, duplicate timestamps, bad flux-error sanitising, rejection of unusable flux;
- detrending that preserves hour-scale transits;
- BLS: boundary-peak flagging, capped period grid, period recovery on long baselines, survival of unusable flux errors;
- vetting: indeterminate (not "EB") results when data are insufficient, distinct-transit counting, secondary-eclipse gating;
- status logic: white noise is **not** a detection, an injected transit **is**;
- pipeline: too-few-cadence rejection, entry points import without shadowing top-level names;
- checkpoint loader with non-tensor metadata; QLP quality mask opt-in; HLSP reader without uncertainties;
- report rendering uses the pipeline's status;
- an end-to-end check that the bundled real QLP light curve for TIC 172518755 recovers its known 3.2087 d period (skipped automatically if the FITS file is absent).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'root'`**
Run from the directory that *contains* `root/`, and use the module form for scripts (`python -m root.main_pipeline`). The dashboard adds the project root to `sys.path` itself, but starting it from the project root is still safest.

**`ModuleNotFoundError: No module named 'root.reporting.pdf_report'`**
In this version of the code, `main_pipeline.py`, `dashboard.py` and one test import `generate_vetting_report` from `root.reporting.pdf_report`, but the file in the project is named `root/reporting/vetting_report.py`. Either rename the file to `pdf_report.py`, or change those three imports to `from root.reporting.vetting_report import generate_vetting_report`. Until one of those is done, the dashboard and CLI will fail at import time.

**Red "UNTRAINED WEIGHTS" or amber "LEGACY WEIGHTS" banner**
The checkpoint is missing or predates the current contract. Confirm `root/weights/tri_branch_tess_net.pt` exists, or retrain with `python -m root.train_all` then `python -m root.train_real`.

**MAST search times out or `ConnectionError`**
`mast.stsci.edu` is blocked or slow (corporate firewall, VPN, proxy). Try another network. Use the synthetic dashboard modes to verify the install without internet.

**`No TESS data found for TIC ...`**
Run with `--list-sectors` (or use the dashboard's **List available sectors** button) and choose a sector that actually has data.

**`only N valid cadences after cleaning (need at least 100)`**
The light curve is too short or too heavily masked to search. Try a different sector, or `all_sectors=True` in `download_tess_lightcurve`.

**Status is `NO_SIGNIFICANT_DETECTION` for a known planet**
The BLS SNR was below 7. Try a different sector, stitch multiple sectors, or lower `--min-snr` (with the understanding that this raises the false-alarm rate).

**Status is `UNCONFIRMED_INSUFFICIENT_TRANSITS` with a long period**
A single ~27-day sector holds only one or two transits of a ~14-day planet. This is expected; use more sectors.

**Streamlit command not found**
Use `python -m streamlit run root/ux/dashboard.py` (as above) so the interpreter that has the dependencies is the one that launches it.

---

## Limitations and honest caveats

- **Modest classifier accuracy.** Accuracy 0.635 and ROC-AUC 0.73 on 200 unseen real TOIs. Treat the AI score as one weak signal among several.
- **Small training set.** Real-TESS fine-tuning used on the order of tens of targets per class, so generalisation is limited.
- **Calibration is approximate.** A single temperature scalar is fitted; the output should not be read as a true probability.
- **Labels are noisy.** CP/KP vs FP/FA from TOI dispositions is a practical proxy, not ground truth.
- **No centroid / pixel-level vetting.** Contamination by nearby eclipsing binaries cannot be excluded.
- **Search range:** periods of 0.5 to 15 days; single-transit or very long-period planets are out of scope.
- **Single-sector default.** One sector often cannot validate long periods.
- **Not a discovery tool.** Candidates need follow-up (additional sectors, centroid tests, spectroscopy) before any claim.

---

## Data sources and references

- **TESS light curves:** NASA MAST (SPOC PDCSAP; QLP and TARS high-level science products), accessed through `lightkurve` and `astroquery`.
- **Labels:** NASA Exoplanet Archive TOI table (TAP service).
- **Method background:** Box Least Squares (Kovács, Zucker & Mazeh 2002); AstroNet-style global/local phase-folded views for transit classification (Shallue & Vanderburg 2018); surrogate-gradient training of spiking networks (Neftci, Mostafa & Zenke 2019).

---

## License

<<<<<<< HEAD
=======
No license file is included with this distribution. Add one (for example MIT or Apache-2.0) before sharing or publishing the repository.
>>>>>>> 3c3a5c3 (vetting report generator modified)
