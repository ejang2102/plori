# PLoRI — Per-pixel Local Rest-referenced Intensity

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22095623.svg)](https://doi.org/10.5281/zenodo.22095623)

PLoRI extracts a cardiac **contraction waveform** and **per-beat metrics** directly from a brightfield video of a beating cardiac organoid. It needs **no fluorescent label and no specialized imaging equipment** — only the raw brightfield video and a handful of standard scientific-Python packages.

This repository accompanies the PLoRI paper and reproduces the method and its baseline comparisons on any input video, including a synthetic clip generated here so the full pipeline can be run without proprietary data.

---

## How it works

For each pixel inside an automatically segmented organoid mask, PLoRI measures how far the pixel's brightness departs from its own **time-local rolling reference**:

```
PLoRI(t) = ⟨ | I(x,t) − median_t( I(x,·), k ) | ⟩_mask ,   k ≈ one beat period
```

- **Per-pixel** and **local rest reference**: each pixel is compared against a moving median of its own recent history, so slow drift and uneven illumination cancel out. This is the key difference from a single fixed reference frame, which saturates and drifts over a multi-beat clip.
- **Rectify then average**: the absolute deviation is taken *before* spatial averaging, so contraction and relaxation do not cancel and there is no mean-brightness polarity ambiguity.
- The beat period `k` is estimated by autocorrelation (the lag of the first autocorrelation peak), in two passes: a frame-difference pass seeds the window, then the PLoRI signal itself refines it.

From the waveform PLoRI derives per-beat metrics: rate (BPM), CD50 (contraction duration at the 50 % level), contraction time and relaxation time (onset→peak and peak→offset at the 10 % level), amplitude, and beat-to-beat variability, together with organoid size, opacity, and drift. See `plori/core.py` for the exact definitions.

> PLoRI values are intensity units (a.u.), not physical displacement. The waveform is a proxy for contraction magnitude; quantitative interpretation is discussed in the paper.

---

## Install

```bash
pip install -r requirements.txt        # numpy, scipy, opencv-python, matplotlib, openpyxl
# optional, for `import plori` from anywhere:
pip install -e .
```

Tested with Python 3.11 (numpy 2.1, scipy 1.17, opencv 4.13).

---

## Quickstart (no data required)

```bash
# 1. synthesize a beating-organoid demo clip
python scripts/make_synthetic.py --out examples/synthetic_beating.mp4

# 2. run PLoRI on it -> a *_ploridata.npz result bundle
python scripts/run_plori.py --video examples/synthetic_beating.mp4 \
    --name demo --cat sample --out-dir examples/out \
    --mpp 1.0 --mag synthetic --scale 1.0

# 3. render a report-card PNG (waveform, frame difference, metrics, QC flags)
python scripts/plori_report.py --glob 'examples/out/**/*_ploridata.npz' \
    --out-dir examples/out

# 4. export a shareable spreadsheet (time series + per-beat table)
python scripts/plori_to_xlsx.py --glob 'examples/out/**/*_ploridata.npz' \
    --out examples/out/PLoRI_demo.xlsx
```

The demo clip beats at 35 BPM; step 2 should report `beats=11 BPM=35`.

### On your own videos

Use the single-clip runner for one file, or the batch runner for a folder:

```bash
python scripts/plori_batch.py --glob '/path/to/videos/*.mp4' \
    --out-dir results/cache --cat mygroup --mpp 0.6 --mag 4X --scale 0.25
```

**Mechanics (`--with-flow`, on by default)** — the runners also compute masked dense optical flow (Farneback, averaged inside the organoid mask) and, over each PLoRI onset/offset beat window, per-beat **max speed** (µm/s), **displacement** (µm, `0.5 ×` the integral of speed), and **strain** (displacement ÷ organoid size). These are written to the npz and spreadsheet. Pass `--no-with-flow` to skip it (faster; the mechanics columns are then blank). This is plain masked optical flow and is independent of the whole-frame ContractionWave baseline; the µm units require a real `--mpp`.

**About `--scale`** — a decode-time downscale factor (speed only). Large full-resolution videos decode and process faster at a smaller scale; the physical outputs (µm/mm) are unaffected because `mpp` is compensated internally by `1/scale`. Defaults: `1.0` for the single-clip runner (`run_plori.py`), `0.25` for the batch runner (`plori_batch.py`). Lower it (e.g. `0.2`–`0.5`) for large recordings; keep `1.0` for already-small clips such as the synthetic demo.

**About `--mpp`** — microns per full-resolution pixel, your microscope's spatial calibration. It only rescales the physical-size outputs (organoid diameter in µm, area in mm², drift in µm); it has **no effect** on any temporal or waveform result (BPM, CD50, CT, RT, and the PLoRI signal itself). So if you only need beat timing, pass `--mpp 1.0` (the `run_plori.py` default) and ignore the size columns. To get real sizes, read a scale bar on a full-resolution frame and use `mpp = known_length_µm / length_in_pixels` (e.g. a 100 µm bar spanning 200 px gives `mpp = 0.5`). Pass the full-resolution value even when downscaling: `--scale` only speeds up decoding, and the script multiplies mpp by `1/scale` internally.

---

## Repository layout

```
plori/                     the package
  core.py                  the PLoRI method (segment / signal / beat metrics / analyze)
  video.py                 video IO helper (read_clip / video_fps)
  baselines/               faithful ports of the compared methods
    musclemotion.py          fixed single-reference frame difference (MuscleMotion)
    contractionwave.py       dense optical-flow magnitude (ContractionWave)
scripts/
  make_synthetic.py        generate the demo video
  run_plori.py             run PLoRI on one video
  plori_batch.py           run PLoRI over a folder of videos
  plori_report.py          render a report-card PNG from a result bundle
  plori_to_xlsx.py         export result bundles to a spreadsheet
examples/                  demo outputs land here (git-ignored)
```

Each `*_ploridata.npz` bundle holds the masks, the PLoRI waveform, the per-beat metric arrays, and organoid size/opacity/drift; the reporting and export scripts read only these bundles (no video needed to re-render).

---

## Baselines

`plori.baselines.musclemotion` and `plori.baselines.contractionwave` are re-implementations of the two published methods PLoRI is compared against in the paper (a fixed single-reference frame difference, and a dense optical-flow magnitude). They are included so the comparison figures can be reproduced from the same input videos.

---

## Data availability

The organoid videos used in the paper are not redistributed here. The synthetic generator (`scripts/make_synthetic.py`) provides a self-contained example so the whole pipeline can be exercised without them; to reproduce the paper's numbers, point the runners at the corresponding source videos.

---

## License

GPL-3.0-or-later (see `LICENSE`). PLoRI extends **MuscleMotion** (Sala et al., 2018), which is distributed under GPL-3.0; this project inherits that license. The **ContractionWave** baseline likewise re-implements a published method. Please cite the original tools alongside this work where appropriate.

## Citation

See `CITATION.cff`. Please cite the accompanying PLoRI paper (details to be filled in on publication).
