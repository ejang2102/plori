#!/usr/bin/env python3
"""MUSCLEMOTION (Sala et al. 2018, Circ Res) — faithful Python port of the AUTOMATIC pipeline.

PROVENANCE: github.com/l-sala/MUSCLEMOTION 'MUSCLEMOTION v1.0.ijm' (ImageJ macro,
BJ van Meer / L Sala / F Burton, LUMC 2017). This ports the macro's automatic path
verbatim where the algorithm is deterministic; the macro's manual/GUI-only steps are
substituted automatically and documented under IRREDUCIBLE below. Tags: ✓ = ported
faithfully from the .ijm (line refs in comments); ~ = automatic substitute for a
manual/GUI step. NOT the original macro — validate equivalence on a clip before
strong claims.

SIGNALS
  contraction (✓ getContractionData, ijm 677-709):
      contraction[i] = mean_xy |I_i - I_ref|, over ALL pixels (the macro multiplies a
      0/255 binary mask onto the 32-bit difference and averages getStatistics over the
      whole image). Absolute a.u. therefore differ from any "mean over mask pixels"
      formulation by a constant scale (fraction of pixels in the mask); ratios and all
      timing metrics are unaffected.
  speed (✓ getSpeedData, ijm 711-763):
      speed[i] = mean_xy |I_i - I_{i-speedWindow}|, speedWindow=2.
  SNR mask (✓ pixelsOfInterest, ijm 641-675):
      max projection of |I_i - I_ref| over the stack -> threshold (mean + 1*SD) ->
      binary mask, multiplied onto the difference stack (maxProject default = true via
      SNRimprovement "Yes, but keep it simple").
  Gaussian (✓ guassianBlur10, ijm 654/691/...): optional sigma=10 blur, default off.

REFERENCE FRAME (✓ getReferenceFrame autodetect, ijm 840-928)
  Macro default autodetectReferenceFrame = "Yes, but keep it simple". Builds
  speedY[i] = mean|I_i - I_{i-speedWindow}| over [autoDetectStart, autoDetectStop],
  forms speedYshift (speedY shifted by 1), radianPoints = sqrt(speedY^2 + speedYshift^2),
  takes the lowValueN(=20) lowest-radius candidates, then among those takes the
  unitySelectionN(=10) whose |speedY/speedYshift - 1| is closest to unity, and picks
  the argmin of speedY*speedYshift*unitySelection. numpy argsort replaces ImageJ
  Array.rankPositions. Defaults: speedWindow=2, autoDetectStart=1, autoDetectStop=300,
  lowValueN=20, unitySelectionN=10.

QUANTIFICATION on the CONTRACTION trace (✓ transientAnalysis, ijm 995-1264)
  peak detect (✓ ijm 1004-1041): perc100 = global max of the trace; perc0 = contraction
      AT the reference frame (yValues[referenceFrameSlice]); threshold = 0.30*(perc100-perc0);
      even window 20; a point is a peak if (y-perc0) > threshold and it is the strict local
      max within +/- window/2.
  baseline (✓ highFreqBaselineDetection=true, ijm 1082-1104): per peak, baseline = MIN of
      y over the PRE-peak window [peak - round((peak-prevPeak)/2), peak); for the first peak
      the window starts at 0.
  onset/offset (✓ ijm 1163-1234): level0 = baseline + 0.10*amplitude. Scan down/up from the
      peak; record the index where 3 consecutive points first fall below level0 (the
      OUTERMOST such index — the macro stores l, not backed up by N-1). time_to_peak =
      |peak-lowDown|, relaxation = |peak-lowUp|, duration = |lowUp-lowDown|, all *1000/fps.
  transient widths (✓ ijm 1238-1247): per percentage p, the TWO-SIDED width at the
      (100-p)% level = (percentageDataUp[p] - percentageDataDown[p]) * 1000/fps, where up/down
      use the same 3-consecutive-below search. Default percentages 10/50/90 (macro default
      binaryFormatPercentages "100010001"). Reported as keys "T<100-p>_width_ms" (i.e.
      90-to-90, 50-to-50, 10-to-10 transient widths) to match the macro's
      "<100-perc>-to-<100-perc> transient (ms)" outputs.

DERIVED EXTENSION (NOT a macro output)
  BPM = 60000 / median(peak-to-peak_ms). The macro reports per-beat "Peak-to-peak time (ms)"
  only; BPM is computed here for the shared batch interface. n_beats = number of detected peaks
  (matches the macro's peak count).

IRREDUCIBLE (macro manual/GUI; auto substitute used here)
  - Manual reference-frame selection (waitForUser, ijm 929-939): replaced by the autodetect
    above (the macro's own default path).
  - All GUI parameter dialogs (Dialog.*, ijm 89-268): replaced by the macro DEFAULT values,
    exposed as module constants / CLI where useful.
  - Visual QC plots (clipping warning ijm 808-815; measured-vs-calculated speed linearity
    speedLinCompare ijm 955-993): not reproduced — informational only, no metric output.

NOTE on --scale / --max-frames: the faithful macro runs at full resolution over the full
recording. These CLI args (downscale / truncate) are PERFORMANCE shortcuts only; a faithful
run uses --scale 1.0 and no --max-frames. They are NOT part of the method.
"""
import argparse
import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
from plori.video import read_clip, video_fps

# --- macro DEFAULT values (ijm 20-36); GUI dialogs replaced by these constants ---
SPEED_WINDOW = 2            # ijm default_speedWindow
AUTODETECT_START = 1        # ijm default_autoDetectStart (1-based frame)
AUTODETECT_STOP = 300       # ijm default_autoDetectStop
LOW_VALUE_N = 20            # ijm default_lowValueN
UNITY_SELECTION_N = 10      # ijm default_unitySelectionN
PEAK_WIN = 20              # ijm default_PeakDetectionWindow
PEAK_THRESH = 0.30          # ijm default_peakThreshold (30%)
ONSET_FRAC = 0.10          # ijm transientAnalysis: 10% level (percentages[0]/100)
NCONSEC = 3                # ijm: 3 consecutive points below to exclude noise
PERCENTAGES = (10, 50, 90)  # ijm default binaryFormatPercentages "100010001"
HIGHFREQ_BASELINE = True    # ijm default_highFreqBaselineDetection "Yes"


def _gray(frames):
    if frames.ndim == 3:
        return frames.astype(np.float64)
    return frames.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def select_reference(gray, speed_window=SPEED_WINDOW,
                     auto_start=AUTODETECT_START, auto_stop=AUTODETECT_STOP,
                     low_value_n=LOW_VALUE_N, unity_selection_n=UNITY_SELECTION_N):
    """✓ getReferenceFrame autodetect (ijm 840-928).

    Returns a 0-based frame index. ImageJ Array.rankPositions -> numpy argsort.
    The macro's loop bounds (d<lowValueN-1, d<unitySelectionN-1) are ported verbatim,
    including their off-by-one, so candidate counts match the macro exactly.
    """
    T = len(gray)
    flat = gray.reshape(T, -1)

    # speedY[i] = mean|I_i - I_{i-speedWindow}| (ijm 857-878, indices over the stack)
    # The macro slices speedY to [autoDetectStart, autoDetectStop] (Array.slice, 0-based
    # half-open), then forms speedYshift = speedY[1:] and trims speedY to drop the last
    # element so both align (ijm 886-889).
    speed_full = np.zeros(T)
    speed_full[speed_window:] = np.abs(flat[speed_window:] - flat[:-speed_window]).mean(1)
    # macro computes speedY over autoDetectStop-autoDetectStart+1 frames starting at frame 1
    # then Array.slice(speedY, autoDetectStart, autoDetectStop). Net effect: a contiguous
    # window of speed values. Clamp to available range (ijm 459-477 sanity checks).
    stop = min(auto_stop, T - speed_window - 1)
    start = min(auto_start, max(stop - 1, 0))
    seg = speed_full[start:stop + 1]
    if len(seg) < 3:
        # too short to run the heuristic; fall back to lowest-speed frame in range
        return int(np.argmin(speed_full[speed_window:]) + speed_window)

    speedY = seg[:-1]                      # ijm Array.trim(speedY, len-1)
    speedYshift = seg[1:]                  # ijm Array.slice(speedYshift, 1)
    n = len(speedY)

    radian = np.sqrt(speedY ** 2 + speedYshift ** 2)   # ijm 891-894
    indicesVal = np.argsort(radian, kind="stable")     # ijm Array.rankPositions(radianPoints)

    lvn = min(low_value_n, n)
    usn = min(unity_selection_n, lvn)

    # unitySelection over the lowValueN lowest-radius candidates (ijm 898-902; loop d<lowValueN-1)
    unitySelection = np.zeros(lvn)
    with np.errstate(divide="ignore", invalid="ignore"):
        for d in range(lvn - 1):
            idx = indicesVal[d]
            unitySelection[d] = abs((speedY[idx] / speedYshift[idx]) - 1.0) if speedYshift[idx] != 0 else np.inf

    indicesUni = np.argsort(unitySelection, kind="stable")  # ijm Array.rankPositions(unitySelection)

    low = None
    low_index = int(indicesVal[0])
    for d in range(usn - 1):                # ijm 906-918; loop d<unitySelectionN-1
        index_trans = indicesUni[d]
        index = indicesVal[index_trans]
        low_value = (speedY[index] * speedYshift[index]) * unitySelection[index_trans]
        if low is None or low_value < low:
            low = low_value
            low_index = int(index)

    # ijm: referenceFrameSlice = lowIndex+1 (1-based slice into the autodetect window,
    # which began at frame `start`). Map back to a 0-based frame index in the full stack.
    return int(start + low_index + 1)


def snr_mask(gray, ref_idx):
    """✓ pixelsOfInterest (ijm 641-675): max projection of |I-I_ref| -> threshold mean+SD."""
    diff = np.abs(gray - gray[ref_idx][None])
    mp = diff.max(axis=0)
    thr = mp.mean() + mp.std()   # ijm: lucaVar = mean + stdDev (ijm 671)
    return mp > thr


def signals(frames, gaussian=False):
    """Return (contraction, speed, ref). Macro averages getStatistics over the WHOLE image
    after multiplying the 0/255 mask onto the difference; emulated by zeroing non-mask pixels
    and averaging over all pixels (so absolute a.u. carry the mask-fraction scale)."""
    g = _gray(frames)
    if gaussian:
        g = np.stack([cv2.GaussianBlur(f, (0, 0), 10) for f in g])
    ref = select_reference(g)
    mask = snr_mask(g, ref).reshape(-1)
    T = len(g)
    gf = g.reshape(T, -1)
    npix = gf.shape[1]

    # contraction = mean over ALL pixels of (|I_i - I_ref| * mask), ijm getContractionData
    diff_ref = np.abs(gf - gf[ref][None])
    contraction = (diff_ref * mask[None]).sum(1) / npix

    # speed = mean over ALL pixels of (|I_i - I_{i-speedWindow}| * mask)
    speed = np.zeros(T)
    diff_sp = np.abs(gf[SPEED_WINDOW:] - gf[:-SPEED_WINDOW])
    speed[SPEED_WINDOW:] = (diff_sp * mask[None]).sum(1) / npix
    return contraction, speed, ref


def detect_peaks(y, ref):
    """✓ transientAnalysis peak detection (ijm 1004-1041).

    perc100 = global max (y.max()); perc0 = y[ref] (contraction at the reference frame);
    threshold = 0.30*(perc100-perc0); even window; strict local max within +/- window/2.
    """
    perc100 = float(y.max())                                  # ijm 1004 (rank-max value)
    perc0 = float(y[ref]) if 0 <= ref < len(y) else float(y.min())  # ijm 1005
    thr = PEAK_THRESH * (perc100 - perc0)
    win = PEAK_WIN + (PEAK_WIN % 2)                           # ijm 1009-1012: force even
    half = win // 2
    peaks = []
    for u in range(half, len(y) - 1 - half):                 # ijm 1016
        if (y[u] - perc0) <= thr:                            # ijm 1017
            continue
        is_max = True
        for r in range(1, half):                            # ijm 1018-1021
            if y[u - r] > y[u] or y[u + r] > y[u]:
                is_max = False
                break
        if is_max:
            peaks.append(u)
    return np.array(peaks, int)


def quantify(contraction, fps):
    """✓ Native MUSCLEMOTION transientAnalysis on the contraction trace (ijm 995-1264)."""
    y = np.asarray(contraction, float)
    T = len(y)
    # NOTE: detect_peaks needs the reference frame; recompute the same metric the macro uses
    # (perc0 at ref). The reference index is carried via the trace's argmax-baseline only in
    # signals(); here we infer perc0 from the global min as a robust stand-in IF ref unknown,
    # but the shared interface passes only (contraction, fps). The macro uses y[ref]; to stay
    # faithful while honoring the interface, treat the trace minimum frame as the rest level
    # ONLY if no ref is available. Since callers pass the contraction computed against the
    # detected ref (where contraction ~ 0), y.argmin() coincides with the ref frame.
    ref = int(np.argmin(y))
    peaks = detect_peaks(y, ref)
    if len(peaks) == 0:
        return [], {"n_beats": 0, "BPM": float("nan"), "time_to_peak_ms_mean": float("nan")}

    # baseline per peak: MIN over PRE-peak window [peak - round((peak-prevPeak)/2), peak)
    # (ijm 1082-1104, highFreqBaselineDetection=true)
    baselines = np.zeros(len(peaks))
    for c, pk in enumerate(peaks):
        if c == 0:
            start_range = 0
        else:
            start_range = pk - int(round((pk - peaks[c - 1]) / 2.0))
        start_range = max(start_range, 0)
        seg = y[start_range:pk] if pk > start_range else y[pk:pk + 1]
        baselines[c] = float(seg.min())

    def find_below(pk, border, direction, level):
        """3-consecutive-below scan; return the OUTERMOST index l where the run completes
        (ijm 1191-1209). direction -1 scans down (l--), +1 scans up (l++)."""
        if direction < 0:
            l = pk
            while l > border:
                if l - 2 >= 0 and y[l] < level and y[l - 1] < level and y[l - 2] < level:
                    return l
                l -= 1
        else:
            l = pk
            while l < border:
                if l + 2 < len(y) and y[l] < level and y[l + 1] < level and y[l + 2] < level:
                    return l
                l += 1
        return None

    beats = []
    for c, pk in enumerate(peaks):
        # peak-to-peak distance defines the search window (ijm 1165-1166, 1181-1188)
        if c < len(peaks) - 1:
            p2p = peaks[c + 1] - peaks[c]
        else:
            p2p = peaks[c] - peaks[c - 1] if c > 0 else PEAK_WIN
        min_border = max(pk - abs(p2p), 2)
        max_border = min(pk + abs(p2p), T - 3)

        perc100 = float(y[pk])                  # ijm 1170
        perc0 = float(baselines[c])             # ijm 1171
        amp = perc100 - perc0

        # percentage levels (ijm 1172-1174)
        levels = {p: (p / 100.0) * (perc100 - perc0) + perc0 for p in PERCENTAGES}
        level0 = (PERCENTAGES[0] / 100.0) * (perc100 - perc0) + perc0  # m==0 level (10%)

        # onset/offset at the 10% level (ijm 1191-1209, m==0 -> lowDown/lowUp)
        low_down = find_below(pk, min_border, -1, level0)
        low_up = find_below(pk, max_border, +1, level0)

        on = low_down if low_down is not None else int(min_border)
        off = low_up if low_up is not None else int(max_border)

        t2pk = abs(pk - on) / fps * 1000.0 if low_down is not None else np.nan
        relax = abs(off - pk) / fps * 1000.0 if low_up is not None else np.nan
        dur = abs(off - on) / fps * 1000.0 if (low_down is not None and low_up is not None) else np.nan

        rec = {"peak_idx": int(pk), "onset_idx": int(on), "offset_idx": int(off),
               "baseline": perc0, "amplitude": amp,
               "time_to_peak_ms": t2pk, "relaxation_ms": relax, "duration_ms": dur}

        # two-sided transient WIDTH at each (100-p)% level (ijm 1238-1247)
        for p in PERCENTAGES:
            down = find_below(pk, min_border, -1, levels[p])
            up = find_below(pk, max_border, +1, levels[p])
            if down is not None and up is not None:
                rec["T%d_width_ms" % (100 - p)] = abs(up - down) / fps * 1000.0
            else:
                rec["T%d_width_ms" % (100 - p)] = np.nan
        beats.append(rec)

    # Peak-to-peak time (ms) is the macro output (ijm 1257); BPM is a derived extension.
    p2p_ms = np.diff(peaks) / fps * 1000.0
    bpm = float(60000.0 / np.median(p2p_ms)) if len(p2p_ms) else np.nan

    keys = ["amplitude", "time_to_peak_ms", "relaxation_ms", "duration_ms"] + \
           ["T%d_width_ms" % (100 - p) for p in PERCENTAGES]
    summary = {"n_beats": int(len(peaks)), "BPM": bpm}
    for k in keys:
        v = np.array([b[k] for b in beats], float)
        v = v[~np.isnan(v)]
        summary[k + "_mean"] = float(v.mean()) if len(v) else np.nan
        summary[k + "_std"] = float(v.std()) if len(v) else np.nan
    return beats, summary


def main():
    ap = argparse.ArgumentParser(
        description="MUSCLEMOTION reimpl. Faithful runs use --scale 1.0 and full length; "
                    "--scale/--max-frames are performance shortcuts, NOT part of the method.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--scale", type=float, default=0.4,
                    help="downscale factor (PERFORMANCE only; faithful=1.0)")
    ap.add_argument("--max-frames", type=int, default=150,
                    help="truncate frame count (PERFORMANCE only; faithful=full length)")
    ap.add_argument("--gaussian", action="store_true", help="sigma=10 Gaussian blur (ijm guassianBlur10)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    frames = read_clip(args.video, scale=args.scale, max_frames=args.max_frames)
    fps = video_fps(args.video)
    contraction, speed, ref = signals(frames, gaussian=args.gaussian)
    beats, summary = quantify(contraction, fps)

    cache = os.path.join(args.output, "cache")
    outd = os.path.join(args.output, "output")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(outd, exist_ok=True)
    np.savez(os.path.join(cache, f"{args.name}_musclemotion.npz"),
             contraction=contraction, speed=speed, ref=ref, fps=fps)
    with open(os.path.join(cache, f"{args.name}_musclemotion_params.json"), "w") as f:
        json.dump({"sample": args.name, "method": "musclemotion", "ref_frame": ref, **summary}, f, indent=1)

    t = np.arange(len(contraction)) / fps
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(t, contraction, color="C1", lw=1.2, label="contraction (|I-I_ref|)")
    for b in beats:
        ax.axvline(b["onset_idx"] / fps, color="g", lw=0.7, alpha=0.5)
        ax.axvline(b["offset_idx"] / fps, color="r", lw=0.7, alpha=0.5)
        ax.plot(b["peak_idx"] / fps, contraction[b["peak_idx"]], "v", color="C3", ms=6)
    ax.set_title("%s — MUSCLEMOTION [reimpl]  %d beats, BPM=%.0f, t2pk=%.0f±%.0f ms"
                 % (args.name, summary["n_beats"], summary.get("BPM", float("nan")),
                    summary.get("time_to_peak_ms_mean", float("nan")),
                    summary.get("time_to_peak_ms_std", float("nan"))), fontsize=9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("mean |ΔI| (a.u.)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(outd, f"{args.name}_musclemotion.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out, "|", json.dumps({k: summary[k] for k in ("n_beats", "BPM") if k in summary}))


if __name__ == "__main__":
    main()
