#!/usr/bin/env python3
"""CONTRACTIONWAVE (Scalzo et al. 2021, Cell Rep Methods) — faithful Python reproduction
of the ORIGINAL automatic pipeline.

PROVENANCE: github.com/marceloqla/ContractionWavePy (authoritative Python source).
✓ faithful to source (file:line) / ~ approximation / [ext] derived extension not in source.

This reproduces the DEFAULT AUTOMATIC path: the GUI, with all controls at their defaults,
calls noise_detection(..., cutoff_val=0.90) then peak_detection(...) on the whole-frame
mean-magnitude scalar (current_case = mag_means). ContractionWave.py ~3842-3844.

------------------------------------------------------------------------------------------
SIGNAL (✓):  dense optical flow, whole-frame mean magnitude, per-frame scalar reduction.
  ContractionWave.py ~837-983 Farneback + scalar reduction; defaults ~1260-1271.
    flow    = cv2.calcOpticalFlowFarneback(prvs, prvs2, None, 0.5, 1, 15, 1, 7, 1.5, 0)
    mag,_   = cv2.cartToPolar(flow[...,0], flow[...,1])
    meanval = abs(mag.mean() * fps * pixel)        # µm/s
  Default segmentationtype=2 (whole frame, NO mask); smoothbeforeregression="never"
  (no pre-smoothing). sig[0]=0 (first frame has no predecessor).

NOISE CLASSIFICATION (✓):  smoothregress.py noise_detection / class_definition /
  probable_signal_from_classes ~46-101, 103-177. cutoff_val=0.90 (GUI default,
  magnitude-units threshold). Points below cutoff -> class 1 (noise); >= cutoff ->
  class 2 (signal). Yields mean_noise, noise_areas (consecutive noise runs),
  filtered_maxfilter_areas (noise areas filtered by mean size, keeping first+last),
  used as the per-beat window bounds.

PEAKDET (✓):  peakdetectpure.py peakdet ~10-91, copied EXACTLY incl. the
  `this < mn and this > 0` minima guard and the extra `if this > mxpos` minima-emission
  block. delta (✓) = mean(non_noise_points_values)/3   (smoothregress.py ~501-503).

BEAT SEGMENTATION & MCS/MRS PAIRING (✓):  smoothregress.py peak_detection ~479-796.
  Per beat (window bounded by the next filtered noise area's median point):
    - two highest maxima in window -> ordered in time: earlier = MCS (fmax, contraction
      speed peak), later = MRS (smax, relaxation speed peak).        (~770-779)
    - trough (min) = strictly between them, the lowest-value point.   (~780-783)
    - start (first) = last index before MCS whose value < mean_noise. (~553, 759-761)
    - end (last) = exponential-area criterion below.                  (~580-765)
  Landmark mapping first/fmax/min/smax/last from draghandlers.py ~386-397.

END POINT, exponential criterion (✓):  smoothregress.py ~580-721; exponential_fit ~9-10.
    exponential_fit(x,a,b,c) = exp(a)*exp(b*x)+c
    curve_fit(p0=(1e-6,1e-6,1), maxfev=150000) over the post-MRS noise-area window;
    total_area = trapz(fit over window); walk forward accumulating fit values until
    cumulative >= total_area*stop_condition_perc (default 0.35) -> that index = end.

DECAY TIMES T10..T90 (✓):  ContractionWave.py genexportdata decay ~6669-6708, on
  RELAXATION SPEED.  For decay in 0.1..0.9: decayed = MRS_speed*(1-decay); within
  fulldata[smax:last+1] take the index whose value is closest to decayed; T = (idx-smax)
  in ms. (We report T10/T50/T90 plus the full set in each beat dict.)

BPM ([ext] — NOT in source):  ContractionWave has no rate/BPM metric. Kept here only for
  cross-tool comparison, clearly labelled as a derived extension:
    BPM = 60 / median(inter-MCS interval in s), using the faithful MCS anchors;
    n_beats = number of detected beats.

------------------------------------------------------------------------------------------
IRREDUCIBLE (source manual / GUI-only) — auto-substituted here, documented:
  * Interactive interval cropping (selectedframes): we use the full clip (max_frames).
  * Manual landmark dragging / add-remove dots (added_noise_dots/removed_noise_dots):
    none applied (both lists empty, as in the default automatic call).
  * Non-default segmentation modes (0 = magnitude-threshold mask, 1 = angle clustering):
    not implemented; we use default mode 2 (whole frame).
  * Optional denoising (FFT / Savitzky-Golay / convolution smoothing): off by default
    (smoothbeforeregression="never"); not implemented.
  * Alternative end modes (peak_detection_threshold, 75%-decay end): not implemented;
    we use the default exponential-area end (end_current_type="exponential").
NOT the original GUI — validate on a clip before strong claims.
"""
import argparse
import os
import sys
import json
import numpy as np
import cv2
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
from plori.video import read_clip, video_fps


# ============================================================ SIGNAL =========
def signal_farneback(frames, fps, px2um):
    """✓ ContractionWave.py ~837-983: dense flow, whole-frame mean magnitude * fps * px.

    Returns a 1-D array of length n_frames with sig[0]=0.
    """
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f.astype(np.uint8) for f in frames]
    T = len(g)
    sig = np.zeros(T)
    for i in range(1, T):
        flow = cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None, 0.5, 1, 15, 1, 7, 1.5, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        sig[i] = abs(mag.mean() * fps * px2um)
    return sig


# ============================================================ PEAKDET ========
def peakdet(v, delta, x=None):
    """✓ EXACT copy of peakdetectpure.py peakdet ~10-91 (incl. `this>0` minima guard
    and the extra `if this > mxpos` minima-emission block). Returns (maxind, minind)."""
    maxind = []
    minind = []
    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan
    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn and this > 0:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxind.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                minind.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True
            if this > mxpos:
                minind.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True

    return [int(a) for a in maxind], [int(a) for a in minind]


def generate_bfderivative_full(data):
    """✓ lineardetectpack.py ~80-93: before-derivative (data[i]-data[i-1]), first = 0."""
    before = [0]
    for i in range(1, len(data)):
        before.append(data[i] - data[i - 1])
    return before


def exponential_fit(x, a, b, c):
    """✓ smoothregress.py ~9-10."""
    return np.exp(a) * np.exp(b * x) + c


# ===================================================== NOISE CLASSIFICATION ==
def noise_definition(data):
    """✓ smoothregress.py ~46-56, fallback branch (no linear-regression module here):
    mean/std/max of the lowest 25% of values. Used for mean_noise baseline."""
    sorted_data = sorted(data)
    sorted_data_25perc = sorted_data[: int(len(sorted_data) / 4)]
    if len(sorted_data_25perc) == 0:
        sorted_data_25perc = sorted_data
    return np.mean(sorted_data_25perc), np.std(sorted_data_25perc), np.max(sorted_data_25perc)


def class_definition(current_case, cutoff_val):
    """✓ smoothregress.py ~94-101: class 1 below cutoff (noise), class 2 at/above (signal)."""
    return [1 if e < cutoff_val else 2 for e in current_case]


def probable_signal_from_classes(current_case, case_classes, filter_noise_area=True,
                                 added_noise_dots=[], removed_noise_dots=[]):
    """✓ smoothregress.py ~103-177. Partition signal vs noise, compute mean_noise,
    noise_areas (consecutive noise runs) and filtered_maxfilter_areas."""
    non_noise_points = [(e, i) for i, e in enumerate(current_case) if case_classes[i] != 1]
    non_noise_points_values = [e[0] for e in non_noise_points]
    non_noise_points_indexes = [e[1] for e in non_noise_points]

    if len(added_noise_dots) > 0:
        noise_points = [(e, i) for i, e in enumerate(current_case)
                        if case_classes[i] == 1 or int(i) in added_noise_dots]
    else:
        noise_points = [(e, i) for i, e in enumerate(current_case) if case_classes[i] == 1]
    if len(noise_points) < 2:
        return None
    if len(removed_noise_dots) > 0:
        noise_points = [a for a in noise_points if int(a[1]) not in removed_noise_dots]

    noise_points_values = [e[0] for e in noise_points]
    noise_points_indexes = [e[1] for e in noise_points]

    mean_noise = np.mean(noise_points_values)
    std_noise = np.std(noise_points_values)
    max_noise = np.max(noise_points_values)

    peak_freq = len(non_noise_points) / len(current_case)
    noise_freq = len(noise_points) / len(current_case)
    peak_to_noise_ratio = len(non_noise_points) / len(noise_points)

    noise_area = False
    noise_areas = []
    for i, e in enumerate(current_case):
        if i in noise_points_indexes and noise_area is True:
            noise_areas[-1].append(i)
        elif i in noise_points_indexes and noise_area is False:
            noise_areas.append([])
            noise_areas[-1].append(i)
            noise_area = True
        else:
            noise_area = False

    mean_noise_area_size = np.mean([len(a) for a in noise_areas])

    if filter_noise_area is True:
        filtered_maxfilter_areas = [noise_areas[0]]
        filtered_maxfilter_areas.extend([a for a in noise_areas[1:-1] if len(a) > mean_noise_area_size])
        filtered_maxfilter_areas.append(noise_areas[-1])
    else:
        filtered_maxfilter_areas = noise_areas.copy()

    filtered_maxfilter_indexes = []
    for a in filtered_maxfilter_areas:
        filtered_maxfilter_indexes.extend(a)
    filtered_maxfilter_values = [current_case[i] for i in filtered_maxfilter_indexes]
    max_filtered_noise = np.max(filtered_maxfilter_values)

    return (non_noise_points, non_noise_points_values, non_noise_points_indexes,
            noise_points, noise_points_values, noise_points_indexes,
            mean_noise, std_noise, max_noise, peak_freq, noise_freq, peak_to_noise_ratio,
            noise_areas, mean_noise_area_size, filtered_maxfilter_areas,
            filtered_maxfilter_indexes, filtered_maxfilter_values, max_filtered_noise)


def noise_detection(current_case, filter_noise_area=True, added_noise_dots=[],
                    removed_noise_dots=[], cutoff_val=None):
    """✓ smoothregress.py ~65-92. Default automatic call uses cutoff_val=0.90."""
    if cutoff_val is None:
        n = int(len(current_case) * 0.3)
        vi = np.argsort(current_case)[-n:]
        cutoff_val = np.mean([current_case[a] for a in vi])
        cutoff_val = float("{:.3f}".format(cutoff_val))
    case_classes = class_definition(current_case, cutoff_val)
    results = probable_signal_from_classes(current_case, case_classes,
                                           filter_noise_area=filter_noise_area,
                                           added_noise_dots=added_noise_dots,
                                           removed_noise_dots=removed_noise_dots)
    if results is not None:
        return results + (cutoff_val,)
    return None


# ============================================================ PEAK DETECTION =
def peak_detection(current_case, delta=False, stop_condition_perc=False, nargs=None):
    """✓ smoothregress.py peak_detection ~479-796 (default automatic path).

    nargs is the tuple returned by noise_detection(...). Returns
    (f_points, s_f_points, t_points, l_points), where s_f_points holds
    [MCS, MRS, MCS, MRS, ...] pairs (fmax/smax per beat)."""
    (non_noise_points, non_noise_points_values, non_noise_points_indexes,
     noise_points, noise_points_values, noise_points_indexes,
     mean_noise, std_noise, max_noise, peak_freq, noise_freq, peak_to_noise_ratio,
     noise_areas, mean_noise_area_size, filtered_maxfilter_areas,
     filtered_maxfilter_indexes, filtered_maxfilter_values,
     max_filtered_noise, cutoff_val) = nargs

    # defaults (expconfigs absent in the automatic call) ~493-499
    endnoisecriteria = 0.9
    smoothbeforeregression = "never"
    local_minimum_check = False

    if delta is False:
        delta = np.mean(non_noise_points_values) / 3
        delta = float("{:.3f}".format(delta))

    maxtab, mintab = peakdet(current_case, delta)
    maxtab = [i for i in maxtab if i not in noise_points_indexes]

    above_before_derivatives = generate_bfderivative_full(current_case)
    all_local_maximums = []
    all_local_minimums = []
    for i in range(len(above_before_derivatives) - 1):
        val = above_before_derivatives[i]
        val2 = above_before_derivatives[i + 1]
        if val > 0.0 and val2 < 0.0:
            all_local_maximums.append(i)
        elif val < 0.0 and val2 > 0.0:
            all_local_minimums.append(i)
    if current_case[0] < np.max(noise_points_values) and current_case[1] > current_case[0]:
        all_local_minimums.insert(0, 0)
    if current_case[-1] < np.max(noise_points_values) and current_case[-2] > current_case[-1]:
        all_local_minimums.append(len(current_case) - 1)

    maxtab = sorted(list(set(maxtab) & set(all_local_maximums)))

    f_points, s_f_points, t_points, l_points = [], [], [], []
    exponential_pops = []

    i = 0
    while (True and len(maxtab) > 0):
        max_1_i = maxtab[i]
        previous_mins = sorted(list(set([j for j, e in enumerate(current_case[:max_1_i]) if e < mean_noise])))

        try:
            max_2_i = maxtab[i + 1]
        except IndexError:
            break

        try:
            filtered_maxfilter_area_above = [a for a in filtered_maxfilter_areas if a[0] > max_2_i][0]
            filtered_maxfilter_area_above_start = filtered_maxfilter_area_above[0]
            filtered_maxfilter_area_above_middle_p = int(np.median(filtered_maxfilter_area_above))
            filtered_maxfilter_area_endpoint = int(np.quantile(filtered_maxfilter_area_above, endnoisecriteria))
            range_maxfilter = range(filtered_maxfilter_area_above_start, filtered_maxfilter_area_endpoint)
        except IndexError:
            break

        after_mins = current_case[filtered_maxfilter_area_above_start:filtered_maxfilter_area_endpoint]

        auto_mode = False
        if stop_condition_perc is False:
            auto_mode = True
            stop_condition_perc = 0.35

        valuesfit = list(after_mins)
        # smoothbeforeregression == "never": no smoothing (default)

        after_point = None
        if len(valuesfit) == 0:
            i += 1
            continue
        elif len(valuesfit) < 2:
            after_point = list(range_maxfilter)[valuesfit.index(np.min(valuesfit))]
        elif len(valuesfit) > 2:
            valuesfitx = np.array(range_maxfilter)
            valuesfitx_highdef = np.linspace(max_2_i, filtered_maxfilter_area_endpoint, 100)
            try:
                popt, _ = curve_fit(exponential_fit, valuesfitx, valuesfit,
                                    p0=(1e-6, 1e-6, 1), maxfev=150000)
            except Exception:
                i += 1
                continue
            exponential_pops.append((valuesfitx_highdef, popt))

            total_area = np.trapz([exponential_fit(point, *popt) for point in range_maxfilter])
            current_area = 0.0
            stop_percentual = total_area * stop_condition_perc
            for point in range_maxfilter:
                after_point = point
                exponential_speed_value = exponential_fit(point, *popt)
                current_area += exponential_speed_value
                is_local_minimum = point in all_local_minimums
                if current_area >= stop_percentual and local_minimum_check is False:
                    break
                elif current_area >= stop_percentual and local_minimum_check is True and is_local_minimum is True:
                    break
        if after_point is None:
            i += 1
            continue

        if len(previous_mins) >= 1:
            f_point = previous_mins[-1]
            f_points.append(f_point)
        else:
            i += 1
            continue
        l_points.append(after_point)

        try:
            in_between = range(f_points[-1], l_points[-1] + 1)
            in_between_maximums = sorted(list(set(in_between) & set(maxtab)))
            in_between_maximums_vals_sort = sorted(
                [(j, current_case[j]) for j in in_between_maximums], key=lambda x: x[1], reverse=True)
            # two highest maxima
            second_point = in_between_maximums_vals_sort[0][0]
            s_f_points.append(second_point)
            fourth_point = in_between_maximums_vals_sort[1][0]
            s_f_points.append(fourth_point)
            # order in time: earlier = MCS (fmax), later = MRS (smax)
            second_point = int(np.min([s_f_points[-1], s_f_points[-2]]))
            fourth_point = int(np.max([s_f_points[-1], s_f_points[-2]]))
            # trough (third point) = lowest-value point strictly between them
            between_defined_maximums = [e for j, e in enumerate(current_case) if j > second_point and j < fourth_point]
            all_between_maxes = [(j + second_point + 1, e) for j, e in enumerate(between_defined_maximums)]
            all_between_maxes_sort = sorted(all_between_maxes, key=lambda x: x[1], reverse=True)
            t_points.append(all_between_maxes_sort[-1][0])
        except IndexError:
            i += 1
            continue

        maxtab_skip_point = sorted([m for m in maxtab if m > filtered_maxfilter_area_above_middle_p])
        if len(maxtab_skip_point) > 0:
            i = maxtab.index(maxtab_skip_point[0])
        else:
            break

    return (f_points, s_f_points, t_points, l_points)


# ============================================================ QUANTIFY =======
def _decay_times(fulldata, smax, last, fps):
    """✓ ContractionWave.py genexportdata decay ~6669-6708, on relaxation speed.

    For decay in 0.1..0.9: decayed = MRS_speed*(1-decay); within fulldata[smax:last+1]
    take the closest-value index; T = (idx - smax)/fps in ms."""
    speed_relax = fulldata[smax]
    filtered_array = np.asarray(fulldata[smax:last + 1])
    out = {}
    for decay in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        decayed_speed_relax = speed_relax * (1 - decay)
        if len(filtered_array) == 0:
            out[int(round(decay * 100))] = np.nan
            continue
        new_idx = int(np.abs(filtered_array - decayed_speed_relax).argmin()) + smax
        out[int(round(decay * 100))] = (new_idx - smax) / fps * 1000.0
    return out


def quantify(sig, fps, cutoff_val=0.90):
    """Faithful ContractionWave automatic quantification.

    Returns (beats, summary). Each beat dict carries the five landmarks
    (start/mcs/mrs/end indices, plus min_idx) and per-beat metrics.
    summary carries n_beats, BPM ([ext]), and per-metric mean/std incl.
    contraction_ms_mean.
    """
    current_case = list(np.asarray(sig, float))
    T = len(current_case)
    rng = max(current_case) - min(current_case)
    if T < 4 or rng <= 0:
        return [], {"n_beats": 0, "BPM": np.nan, "contraction_ms_mean": np.nan}

    nargs = noise_detection(current_case, filter_noise_area=True,
                            added_noise_dots=[], removed_noise_dots=[], cutoff_val=cutoff_val)
    if nargs is None:
        return [], {"n_beats": 0, "BPM": np.nan, "contraction_ms_mean": np.nan}
    mean_noise = nargs[6]

    f_points, s_f_points, t_points, l_points = peak_detection(current_case, nargs=nargs)

    n = min(len(f_points), len(t_points), len(l_points), len(s_f_points) // 2)
    beats = []
    for k in range(n):
        first = int(f_points[k])
        # s_f_points stores pairs; per beat: [2k]=earlier(MCS/fmax) and [2k+1]=later(MRS/smax)
        a, b = int(s_f_points[2 * k]), int(s_f_points[2 * k + 1])
        fmax = min(a, b)   # MCS
        smax = max(a, b)   # MRS
        mn = int(t_points[k])
        last = int(l_points[k])

        mcs_val = float(current_case[fmax])
        mrs_val = float(current_case[smax])
        rec = {
            "start_idx": first, "mcs_idx": fmax, "min_idx": mn, "mrs_idx": smax, "end_idx": last,
            "MCS": mcs_val, "MRS": mrs_val,
            "MCS_MRS_diff": abs(mcs_val - mrs_val),
            # advanced_parameters (draghandlers.py ~175-196), in ms / µm·s units
            "duration_ms": (last - first) / fps * 1000.0,         # CRT
            "contraction_ms": (mn - first) / fps * 1000.0,        # CT (start->trough)
            "relaxation_ms": (last - mn) / fps * 1000.0,          # RT (trough->end)
            "CTP_ms": (fmax - first) / fps * 1000.0,
            "RTP_ms": (smax - mn) / fps * 1000.0,
            "TBC_RMS_ms": (smax - fmax) / fps * 1000.0,
            "CRA": float(np.trapz([e for e in current_case[first:last + 1]])),
            "SA": float(np.trapz([e for e in current_case[mn:last + 1]])),
        }
        dts = _decay_times(current_case, smax, last, fps)
        for p, v in dts.items():
            rec["T%d_ms" % p] = float(v)
        beats.append(rec)

    mcs_idx = [b["mcs_idx"] for b in beats]
    bpm = float(60.0 / np.median(np.diff(mcs_idx) / fps)) if len(mcs_idx) >= 2 else np.nan

    metric_keys = ["MCS", "MRS", "MCS_MRS_diff", "duration_ms", "contraction_ms",
                   "relaxation_ms", "CTP_ms", "RTP_ms", "TBC_RMS_ms", "CRA", "SA",
                   "T10_ms", "T50_ms", "T90_ms"]
    summary = {"n_beats": len(beats), "BPM": bpm}
    for key in metric_keys:
        vals = np.array([b[key] for b in beats], float)
        vals = vals[~np.isnan(vals)]
        summary[key + "_mean"] = float(vals.mean()) if len(vals) else np.nan
        summary[key + "_std"] = float(vals.std()) if len(vals) else np.nan
    return beats, summary


# ============================================================ MAIN ===========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mpp", type=float, required=True)
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--cutoff-val", type=float, default=0.90,
                    help="noise classification threshold (GUI default 0.90, magnitude units)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    frames = read_clip(args.video, scale=args.scale, max_frames=args.max_frames)
    fps = video_fps(args.video)
    px2um = args.mpp / args.scale
    sig = signal_farneback(frames, fps, px2um)
    beats, summary = quantify(sig, fps, cutoff_val=args.cutoff_val)

    cache = os.path.join(args.output, "cache")
    outd = os.path.join(args.output, "output")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(outd, exist_ok=True)
    np.savez(os.path.join(cache, f"{args.name}_contractionwave.npz"), signal=sig, fps=fps, px2um=px2um)
    with open(os.path.join(cache, f"{args.name}_contractionwave_params.json"), "w") as f:
        json.dump({"sample": args.name, "method": "contractionwave",
                   "cutoff_val": args.cutoff_val, **summary}, f, indent=1)

    t = np.arange(len(sig)) / fps
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(t, sig, color="C0", lw=1.2, label="dense-flow speed (µm/s)")
    for b in beats:
        ax.axvline(b["start_idx"] / fps, color="g", lw=0.7, alpha=0.5)
        ax.axvline(b["end_idx"] / fps, color="r", lw=0.7, alpha=0.5)
        ax.plot(b["mcs_idx"] / fps, sig[b["mcs_idx"]], "v", color="C3", ms=6)
        ax.plot(b["mrs_idx"] / fps, sig[b["mrs_idx"]], "^", color="C4", ms=5)
    ax.set_title("%s — CONTRACTIONWAVE [reimpl]  %d beats, BPM=%.0f, contr=%.0f±%.0f ms"
                 % (args.name, summary["n_beats"], summary.get("BPM", float("nan")),
                    summary.get("contraction_ms_mean", float("nan")),
                    summary.get("contraction_ms_std", float("nan"))), fontsize=9)
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed (µm/s)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(outd, f"{args.name}_contractionwave.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out, "| n_beats", summary["n_beats"], "BPM", round(summary.get("BPM", float("nan")), 1))


if __name__ == "__main__":
    main()
