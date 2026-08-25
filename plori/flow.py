"""Masked dense optical flow — a per-frame motion measure computed INSIDE the
organoid mask.

This is plain dense optical flow (Farneback); it is deliberately kept separate
from `plori.baselines.contractionwave`, which is a faithful port of the
ContractionWave tool and reduces motion over the WHOLE frame. Here the spatial
reduction is over the organoid mask only (`mag[mask].mean()`). ContractionWave is
one whole-frame implementation of this technique; this masked variant is
independent of that tool.

It is used to derive per-beat mechanics over PLoRI onset/offset windows:
  max speed     = peak in-mask flow magnitude within a beat            (µm/s)
  displacement  = 0.5 * integral of speed over onset->offset           (µm)
  strain        = displacement / organoid size                         (dimensionless)

All physical units assume a real µm-per-pixel calibration (px2um = mpp / scale);
with px2um = 1 the same quantities are expressed in pixels instead of µm.
"""
import numpy as np
import cv2

# Same Farneback parameters as the ContractionWave port, so the underlying flow
# estimate is comparable; only the spatial reduction (mask vs whole frame) differs.
_FARNEBACK = dict(pyr_scale=0.5, levels=1, winsize=15, iterations=1,
                  poly_n=7, poly_sigma=1.5, flags=0)


def masked_flow_speed(g, mask, fps, px2um):
    """Per-frame speed (µm/s) = mean Farneback flow magnitude inside `mask`,
    scaled by fps and px2um. `g` is a (T,H,W) gray stack, `mask` a (H,W) bool.
    Returns a length-T array with speed[0] = 0."""
    g = np.asarray(g)
    m = np.asarray(mask, bool)
    T = len(g)
    speed = np.zeros(T)
    if not m.any():
        return speed
    prev = g[0].astype(np.uint8)
    for i in range(1, T):
        cur = g[i].astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None,
                                            _FARNEBACK["pyr_scale"], _FARNEBACK["levels"],
                                            _FARNEBACK["winsize"], _FARNEBACK["iterations"],
                                            _FARNEBACK["poly_n"], _FARNEBACK["poly_sigma"],
                                            _FARNEBACK["flags"])
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        speed[i] = abs(float(mag[m].mean()) * fps * px2um)
        prev = cur
    return speed


def beat_mechanics(speed, ons, offs, fps, size_um):
    """Per-beat mechanics from a masked-flow speed series (µm/s) and the PLoRI
    onset/offset frame times (`ons`, `offs`, paired per beat).

      max_speed    = peak speed within [onset, offset]                       (µm/s)
      displacement = 0.5 * integral(speed dt) over [onset, offset]           (µm)
                     The factor 0.5 turns the rectified round-trip path (the
                     magnitude rises during both contraction and relaxation)
                     into a one-way excursion, assuming the two are symmetric.
      strain       = displacement / size_um                                  (dimensionless)

    Returns per-beat arrays (max_speed / displacement / strain) and their medians.
    Beats whose window is too short are skipped."""
    speed = np.asarray(speed, float)
    T = len(speed)
    ons = np.asarray(ons, float); offs = np.asarray(offs, float)
    ms = []; disp = []; strain = []
    for on, off in zip(ons, offs):
        a = int(np.clip(np.floor(on), 0, T - 1))
        b = int(np.clip(np.ceil(off), 0, T - 1))
        if b - a < 2:
            continue
        seg = speed[a:b + 1]
        ms.append(float(seg.max()))
        d = 0.5 * float(np.trapz(seg, dx=1.0 / fps))          # µm (fps cancels: this is a distance)
        disp.append(d)
        strain.append(d / size_um if size_um else float("nan"))
    ms = np.array(ms); disp = np.array(disp); strain = np.array(strain)
    med = lambda x: float(np.median(x)) if len(x) else float("nan")
    return dict(max_speed=ms, displacement=disp, strain=strain,
                max_speed_med=med(ms), displacement_med=med(disp), strain_med=med(strain))
