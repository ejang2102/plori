"""Generate a synthetic beating-organoid video for a self-contained PLoRI demo.

Renders a dark, textured disc on a light background whose radius and interior
brightness pulse periodically (a short contraction phase and a longer return).
No proprietary data is needed to exercise the full pipeline; the segmentation,
period estimation, and beat metrics all run on this clip.

Example:
    python scripts/make_synthetic.py --out examples/synthetic_beating.mp4
"""
import os
import argparse
import numpy as np
import cv2


def beat_transient(phase, contract_frac, relax_frac):
    """Per-beat envelope in [0, 1] on a smooth raised-cosine shape: rise 0->1 over
    the first `contract_frac` of the beat cycle, fall 1->0 over the next
    `relax_frac`, then held at 0 until the next beat. The rise/fall asymmetry is set
    directly by the two fractions (e.g. relax_frac larger than contract_frac gives
    a longer fall than rise)."""
    c, r = contract_frac, relax_frac
    if phase < c:
        return 0.5 * (1.0 - np.cos(np.pi * phase / c))
    if phase < c + r:
        return 0.5 * (1.0 + np.cos(np.pi * (phase - c) / r))
    return 0.0


def main():
    ap = argparse.ArgumentParser(description="Synthesize a beating-organoid demo video")
    ap.add_argument("--out", required=True, help="output .mp4 path")
    ap.add_argument("--size", type=int, default=256, help="frame height=width (px)")
    ap.add_argument("--fps", type=float, default=15.0, help="frames per second")
    ap.add_argument("--duration", type=float, default=18.0, help="clip length (s)")
    ap.add_argument("--bpm", type=float, default=35.0, help="beat rate (beats per minute)")
    ap.add_argument("--radius-amp", type=float, default=0.10, help="fractional radius shrink at peak contraction")
    ap.add_argument("--contract-frac", type=float, default=0.08, help="fraction of each beat cycle spent on the rising (contraction) phase")
    ap.add_argument("--relax-frac", type=float, default=0.14, help="fraction of each beat cycle spent on the falling (relaxation) phase "
                    "(contract-frac + relax-frac must be <= 1); the remaining fraction of the cycle is held at 0")
    ap.add_argument("--seed", type=int, default=0, help="texture/noise seed")
    a = ap.parse_args()
    if a.contract_frac + a.relax_frac > 1.0:
        ap.error("--contract-frac + --relax-frac must be <= 1 (they are fractions of one beat cycle)")

    H = W = int(a.size)
    T = int(round(a.fps * a.duration))
    rate_hz = a.bpm / 60.0
    rng = np.random.default_rng(a.seed)

    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r0 = 0.34 * H

    # static interior texture (speckle), smoothed so it survives downscaling
    texture = rng.normal(0, 1, (H, W)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), 3.0)
    texture = 18.0 * texture / (texture.std() + 1e-6)

    bg, inside = 205.0, 70.0  # background bright, organoid dark (darkness-Otsu segmentable)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H), isColor=True)
    if not vw.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {a.out}")

    for t in range(T):
        phase = (t / a.fps * rate_hz) % 1.0
        L = float(beat_transient(phase, a.contract_frac, a.relax_frac))  # 0 between beats, 1 at peak contraction
        r = r0 * (1.0 - a.radius_amp * L)            # contract = smaller radius
        edge = 1.0 / (1.0 + np.exp((dist - r) / 1.5))  # soft disc membership in [0,1]
        interior = inside - 22.0 * L + texture       # darker at peak contraction
        img = bg * (1.0 - edge) + interior * edge
        img = img + rng.normal(0, 1.6, (H, W))       # mild sensor noise
        frame = np.clip(img, 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    vw.release()
    print(f"wrote {a.out}  ({T} frames, {a.fps:g} fps, {a.duration:g}s, {a.bpm:g} BPM)")


if __name__ == "__main__":
    main()
