"""Run PLoRI on a single brightfield organoid video.

Reads one video, computes the PLoRI contraction signal and per-beat
metrics, and writes a `*_ploridata.npz` result bundle. Render it to a report card
with `plori_report.py`, or export a spreadsheet with `plori_to_xlsx.py`.

Example (using the bundled synthetic clip):
    python scripts/make_synthetic.py --out examples/synthetic_beating.mp4
    python scripts/run_plori.py --video examples/synthetic_beating.mp4 \
        --name demo --out-dir examples/out --mpp 1.0 --mag synthetic --scale 1.0
    python scripts/plori_report.py --glob 'examples/out/**/*_ploridata.npz' --out-dir examples/out

`--mpp` is microns per full-resolution pixel; pass 1.0 if you only care about the
temporal metrics (rate, durations) and not the physical size in mm/µm.
"""
import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plori.video import read_clip, video_fps
from plori import core as PL

cv2.setNumThreads(int(os.environ.get("CV2_THREADS", "4")))


def main():
    ap = argparse.ArgumentParser(description="Run PLoRI on one video -> ploridata.npz")
    ap.add_argument("--video", required=True, help="path to a brightfield organoid video")
    ap.add_argument("--name", default="", help="sample name (default: file stem)")
    ap.add_argument("--cat", default="sample", help="category/group label (subfolder in --out-dir)")
    ap.add_argument("--out-dir", required=True, help="output root; writes {cat}/{name}/{cat}_{name}_ploridata.npz")
    ap.add_argument("--mpp", type=float, default=1.0, help="microns per full-resolution pixel (1.0 = size in pixels)")
    ap.add_argument("--mag", default="NA", help="magnification label, stored in the result for reference")
    ap.add_argument("--scale", type=float, default=1.0, help="decode downscale factor (speed only)")
    ap.add_argument("--pixcap", type=int, default=3000, help="max pixels sampled inside the mask")
    ap.add_argument("--dilate", type=int, default=3, help="mask dilation radius (px, at --scale)")
    ap.add_argument("--mask-mode", default="union", choices=["median", "union"],
                    help="signal mask: 'union' (default) OR of per-frame masks; 'median' mask of the temporal median")
    ap.add_argument("--save-perpixel", action=argparse.BooleanOptionalAction, default=True,
                    help="store per-pixel raw intensities so any sub-mask signal can be re-derived offline")
    ap.add_argument("--max-frames", type=int, default=1500, help="cap frames decoded")
    ap.add_argument("--dynamic-range-filter", action="store_true",
                    help="per-pixel dynamic-range filter (off by default): before averaging, drop the lowest --drop-frac of pixels "
                         "by p95-p50 intensity spread (excludes pixels with negligible temporal variation). For very weak beats only.")
    ap.add_argument("--drop-frac", type=float, default=0.25,
                    help="fraction of the lowest-spread pixels the dynamic-range filter removes (0.25 = narrowest quartile)")
    ap.add_argument("--with-flow", action=argparse.BooleanOptionalAction, default=True,
                    help="also compute masked dense optical flow (Farneback, inside the organoid mask) and per-beat "
                         "mechanics over the PLoRI onset/offset windows: max speed (µm/s), displacement (µm), strain. "
                         "Independent of the ContractionWave baseline. On by default (--no-with-flow to skip; the µm "
                         "units need a real --mpp). Slower (optical flow per frame).")
    ap.add_argument("--fill", action=argparse.BooleanOptionalAction, default=True,
                    help="fill interior mask holes (binary_fill_holes): bright translucent centres that Otsu drops as "
                         "background are closed into the full organoid footprint → corrects both the union signal mask and "
                         "size/area. On by default; pass --no-fill to disable. Opacity (dark-core based) is unchanged.")
    a = ap.parse_args()

    name = a.name or os.path.splitext(os.path.basename(a.video))[0].replace(" ", "_")
    frames = read_clip(a.video, scale=a.scale, max_frames=a.max_frames)
    fps = video_fps(a.video)
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
                  for f in frames]).astype(np.float32)

    out = PL.analyze(g, fps, a.mpp, a.mag, a.scale, a.pixcap, a.dilate,
                     a.mask_mode, a.save_perpixel,
                     dynrange_filter=a.dynamic_range_filter, drop_frac=a.drop_frac, fill=a.fill)
    out.update(cat=a.cat, name=name, typ=a.cat, video=a.video)

    out["with_flow"] = bool(a.with_flow)
    if a.with_flow:
        from plori import flow as FL
        px2um = a.mpp / a.scale
        speed = FL.masked_flow_speed(g, out["mask"], fps, px2um)
        out["flow_speed"] = speed
        out.update(FL.beat_mechanics(speed, out["ons"], out["offs"], fps, float(out["size_um"])))

    od = os.path.join(a.out_dir, a.cat, name)
    os.makedirs(od, exist_ok=True)
    saver = np.savez_compressed if "pixraw" in out else np.savez
    fp = os.path.join(od, f"{a.cat}_{name}_ploridata.npz")
    saver(fp, **out)   # save BEFORE any summary print: a stdout encoding error must never lose a finished result

    print(f"[{a.cat}] {name}: fps={fps:.2f} T={len(g)} beats={len(out['pk'])} "
          f"k={int(out['k'])} BPM={float(out['bpm_pk']):.0f} "
          f"CD50={float(out['cd50_med']):.0f}ms drift={float(out['drift_um']):.0f}um")
    if a.with_flow:
        print(f"    masked-flow: max_speed={out['max_speed_med']:.2f}um/s "
              f"displacement={out['displacement_med']:.2f}um strain={out['strain_med']:.4f} (per-beat medians)")
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()
