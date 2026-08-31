"""PLoRI batch — run over a folder of videos. Globs videos -> plori.core.analyze
-> {out-dir}/{cat}/{name}/{cat}_{name}_ploridata.npz. Render with plori_report.py.

Output layout: {out-dir}/{cat}/{name}/{cat}_{name}_ploridata.npz.

Examples:
  # Group given literally (cat=mygroup), sample name extracted from the filename via regex.
  python scripts/plori_batch.py \
    --glob '/path/to/videos/*.mp4' \
    --out-dir results/cache --cat mygroup --name-re 'sample_\\d+' --mpp 0.6 --mag 4X --scale 0.25
  # Group taken from the parent directory name, sample name = file stem.
  python scripts/plori_batch.py \
    --glob '/path/to/videos/**/*.mp4' \
    --out-dir results/cache --cat @parent --mpp 0.6 --mag 4X --scale 0.25
"""
import os, re, glob, argparse, numpy as np, cv2, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plori.video import read_clip, video_fps
from plori import core as PL
cv2.setNumThreads(int(os.environ.get("CV2_THREADS", "4")))


def resolve_cat(video, cat_arg):
    return os.path.basename(os.path.dirname(video)) if cat_arg == "@parent" else cat_arg


def resolve_name(video, name_re):
    base = os.path.basename(video)
    if name_re:
        m = re.search(name_re, base)
        if not m:
            raise ValueError(f"--name-re {name_re!r} no match in {base!r}")
        return m.group(0)
    return os.path.splitext(base)[0].replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="video glob (recursive ** supported)")
    ap.add_argument("--out-dir", required=True, help="cache root; writes {cat}/{name}/{cat}_{name}_ploridata.npz")
    ap.add_argument("--cat", required=True, help="category label (e.g. a group name) or '@parent' (use the parent directory name)")
    ap.add_argument("--name-re", default="", help="regex to extract sample name from basename; empty = file stem")
    ap.add_argument("--mpp", type=float, required=True, help="microns per full-resolution pixel (your microscope's calibration; the script applies --scale internally)")
    ap.add_argument("--mag", default="4X")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--pixcap", type=int, default=3000)
    ap.add_argument("--dilate", type=int, default=3)
    ap.add_argument("--mask-mode", default="median", choices=["median", "union"])
    ap.add_argument("--save-perpixel", action=argparse.BooleanOptionalAction, default=True,
                    help="store per-pixel signal (union superset) → derive any mask aggregate offline (bigger files). "
                         "on by default; pass --no-save-perpixel to skip")
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--dynamic-range-filter", action="store_true", help="per-pixel dynamic-range filter (off by default): before averaging, drop the lowest --drop-frac of pixels by p95-p50 intensity spread (excludes pixels with negligible temporal variation)")
    ap.add_argument("--drop-frac", type=float, default=0.25, help="fraction of the lowest-spread pixels the dynamic-range filter removes (0.25 = narrowest quartile)")
    ap.add_argument("--with-flow", action=argparse.BooleanOptionalAction, default=True,
                    help="also compute masked dense optical flow (inside the organoid mask) and per-beat mechanics "
                         "over the PLoRI onset/offset windows: max speed (µm/s), displacement (µm), strain. On by "
                         "default (--no-with-flow to skip; µm units need a real --mpp). Slower.")
    ap.add_argument("--fill", action=argparse.BooleanOptionalAction, default=True,
                    help="fill interior mask holes (binary_fill_holes): bright translucent centres that Otsu drops as "
                         "background are closed into the full organoid footprint → corrects both the union signal mask and "
                         "size/area. On by default; pass --no-fill to disable. Opacity (dark-core based) is unchanged.")
    ap.add_argument("--only", default="", help="comma names to restrict")
    a = ap.parse_args()
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    vids = sorted(glob.glob(a.glob, recursive=True))
    if not vids:
        print(f"no videos: {a.glob}"); return
    print(f"{len(vids)} videos -> {a.out_dir}  (cat={a.cat} scale={a.scale} mask_mode={a.mask_mode} mpp={a.mpp})", flush=True)
    ok = 0
    for p in vids:
        try:
            cat = resolve_cat(p, a.cat); name = resolve_name(p, a.name_re)
        except ValueError as e:
            print(f"skip {p}: {e}", flush=True); continue
        if only and name not in only:
            continue
        try:
            frames = read_clip(p, scale=a.scale, max_frames=a.max_frames); fps = video_fps(p)
            g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f for f in frames]).astype(np.float32)
            out = PL.analyze(g, fps, a.mpp, a.mag, a.scale, a.pixcap, a.dilate, a.mask_mode, a.save_perpixel,
                             dynrange_filter=a.dynamic_range_filter, drop_frac=a.drop_frac, fill=a.fill)
            out.update(cat=cat, name=name, typ=cat, video=p)
            out["with_flow"] = bool(a.with_flow)
            if a.with_flow:
                from plori import flow as FL
                speed = FL.masked_flow_speed(g, out["mask"], fps, a.mpp / a.scale)
                out["flow_speed"] = speed
                out.update(FL.beat_mechanics(speed, out["ons"], out["offs"], fps, float(out["size_um"])))
            od = os.path.join(a.out_dir, cat, name); os.makedirs(od, exist_ok=True)
            saver = np.savez_compressed if "pixraw" in out else np.savez   # save the per-pixel raw signal compressed
            saver(os.path.join(od, f"{cat}_{name}_ploridata.npz"), **out)
            ok += 1
            print(f"[{cat}] {name}: fps={fps:.2f} T={len(g)} beats={len(out['pk'])} k={int(out['k'])} "
                  f"BPM={float(out['bpm_pk']):.0f} CD50={float(out['cd50_med']):.0f}ms drift={float(out['drift_um']):.0f}um", flush=True)
        except Exception as e:
            import traceback; print(f"{name} FAIL: {e}\n{traceback.format_exc()[:300]}", flush=True)
    print(f"DONE {ok}/{len(vids)}", flush=True)


if __name__ == "__main__":
    main()
