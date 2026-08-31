"""PLoRI report card renderer (fast, no video needed) — reads {cat}_{name}_ploridata.npz
(produced by plori_batch.py) and renders the summary PNG.

Thumbnail: first frame + median/union area overlap + last-frame outline (red) + drift (with warning).
Panels: thumbnail | metrics text + QC | PLoRI (peaks, onset/offset) | masked-flow speed (with --with-flow) or frame difference | d(PLoRI)/dt.
Usage:
  python scripts/plori_report.py \
    --glob 'results/**/*_ploridata.npz' --out-dir results/reports
"""
import os, glob, argparse, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plori import core as P


def _get(d, *keys, default=""):
    for k in keys:
        if k in d.files:
            return str(d[k])
    return default


def render(fp, out_dir, by_cat=False):
    d = np.load(fp, allow_pickle=True)
    cat = _get(d, "cat", "category", "typ"); name = _get(d, "name")
    tag = f"{cat}_{name}" if cat else name
    med = d["med"]; mask = d["mask"]; cs = d["cs"]; y = np.asarray(d["plori"], float); dd = d["dd"]
    fps = float(d["fps"]); T = len(y)
    pk, ons, offs = d["pk"], d["ons"], d["offs"]; ivl, ct, rt = d["ivl_ms"], d["ct"], d["rt"]
    bpm_pk, bpm_ac, iCV = float(d["bpm_pk"]), float(d["bpm_ac"]), float(d["iCV"])
    ctCV, rtCV = float(d["ctCV"]), float(d["rtCV"])
    opacity_class = str(d["opacity_class"]); flags = [f for f in str(d["flags"]).split("|") if f]
    t = np.arange(T) / fps; fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1.0, 1.6, 1.6], height_ratios=[1.2, 1.0, 1.0], hspace=0.35, wspace=0.25)
    # --- thumbnail ---
    axI = fig.add_subplot(gs[0, 0])
    if "frame0" in d.files and "mask_union" in d.files:                # new cache: actual first frame + union mask (analysis mask) + last-frame outline
        axI.imshow(d["frame0"], cmap="gray")
        mu = d["mask_union"]
        axI.imshow(np.ma.masked_where(~mu, mu.astype(float)), cmap=ListedColormap(["lime"]), alpha=0.28, vmin=0, vmax=1)
        axI.contour(mu, [0.5], colors="lime", linewidths=1.0)          # area fill + distinct outline
        if "mask_last" in d.files: axI.contour(d["mask_last"], [0.5], colors="red", linewidths=1.2)
        top = "first frame"; cap = "green fill=analysis mask  red=last-frame outline"
    else:                                                              # old cache: fall back to the median image (labeled explicitly)
        axI.imshow(med, cmap="gray"); axI.contour(mask, [0.5], colors="#2ca02c", linewidths=1.3)
        top = "median (rest) image"; cap = "green=mask"
    axI.plot(cs[:, 0], cs[:, 1], "-", color="#ff7f0e", lw=0.9)
    dr = float(d["drift_um"]); warn = "  HIGH DRIFT" if dr > 0.15 * float(d["size_um"]) else ""
    axI.set_title(f"{top} · drift {dr:.0f}µm{warn}", fontsize=8)
    axI.set_xlabel(cap, fontsize=7); axI.set_xticks([]); axI.set_yticks([])
    badge = [f for f in flags if f != "high_drift"]                    # high_drift is already shown in the title → avoid a duplicate chip
    if badge:                                                          # QC badge = top-left of thumbnail (does not cover the text panel)
        FLAG_LABEL = {"low_signal": "failure suspected", "irregular_rhythm": "irregular rhythm"}
        axI.text(0.03, 0.97, "QC: " + " · ".join(FLAG_LABEL.get(f, f) for f in badge), transform=axI.transAxes,
                 va="top", ha="left", color="white", fontsize=7.5, bbox=dict(fc="#d62728", ec="none", pad=2), zorder=5)
    # --- metrics text ---
    axT = fig.add_subplot(gs[1:, 0]); axT.axis("off")
    ivl_med = float(np.median(ivl))
    mmode = str(d["mask_mode"]) if "mask_mode" in d.files else "?"                    # analysis mask and dynamic-range condition (shown on the card)
    dr_status = f"dyn-range drop {float(d['drop_frac']):.2f}" if ("dynrange_filter" in d.files and bool(d["dynrange_filter"])) else "dyn-range off"
    txt = (f"[{cat}]\n[{opacity_class}]\n\nVIDEO\n {str(d['mag'])}  fps {fps:.2f}  {float(d['dur']):.0f}s ({int(d['nframes'])}f)\n"
           f" mpp {float(d['mpp']):.4f}\n MASK {mmode} · {dr_status}\n\nRHYTHM\n BPM {bpm_pk:.1f} (ac {bpm_ac:.1f})\n interval {ivl_med:.0f}ms (CV {iCV:.1f}%)\n\n"
           f"DURATION\n CD50 {float(d['cd50_med']):.0f}ms (CV {float(d['cd50CV']):.0f}%)\n"
           f" CT {float(d['ct_med']):.0f}ms (CV {ctCV:.0f}%)\n RT {float(d['rt_med']):.0f}ms (CV {rtCV:.0f}%)\n\n"
           f"AMPLITUDE\n med {float(d['amp_med']):.1f}  CV {float(d['ampCV']):.0f}%\n\n"
           f"AREA {float(d['area_mm2']):.2f}mm²\nOPACITY {float(d['opacity']):.0f}%")
    axT.text(0.02, 0.99, tag, va="top", ha="left", fontsize=11, fontweight="bold", family="monospace")
    axT.text(0.02, 0.92, txt, va="top", ha="left", fontsize=10, family="monospace")
    # --- signal + derivative + histograms ---
    axL = fig.add_subplot(gs[0, 1:]); axL.plot(t, y, color="#7f0000", lw=0.8)
    for on, off in zip(ons, offs): axL.axvspan(on / fps, off / fps, color="#2ca02c", alpha=0.08)
    axL.plot(pk / fps, y[pk], "v", color="#1f77b4", ms=4)
    axL.set_ylabel("PLoRI (a.u.)", fontsize=8)
    axL.set_title("PLoRI  v=peak  shade=transient", fontsize=9); axL.grid(alpha=0.3); axL.margins(x=0); axL.set_xticklabels([])
    hasff = "pixraw" in d.files and "mask_union" in d.files
    has_flow = "flow_speed" in d.files
    # middle: masked optical-flow speed (with --with-flow, replacing frame-diff); else frame-diff from pixraw; else d(PLoRI)/dt
    axM = fig.add_subplot(gs[1, 1:])
    if has_flow:
        spd = np.asarray(d["flow_speed"], float)
        axM.plot(t, spd, color="tab:red", lw=0.8)
        for on, off in zip(ons, offs):                              # mark each beat's peak speed
            a = int(max(0, np.floor(on))); b = int(min(T - 1, np.ceil(off)))
            if b - a >= 2:
                jj = a + int(np.argmax(spd[a:b + 1])); axM.plot(t[jj], spd[jj], "v", color="k", ms=3)
        axM.set_ylabel("masked flow speed (µm/s)", fontsize=8)
        axM.set_title(f"masked optical flow  ·  max {float(d['max_speed_med']):.1f} µm/s  "
                      f"disp {float(d['displacement_med']):.2f} µm  strain {float(d['strain_med']):.3f}  (per-beat median)", fontsize=9)
        axM.set_xticklabels([])
    elif hasff:
        fdiff = P.derive(d["pixraw"], d["pix_idx"], d["mask_union"], fps, "ftf")
        axM.plot(t, fdiff, color="tab:green", lw=0.8); axM.set_ylabel("frame-diff (a.u./frame)", fontsize=8)
        axM.set_title("frame difference  |I(t)−I(t−1)|", fontsize=9); axM.set_xticklabels([])
    else:
        axM.plot(t, dd, color="#9467bd", lw=0.9); axM.axhline(0, color="0.6", lw=0.6, ls="--")
        axM.set_ylabel("d(PLoRI)/dt (a.u./s)", fontsize=8); axM.set_title("d(PLoRI)/dt", fontsize=9); axM.set_xlabel("time (s)")
    axM.grid(alpha=0.3); axM.margins(x=0)
    # bottom (former histogram slot): d(PLoRI)/dt
    if hasff or has_flow:
        axD = fig.add_subplot(gs[2, 1:]); axD.plot(t, dd, color="#9467bd", lw=0.9); axD.axhline(0, color="0.6", lw=0.6, ls="--")
        axD.set_ylabel("d(PLoRI)/dt (a.u./s)", fontsize=8); axD.set_title("d(PLoRI)/dt  (+contract / −relax)", fontsize=9)
        axD.grid(alpha=0.3); axD.margins(x=0); axD.set_xlabel("time (s)")
    fig.suptitle(f"PLoRI report — {tag}   BPM {bpm_pk:.1f}  iCV {iCV:.1f}%  CD50 {float(d['cd50_med']):.0f}ms", fontsize=13)
    od = os.path.join(out_dir, cat) if (by_cat and cat) else out_dir      # --by-cat → {out-dir}/{cat}/
    os.makedirs(od, exist_ok=True); out = f"{od}/report_{tag}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob of *_ploridata.npz (recursive ** supported)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--by-cat", action="store_true", help="write into a {out-dir}/{cat}/ subfolder")
    a = ap.parse_args()
    fps = sorted(glob.glob(a.glob, recursive=True))
    if not fps: print(f"no ploridata: {a.glob}"); return
    for fp in fps:
        try: print(render(fp, a.out_dir, a.by_cat), flush=True)
        except Exception as e: print(f"{os.path.basename(fp)} FAIL: {e}", flush=True)
    print(f"DONE {len(fps)} reports", flush=True)


if __name__ == "__main__":
    main()
