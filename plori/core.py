"""PLoRI — Per-pixel Local Rest-referenced Intensity. Drift-robust, label-free contraction
signal from video: per masked pixel, |I(t) − medfiltₜ(I,k)| (time-local rolling reference,
k≈period), spatially averaged over an organoid mask. The per-pixel + local rest-reference +
rectify-then-average combination avoids MuscleMotion's single-reference saturation and the
mean-brightness polarity ambiguity.

Core API:
  segment(gray_img, dilate)                 -> organoid mask (darkness Otsu + largest CC + morph)
  plori_signal(gfm, fps)                    -> (signal, k)   2-pass period, per-pixel medfilt
  beat_metrics(signal, fps)                 -> dict of peaks/onset-offset/CD50/CT/RT/BPM/CV/flags
  analyze(g, fps, mpp, mag, fullres, ...)   -> full result dict (mask, signal, metrics, size/opacity/drift)
  derive(pixraw, pix_idx, mask, fps, kind)  -> aggregate for any sub-mask from stored raw (PLoRI/ftf/raw)
`g` is a (T,H,W) float32 gray stack. mpp = µm/full-res-px; the caller reads at `scale`.

Derived signals — all are intensity a.u. (NOT physical µm; the kinematic names are loose analogies,
so figure axes label the literal quantity, not velocity/speed):
  disp     ≈ PLoRI  = ⟨|I − medfiltₜ(I,k)|⟩_mask       contraction-magnitude proxy   [a.u.]
  velocity ≈ d(PLoRI)/dt                                signed rate (+contract/−relax) [a.u./s]  (field `dd`)
  speed    ≈ frame-diff = ⟨|I(t) − I(t−1)|⟩_mask        inter-frame motion magnitude   [a.u./frame]
PLoRI/disp is stored; d/dt is its derivative; frame-diff/speed needs raw intensities (save_perpixel →
derive(kind='ftf')). frame-diff has per-beat double-peaks (contract+relax); rectified PLoRI yields
one bump per beat. ('ftf' = frame-to-frame, the internal derive() kind string.)
"""
import os, math, numpy as np, cv2
from scipy.ndimage import median_filter, binary_fill_holes
from scipy.signal import find_peaks
K_MULT = float(os.environ.get("PLORI_K_MULT", "1.0"))   # moving-reference window multiplier: window = odd(K_MULT * period). Override via PLORI_K_MULT.
# --- beat-quality QC thresholds (PROVISIONAL; calibrated on a limited reference set, see paper limitations) ---
# low_signal ("failure suspected") = amp_med < QC_AMP_MIN AND snr < QC_SNR_MIN: BOTH detection axes at the floor.
#   The AND is deliberate — a beat with healthy amplitude OR healthy SNR is rescued; only when both fail is the
#   signal flagged. "suspected" because a signal at the noise floor does not prove contraction has ceased (fine
#   sub-threshold activity cannot be ruled out). Calibrated to separate beating (n=53) from no-beat (n=40) organoids.
# irregular_rhythm = iCV > QC_ICV_MAX AND not low_signal: high beat-to-beat interval variability among DETECTABLE
#   beats only (no-beat noise inflates iCV via spurious peaks, so it must be gated behind low_signal). This is a
#   descriptive regularity readout, NOT a diagnosis of arrhythmia. QC_ICV_MAX set by visual inspection.
QC_AMP_MIN = 0.7
QC_SNR_MIN = 5.6
QC_ICV_MAX = 20.0
def autocorr_period(sig, fps, fmin=0.1, fmax=4.0):
    """Period (frames) = lag of the first autocorrelation peak in [fps/fmax, fps/fmin]."""
    x = sig - sig.mean()
    if np.allclose(x, 0):
        return int(fps)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = max(2, int(fps / fmax))
    hi = min(len(ac) - 1, int(fps / fmin))
    if hi <= lo:
        return int(fps)
    pk, _ = find_peaks(ac[lo:hi + 1], height=0.1)
    if len(pk) == 0:
        return int(np.clip(lo + int(np.argmax(ac[lo:hi + 1])), 2, hi))
    return int(lo + pk[0])
def _odd(k,T): k=int(k)|1; return max(5,min(k,(T//2)*2-1))
def _grid_subsample(mask, pixcap):
    """Uniform 2D-grid subsample: select ~pixcap flat (H*W) indices from within mask(H,W) on a
    regular grid. This avoids the spatial bias of 1D linspace (row-major) subsampling and the moiré
    it produces in per-pixel visualizations. Since count(stride) decreases monotonically with stride,
    pick the densest (smallest) stride that still stays at or below pixcap."""
    idx=np.flatnonzero(mask.reshape(-1))
    if not pixcap or len(idx)<=pixcap: return idx
    H,W=mask.shape
    def sel_for(s):
        gm=np.zeros((H,W),bool); gm[::s,::s]=True; return np.flatnonzero((gm&mask).reshape(-1))
    s=max(1,int(round(np.sqrt(len(idx)/pixcap))))
    while len(sel_for(s))>pixcap: s+=1                     # too many: make the grid coarser
    while s>1 and len(sel_for(s-1))<=pixcap: s-=1          # room to spare: make it denser (staying ≤pixcap)
    return sel_for(s)
def _dynamic_range_mask(gf, mask, drop_frac=0.25, min_n=50):
    """Dynamic-range filter (k-free): score each pixel by its intensity spread s(x)=p95−p50 (temporal
    dynamic range) and drop the lowest drop_frac of pixels (dark, low-texture, non-beating). Unlike an
    activity measure (Σ|ΔI|), the spread is not fooled by high-frequency noise, and it is independent
    of brightness and beat rate (slow beats keep their spread). drop_frac=0.25 excludes the narrowest
    25%. If too few pixels remain, fall back to the original mask."""
    H,W=mask.shape; uflat=np.flatnonzero(mask.reshape(-1))
    if len(uflat)==0 or drop_frac<=0: return mask
    sub=gf[:,uflat].astype(np.float32)
    spread=np.percentile(sub,95,axis=0)-np.percentile(sub,50,axis=0)    # (len(uflat),) p95−p50
    thr=float(np.quantile(spread,drop_frac))                           # cut at the lowest drop_frac
    keep=spread>thr
    if keep.sum()<min_n: return mask
    active=np.zeros(H*W,bool); active[uflat[keep]]=True
    return active.reshape(H,W)
def _lcc(m):
    n,lab,st,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
    return m if n<=1 else lab==(1+int(np.argmax(st[1:,cv2.CC_STAT_AREA])))
def segment(gray_img, dilate=3, fill=False):
    """Organoid mask: darkness Otsu on the (blurred) image -> largest CC -> close/open (+dilate).
    Works for opaque and translucent-but-darker organoids alike (both darker than background).
    fill=True: binary_fill_holes closes interior holes (bright translucent centres that Otsu drops as
    background) -> full organoid footprint. Applied before dilate."""
    gi=cv2.GaussianBlur(gray_img.astype(np.float32),(0,0),2)
    thr,_=cv2.threshold(gi.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); m=gi<thr
    m=_lcc(m); m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); m=_lcc(m>0)
    if dilate>0: m=cv2.dilate(m.astype(np.uint8),np.ones((2*dilate+1,2*dilate+1),np.uint8))>0
    if fill: m=binary_fill_holes(m)   # fill after dilate: dilation can seal a narrow open-bay mouth, turning it into a closed hole only then
    return m
def plori_signal(gfm, fps):
    """PLoRI over a (T, n_pixels) masked (optionally downsampled) stack. 2-pass period: ftf
    (frame-diff) can 2x-lock -> provisional PLoRI, re-estimate period from the single-humped
    PLoRI, then final. Returns (signal, k)."""
    T=gfm.shape[0]
    ftf=np.zeros(T); ftf[1:]=np.abs(np.diff(gfm,axis=0)).mean(1); k0=_odd(1.5*autocorr_period(ftf,fps),T)
    s0=np.abs(gfm-median_filter(gfm,(k0,1),mode="reflect")).mean(1); k=_odd(K_MULT*autocorr_period(s0,fps),T)
    sig=np.abs(gfm-median_filter(gfm,(k,1),mode="reflect")).mean(1).astype(np.float64)
    return sig,k
def plori_perpixel(gfm, fps):
    """Like plori_signal but returns the PER-PIXEL rectified deviation R=(T,N) float32 (not spatially
    averaged) plus k. Aggregate over any sub-mask = R[:, sel].mean(1) — lets one stored union-superset
    R derive median/union/dynamic-range/regional aggregates without re-reading the video. See aggregate_for()."""
    T=gfm.shape[0]
    ftf=np.zeros(T); ftf[1:]=np.abs(np.diff(gfm,axis=0)).mean(1); k0=_odd(1.5*autocorr_period(ftf,fps),T)
    s0=np.abs(gfm-median_filter(gfm,(k0,1),mode="reflect")).mean(1); k=_odd(K_MULT*autocorr_period(s0,fps),T)
    R=np.abs(gfm-median_filter(gfm,(k,1),mode="reflect")).astype(np.float32)
    return R,k
def aggregate_for(perpixel, pix_idx, mask):
    """Aggregate over a (H,W) bool sub-mask from a stored per-pixel array (T,N): mean of the columns
    whose pixel is in `mask`. Works for any per-pixel quantity (R, raw, ...). pix_idx = flat H*W indices."""
    sel=np.asarray(mask).reshape(-1)[np.asarray(pix_idx)]
    return perpixel[:, sel].mean(1).astype(np.float64) if int(sel.sum())>=1 else perpixel.mean(1).astype(np.float64)
def derive(pixraw, pix_idx, mask, fps, kind="plori", k=None):
    """Derive an aggregate signal for any sub-mask from stored RAW per-pixel intensities `pixraw` (T,N uint8;
    save_perpixel). No video needed. kind: 'plori'=|I−medfilt_t(I,k)| (k from npz or re-estimated) · 'ftf'=
    |I(t)−I(t−1)| (speed) · 'raw'=mean intensity. Enables PLoRI/ftf/MM/regional offline from one stored array."""
    sel=np.asarray(mask).reshape(-1)[np.asarray(pix_idx)]
    if int(sel.sum())<1: sel=np.ones(len(pix_idx),bool)
    sub=np.asarray(pixraw)[:, sel].astype(np.float32); T=len(sub)
    if kind=="raw": return sub.mean(1).astype(np.float64)
    if kind=="ftf":
        f=np.zeros(T); f[1:]=np.abs(np.diff(sub,axis=0)).mean(1); return f.astype(np.float64)
    if k is None: _,k=plori_signal(sub,fps)
    return np.abs(sub-median_filter(sub,(int(k)|1,1),mode="reflect")).mean(1).astype(np.float64)
def _cb(y,a,lo,lv):
    for j in range(a,lo,-1):
        if y[j]>=lv>y[j-1]: return (j-1)+(lv-y[j-1])/(y[j]-y[j-1])
    return float(lo)
def _ca(y,a,hi,lv):
    for j in range(a,hi):
        if y[j]>=lv>y[j+1]: return j+(y[j]-lv)/(y[j]-y[j+1])
    return float(hi)
def _cv(x): x=np.asarray(x,float); return 100*np.std(x)/np.mean(x) if len(x)>1 and np.mean(x) else float("nan")
def _locmed(y,c,w=2): c=int(np.clip(round(c),0,len(y)-1)); a=max(0,c-w); b=min(len(y),c+w+1); return float(np.median(y[a:b]))  # local median at a split point (±w frames; center is clamped in-range to avoid an empty slice for edge beats)
def beat_metrics(y, fps, split_frac=0.6):
    """Peaks (raw signal, autocorr distance, prominence) + per-beat CD50(FWHM), CT(onset->peak)
    and RT(peak->offset) at the 10% level, amplitude. BPM from peak intervals (robust);
    autocorr BPM as cross-check. QC flags: low_signal ("failure suspected") / irregular_rhythm /
    period_lock / amp_var / rest_nonreturn. Also returns `snr` = (p95 - median)/MAD of the signal.

    The per-beat baseline is the local median at a split point (split_frac), rather than the minimum
    of a half peak-to-peak window (which is fragile to a single noisy sample). Because contraction
    rises sharply while relaxation is gradual, the relaxation trough sits later than the midpoint
    between peaks (split_frac>0.5). The split points s_pre=peak−(1−f)·pre_interval and
    s_post=peak+f·post_interval (edge beats substitute the median interval) define the baseline via
    the median in their neighborhood, and onset/offset are found by the 10% crossings within
    [s_pre,s_post]. This markedly improves the beat-to-beat consistency of the onset/offset times
    (CT, RT), while CD50 (the 50% level) is essentially unchanged."""
    T=len(y); p_ac=autocorr_period(y,fps); rng=np.percentile(y,95)-np.percentile(y,5)
    pk,_=find_peaks(y,prominence=0.3*max(rng,1e-9),distance=max(2,int(0.5*p_ac)))
    ivl=np.diff(pk)/fps; bpm_pk=60/np.median(ivl); bpm_ac=fps/p_ac*60
    Pmed=float(np.median(np.diff(pk))) if len(pk)>1 else p_ac
    dd=np.gradient(median_filter(y,_odd(0.12*p_ac,T)))*fps
    cd50=[];ct=[];rt=[];amp=[];ons=[];offs=[];on50=[];off50=[];lv50=[];bnd=0
    for i,apx in enumerate(pk):
        pre=(pk[i]-pk[i-1]) if i>0 else Pmed; post=(pk[i+1]-pk[i]) if i+1<len(pk) else Pmed
        s_pre=apx-(1-split_frac)*pre; s_post=apx+split_frac*post          # split-point baseline (replaces the .min() approach)
        lo=int(np.clip(round(s_pre),0,T-1)); hi=int(np.clip(round(s_post),0,T-1))
        if lo>=apx or hi<=apx: continue
        bpre=_locmed(y,s_pre); bpost=_locmed(y,s_post); base=min(bpre,bpost); A=y[apx]-base
        if A<=0: continue
        amp.append(A); l50=base+0.5*A; c50a=_cb(y,apx,lo,l50); c50b=_ca(y,apx,hi,l50)
        cd50.append((c50b-c50a)/fps*1000); on50.append(c50a); off50.append(c50b); lv50.append(l50)
        on=_cb(y,apx,lo,bpre+0.1*(y[apx]-bpre)); off=_ca(y,apx,hi,bpost+0.1*(y[apx]-bpost)); ons.append(on); offs.append(off)
        ct.append((apx-on)/fps*1000); rt.append((off-apx)/fps*1000)
        if on<=lo or off>=hi: bnd+=1
    ampCV=_cv(amp); icv=_cv(ivl*1000); amp_med=float(np.median(amp)) if len(amp) else 0.0
    mad=np.median(np.abs(y-np.median(y)))*1.4826                          # robust noise scale
    snr=float((np.percentile(y,95)-np.median(y))/mad) if mad>0 else float("inf")
    low_signal=(amp_med<QC_AMP_MIN) and (snr<QC_SNR_MIN)                  # BOTH detection axes at floor → "failure suspected"
    irregular=(icv>QC_ICV_MAX) and not low_signal                        # high interval variability among DETECTABLE beats only
    flags=[f for f,c in [("low_signal",low_signal),("irregular_rhythm",irregular),
                         ("period_lock",abs(bpm_ac/bpm_pk-1)>0.2),("amp_var",ampCV>50),
                         ("rest_nonreturn",bnd/max(1,len(pk))>0.3)] if c]
    return dict(pk=pk,ons=np.array(ons),offs=np.array(offs),on50=np.array(on50),off50=np.array(off50),lv50=np.array(lv50),dd=dd,ivl_ms=ivl*1000,
        cd50=np.array(cd50),ct=np.array(ct),rt=np.array(rt),amp=np.array(amp),
        bpm_pk=bpm_pk,bpm_ac=bpm_ac,iCV=icv,cd50_med=np.median(cd50),cd50CV=_cv(cd50),
        ct_med=np.median(ct),ctCV=_cv(ct),rt_med=np.median(rt),rtCV=_cv(rt),
        amp_med=amp_med,ampCV=ampCV,snr=snr,flags="|".join(flags))
def analyze(g, fps, mpp, mag, scale, pixcap=3000, dilate=3, mask_mode="union", save_perpixel=False, dynrange_filter=False, drop_frac=0.25, fill=True):
    """Full PLoRI analysis of a gray stack `g` (T,H,W). Returns a dict ready to
    savez as ploridata.npz: median image, first frame, masks (median/union/last), centroid path,
    PLoRI signal, beat metrics, and organoid area(mm²)/equiv-diam(µm)/opacity/drift + video meta.

    mask_mode selects the SIGNAL mask: 'union' (default) = OR of Otsu over ~15 evenly-spaced frames
    (each frame sharp; covers the drift-swept path incl. protrusions); 'median' = Otsu on the temporal-
    median image (drift-averaged core; smears/under-segments under drift). Union is the default because
    it gives equal or better beat-to-beat CD50/CT/RT consistency at the same rhythm/rest noise, and the
    drift-crossing noise one might fear from a union mask is removed by the per-pixel moving reference.

    save_perpixel=True stores the RAW per-pixel intensities `pixraw` (T,N uint8) + `pix_idx`, computed on
    the UNION superset (pixcapped) → any sub-mask signal is derivable offline WITHOUT the video via
    derive(): PLoRI (any mask), ftf/speed (|I(t)−I(t−1)|), MM fixed-ref, raw, regional. uint8
    is smaller than the rectified R and compresses well."""
    T,H,W=g.shape; med=np.median(g,0)
    mask_median=segment(med,dilate)
    px=mpp/scale; union=np.zeros((H,W),bool); frame_diam=[]
    for i in range(0,T,max(1,T//15)):
        mi=segment(g[i],dilate,fill=fill); union|=mi             # fill: close bright-interior holes → union signal mask + size both use the footprint
        frame_diam.append(2*math.sqrt(mi.sum()*px*px/math.pi))   # per-frame equiv-diameter (µm); union OVER-estimates size under drift
    if fill: union=binary_fill_holes(union)                      # also close holes enclosed only in the OR union (per-frame open bays that the moving rim seals)
    mask_last=segment(g[-1],dilate,fill=fill)
    mask = union if mask_mode=="union" else (segment(med,dilate,fill=True) if fill else mask_median)  # signal mask honours --fill in median mode too; mask_median stays unfilled for opacity
    gf=g.reshape(T,-1)
    sig_mask = _dynamic_range_mask(gf, mask, drop_frac) if dynrange_filter else mask   # dynamic-range filter: drop the narrow-spread (p95−p50) lowest fraction
    pixraw=pix_idx=None
    if save_perpixel:                                                 # sample points = 2D grid over union∩active → store raw
        idxU=_grid_subsample(sig_mask, pixcap)                        # union → 2D grid → dynamic-range filter (removes non-signal pixels)
        gfmU=gf[:,idxU].astype(np.float32)
        R,k=plori_perpixel(gfmU,fps); pix_idx=idxU
        y=aggregate_for(R,pix_idx,sig_mask)                           # aggregate = mean over the active grid points
        pixraw=np.clip(gfmU,0,255).astype(np.uint8)                  # store original intensities (uint8) → derive PLoRI/ftf/MM
    else:
        idx=_grid_subsample(sig_mask, pixcap)                        # union → 2D grid → dynamic-range filter (removes non-signal pixels)
        y,k=plori_signal(gf[:,idx].astype(np.float32),fps)
    m=beat_metrics(y,fps)
    size=float(np.median(frame_diam)); area_mm2=math.pi*(size/2)**2/1e6   # size = median of per-frame equiv-diameters (drift/phase-robust; not the drift-swept union)
    opac=100*(med[~union].mean()-med[mask_median].mean())/med[~union].mean()   # opacity is measured against the fixed median core
    cs=np.array([np.array(np.where(segment(g[i],dilate))).mean(1)[::-1] for i in range(0,T,max(1,T//25))])
    drift=float(np.linalg.norm(cs[-1]-cs[0])*px)
    out=dict(med=med.astype(np.uint8),frame0=np.clip(g[0],0,255).astype(np.uint8),
        mask=mask,mask_median=mask_median,mask_union=union,mask_last=mask_last,mask_mode=mask_mode,
        mask_active=sig_mask,dynrange_filter=bool(dynrange_filter),drop_frac=float(drop_frac) if dynrange_filter else 0.0,  # dynamic-range-filtered (p95−p50) active mask
        cs=cs,plori=y,fps=fps,k=k,
        area_mm2=area_mm2,size_um=size,opacity=opac,drift_um=drift,area_frac=mask.sum()/(H*W),fill=bool(fill),
        mag=mag,mpp=mpp,fullres=f"{int(round(W/scale))}x{int(round(H/scale))}",dur=T/fps,nframes=T,
        opacity_class=("opaque" if opac>80 else "translucent"))
    if pixraw is not None: out.update(pixraw=pixraw, pix_idx=pix_idx.astype(np.int32))
    out.update(m)
    if drift > 0.15 * size:                                  # high_drift: 6th QC flag (same threshold as plori_report's HIGH DRIFT badge, dr>0.15*size_um)
        out["flags"] = f"{out['flags']}|high_drift" if out["flags"] else "high_drift"
    return out
