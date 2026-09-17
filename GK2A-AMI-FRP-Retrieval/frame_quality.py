"""
frame_quality.py -- detect AOI-wide artifact frames in the GK2A FF product.

Some GK2A FF scans flag hundreds-to-thousands of cells across the whole AOI in a
single 10-min frame -- a diurnal (~24 h / 144-frame) sensor/illumination artifact,
NOT fire. Left in, each such frame injects ~thousands of spurious FP_false_alarm
labels (Sancheong: 7 frames x ~1600 cells ~= 11k false FP).

The test is physical and uses BOTH conditions, so it never removes a genuine
large-fire scan:
  (a) the frame flags far more cells than the fire could occupy
        ff[t].sum() > max(abs_floor, k * truth_cells_over_window)
  (b) those flags are spatially diffuse -- mostly OFF the burn
        off_burn_fraction(t) > off_frac    (flags outside truth+buffer)

Validated separation on the 2025 events:
  Sancheong artifact frames: ~1660 flags, off-burn frac 0.98  -> dropped
  Uiseong real mega-fire    : ~100-270 flags, off-burn frac 0.03-0.11 -> KEPT
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation


def artifact_frames(ff_flag, truth_2d, k=3.0, abs_floor=50, off_frac=0.8,
                    buffer_cells=1):
    """-> boolean mask [T] True where frame t is an AOI-wide artifact to DROP.

    ff_flag    [T,H,W] bool/0-1 product flags
    truth_2d   [H,W]   bool KFS burned-area truth (union over frames)
    k          flag-count multiple of truth size that counts as 'too many'
    abs_floor  minimum absolute flag count to ever consider a frame suspect
    off_frac   minimum fraction of flags OFF the burn to call it diffuse
    buffer_cells  dilate truth by this many cells before measuring 'on-burn'
    """
    ff = np.asarray(ff_flag).astype(bool)
    T = ff.shape[0]
    truth = np.asarray(truth_2d).astype(bool)
    tb = binary_dilation(truth, iterations=int(buffer_cells)) if buffer_cells else truth
    n_truth = int(truth.sum())
    thresh = max(abs_floor, k * max(n_truth, 1))
    bad = np.zeros(T, bool)
    for t in range(T):
        tot = int(ff[t].sum())
        if tot <= thresh:
            continue                                   # not enough flags to be suspect
        on = int((ff[t] & tb).sum())
        off_fraction = 1.0 - on / max(tot, 1)
        if off_fraction > off_frac:                    # diffuse + over-flagged -> artifact
            bad[t] = True
    return bad


def report(ff_flag, truth_2d, **kw):
    """Print a per-frame summary of the artifact frames found (for auditing)."""
    bad = artifact_frames(ff_flag, truth_2d, **kw)
    ff = np.asarray(ff_flag).astype(bool)
    idx = np.where(bad)[0]
    if not len(idx):
        print("  no artifact frames"); return bad
    for t in idx:
        tot = int(ff[t].sum())
        print(f"  DROP frame {t:4d}: {tot} flags")
    if len(idx) > 1:
        print(f"  ({len(idx)} frames; index spacing {np.diff(idx).tolist()})")
    return bad