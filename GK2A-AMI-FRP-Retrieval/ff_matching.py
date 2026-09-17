"""
Parallax-aware multi-buffer matching between the GK2A FF product and the
KFS-confirmed fire events (PRIMARY ground truth), on the common 2-km FF grid.
VIIRS active fire is used as SECONDARY satellite confirmation (a covariate /
candidate support / tie-breaker), not as the yardstick.

Product EVALUATION only (Fig. 3 stage 3):
  * match_at_buffer / confusion_curve  -- POD / FAR vs matching tolerance
    (0 / 2 / 4 / 6 km): separates genuine FF error from geolocation / footprint
    mismatch.
  * confusion_map      -- per-cell region (TN/TP/FN/FP) for tolerance & maps.
  * candidate_labels   -- 3-class model target on candidate event-cells only
    (KFS-fire u FF-flag); VIIRS may be passed to confirm/extend candidates.
  * min_match_buffer + parallax_attribution + displacement_risk_km -- the
    cross-check: do buffered matches concentrate where the predicted displacement
    (tan(VZA) x height) is large?  terrain relief -> fire collocation;
    cloud-top height -> cloud-mask displacement.

`truth` here is the KFS-confirmed fire field (gridded to the FF grid).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation

import ff_config as C

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from parallax import parallax_offset
    _HAVE_PARALLAX = True
except Exception:
    _HAVE_PARALLAX = False


def _disk(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    if r < 1:
        return np.array([[True]])
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= (radius_px * radius_px)


def match_at_buffer(ff_flag, truth, buffer_km, res_km=C.GRID_RES_KM) -> dict:
    """Confuse FF against KFS truth allowing tolerance `buffer_km`."""
    ff = np.asarray(ff_flag, dtype=bool)
    tr = np.asarray(truth, dtype=bool)
    se = _disk(buffer_km / res_km)
    truth_within = binary_dilation(tr, se)
    ff_within    = binary_dilation(ff, se)

    tp_mask = ff & truth_within
    fp_mask = ff & ~truth_within
    fn_mask = tr & ~ff_within

    n_truth = int(tr.sum())
    n_truth_hit = int((tr & ff_within).sum())
    tp, fp, fn = int(tp_mask.sum()), int(fp_mask.sum()), int(fn_mask.sum())
    pod = n_truth_hit / n_truth if n_truth else float("nan")
    far = fp / (tp + fp) if (tp + fp) else float("nan")
    return dict(buffer_km=buffer_km, POD=pod, FAR=far, TP=tp, FP=fp, FN=fn,
                tp_mask=tp_mask, fp_mask=fp_mask, fn_mask=fn_mask)


def confusion_curve(ff_flag, truth, buffers=C.MATCH_BUFFERS_KM, res_km=C.GRID_RES_KM):
    return [match_at_buffer(ff_flag, truth, b, res_km) for b in buffers]


def confusion_map(ff_flag, truth, buffer_km=C.OPERATING_BUFFER_KM,
                  res_km=C.GRID_RES_KM) -> np.ndarray:
    """Per-cell confusion region: 0 TN, 1 TP, 2 FN, 3 FP (CONFUSION_CLASSES)."""
    m = match_at_buffer(ff_flag, truth, buffer_km, res_km)
    cm = np.zeros(np.shape(ff_flag), dtype=np.int64)
    cm[m["tp_mask"]] = 1
    cm[m["fn_mask"]] = 2
    cm[m["fp_mask"]] = 3
    return cm


def viirs_support_mask(viirs, buffer_km=C.OPERATING_BUFFER_KM, res_km=C.GRID_RES_KM):
    """SECONDARY confirmation field: cells within `buffer_km` of a VIIRS active-fire
    detection. Used to support / extend candidate cells (Fig. 3 stage 4), never as
    the primary yardstick. Returns a bool grid (all-False if viirs is None)."""
    if viirs is None:
        return None
    return binary_dilation(np.asarray(viirs, bool), _disk(buffer_km / res_km))


def candidate_labels(ff_flag, truth, buffer_km=C.OPERATING_BUFFER_KM,
                     res_km=C.GRID_RES_KM, viirs=None) -> np.ndarray:
    """
    3-class model target on candidate event-cells only (KFS-fire u FF-flag):
        TP_hit=0, FN_miss=1, FP_false_alarm=2 ; non-candidate cells = -1.

    If `viirs` (secondary) is supplied, FF-flag cells confirmed by a nearby VIIRS
    detection are kept as candidates even when they fall just outside the KFS
    polygon at this buffer (they remain FP unless inside KFS truth) -- i.e. VIIRS
    only ever *supports* a candidate, the KFS field still decides hit/miss.
    """
    cm = confusion_map(ff_flag, truth, buffer_km, res_km)
    lab = np.full(cm.shape, -1, dtype=np.int64)
    for conf, cls in C.CONFUSION_TO_CLASS.items():
        lab[cm == conf] = cls
    sup = viirs_support_mask(viirs, buffer_km, res_km)
    if sup is not None:
        # ensure VIIRS-confirmed FF flags are present as candidates (FP by default)
        ff = np.asarray(ff_flag, bool)
        add = ff & sup & (lab < 0)
        lab[add] = C.CONFUSION_TO_CLASS[3]   # FP_false_alarm
    return lab


def min_match_buffer(ff_flag, truth, buffers=C.MATCH_BUFFERS_KM, res_km=C.GRID_RES_KM):
    """Smallest buffer at which each true-fire pixel first matches an FF flag."""
    tr = np.asarray(truth, dtype=bool)
    ff = np.asarray(ff_flag, dtype=bool)
    out = np.full(tr.shape, np.inf, dtype=np.float64)
    for b in sorted(buffers):
        within = binary_dilation(ff, _disk(b / res_km))
        newly = tr & within & ~np.isfinite(out)
        out[newly] = b
    out[~tr] = np.nan
    return out


def displacement_risk_km(lat, lon, height_km) -> np.ndarray:
    """
    Predicted displacement magnitude (km) = |parallax offset| for the given height.
    Pass terrain relief for fire collocation, or cloud-top height for cloud-mask
    displacement. Requires the parent parallax module.
    """
    if not _HAVE_PARALLAX:
        raise RuntimeError("parallax.py not importable; check package layout.")
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    dlat, dlon = parallax_offset(lat, lon, height_km, C.SUB_LON)
    return np.sqrt((dlat * 111.32) ** 2 + (dlon * 111.32 * np.cos(np.radians(lat))) ** 2)

# backwards-compatible name
predicted_parallax_km = displacement_risk_km


def parallax_attribution(min_buffer_map, displacement_km) -> dict:
    """Spearman(min-match-buffer, predicted displacement) over true-fire pixels."""
    from scipy.stats import spearmanr
    tr = np.isfinite(min_buffer_map)
    mb = min_buffer_map[tr]; px = np.asarray(displacement_km)[tr]
    ok = np.isfinite(mb) & np.isfinite(px)
    if ok.sum() < 3:
        return dict(rho=float("nan"), p=float("nan"),
                    disp_exact=float("nan"), disp_buffered=float("nan"), n=int(ok.sum()))
    rho, p = spearmanr(mb[ok], px[ok])
    exact = mb[ok] <= 0
    return dict(rho=float(rho), p=float(p),
                disp_exact=float(np.nanmean(px[ok][exact])) if exact.any() else float("nan"),
                disp_buffered=float(np.nanmean(px[ok][~exact])) if (~exact).any() else float("nan"),
                n=int(ok.sum()))


if __name__ == "__main__":
    H = W = 40
    truth = np.zeros((H, W), bool); truth[20:23, 20:23] = True     # KFS-confirmed fire
    ff = np.zeros((H, W), bool); ff[22:25, 22:25] = True; ff[5, 30] = True
    viirs = np.zeros((H, W), bool); viirs[5, 30] = True            # VIIRS confirms the lone flag
    print("confusion vs matching tolerance:")
    for m in confusion_curve(ff, truth):
        print(f"  buffer {m['buffer_km']:.0f} km | POD {m['POD']:.2f}  FAR {m['FAR']:.2f} "
              f"| TP {m['TP']}  FP {m['FP']}  FN {m['FN']}")
    lab = candidate_labels(ff, truth, 2.0, viirs=viirs)
    u, c = np.unique(lab[lab >= 0], return_counts=True)
    print("candidate labels @2 km (VIIRS-supported):",
          dict(zip([C.OUTCOME_CLASSES[i] for i in u], c)))
