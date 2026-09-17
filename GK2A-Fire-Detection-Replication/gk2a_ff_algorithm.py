# -*- coding: utf-8 -*-
"""
GK2A FF replication helper - FIXED VERSION

Key fixes vs the previous version (these are why detections were too few)
-------------------------------------------------------------------------
FIX 1 (largest impact): Context-test RMSD is now computed around the LOCAL
       MEDIAN of valid neighbours, NOT against the base-plane. Computing RMSD
       against the base-plane inflated the denominator near fires (because the
       base-plane difference already contains the thermal anomaly), which made
       the normalized context tests far too strict and suppressed real fires.
       This matches the MODIS / GK2A contextual definition (Giglio et al. 2003).
       A toggle `rmsd_mode` is provided so you can compare both definitions.

FIX 2: The base-plane now EXCLUDES obviously hot pixels before median
       filtering, so a large multi-pixel fire does not raise its own
       background and partially mask itself.

FIX 3: When computing context-window statistics, background-fire pixels
       (potential-fire candidates and obviously hot pixels) are EXCLUDED from
       the neighbour set, exactly as MODIS excludes "background fires" from
       the mean/MAD. This stops clustered fires from cancelling each other.

FIX 4: Diagnostic counters report how many candidates are dropped at each
       stage (potential-fire, validity gate, context test, rejections), so
       you can see precisely where pixels are lost. Enable with verbose=True.

FIX 5: Candidates dropped at the validity gate are now flagged in DQF
       (DQF.PROB_CLOUD) instead of silently vanishing, so the DQF map and the
       fire list stay consistent.

Inputs are brightness temperatures / reflectances already converted from GK2A
L1B. This function does not read raw DN values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
import warnings

import numpy as np
from scipy.ndimage import generic_filter
from scipy.stats import linregress


class DQF:
    OUT_OF_RANGE = 0
    INVALID = 1
    LAND = 2
    WATER = 3
    CLOUD = 4
    CLOUD_REJ = 5
    URBAN_REJ = 6
    POT_FIRE = 7
    FIRE = 8
    ABS_FIRE = 9
    IND_HEAT = 10
    STABILITY = 12
    PROB_CLOUD = 13


DQF_DESCRIPTIONS = {
    DQF.OUT_OF_RANGE: "Invalid - outside observation range (VZA > 70 deg)",
    DQF.INVALID: "Invalid - masked area or missing input data",
    DQF.LAND: "Land",
    DQF.WATER: "Water",
    DQF.CLOUD: "Cloud",
    DQF.CLOUD_REJ: "Rejected by cloud test",
    DQF.URBAN_REJ: "Rejected by baresoil / urban / coastal test",
    DQF.POT_FIRE: "Potential fire",
    DQF.FIRE: "Fire",
    DQF.ABS_FIRE: "Absolute fire",
    DQF.IND_HEAT: "Industrial heat detection",
    DQF.STABILITY: "Stability test",
    DQF.PROB_CLOUD: "Dropped (insufficient valid background)",
}


@dataclass
class FFThresholds:
    # Geometry
    sza_day_night: float = 85.0
    vza_valid: float = 70.0

    # Topography
    topo_ref_height: float = 100.0
    topo_lapse_swir: float = -7.0e-3  # K/m
    topo_lapse_tir: float = -6.0e-3   # K/m
    topo_pair_dist_min_m: float = 200000.0
    topo_pair_dist_max_m: float = 400000.0
    topo_outlier_sigma: float = 2.58

    # Base-plane (15x15 median)
    base_window: int = 15
    # FIX 2: exclude pixels hotter than this percentile of clear-land SWIR
    # from the base-plane median, so big fires do not raise their own background.
    base_hot_percentile: float = 98.0

    # Absolute fire
    abs_fire_day: float = 350.0
    abs_fire_night: float = 320.0

    # Potential fire
    pot_swir_dev: float = 2.0
    pot_dt_dev: float = 2.0
    pot_nir_refl: float = 0.35

    # Context test (day)
    ctx_day_alpha: float = 2.5
    ctx_day_beta: float = 6.3
    ctx_day_gamma: float = 4.0
    ctx_day_tau: float = 2.5
    # Context test (night)
    ctx_night_alpha: float = 2.0
    ctx_night_beta: float = 4.0
    ctx_night_gamma: float = 2.0
    ctx_night_tau: float = 2.0

    # Context window: 7x7 start, 15x15 max
    ctx_half_init: int = 3
    ctx_half_max: int = 7
    ctx_min_valid: int = 8
    ctx_min_prop: float = 0.25

    # Background-fire exclusion (FIX 3): pixels hotter than the local median
    # by more than this many K are treated as background fires and excluded
    # from the neighbour statistics.
    bg_fire_excess_swir: float = 8.0

    # Cloud rejection: if cloud fraction > 0.1, use 15x15 minus inner 7x7
    cloud_fraction_threshold: float = 0.10
    cloud_rej_half: int = 7
    cloud_rej_inner_half: int = 3

    # Optional urban / baresoil / coastal rejection (theoretical branch)
    fire_fraction_threshold: float = 0.10
    urb_stage1_alpha: float = 12.5
    urb_stage1_beta: float = 15.5
    urb_stage2_ndvi: float = -3.0
    urb_stage2_nbr: float = -2.0

    # RMSD definition for the context test:
    #   "local_median" (FIX 1, correct, default) -> RMSD about the local median
    #   "base_plane"   (old behaviour)            -> RMSD about the base-plane
    rmsd_mode: str = "local_median"


THRESHOLDS = FFThresholds()


@dataclass
class FFResult:
    FF: np.ndarray
    DQF_FF: np.ndarray
    T_SWIR_c: np.ndarray
    T_TIR_c: np.ndarray
    delta_T: np.ndarray
    base_SWIR: np.ndarray
    base_dT: np.ndarray
    lapse_SWIR: float
    lapse_TIR: float


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _as_float(a):
    return np.asarray(a, dtype=np.float64)


def _as_bool(a):
    return np.asarray(a, dtype=bool)


def _check_shape(name0, arr0, **others):
    shape = np.shape(arr0)
    for name, arr in others.items():
        if arr is None:
            continue
        if np.shape(arr) != shape:
            raise ValueError(
                f"{name} shape {np.shape(arr)} does not match "
                f"{name0} shape {shape}")


def _nanmedian_filter(image: np.ndarray, size: int) -> np.ndarray:
    def _nanmedian(values):
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan
        return float(np.median(values))

    return generic_filter(image, _nanmedian, size=size, mode="nearest")


def _estimate_lapse_rate(
    bt: np.ndarray,
    dem: np.ndarray,
    cloud_mask: np.ndarray,
    ref_rows: Sequence[int],
    ref_cols: Sequence[int],
    pixel_size_m: float,
    T: FFThresholds,
) -> float:
    """
    Estimate scene lapse rate using the ATBD pair idea:
    reference pixels + random neighbours at 200-400 km.
    """
    h, w = bt.shape
    min_px = T.topo_pair_dist_min_m / pixel_size_m
    max_px = T.topo_pair_dist_max_m / pixel_size_m
    rng = np.random.default_rng(0)

    dz_list = []
    dt_list = []

    for r0, c0 in zip(ref_rows, ref_cols):
        if r0 < 0 or r0 >= h or c0 < 0 or c0 >= w:
            continue
        if (cloud_mask[r0, c0] or not np.isfinite(bt[r0, c0])
                or not np.isfinite(dem[r0, c0])):
            continue

        ang = rng.uniform(0.0, 2.0 * np.pi, 400)
        rad = rng.uniform(min_px, max_px, 400)
        rr = (r0 + rad * np.sin(ang)).astype(int)
        cc = (c0 + rad * np.cos(ang)).astype(int)

        # Filter to in-bounds pixels FIRST, then index the arrays.
        in_bounds = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr = rr[in_bounds]
        cc = cc[in_bounds]
        if rr.size < 2:
            continue

        ok = (
            (~cloud_mask[rr, cc]) &
            np.isfinite(bt[rr, cc]) &
            np.isfinite(dem[rr, cc])
        )
        rr = rr[ok]
        cc = cc[ok]
        if rr.size < 2:
            continue

        use_n = min(100, rr.size)
        idx = rng.choice(rr.size, use_n, replace=False)
        dz_list.extend((dem[rr[idx], cc[idx]] - dem[r0, c0]).tolist())
        dt_list.extend((bt[rr[idx], cc[idx]] - bt[r0, c0]).tolist())

    if len(dz_list) < 10:
        return np.nan

    dz = np.asarray(dz_list, dtype=np.float64)
    dt = np.asarray(dt_list, dtype=np.float64)

    slope0, intercept0 = linregress(dz, dt)[:2]
    resid = dt - (slope0 * dz + intercept0)
    sigma = np.nanstd(resid)
    if not np.isfinite(sigma) or sigma == 0.0:
        return np.nan

    keep = np.abs(resid) < (T.topo_outlier_sigma * sigma)
    if keep.sum() < 5:
        return np.nan

    slope = linregress(dz[keep], dt[keep])[0]
    return float(slope)


def _apply_topo_correction(bt, dem, lapse, ref_h):
    return bt - lapse * (dem - ref_h)


def _window_stats(
    field: np.ndarray,
    valid_mask: np.ndarray,
    row: int,
    col: int,
    half: int,
    base_field: Optional[np.ndarray] = None,
    rmsd_mode: str = "local_median",
    bg_exclude_excess: Optional[float] = None,
    exclude_inner_half: Optional[int] = None,
):
    """
    Compute background median and RMSD of `field` in a (2*half+1)^2 window.

    median : median of valid neighbour values (centre excluded)

    rmsd   : depends on rmsd_mode:
               "local_median" (FIX 1) -> sqrt(mean((field - median)^2))
                                          over valid neighbours
               "base_plane"            -> sqrt(mean((field - base_field)^2))
                                          over valid neighbours
                                          (requires base_field)

    Background-fire exclusion (FIX 3): if bg_exclude_excess is given, neighbour
    pixels hotter than (local median + bg_exclude_excess) are dropped before
    computing the statistics. This mimics the MODIS exclusion of background
    fire pixels from the mean / MAD.

    Returns
    -------
    median, rmsd, n_valid
        rmsd clipped to >= 1e-6 to avoid divide-by-zero.
    """
    h, w = field.shape
    r0 = max(0, row - half)
    r1 = min(h, row + half + 1)
    c0 = max(0, col - half)
    c1 = min(w, col + half + 1)

    patch = field[r0:r1, c0:c1]
    vmask = valid_mask[r0:r1, c0:c1].copy()

    cr = row - r0
    cc = col - c0
    if 0 <= cr < vmask.shape[0] and 0 <= cc < vmask.shape[1]:
        vmask[cr, cc] = False  # exclude the candidate pixel itself

    if exclude_inner_half is not None:
        ir0 = max(0, cr - exclude_inner_half)
        ir1 = min(vmask.shape[0], cr + exclude_inner_half + 1)
        ic0 = max(0, cc - exclude_inner_half)
        ic1 = min(vmask.shape[1], cc + exclude_inner_half + 1)
        vmask[ir0:ir1, ic0:ic1] = False

    finite = np.isfinite(patch)
    use = vmask & finite
    if int(use.sum()) == 0:
        return np.nan, np.nan, 0

    vals = patch[use]
    med = float(np.median(vals))

    # FIX 3: drop background-fire neighbours before computing statistics
    if bg_exclude_excess is not None:
        keep = vals <= (med + bg_exclude_excess)
        if keep.sum() >= 1:
            vals = vals[keep]
            med = float(np.median(vals))

    n = int(vals.size)
    if n == 0:
        return np.nan, np.nan, 0

    if rmsd_mode == "base_plane":
        if base_field is None:
            raise ValueError("base_field required for rmsd_mode='base_plane'")
        patch_base = base_field[r0:r1, c0:c1]
        finite_b = np.isfinite(patch) & np.isfinite(patch_base)
        use_b = vmask & finite_b
        diffs = (patch - patch_base)[use_b]
        if diffs.size == 0:
            return med, 1e-6, n
        rmsd = float(np.sqrt(np.mean(diffs ** 2)))
    else:
        # "local_median" (correct default)
        rmsd = float(np.sqrt(np.mean((vals - med) ** 2)))

    return med, max(rmsd, 1e-6), n


def make_fire_report(ff: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Return a list of (row, col, lat, lon) for detected fire pixels."""
    rows, cols = np.where(ff == 1)
    out = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        out.append((r, c, float(lat[r, c]), float(lon[r, c])))
    return out


# -----------------------------------------------------------------------------
# Main algorithm
# -----------------------------------------------------------------------------

def run_ff_algorithm(
    T_SWIR,
    T_TIR,
    refl_red,
    refl_nir,
    sza,
    vza,
    land_mask,
    cloud_mask,
    dem,
    prev_dqf: Optional[np.ndarray] = None,
    industrial_mask: Optional[np.ndarray] = None,
    baresoil_mask: Optional[np.ndarray] = None,
    topo_ref_rows: Optional[Sequence[int]] = None,
    topo_ref_cols: Optional[Sequence[int]] = None,
    pixel_size_m: float = 2000.0,
    refl_138: Optional[np.ndarray] = None,
    use_theoretical_urban_rejection: bool = False,
    thresholds: FFThresholds = THRESHOLDS,
    verbose: bool = True,
) -> FFResult:
    """
    Parameters
    ----------
    T_SWIR : Ch07 BT [K]
    T_TIR  : Ch14 BT [K]
    refl_red : Ch03 TOA reflectance [0-1]
    refl_nir : Ch04 TOA reflectance [0-1]
    sza, vza : solar / view zenith angle [degree]
    land_mask : True for land
    cloud_mask : True for cloud
    dem : DEM [m]

    Optional
    --------
    prev_dqf : previous DQF_FF image for the stability test
    industrial_mask : boolean mask for industrial heat pixels on your grid
    baresoil_mask : boolean mask of bare soil / urban / cropland pixels
    topo_ref_rows, topo_ref_cols : reference pixel indices for topographic step
    refl_138 : Ch05 1.38 um reflectance (only if urban rejection enabled)
    use_theoretical_urban_rejection : enable NDVI/NBR urban rejection branch
    thresholds : FFThresholds instance
    verbose : print stage-by-stage candidate accounting (FIX 4)
    """
    T = thresholds

    T_SWIR = _as_float(T_SWIR)
    T_TIR = _as_float(T_TIR)
    refl_red = _as_float(refl_red)
    refl_nir = _as_float(refl_nir)
    sza = _as_float(sza)
    vza = _as_float(vza)
    dem = _as_float(dem)
    land_mask = _as_bool(land_mask)
    cloud_mask = _as_bool(cloud_mask)

    _check_shape(
        "T_SWIR", T_SWIR,
        T_TIR=T_TIR, refl_red=refl_red, refl_nir=refl_nir,
        sza=sza, vza=vza, dem=dem,
        land_mask=land_mask, cloud_mask=cloud_mask,
        prev_dqf=prev_dqf, industrial_mask=industrial_mask,
        baresoil_mask=baresoil_mask, refl_138=refl_138,
    )

    h, w = T_SWIR.shape

    if industrial_mask is None:
        industrial_mask = np.zeros((h, w), dtype=bool)
    else:
        industrial_mask = _as_bool(industrial_mask)

    if baresoil_mask is None:
        baresoil_mask = np.zeros((h, w), dtype=bool)
    else:
        baresoil_mask = _as_bool(baresoil_mask)

    if use_theoretical_urban_rejection:
        if refl_138 is None:
            raise ValueError(
                "refl_138 is required when "
                "use_theoretical_urban_rejection=True")
        refl_138 = _as_float(refl_138)

    # ------------------------------------------------------------------
    # DQF initialization
    # ------------------------------------------------------------------
    DQF_arr = np.full((h, w), DQF.INVALID, dtype=np.int16)

    numeric_missing = (
        ~np.isfinite(T_SWIR) | ~np.isfinite(T_TIR) |
        ~np.isfinite(refl_red) | ~np.isfinite(refl_nir) |
        ~np.isfinite(sza) | ~np.isfinite(vza) | ~np.isfinite(dem)
    )

    out_of_range = vza > T.vza_valid
    valid_obs = (~numeric_missing) & (~out_of_range)

    DQF_arr[out_of_range] = DQF.OUT_OF_RANGE

    water_px = valid_obs & (~land_mask)
    cloud_px = valid_obs & land_mask & cloud_mask
    bare_px = valid_obs & land_mask & (~cloud_mask) & baresoil_mask
    land_px = valid_obs & land_mask & (~cloud_mask) & (~baresoil_mask)

    DQF_arr[water_px] = DQF.WATER
    DQF_arr[cloud_px] = DQF.CLOUD
    DQF_arr[bare_px] = DQF.URBAN_REJ
    DQF_arr[land_px] = DQF.LAND

    # Day / night
    is_day = sza < T.sza_day_night
    is_night = ~is_day

    # ------------------------------------------------------------------
    # Topographic correction
    # ------------------------------------------------------------------
    if (topo_ref_rows is not None and topo_ref_cols is not None
            and len(topo_ref_rows) > 0):
        lr_swir = _estimate_lapse_rate(
            T_SWIR, dem, cloud_mask, topo_ref_rows, topo_ref_cols,
            pixel_size_m, T)
        lr_tir = _estimate_lapse_rate(
            T_TIR, dem, cloud_mask, topo_ref_rows, topo_ref_cols,
            pixel_size_m, T)
        if np.isnan(lr_swir):
            warnings.warn("SWIR lapse-rate estimation failed, fallback -7 K/km")
            lr_swir = T.topo_lapse_swir
        if np.isnan(lr_tir):
            warnings.warn("TIR lapse-rate estimation failed, fallback -6 K/km")
            lr_tir = T.topo_lapse_tir
    else:
        warnings.warn(
            "topo_ref_rows/topo_ref_cols not supplied; using -7/-6 K/km "
            "fallback (not the strict ATBD topographic replication).")
        lr_swir = T.topo_lapse_swir
        lr_tir = T.topo_lapse_tir

    T_SWIR_c = _apply_topo_correction(T_SWIR, dem, lr_swir, T.topo_ref_height)
    T_TIR_c = _apply_topo_correction(T_TIR, dem, lr_tir, T.topo_ref_height)
    delta_T = T_SWIR_c - T_TIR_c

    # ------------------------------------------------------------------
    # Base-planes (15x15 median) with hot-pixel exclusion (FIX 2)
    # ------------------------------------------------------------------
    clear_land = land_px.copy()

    if clear_land.sum() > 0:
        hot_thresh = np.nanpercentile(T_SWIR_c[clear_land],
                                      T.base_hot_percentile)
    else:
        hot_thresh = np.inf
    hot_seed = clear_land & (T_SWIR_c > hot_thresh)

    base_input_swir = np.where(clear_land & ~hot_seed, T_SWIR_c, np.nan)
    base_input_dt = np.where(clear_land & ~hot_seed, delta_T, np.nan)

    base_SWIR = _nanmedian_filter(base_input_swir, T.base_window)
    base_dT = _nanmedian_filter(base_input_dt, T.base_window)

    # ------------------------------------------------------------------
    # Absolute fire and potential fire
    # ------------------------------------------------------------------
    abs_fire = (
        (clear_land & is_day & (T_SWIR_c > T.abs_fire_day)) |
        (clear_land & is_night & (T_SWIR_c > T.abs_fire_night))
    )
    DQF_arr[abs_fire] = DQF.ABS_FIRE

    refl_nir = np.clip(refl_nir, 1e-6, 2.0)
    pot_swir = (T_SWIR_c - base_SWIR) > T.pot_swir_dev
    pot_dt = (delta_T - base_dT) > T.pot_dt_dev
    nir_ok = is_night | (is_day & (refl_nir < T.pot_nir_refl))

    pot_fire = clear_land & pot_swir & pot_dt & nir_ok
    DQF_arr[pot_fire & (~abs_fire)] = DQF.POT_FIRE

    # Background-fire mask for window-stat exclusion (FIX 3):
    # candidate / potential-fire pixels and absolute-fire pixels are
    # excluded from neighbour statistics so clustered fires do not cancel.
    bg_fire_mask = pot_fire | abs_fire

    # valid-background mask = clear land AND not a background fire pixel
    bg_valid = clear_land & (~bg_fire_mask)

    # Optional NDVI / NBR branch
    if use_theoretical_urban_rejection:
        refl_red = np.clip(refl_red, 1e-6, 2.0)
        refl_138 = np.clip(refl_138, 1e-6, 2.0)
        NDVI = (refl_nir - refl_red) / (refl_nir + refl_red)
        NBR = (refl_nir - refl_138) / (refl_nir + refl_138)
    else:
        NDVI = NBR = None

    # ------------------------------------------------------------------
    # Context test + cloud rejection + optional daytime urban rejection
    # ------------------------------------------------------------------
    confirmed = abs_fire.copy()
    cand_rows, cand_cols = np.where(pot_fire & (~abs_fire))

    # Diagnostic counters (FIX 4)
    n_cand = len(cand_rows)
    n_drop_validity = 0
    n_drop_context = 0
    n_drop_cloudrej = 0
    n_drop_urban = 0
    n_confirm_context = 0

    for r, c in zip(cand_rows.tolist(), cand_cols.tolist()):
        day = bool(is_day[r, c])
        alpha = T.ctx_day_alpha if day else T.ctx_night_alpha
        beta = T.ctx_day_beta if day else T.ctx_night_beta
        gamma = T.ctx_day_gamma if day else T.ctx_night_gamma
        tau = T.ctx_day_tau if day else T.ctx_night_tau

        # Cloud fraction in the initial 7x7 window (centre excluded)
        rr0 = max(0, r - T.ctx_half_init)
        rr1 = min(h, r + T.ctx_half_init + 1)
        cc0 = max(0, c - T.ctx_half_init)
        cc1 = min(w, c + T.ctx_half_init + 1)
        cloud_patch = cloud_mask[rr0:rr1, cc0:cc1].copy()
        ccr = r - rr0
        ccc = c - cc0
        if 0 <= ccr < cloud_patch.shape[0] and 0 <= ccc < cloud_patch.shape[1]:
            cloud_patch[ccr, ccc] = False
        cloud_fraction = float(cloud_patch.mean()) if cloud_patch.size else 0.0

        used_cloud_rejection = cloud_fraction > T.cloud_fraction_threshold

        if used_cloud_rejection:
            half = T.cloud_rej_half
            med_s, rmsd_s, nv_s = _window_stats(
                T_SWIR_c, bg_valid, r, c, half,
                base_field=base_SWIR, rmsd_mode=T.rmsd_mode,
                bg_exclude_excess=T.bg_fire_excess_swir,
                exclude_inner_half=T.cloud_rej_inner_half,
            )
            med_d, rmsd_d, nv_d = _window_stats(
                delta_T, bg_valid, r, c, half,
                base_field=base_dT, rmsd_mode=T.rmsd_mode,
                bg_exclude_excess=None,
                exclude_inner_half=T.cloud_rej_inner_half,
            )
            n_valid = min(nv_s, nv_d)
            if n_valid == 0 or not np.isfinite(rmsd_s) or not np.isfinite(rmsd_d):
                DQF_arr[r, c] = DQF.CLOUD_REJ
                n_drop_cloudrej += 1
                continue
        else:
            half = T.ctx_half_init
            med_s = rmsd_s = med_d = rmsd_d = np.nan
            n_valid = 0
            passed_validity = False

            while half <= T.ctx_half_max:
                med_s, rmsd_s, nv_s = _window_stats(
                    T_SWIR_c, bg_valid, r, c, half,
                    base_field=base_SWIR, rmsd_mode=T.rmsd_mode,
                    bg_exclude_excess=T.bg_fire_excess_swir)
                med_d, rmsd_d, nv_d = _window_stats(
                    delta_T, bg_valid, r, c, half,
                    base_field=base_dT, rmsd_mode=T.rmsd_mode,
                    bg_exclude_excess=None)
                n_valid = min(nv_s, nv_d)
                win_total = (2 * half + 1) ** 2 - 1
                prop = n_valid / win_total if win_total > 0 else 0.0
                if n_valid >= T.ctx_min_valid and prop > T.ctx_min_prop:
                    passed_validity = True
                    break
                half += 1

            if ((not passed_validity) or n_valid == 0
                    or not np.isfinite(rmsd_s) or not np.isfinite(rmsd_d)):
                # FIX 5: flag instead of silently dropping
                DQF_arr[r, c] = DQF.PROB_CLOUD
                n_drop_validity += 1
                continue

        d_swir = T_SWIR_c[r, c] - med_s
        d_dt = delta_T[r, c] - med_d

        pass_context = (
            (d_swir / rmsd_s > alpha) and
            (d_dt / rmsd_d > beta) and
            (d_swir > gamma) and
            (d_dt > tau)
        )
        if not pass_context:
            if used_cloud_rejection:
                DQF_arr[r, c] = DQF.CLOUD_REJ
                n_drop_cloudrej += 1
            else:
                n_drop_context += 1
            continue

        # Optional daytime urban / baresoil / coastal rejection
        if day and use_theoretical_urban_rejection:
            ur0 = max(0, r - T.ctx_half_init)
            ur1 = min(h, r + T.ctx_half_init + 1)
            uc0 = max(0, c - T.ctx_half_init)
            uc1 = min(w, c + T.ctx_half_init + 1)
            fire_fraction = float(pot_fire[ur0:ur1, uc0:uc1].mean())

            if fire_fraction > T.fire_fraction_threshold:
                stage1 = ((d_swir / rmsd_s > T.urb_stage1_alpha) and
                          (d_dt / rmsd_d > T.urb_stage1_beta))
                if not stage1:
                    med_ndvi, rmsd_ndvi, n_ndvi = _window_stats(
                        NDVI, bg_valid, r, c, half,
                        rmsd_mode="local_median")
                    med_nbr, rmsd_nbr, n_nbr = _window_stats(
                        NBR, bg_valid, r, c, half,
                        rmsd_mode="local_median")
                    if (min(n_ndvi, n_nbr) == 0
                            or not np.isfinite(rmsd_ndvi)
                            or not np.isfinite(rmsd_nbr)):
                        DQF_arr[r, c] = DQF.URBAN_REJ
                        n_drop_urban += 1
                        continue
                    stage2 = (
                        ((NDVI[r, c] - med_ndvi) / rmsd_ndvi
                         > T.urb_stage2_ndvi) and
                        ((NBR[r, c] - med_nbr) / rmsd_nbr
                         > T.urb_stage2_nbr)
                    )
                    if not stage2:
                        DQF_arr[r, c] = DQF.URBAN_REJ
                        n_drop_urban += 1
                        continue

        confirmed[r, c] = True
        DQF_arr[r, c] = DQF.FIRE
        n_confirm_context += 1

    # ------------------------------------------------------------------
    # Industrial heat mask (user-supplied exact mask on current grid)
    # ------------------------------------------------------------------
    hit_industry = confirmed & industrial_mask
    confirmed[hit_industry] = False
    DQF_arr[hit_industry] = DQF.IND_HEAT
    n_industry = int(hit_industry.sum())

    # ------------------------------------------------------------------
    # Stability test: previous image, 3x3 neighbourhood
    # ------------------------------------------------------------------
    n_stability = 0
    if prev_dqf is not None:
        prev_fire = np.isin(
            prev_dqf, [DQF.POT_FIRE, DQF.FIRE, DQF.ABS_FIRE, DQF.STABILITY])
        rows, cols = np.where(confirmed)
        for r, c in zip(rows.tolist(), cols.tolist()):
            rr0 = max(0, r - 1)
            rr1 = min(h, r + 2)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + 2)
            if not prev_fire[rr0:rr1, cc0:cc1].any():
                confirmed[r, c] = False
                DQF_arr[r, c] = DQF.STABILITY
                n_stability += 1

    FF = confirmed.astype(np.uint8)

    if verbose:
        n_abs = int(abs_fire.sum())
        print("---- FF candidate accounting ----")
        print(f"  absolute fires              : {n_abs}")
        print(f"  potential-fire candidates   : {n_cand}")
        print(f"    dropped - validity gate   : {n_drop_validity}")
        print(f"    dropped - context test    : {n_drop_context}")
        print(f"    dropped - cloud rejection : {n_drop_cloudrej}")
        print(f"    dropped - urban rejection : {n_drop_urban}")
        print(f"    confirmed by context test : {n_confirm_context}")
        print(f"  removed - industrial heat   : {n_industry}")
        print(f"  removed - stability test    : {n_stability}")
        print(f"  FINAL fire pixels           : {int(FF.sum())}")
        print(f"  rmsd_mode                   : {T.rmsd_mode}")
        print("---------------------------------")

    return FFResult(
        FF=FF,
        DQF_FF=DQF_arr,
        T_SWIR_c=T_SWIR_c,
        T_TIR_c=T_TIR_c,
        delta_T=delta_T,
        base_SWIR=base_SWIR,
        base_dT=base_dT,
        lapse_SWIR=float(lr_swir),
        lapse_TIR=float(lr_tir),
    )


def summarise(result: FFResult):
    unique, counts = np.unique(result.DQF_FF, return_counts=True)
    out = {}
    for flag, n in zip(unique.tolist(), counts.tolist()):
        out[int(flag)] = {
            "description": DQF_DESCRIPTIONS.get(int(flag), "unknown"),
            "pixels": int(n),
        }
    out["total_fire_pixels"] = int(result.FF.sum())
    out["lapse_SWIR_K_per_km"] = round(result.lapse_SWIR * 1000.0, 3)
    out["lapse_TIR_K_per_km"] = round(result.lapse_TIR * 1000.0, 3)
    return out