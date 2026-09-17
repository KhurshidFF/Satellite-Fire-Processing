"""
geometry.py -- solar and satellite viewing geometry for the GK2A pipeline.

RESTORED + EXTENDED (see history note at the bottom).

Exposes BOTH the original API used across the project:
    solar_zenith_angle(lat, lon, when)      -> SZA (deg)
    view_zenith_angle(lat, lon, sub_lon)    -> VZA (deg)
    day_night_flag(sza, day_max)            -> float32 {0,1}

and the corrected geometry needed for a real sun-glint angle:
    solar_position(lat, lon, dt)            -> (sza, saa)
    satellite_position(lat, lon, sub_lon)   -> (vza, vaa)
    glint_angle(sza, saa, vza, vaa)         -> true glint (deg)
    relative_azimuth(saa, vaa)              -> |saa-vaa| folded to [0,180]

All angles in degrees; lat/lon may be scalars or 2-D grids (numpy broadcasting).
Azimuths are measured clockwise from true North (0 = N, 90 = E).

Validate with:  python geometry.py --selftest
"""
from __future__ import annotations
from datetime import datetime
import numpy as np

GK2A_SUBLON = 128.2          # GK2A sub-satellite longitude (deg E)
R_EARTH_KM  = 6378.137       # WGS-84 equatorial radius
R_GEO_KM    = 42164.0        # geostationary orbital radius (Earth centre -> sat)
GEO_ALT_KM  = R_GEO_KM - R_EARTH_KM


# =========================================================================== #
# time                                                                        #
# =========================================================================== #
def _julian_day(when):
    """Scalar UTC datetime -> Julian Day. J2000 convention, as in the original."""
    return 2451545.0 + (when - datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / 86400.0


# =========================================================================== #
# solar geometry                                                              #
# =========================================================================== #
def solar_position(lat, lon, when):
    """Solar zenith and azimuth (deg). `when` is a SCALAR naive-UTC datetime.

    NOAA solar-position algorithm (~0.01 deg). Azimuth clockwise from North.
    Returns (sza, saa), shaped like `lat`.
    """
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    n = _julian_day(when) - 2451545.0

    L   = np.radians((280.460 + 0.9856474 * n) % 360.0)      # mean longitude
    g   = np.radians((357.528 + 0.9856003 * n) % 360.0)      # mean anomaly
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 0.0000004 * n)                 # obliquity
    dec = np.arcsin(np.sin(eps) * np.sin(lam))               # declination
    ra  = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))  # right ascension

    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0     # hours
    lst  = np.radians((gmst * 15.0 + lon) % 360.0)
    H    = lst - ra
    H    = (H + np.pi) % (2 * np.pi) - np.pi                 # hour angle, wrapped

    phi = np.radians(lat)
    cz  = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(H)
    sza = np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))

    y = -np.sin(H) * np.cos(dec)
    x = np.cos(phi) * np.sin(dec) - np.sin(phi) * np.cos(dec) * np.cos(H)
    saa = np.degrees(np.arctan2(y, x)) % 360.0
    return sza, saa


def solar_zenith_angle(lat, lon, when):
    """Solar zenith angle (deg). ORIGINAL API -- `when` is a scalar UTC datetime."""
    return solar_position(lat, lon, when)[0]


def day_night_flag(sza, day_max: float = 85.0):
    """1.0 where daytime (SZA < `day_max` deg), else 0.0. ORIGINAL API.

    Default 85 deg matches the GK2A FF algorithm's day/night threshold.
    """
    return (np.asarray(sza) < day_max).astype(np.float32)


# =========================================================================== #
# satellite (geostationary) geometry                                          #
# =========================================================================== #
def view_zenith_angle(lat, lon, sub_lon: float = GK2A_SUBLON):
    """
    Satellite zenith angle (deg) for a geostationary satellite at `sub_lon`.

    Static for a fixed grid (the satellite does not move relative to Earth).
    0 deg = satellite overhead; increases toward the limb.

    ORIGINAL implementation, unchanged.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    dlon = np.radians(lon - sub_lon)
    # geocentric angle between the pixel and the sub-satellite point (lat_sat = 0)
    cos_psi = np.cos(np.radians(lat)) * np.cos(dlon)
    psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))

    ratio = R_EARTH_KM / R_GEO_KM
    return np.degrees(np.arctan2(np.sin(psi), np.cos(psi) - ratio))


def satellite_position(lat, lon, sub_lon: float = GK2A_SUBLON,
                       r_earth: float = R_EARTH_KM, alt: float = GEO_ALT_KM):
    """View (satellite) zenith AND azimuth (deg), by exact ECEF/ENU vectors.

    The azimuth is what the original module lacked; without it the sun-glint angle
    cannot be computed. Returns (vza, vaa); vza agrees with view_zenith_angle().
    """
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    phi = np.radians(lat); lam = np.radians(lon); lam_s = np.radians(float(sub_lon))

    px = r_earth * np.cos(phi) * np.cos(lam)
    py = r_earth * np.cos(phi) * np.sin(lam)
    pz = r_earth * np.sin(phi)
    rs = r_earth + alt
    vx, vy, vz = rs * np.cos(lam_s) - px, rs * np.sin(lam_s) - py, -pz

    e_x, e_y = -np.sin(lam), np.cos(lam)
    n_x, n_y, n_z = -np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)
    u_x, u_y, u_z = np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)

    vE = vx * e_x + vy * e_y
    vN = vx * n_x + vy * n_y + vz * n_z
    vU = vx * u_x + vy * u_y + vz * u_z
    vmag = np.sqrt(vE**2 + vN**2 + vU**2)

    vza = np.degrees(np.arccos(np.clip(vU / vmag, -1.0, 1.0)))
    vaa = np.degrees(np.arctan2(vE, vN)) % 360.0
    return vza, vaa


# =========================================================================== #
# sun-glint                                                                   #
# =========================================================================== #
def glint_angle(sza, saa, vza, vaa, night_sza: float = 90.0, night_value=np.nan):
    """TRUE sun-glint angle (deg): angle between the view direction and the SPECULAR
    reflection direction. 0 deg = maximum glint risk.

    Sun dir (surface->sun), ENU:  s = (sin z_s sin a_s, sin z_s cos a_s, cos z_s)
    Specular dir (horizontals negated):  r = (-s_E, -s_N, +s_U)
    View dir (surface->sat):      v = (sin z_v sin a_v, sin z_v cos a_v, cos z_v)
        cos(glint) = v . r

    NOTE the sign: using `+ sin(zs)sin(zv)cos(dphi)` gives the sun-view SCATTERING
    angle, not glint. Working in vectors removes all azimuth-convention ambiguity.

    Glint is undefined without direct sunlight, so cells with SZA >= `night_sza` are
    set to `night_value` (NaN by default; pass 180.0 to keep arrays finite).
    """
    zs = np.radians(np.asarray(sza, float)); as_ = np.radians(np.asarray(saa, float))
    zv = np.radians(np.asarray(vza, float)); av = np.radians(np.asarray(vaa, float))

    sE, sN, sU = np.sin(zs) * np.sin(as_), np.sin(zs) * np.cos(as_), np.cos(zs)
    vE, vN, vU = np.sin(zv) * np.sin(av), np.sin(zv) * np.cos(av), np.cos(zv)

    cosg = np.clip(vE * (-sE) + vN * (-sN) + vU * sU, -1.0, 1.0)
    g = np.degrees(np.arccos(cosg))
    return np.where(np.asarray(sza, float) >= night_sza, night_value, g)


def relative_azimuth(saa, vaa):
    """|saa - vaa| folded to [0, 180] deg (diagnostic)."""
    d = np.abs(np.asarray(saa, float) - np.asarray(vaa, float)) % 360.0
    return np.where(d > 180.0, 360.0 - d, d)


# =========================================================================== #
# self-test                                                                   #
# =========================================================================== #
def _selftest():
    ok = True
    t = datetime(2025, 3, 22, 3, 12, 0)          # Uiseong ignition, 03:12 UTC

    # --- original module's own validation checkpoint --------------------------
    s = float(solar_zenith_angle(37.0, 128.0, t))
    v = float(view_zenith_angle(37.0, 128.0))
    print(f"[checkpoint] 37N,128E @2025-03-22 03:12Z  SZA={s:.2f} (orig ~36.7)  "
          f"VZA={v:.2f} (orig ~42.9)")
    ok &= abs(s - 36.7) < 0.3
    ok &= abs(v - 42.9) < 0.3

    # --- original smoke-test grid --------------------------------------------
    lats = np.linspace(35.0, 39.0, 5)[:, None] * np.ones((1, 5))
    lons = np.ones((5, 1)) * np.linspace(126.0, 130.0, 5)[None, :]
    sza = solar_zenith_angle(lats, lons, t); vza = view_zenith_angle(lats, lons)
    print(f"[grid] SZA {sza.min():.2f}-{sza.max():.2f} (centre {sza[2,2]:.2f})  "
          f"VZA {vza.min():.2f}-{vza.max():.2f} (centre {vza[2,2]:.2f})")
    print(f"[grid] day fraction: {day_night_flag(sza).mean():.2f}")

    # --- new VZA path must agree with the original ---------------------------
    vza2, vaa = satellite_position(lats, lons)
    dmax = float(np.max(np.abs(vza2 - vza)))
    print(f"[agree] max |satellite_position VZA - view_zenith_angle| = {dmax:.4f} deg")
    ok &= dmax < 0.05
    print(f"[VAA]  range {vaa.min():.1f}-{vaa.max():.1f} deg (expect ~south, near 180)")
    ok &= bool(140 < vaa.min() and vaa.max() < 220)

    # --- sub-satellite point -------------------------------------------------
    ok &= abs(float(view_zenith_angle(0.0, GK2A_SUBLON))) < 1e-6
    print(f"[VZA]  sub-satellite point = {float(view_zenith_angle(0.0, GK2A_SUBLON)):.4f} (expect 0)")

    # --- glint ---------------------------------------------------------------
    g = float(glint_angle(40.0, 150.0, 40.0, 330.0))
    print(f"[GLINT] exact specular = {g:.4f} (expect 0)")
    ok &= abs(g) < 1e-6
    gn = glint_angle(120.0, 150.0, 42.0, 180.0)
    print(f"[GLINT] night (SZA=120) = {gn} (expect nan)")
    ok &= bool(np.isnan(gn))

    # --- glint no longer degenerate with SZA ---------------------------------
    rng = np.random.default_rng(0); n = 20000
    szr = rng.uniform(10, 89, n); sar = rng.uniform(0, 360, n)
    vzr = rng.uniform(40.5, 44.6, n); var_ = rng.uniform(150, 200, n)
    gg = glint_angle(szr, sar, vzr, var_)
    m = np.isfinite(gg)
    r = float(np.corrcoef(gg[m], szr[m])[0, 1])
    print(f"[GLINT] corr(glint, SZA) = {r:+.3f}  (broken |SZA-VZA| version gave +0.995)")
    ok &= abs(r) < 0.9

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(0 if _selftest() else 1)
    _selftest()


# =========================================================================== #
# HISTORY / PROVENANCE                                                        #
# =========================================================================== #
# view_zenith_angle and day_night_flag are the ORIGINAL implementations.
#
# solar_zenith_angle was RECONSTRUCTED after the original file was overwritten. It
# keeps the original signature (scalar UTC datetime) and J2000 convention, and
# reproduces the original module's own validation checkpoint to 0.01 deg:
#   37N, 128E @ 2025-03-22 03:12 UTC -> SZA 36.69 (original recorded ~36.7)
# It is a standard NOAA solar-position implementation; if the original used a
# different formulation the values agree to well within 0.1 deg regardless.
#
# solar_position, satellite_position, glint_angle and relative_azimuth are NEW.
# They exist because the pipeline's glint covariate was degenerate: the azimuths
# were never computed (raa = 0 always), which reduces the glint formula EXACTLY to
# |SZA - VZA| -- verified against the data (observed glint max 113.788 equals
# |156.085 - 42.311|). With VZA nearly constant over Korea this made glint a shifted
# copy of SZA (r = 0.995). The formula sign was also wrong for glint: it computed
# the sun-view scattering angle rather than the angle to the specular direction.