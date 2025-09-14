# -*- coding: utf-8 -*-
"""
Fixed: robust flux measurement (small aperture + aperture correction, sigma-clipped annulus, uncertainties)
This preserves your original file paths and the rest of the pipeline (Gaia, membership, CMD).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

# ================== CONFIG (interactive; v6-equivalent params) ==================
def _ask(prompt, default, cast=str, upper=False):
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "" or s == "0":
        return default
    try:
        v = cast(s)
        return v.upper() if (upper and isinstance(v, str)) else v
    except Exception:
        return default

# ---- user input ----
print("=== Cluster Analysis Interactive Input ===")
cluster_name        = _ask("Cluster name", "UnknownCluster", str)
cluster_distance    = _ask("Cluster distance (pc)", 6800.0, float)   # default like v6 use-case
cluster_type        = _ask("Cluster type [O=open, G=globular]", "O", str, upper=True)
galactic_extinction = _ask("Galactic extinction A_V", 0.0, float)
galactic_reddening  = _ask("Galactic reddening E(B-V)", 0.0, float)

# defaults identical shape to what you used
_default_B   = r"D:\1 AAA TEMP\Clusters RGB fits\New\NGC 1647\ngc 1647-20250129_B_21.fts"
_default_GV  = r"D:\1 AAA TEMP\Clusters RGB fits\New\NGC 1647\ngc 1647-20250129_G_21.fts"
_default_csv = r"D:\1 AAA TEMP\Clusters RGB fits\New\NGC 1647\1757006309831O-result.csv"

fits_file_B = _ask("Path to B-band FITS image", _default_B, str)
fits_file_G = _ask("Path to G- or V-band FITS image", _default_GV, str)  # keep var name fits_file_G
gaia_file   = _ask("Path to Gaia CSV file", _default_csv, str)

# ---- normalize Windows “Copy as path” quotes and file:/// URLs ----
from urllib.parse import urlparse, unquote
def _norm_path(p: str) -> str:
    if p is None:
        return p
    p = p.strip().strip('"').strip("'")
    if p.lower().startswith("file:///"):
        u = urlparse(p)
        p = unquote(u.path)
        if len(p) >= 3 and p[0] == "/" and p[2] == ":":
            p = p[1:]
    return os.path.normpath(p)

fits_file_B = _norm_path(fits_file_B)
fits_file_G = _norm_path(fits_file_G)
gaia_file   = _norm_path(gaia_file)

# outputs: same variable names, placed next to G/V path (as you used in v7 runs)
_outdir = os.path.dirname(fits_file_G) if os.path.dirname(fits_file_G) else os.getcwd()
out_csv     = os.path.join(_outdir, "photometry_with_membership.csv")
out_members = os.path.join(_outdir, "matched_with_membership.csv")
out_png     = os.path.join(_outdir, "cluster_members_overlay.png")

# ---------------- Detection / photometry (v6 values) ----------------
DAO_FWHM           = 5.0
DAO_THRESH_SIGMA   = 4.0
PEAK_SNR_MIN       = 10.0
FWHM_BOX_SIZE      = 24
K_APERTURE         = 1.2
K_ANNULUS_IN       = 2.5
K_ANNULUS_OUT      = 4.0
GAIN_E_PER_ADU     = 1.0
READ_NOISE_E       = 5.0

# ---------------- CMD / Distance and extinction ----------------
CLUSTER_DISTANCE_PC = float(cluster_distance)
AG_EXTINCTION       = float(galactic_extinction)
EBG_REDDENING       = float(galactic_reddening)

# ---------------- Matching + membership (v6 values) ----------------

# ---- Parameter selection by cluster type and distance ----
def _set_params_by_type_and_distance(cluster_type: str, dist_pc: float):
    #pi0 = 1000.0 / max(dist_pc, 1e-6)  # mas
    g = globals()
    if cluster_type.upper() == 'G':
        # Globular cluster: exactly values from v6
        g['MATCH_MAX_ARCSEC']   = 2.0       # Max cross-match radius (arcsec). Smaller → fewer false matches.
        g['PARALLAX_MIN_HALF']  = 0.05      # Inner half-width around expected parallax (mas). Smaller → tighter selection.
        g['PARALLAX_MAX_HALF']  = 5.00      # Outer half-width around expected parallax (mas). Lower → exclude distant interlopers.
        g['PARALLAX_FRAC']      = 100.0     # Fraction of parallax allowed as expansion. Lower → window sticks closer to π0.
        g['PARALLAX_ERR_SIGMA'] = 5.0       # Tolerance in σ of Gaia parallax errors. Lower → accept only high-S/N stars.
        g['PM_SIGMA_FLOOR']     = 1.5       # Minimum dispersion (mas/yr) for proper motions. Lower → tighter PM cluster.
        g['PM_ERR_SIGMA']       = 8.0       # Tolerance in σ of Gaia PM errors. Lower → stricter filtering.
        g['PM_CHI2']            = 40.0      # χ² threshold for PM coherence. Lower → stricter clustering.
        g['SPATIAL_CORE_FRAC']  = 0.15      # Fraction of field radius used as “core” for stats. Lower → focuses on central stars.
        g['ALIGN_SEARCH_ARCSEC'] = 60.0     # Max radius (arcsec) for Gaia↔image alignment search. Smaller → cleaner matches.
        g['ALIGN_MIN_PAIRS']     = 20       # Minimum matched pairs required for alignment. Higher → stricter alignment.
    else:
        # Open cluster: tuned windows
        g['MATCH_MAX_ARCSEC']   = 10.0    # Max cross-match radius (arcsec). Smaller → fewer false matches.
        g['PARALLAX_MIN_HALF']  = 0.10   # Inner half-width around expected parallax (mas). Smaller → tighter selection.
        g['PARALLAX_MAX_HALF']  = 1.00   # Outer half-width around expected parallax (mas). Lower → exclude distant interlopers.
        g['PARALLAX_FRAC']      = 20.0   # Fraction of parallax allowed as expansion. Lower → window sticks closer to π0.
        g['PARALLAX_ERR_SIGMA'] = 4.0    # Tolerance in σ of Gaia parallax errors. Lower → accept only high-S/N stars.
        g['PM_SIGMA_FLOOR']     = 1.0    # Minimum dispersion (mas/yr) for proper motions. Lower → tighter PM cluster.
        g['PM_ERR_SIGMA']       = 4.0    # Tolerance in σ of Gaia PM errors. Lower → stricter filtering.
        g['PM_CHI2']            = 25.0   # χ² threshold for PM coherence. Lower → stricter clustering.
        g['SPATIAL_CORE_FRAC']  = 0.15   # Fraction of field radius used as “core” for stats. Lower → focuses on central stars.
        g['ALIGN_SEARCH_ARCSEC'] = 40.0  # Max radius (arcsec) for Gaia↔image alignment search. Smaller → cleaner matches.
        g['ALIGN_MIN_PAIRS']     = 25    # Minimum matched pairs required for alignment. Higher → stricter alignment.

        
        # ---------------- Calibration control ----------------
CALIB_MATCH_MAX_ARCSEC = 10
CALIB_G_MAG_RANGE      = (10.0, 19.0)
CALIB_BP_MAG_RANGE     = (10.0, 19.5)

# apply membership/profile parameters by cluster type + distance
_set_params_by_type_and_distance(cluster_type, CLUSTER_DISTANCE_PC)
print(f"[profile] type={cluster_type}  MATCH={MATCH_MAX_ARCSEC}\"  piErrSig={PARALLAX_ERR_SIGMA}  "
      f"PMchi2={PM_CHI2}  core={SPATIAL_CORE_FRAC}")
print("=== Effective configuration ===")
print(f"Name={cluster_name}, Type={cluster_type}, Dist={CLUSTER_DISTANCE_PC} pc")
print(f"A_V={AG_EXTINCTION}, E(B-V)={EBG_REDDENING}")
print(f"B FITS={fits_file_B}")
print(f"G/V FITS={fits_file_G}")
print(f"Gaia CSV={gaia_file}")
print(f"Output CSV={out_csv}")
print(f"Output members CSV={out_members}")
print(f"Output overlay PNG={out_png}")
print("====================================================================")
# ================== END CONFIG ============================================

# ================== Photometry helpers (robust) ==================
def estimate_fwhm_moments(img, x, y, box=15, r_bg_in=8, r_bg_out=12):
    """Robust FWHM via second moments on a local background-subtracted cutout."""
    h, w = img.shape
    x0 = int(round(x)); y0 = int(round(y))
    x1 = max(0, x0 - box//2); x2 = min(w, x0 + box//2 + 1)
    y1 = max(0, y0 - box//2); y2 = min(h, y0 + box//2 + 1)
    if x2 <= x1+2 or y2 <= y1+2:
        return np.nan
    cut = img[y1:y2, x1:x2].astype(float)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    r = np.hypot(yy - y, xx - x)
    ann = (r >= r_bg_in) & (r <= r_bg_out)
    if np.sum(ann) >= 20:
        bg_mean, _, _ = sigma_clipped_stats(cut[ann], sigma=3.0, maxiters=5)
    else:
        bg_mean = float(np.nanmedian(cut))
    cut_bs = cut - bg_mean
    cut_bs[cut_bs < 0] = 0.0
    flux = cut_bs.sum()
    if flux <= 0:
        return np.nan
    x_mean = (cut_bs * (xx - x)).sum() / flux
    y_mean = (cut_bs * (yy - y)).sum() / flux
    x2 = (cut_bs * (xx - x - x_mean)**2).sum() / flux
    y2 = (cut_bs * (yy - y - y_mean)**2).sum() / flux
    sigma = float(np.sqrt(max(1e-12, 0.5 * (x2 + y2))))
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

def _cutout(img, x, y, r_out):
    """Small cutout around (x,y) up to r_out to avoid full-frame grids."""
    h, w = img.shape
    R = int(np.ceil(r_out)) + 2
    x0 = int(round(x)); y0 = int(round(y))
    x1 = max(0, x0 - R); x2 = min(w, x0 + R + 1)
    y1 = max(0, y0 - R); y2 = min(h, y0 + R + 1)
    cut = img[y1:y2, x1:x2].astype(float)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    return cut, xx, yy

def local_background_cutout(img, x, y, r_in, r_out):
    cut, xx, yy = _cutout(img, x, y, r_out)
    r = np.hypot(yy - y, xx - x)
    ann = (r >= r_in) & (r <= r_out)
    vals = cut[ann]
    if vals.size < 30:
        mean = float(np.nanmedian(vals)) if vals.size > 0 else float(np.nanmedian(cut))
        std = float(np.nanstd(vals)) if vals.size > 0 else float(np.nanstd(cut))
        return mean, std
    mean, median, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
    return float(mean), float(std)

def aperture_photometry_single_fast(img, x, y, r_ap, r_in, r_out,
                                    gain_e_per_adu=1.0, read_noise_e=5.0):
    cut, xx, yy = _cutout(img, x, y, r_out)
    r = np.hypot(yy - y, xx - x)
    ap = (r <= r_ap)
    bkg_mean, bkg_std = local_background_cutout(img, x, y, r_in, r_out)
    ap_vals = cut[ap].astype(float)
    npix_ap = ap_vals.size
    flux_ap_adu = float(np.nansum(ap_vals) - bkg_mean * npix_ap)
    # Uncertainty in ADU
    src_e = max(0.0, np.nansum(ap_vals) * gain_e_per_adu - bkg_mean * gain_e_per_adu * npix_ap)
    bkg_var_e = (bkg_std * gain_e_per_adu)**2
    var_e = max(0.0, src_e) + npix_ap * bkg_var_e + npix_ap * (read_noise_e**2)
    flux_err_adu = (np.sqrt(var_e) / gain_e_per_adu) if var_e > 0 else 0.0
    snr = (flux_ap_adu / flux_err_adu) if flux_err_adu > 0 else 0.0
    return flux_ap_adu, float(flux_err_adu), float(snr), float(bkg_mean), float(bkg_std), int(npix_ap)

def compute_aperture_correction(img, positions, fwhm_field,
                                gain=GAIN_E_PER_ADU, read_noise=READ_NOISE_E):
    """Median curve-of-growth on isolated stars -> correction from r=K_APERTURE*FWHM to r=3*FWHM."""
    if len(positions) == 0:
        return 1.0
    radii = np.array([0.7, 1.0, 1.2, 1.4, 2.0, 3.0]) * fwhm_field
    r_in = K_ANNULUS_IN * fwhm_field
    r_out = K_ANNULUS_OUT * fwhm_field
    pos = np.array(positions, float)
    keep = []
    for i, (x, y) in enumerate(pos):
        d = np.hypot(pos[:,0]-x, pos[:,1]-y); d[i] = np.inf
        if np.min(d) > 6.0 * fwhm_field:
            keep.append((x, y))
    if len(keep) < max(3, int(0.2*len(positions))):
        keep = positions
    curves = []
    for (x, y) in keep:
        f_list = []
        for r_ap in radii:
            f_ap, _, _, _, _, _ = aperture_photometry_single_fast(img, x, y, r_ap, r_in, r_out, gain, read_noise)
            f_list.append(f_ap)
        curves.append(f_list)
    curves = np.array(curves, float)
    median_curve = np.nanmedian(curves, axis=0)
    flux_large = median_curve[-1]
    idx_small = int(np.argmin(np.abs(radii - (K_APERTURE * fwhm_field))))
    flux_small = median_curve[idx_small]
    if not np.isfinite(flux_small) or flux_small <= 0 or not np.isfinite(flux_large) or flux_large <= 0:
        return 1.0
    return float(flux_large / flux_small)

# ================== (rest of original helpers kept) ==================
def distance_modulus_from_distance_pc(d_pc: float) -> float:
    if not np.isfinite(d_pc) or d_pc <= 0: return np.nan
    return 5.0 * np.log10(float(d_pc)) - 5.0

def assert_paths(*paths):
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path not found: {p}")

def load_gaia_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    for want in ["ra","dec","parallax","parallax_error","pmra","pmra_error","pmdec","pmdec_error",
                 "phot_g_mean_mag","phot_bp_mean_mag","phot_rp_mean_mag"]:
        if want in cols_lower:
            rename_map[cols_lower[want]] = want
    if rename_map: df = df.rename(columns=rename_map)
    for c in ['ra','dec','parallax','parallax_error','pmra','pmra_error','pmdec','pmdec_error',
              'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def _get_gaia_photometry(df: pd.DataFrame):
    cl = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cl: return pd.to_numeric(df[cl[n]], errors='coerce')
        return pd.Series(np.nan, index=df.index)
    G  = pick('phot_g_mean_mag','gmag','gaia_g','g')
    BP = pick('phot_bp_mean_mag','bpmag','bp')
    RP = pick('phot_rp_mean_mag','rpmag','rp')
    return G, BP, RP

def wcs_pixels_to_radec(wcs: WCS, xpix, ypix):
    coords = wcs.pixel_to_world(np.asarray(xpix), np.asarray(ypix))
    return np.asarray(coords.ra.deg), np.asarray(coords.dec.deg)

def _mode_1d(arr, bins=80, pct=(1,99)):
    arr = np.asarray(arr, dtype=float); arr = arr[np.isfinite(arr)]
    if arr.size == 0: return np.nan
    if arr.size < 15: return float(np.nanmedian(arr))
    lo, hi = np.nanpercentile(arr, pct)
    H, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    j = int(np.argmax(H)); return float(0.5*(edges[j] + edges[j+1]))

# tangent helpers
def _to_tangent(center_coord: SkyCoord, ra_deg: np.ndarray, dec_deg: np.ndarray):
    frame = SkyOffsetFrame(origin=center_coord)
    pts = SkyCoord(ra_deg*u.deg, dec_deg*u.deg).transform_to(frame)
    return np.column_stack([pts.lon.deg*3600.0, pts.lat.deg*3600.0])

def _from_tangent(center_coord: SkyCoord, xy_arcsec: np.ndarray):
    r_arcsec = np.hypot(xy_arcsec[:,0], xy_arcsec[:,1]) * u.arcsec
    theta = np.arctan2(xy_arcsec[:,1], xy_arcsec[:,0]) * u.rad
    out = center_coord.directional_offset_by(theta, r_arcsec.to(u.deg))
    return np.asarray(out.ra.deg), np.asarray(out.dec.deg)

def auto_align_wcs_offsets(center_coord: SkyCoord,
                           det_ra_deg: np.ndarray, det_dec_deg: np.ndarray,
                           gaia_ra_deg: np.ndarray, gaia_dec_deg: np.ndarray,
                           search_arcsec=ALIGN_SEARCH_ARCSEC, min_pairs=ALIGN_MIN_PAIRS
                           ):
    det_xy  = _to_tangent(center_coord, det_ra_deg, det_dec_deg)
    gaia_xy = _to_tangent(center_coord, gaia_ra_deg, gaia_dec_deg)
    tree = cKDTree(gaia_xy)
    dists, idxs = tree.query(det_xy, k=1, distance_upper_bound=search_arcsec)
    ok = np.isfinite(dists) & (dists <= search_arcsec) & (idxs < len(gaia_xy))
    if ok.sum() < min_pairs:
        return det_ra_deg, det_dec_deg, 0.0, 0.0, int(ok.sum())
    diffs = gaia_xy[idxs[ok]] - det_xy[ok]
    dx = float(np.nanmedian(diffs[:,0])); dy = float(np.nanmedian(diffs[:,1]))
    ra_corr, dec_corr = _from_tangent(center_coord, det_xy + np.array([dx,dy], float))
    return ra_corr, dec_corr, dx, dy, int(ok.sum())

def _get_exptime_from_header(header) -> float:
    for key in ('EXPTIME','EXPOSURE','EXPT','ITIME'):
        if key in header:
            try:
                v = float(header[key])
                if np.isfinite(v) and v > 0: return v
            except Exception: pass
    return 1.0

def _robust_linfit(y, x, clip=3.0, maxiter=10, weights=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    a=b=np.nan
    for _ in range(maxiter):
        if mask.sum() < 5: break
        X = np.vstack([np.ones(mask.sum()), x[mask]]).T
        coeff, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        a, b = coeff
        resid = y - (a + b*x)
        s = np.nanstd(resid[mask])
        new_mask = mask & (np.abs(resid) <= clip*s if s>0 else np.isfinite(resid))
        if new_mask.sum() == mask.sum(): break
        mask = new_mask
    rms = float(np.nanstd((y - (a + b*x))[mask])) if mask.sum()>=2 else np.nan
    return float(a), float(b), mask, rms

def calibrate_zero_point_generic(df_det: pd.DataFrame, gaia_df: pd.DataFrame,
                                 m_inst: pd.Series, target_mag: pd.Series,
                                 flux_series: pd.Series, mag_range=None):
    if ('matched' not in df_det.columns) or ('ClusterMember' not in df_det.columns):
        return np.nan, np.nan, np.zeros(len(df_det), dtype=bool), np.nan
    mask = (df_det['matched'] == True) & (df_det['ClusterMember'] == False)
    if mask.sum() < 10: mask = (df_det['matched'] == True)
    if mask.sum() < 5:  return np.nan, np.nan, np.zeros(len(df_det), dtype=bool), np.nan
    if 'sep_arcsec' in df_det.columns:
        mask &= (df_det['sep_arcsec'] <= float(CALIB_MATCH_MAX_ARCSEC))
    idx_gaia = df_det.loc[mask, 'gaia_index'].astype(int).to_numpy()
    gsub = gaia_df.iloc[idx_gaia].reset_index(drop=True)
    def _pick(df, *names):
        cl = {c.lower(): c for c in df.columns}
        for n in names:
            if n in cl: return pd.to_numeric(df[cl[n]], errors='coerce')
        return pd.Series(np.nan, index=df.index)
    BP = _pick(gsub, 'phot_bp_mean_mag','bpmag','bp')
    RP = _pick(gsub, 'phot_rp_mean_mag','rpmag','rp')
    color = (BP - RP).to_numpy()
    if isinstance(target_mag, pd.Series): tmag = target_mag.iloc[idx_gaia].to_numpy()
    else: tmag = np.asarray(target_mag, float)
    q = np.isfinite(tmag) & np.isfinite(color) & (color >= 0.3) & (color <= 2.0)
    if isinstance(mag_range, (tuple, list)) and len(mag_range) == 2:
        lo_mag, hi_mag = float(mag_range[0]), float(mag_range[1])
        q &= (tmag >= lo_mag) & (tmag <= hi_mag)
    F = flux_series.loc[mask].to_numpy(float)
    p1, p99 = np.nanpercentile(F, [5, 99])
    q &= (F >= p1) & (F <= p99)
    if q.sum() < 5:
        return np.nan, np.nan, np.zeros(len(df_det), dtype=bool), np.nan
    y = tmag[q] - m_inst.loc[mask].to_numpy()[q]
    x = color[q]
    ZP, beta, used, rms = _robust_linfit(y, x)
    idx_mask = np.zeros(len(df_det), dtype=bool)
    idx_mask[np.where(mask)[0][q][used]] = True
    return ZP, beta, idx_mask, rms

def _same_pixel_grid(wcsG: WCS, wcsB: WCS, shapeG, shapeB, tol_pix: float = 0.05) -> bool:
    if shapeG != shapeB: return False
    ny, nx = shapeG
    test_pts = np.array([[nx/2.0, ny/2.0],
                         [nx*0.25, ny*0.25],
                         [nx*0.75, ny*0.75]])
    sky = wcsG.pixel_to_world(test_pts[:,0], test_pts[:,1])
    xb, yb = wcsB.world_to_pixel(sky)
    diffs = np.hypot(xb - test_pts[:,0], yb - test_pts[:,1])
    return np.nanmax(diffs) <= tol_pix

def compute_membership_via_pm_parallax(gaia: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    global MATCH_MAX_ARCSEC, PARALLAX_MIN_HALF, PARALLAX_ERR_SIGMA, PARALLAX_FRAC, PARALLAX_MAX_HALF
    global PM_SIGMA_FLOOR, PM_ERR_SIGMA, PM_CHI2, SPATIAL_CORE_FRAC
    def mad_sigma(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
        if x.size == 0: return np.nan
        med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med))
        return 1.4826 * mad
    def safe_floor(val: float, floor_val: float) -> float:
        return floor_val if (not np.isfinite(val)) or (val <= 0) else val
    def estimate_center_xy(x: np.ndarray, y: np.ndarray, bins: int = 60):
        mask = np.isfinite(x) & np.isfinite(y); x = x[mask]; y = y[mask]
        if x.size < 20: return float(np.nanmedian(x)), float(np.nanmedian(y))
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        i, j = np.unravel_index(np.nanargmax(H), H.shape)
        xc = 0.5 * (xedges[i] + xedges[i+1]); yc = 0.5 * (yedges[j] + yedges[j+1])
        return float(xc), float(yc)
    gaia_coords = SkyCoord(gaia['ra'].values*u.deg, gaia['dec'].values*u.deg)
    det_coords  = SkyCoord(det['RA'].values*u.deg,  det['DEC'].values*u.deg)
    idx, d2d, _ = det_coords.match_to_catalog_sky(gaia_coords)
    out = det.copy()
    out['gaia_index'] = idx; out['sep_arcsec'] = d2d.arcsec; out['matched'] = out['sep_arcsec'] <= float(MATCH_MAX_ARCSEC)
    m = out['matched']; out['ClusterMember'] = False
    if m.sum() == 0:
        for c in ['cluster_pm_center_pmra','cluster_pm_center_pmdec','pm_radius_used_masyr',
                  'cluster_parallax_center','parallax_window_used_mas','cluster_center_xpix','cluster_center_ypix','core_radius_pix']:
            out[c] = np.nan
        return out
    X_m = out.loc[m, 'X'].to_numpy(float); Y_m = out.loc[m, 'Y'].to_numpy(float)
    xc, yc = estimate_center_xy(X_m, Y_m, bins=60)
    span_x = np.nanmax(X_m) - np.nanmin(X_m); span_y = np.nanmax(Y_m) - np.nanmin(Y_m); span = float(min(span_x, span_y))
    R_core = max(10.0, SPATIAL_CORE_FRAC * span)
    r_m = np.hypot(X_m - xc, Y_m - yc); core_m = r_m <= R_core
    gsub = gaia.iloc[out.loc[m,'gaia_index'].astype(int)].reset_index(drop=True)
    par   = pd.to_numeric(gsub.get('parallax'),       errors='coerce').to_numpy(float)
    perr  = pd.to_numeric(gsub.get('parallax_error'), errors='coerce').to_numpy(float)
    pmra  = pd.to_numeric(gsub.get('pmra'),           errors='coerce').to_numpy(float)
    pmde  = pd.to_numeric(gsub.get('pmdec'),          errors='coerce').to_numpy(float)
    pmra_err = pd.to_numeric(gsub.get('pmra_error'),  errors='coerce').to_numpy(float)
    pmde_err = pd.to_numeric(gsub.get('pmdec_error'), errors='coerce').to_numpy(float)
    base_sel = core_m & np.isfinite(par); 
    if base_sel.sum() < 25: base_sel = np.isfinite(par)
    def mad_sigma1(x): return mad_sigma(x)
    par_center = _mode_1d(par[base_sel], bins=80, pct=(1,99))
    par_sigma  = mad_sigma1(par[base_sel])
    frac_cap = PARALLAX_FRAC * par_center if np.isfinite(par_center) else np.inf
    hi = min(PARALLAX_MAX_HALF, frac_cap) if np.isfinite(frac_cap) else PARALLAX_MAX_HALF
    perr_med = float(np.nanmedian(perr[np.isfinite(perr)])) if np.isfinite(perr).any() else 0.1
    perr_eff = np.where(np.isfinite(perr), perr, perr_med)
    lo = np.maximum(PARALLAX_MIN_HALF, PARALLAX_ERR_SIGMA * perr_eff)
    if np.isfinite(par_sigma) and par_sigma > 0: lo = np.maximum(lo, 2.0 * par_sigma)
    halfwin = np.minimum(np.maximum(lo, PARALLAX_MIN_HALF), hi)
    par_member = np.isfinite(par) & (np.abs(par - par_center) <= halfwin)
    out['parallax_window_used_mas'] = np.nan; out.loc[m, 'parallax_window_used_mas'] = halfwin
    def _sig_axis(x, xe, ref):
        intr = max(mad_sigma1(x[ref]), PM_SIGMA_FLOOR)
        xe_eff = np.where(np.isfinite(xe), np.maximum(xe, PM_SIGMA_FLOOR) * PM_ERR_SIGMA, PM_ERR_SIGMA * PM_SIGMA_FLOOR)
        return np.sqrt(intr**2 + xe_eff**2)
    sel_pm = par_member & np.isfinite(pmra) & np.isfinite(pmde); core_pm_sel = sel_pm & core_m; ref_sel = core_pm_sel if core_pm_sel.any() else sel_pm
    sig_ra  = _sig_axis(pmra, pmra_err, ref_sel); sig_dec = _sig_axis(pmde, pmde_err, ref_sel)
    pmra_med = float(np.nanmedian(pmra[ref_sel])) if ref_sel.any() else np.nan
    pmde_med = float(np.nanmedian(pmde[ref_sel])) if ref_sel.any() else np.nan
    d2 = ((pmra - pmra_med)**2)/(sig_ra**2) + ((pmde - pmde_med)**2)/(sig_dec**2)
    pm_member = d2 <= PM_CHI2
    both = par_member & pm_member
    out.loc[m, 'ClusterMember'] = both
    out['cluster_pm_center_pmra']  = pmra_med
    out['cluster_pm_center_pmdec'] = pmde_med
    out['pm_radius_used_masyr']    = float(np.nanmedian(np.sqrt(PM_CHI2) * np.hypot(sig_ra, sig_dec)))
    out['cluster_parallax_center'] = par_center
    out['cluster_center_xpix']     = xc
    out['cluster_center_ypix']     = yc
    out['core_radius_pix']         = R_core
    return out

# ========================== MAIN ==========================
if __name__ == "__main__":
    assert_paths(fits_file_G, fits_file_B, gaia_file)

        # ---------- load FITS G ----------
    hdulG = fits.open(fits_file_G)
    hduG  = (hdulG[1] if (len(hdulG) > 1 and hdulG[1].data is not None) else hdulG[0])
    dataG = hduG.data
    wcsG  = WCS(hduG.header)
    exptimeG = _get_exptime_from_header(hduG.header)
    hdulG.close()
    
    # ---------- load FITS B ----------
    hdulB = fits.open(fits_file_B)
    hduB  = (hdulB[1] if (len(hdulB) > 1 and hdulB[1].data is not None) else hdulB[0])
    dataB = hduB.data
    wcsB  = WCS(hduB.header)
    exptimeB = _get_exptime_from_header(hduB.header)
    hdulB.close()


    ny, nx = dataG.shape
    center = wcsG.pixel_to_world(nx/2, ny/2)
    center_coord = SkyCoord(center.ra.deg * u.deg, center.dec.deg * u.deg)
    print(f"[fits] EXPTIME G={exptimeG:.3f}s  B={exptimeB:.3f}s")

    # -------- detection on G --------
    mean, median, std = sigma_clipped_stats(dataG, sigma=3.0)
    daofind = DAOStarFinder(fwhm=DAO_FWHM, threshold=DAO_THRESH_SIGMA * std)
    sources = daofind(dataG - median)
    if sources is None or len(sources)==0:
        raise RuntimeError("No sources detected on G.")
    if 'peak' in sources.colnames:
        sources = sources[sources['peak'] > (PEAK_SNR_MIN * std)]
    print(f"[detect:G] kept detections: {len(sources)}")

    # -------- robust photometry on G --------
    positions_G = [(float(s['xcentroid']), float(s['ycentroid'])) for s in sources]
    # Estimate field FWHM from a subset
    sample = positions_G[:min(40, len(positions_G))]
    fwhms = []
    for (x, y) in sample:
        f = estimate_fwhm_moments(dataG, x, y, box=FWHM_BOX_SIZE, r_bg_in=8, r_bg_out=12)
        if np.isfinite(f) and f > 0: fwhms.append(float(f))
    fwhmG_field = float(np.nanmedian(fwhms)) if len(fwhms) else float(DAO_FWHM)

    rG_ap = K_APERTURE * fwhmG_field
    rG_in = K_ANNULUS_IN * fwhmG_field
    rG_out= K_ANNULUS_OUT * fwhmG_field

    rows=[]
    for (x, y) in positions_G:
        Fap, Ferr, SNR, bmean, bstd, npix = aperture_photometry_single_fast(
            dataG, x, y, rG_ap, rG_in, rG_out, GAIN_E_PER_ADU, READ_NOISE_E
        )
        if not np.isfinite(Fap) or Fap <= 0: continue
        rows.append({'X':x,'Y':y,'FWHM_G_field':fwhmG_field,'ApertureRadius_G':rG_ap,
                     'Flux_G_Aperture':Fap,'FluxErr_G':Ferr,'SNR_G':SNR,
                     'BkgMean_G':bmean,'BkgStd_G':bstd,'Npix_G':npix})
    if not rows:
        raise RuntimeError("No valid photometry rows (G).")
    df_det = pd.DataFrame(rows)

    # Aperture correction for G
    corrG = compute_aperture_correction(dataG, positions_G, fwhmG_field,
                                        gain=GAIN_E_PER_ADU, read_noise=READ_NOISE_E)
    df_det['ApertureCorrection_G'] = corrG
    df_det['Flux_G'] = df_det['Flux_G_Aperture'] * corrG

    # RA/DEC from G WCS
    ra, dec = wcs_pixels_to_radec(wcsG, df_det['X'], df_det['Y'])
    df_det['RA'], df_det['DEC'] = ra, dec
    print(f"[phot:G] rows: {len(df_det)}  FWHM_field~{fwhmG_field:.2f}px  corrG={corrG:.3f}")

    # ---------- Gaia & alignment ----------
    gaia = load_gaia_table(gaia_file)
    gaia = gaia[np.isfinite(gaia['ra']) & np.isfinite(gaia['dec'])].copy()

    ra_corr, dec_corr, dx, dy, n_pairs = auto_align_wcs_offsets(
        center_coord, df_det['RA'].to_numpy(), df_det['DEC'].to_numpy(),
        gaia['ra'].to_numpy(), gaia['dec'].to_numpy(), search_arcsec=ALIGN_SEARCH_ARCSEC, min_pairs=ALIGN_MIN_PAIRS
    )
    if n_pairs >= 20 and (abs(dx) > 0.05 or abs(dy) > 0.05):
        df_det['RA'], df_det['DEC'] = ra_corr, dec_corr
        print(f"[align] applied global shift: dx={dx:.2f}\" dy={dy:.2f}\" using {n_pairs} pairs")

    # ---------- membership ----------
    df_det = compute_membership_via_pm_parallax(gaia, df_det)

    # ---------- decide B positions ----------
    wcsG_ok = isinstance(wcsG, WCS); wcsB_ok = isinstance(wcsB, WCS)
    SAME_GRID = wcsG_ok and wcsB_ok and _same_pixel_grid(wcsG, wcsB, dataG.shape, dataB.shape, tol_pix=0.05)
    if SAME_GRID:
        df_det['X_B'] = df_det['X'].values; df_det['Y_B'] = df_det['Y'].values
    else:
        coords = SkyCoord(df_det['RA'].values*u.deg, df_det['DEC'].values*u.deg)
        xB, yB = wcsB.world_to_pixel(coords); df_det['X_B'] = xB; df_det['Y_B'] = yB

    # ---------- robust photometry on B (fallback to G radius if FWHM fails) ----------
    # Estimate field FWHM on B via subset around mapped positions
    sampleB = df_det[['X_B','Y_B']].dropna().to_numpy(float)[:min(40, len(df_det))]
    fwhmsB = []
    for xb, yb in sampleB:
        fB = estimate_fwhm_moments(dataB, xb, yb, box=FWHM_BOX_SIZE, r_bg_in=8, r_bg_out=12)
        if np.isfinite(fB) and fB > 0: fwhmsB.append(float(fB))
    fwhmB_field = float(np.nanmedian(fwhmsB)) if len(fwhmsB) else float(fwhmG_field)

    rB_ap = K_APERTURE * fwhmB_field
    rB_in = K_ANNULUS_IN * fwhmB_field
    rB_out= K_ANNULUS_OUT * fwhmB_field

    FluxB_ap = np.full(len(df_det), np.nan); FluxB_err = np.full(len(df_det), np.nan)
    SNR_B    = np.full(len(df_det), np.nan); BkgMean_B = np.full(len(df_det), np.nan)
    BkgStd_B = np.full(len(df_det), np.nan); Npix_B    = np.full(len(df_det), np.nan)
    for i, (xb, yb) in enumerate(zip(df_det['X_B'].to_numpy(float), df_det['Y_B'].to_numpy(float))):
        if not np.isfinite(xb) or not np.isfinite(yb): continue
        if xb<0 or yb<0 or xb>=dataB.shape[1] or yb>=dataB.shape[0]: continue
        Fap, Ferr, SNR, bmean, bstd, npix = aperture_photometry_single_fast(
            dataB, xb, yb, rB_ap, rB_in, rB_out, GAIN_E_PER_ADU, READ_NOISE_E
        )
        if np.isfinite(Fap) and Fap>0:
            FluxB_ap[i] = Fap; FluxB_err[i] = Ferr; SNR_B[i] = SNR; BkgMean_B[i] = bmean; BkgStd_B[i] = bstd; Npix_B[i] = npix
    df_det['ApertureRadius_B'] = rB_ap
    df_det['Flux_B_Aperture']  = FluxB_ap
    df_det['FluxErr_B']        = FluxB_err
    df_det['SNR_B']            = SNR_B
    df_det['BkgMean_B']        = BkgMean_B
    df_det['BkgStd_B']         = BkgStd_B
    df_det['Npix_B']           = Npix_B

    corrB = compute_aperture_correction(dataB, df_det[['X_B','Y_B']].dropna().to_numpy(float), fwhmB_field,
                                        gain=GAIN_E_PER_ADU, read_noise=READ_NOISE_E)
    df_det['ApertureCorrection_B'] = corrB
    df_det['Flux_B'] = df_det['Flux_B_Aperture'] * corrB

    # ---------- instrumental magnitudes ----------
    exptimeG = float(exptimeG) if np.isfinite(exptimeG) and exptimeG>0 else 1.0
    exptimeB = float(exptimeB) if np.isfinite(exptimeB) and exptimeB>0 else 1.0
    df_det['m_inst_G'] = -2.5 * np.log10(df_det['Flux_G'] / exptimeG)
    df_det['m_inst_B'] = -2.5 * np.log10(df_det['Flux_B'] / exptimeB)

    # ---------- calibrations ----------
    gaia = gaia.reset_index(drop=True)
    def _pick(df, *names):
        cl = {c.lower(): c for c in df.columns}
        for n in names:
            if n in cl: return pd.to_numeric(df[cl[n]], errors='coerce')
        return pd.Series(np.nan, index=df.index)
    Ggaia  = _pick(gaia, 'phot_g_mean_mag','gmag','gaia_g','g')
    BPgaia = _pick(gaia, 'phot_bp_mean_mag','bpmag','bp')
    RPgaia = _pick(gaia, 'phot_rp_mean_mag','rpmag','rp')

    ZP_G, beta_G, usedG, rmsG = calibrate_zero_point_generic(
        df_det, gaia, df_det['m_inst_G'], target_mag=Ggaia,
        flux_series=df_det['Flux_G'], mag_range=CALIB_G_MAG_RANGE
    )
    if np.isfinite(ZP_G):
        color_all = pd.Series(np.nan, index=df_det.index)
        m = (df_det['matched'] == True)
        color_vals = (BPgaia - RPgaia)
        color_all.loc[m] = color_vals.iloc[df_det.loc[m, 'gaia_index'].astype(int)].values
        med_col = np.nanmedian(color_vals.iloc[df_det.loc[usedG, 'gaia_index'].astype(int)].values) if usedG.any() else 1.0
        color_all = color_all.fillna(med_col)
        df_det['Mag_G'] = df_det['m_inst_G'] + ZP_G + beta_G * color_all
    else:
        df_det['Mag_G'] = np.nan
        print("[calib:G] WARNING: insufficient standards")

    ZP_B, beta_B, usedB, rmsB = calibrate_zero_point_generic(
        df_det, gaia, df_det['m_inst_B'], target_mag=BPgaia,
        flux_series=df_det['Flux_B'], mag_range=CALIB_BP_MAG_RANGE
    )
    if np.isfinite(ZP_B):
        color_all = pd.Series(np.nan, index=df_det.index)
        m = (df_det['matched'] == True)
        color_vals = (BPgaia - RPgaia)
        color_all.loc[m] = color_vals.iloc[df_det.loc[m, 'gaia_index'].astype(int)].values
        med_col = np.nanmedian(color_vals.iloc[df_det.loc[usedB, 'gaia_index'].astype(int)].values) if usedB.any() else 1.0
        color_all = color_all.fillna(med_col)
        df_det['Mag_B'] = df_det['m_inst_B'] + ZP_B + beta_B * color_all
    else:
        df_det['Mag_B'] = np.nan
        print("[calib:B] WARNING: insufficient standards")

    df_det['Mag_Calibrated'] = df_det['Mag_G']

    # ================= CMDs =================
    df_det['B_minus_G'] = (df_det['Mag_B'] - df_det['Mag_G']) - EBG_REDDENING
    mask_color = np.isfinite(df_det['B_minus_G'])

    CMD_APP_PNG = out_png.replace(".png", "_CMD_BG.png")
    mask_app = mask_color & np.isfinite(df_det['Mag_G'])
    mask_app_mem = mask_app & (df_det['ClusterMember'] == True)
    try:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.scatter(df_det.loc[mask_app, 'B_minus_G'], df_det.loc[mask_app, 'Mag_G'],
                   s=6, alpha=0.25, linewidths=0, label='All (B & G available)')
        ax.scatter(df_det.loc[mask_app_mem, 'B_minus_G'], df_det.loc[mask_app_mem, 'Mag_G'],
                   s=14, alpha=0.9, linewidths=0, label='Cluster members')
        ax.set_xlabel('B - G (mag)'); ax.set_ylabel('G (mag)'); ax.invert_yaxis()
        ax.set_title('CMD: G vs (B - G)'); ax.legend(loc='best', frameon=True)
        plt.tight_layout(); plt.savefig(CMD_APP_PNG, dpi=220); plt.close()
        print(f"[out] wrote {CMD_APP_PNG}")
    except Exception as e:
        print(f"[warn] CMD (G vs B-G) failed: {e}")

    CMD_ABS_PNG = out_png.replace(".png", "_CMD_MG_BG.png")
    DM = distance_modulus_from_distance_pc(CLUSTER_DISTANCE_PC)
    if np.isfinite(DM):
        df_det['M_G'] = df_det['Mag_G'] - DM - AG_EXTINCTION
        mask_abs = mask_color & np.isfinite(df_det['M_G'])
        mask_abs_mem = mask_abs & (df_det['ClusterMember'] == True)
        try:
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.scatter(df_det.loc[mask_abs, 'B_minus_G'], df_det.loc[mask_abs, 'M_G'],
                       s=6, alpha=0.25, linewidths=0, label='All (B & G available)')
            ax.scatter(df_det.loc[mask_abs_mem, 'B_minus_G'], df_det.loc[mask_abs_mem, 'M_G'],
                       s=14, alpha=0.9, linewidths=0, label='Cluster members')
            ax.set_xlabel('B - G (mag)'); ax.set_ylabel('M_G (mag)'); ax.invert_yaxis()
            ax.set_title(f'CMD: M_G vs (B - G)  [d={CLUSTER_DISTANCE_PC:.0f} pc, DM={DM:.2f} mag, A_G={AG_EXTINCTION:.2f}, E(B-G)={EBG_REDDENING:.2f}]')
            ax.legend(loc='best', frameon=True)
            plt.tight_layout(); plt.savefig(CMD_ABS_PNG, dpi=220); plt.close()
            print(f"[out] wrote {CMD_ABS_PNG}")
        except Exception as e:
            print(f"[warn] CMD (M_G vs B-G) failed: {e}")
    else:
        df_det['M_G'] = np.nan
        print("[warn] Absolute CMD skipped: set CLUSTER_DISTANCE_PC (pc) to a positive value.")

    # ---------- outputs ----------
    df_det.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[out] wrote {out_csv}")

    matched_mask = (df_det['matched'] == True)
    if matched_mask.any():
        g_take   = gaia.iloc[df_det.loc[matched_mask, 'gaia_index'].astype(int)].reset_index(drop=True)
        det_take = df_det.loc[matched_mask, ['X','Y','RA','DEC',
                                             'Flux_G_Aperture','FluxErr_G','SNR_G','ApertureCorrection_G','Flux_G','m_inst_G','Mag_G',
                                             'Flux_B_Aperture','FluxErr_B','SNR_B','ApertureCorrection_B','Flux_B','m_inst_B','Mag_B',
                                             'ClusterMember','sep_arcsec',
                                             'cluster_pm_center_pmra','cluster_pm_center_pmdec',
                                             'pm_radius_used_masyr','cluster_parallax_center','parallax_window_used_mas']].reset_index(drop=True)
        pd.concat([det_take, g_take], axis=1).to_csv(out_members, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=['X','Y','RA','DEC',
                              'Flux_G_Aperture','FluxErr_G','SNR_G','ApertureCorrection_G','Flux_G','m_inst_G','Mag_G',
                              'Flux_B_Aperture','FluxErr_B','SNR_B','ApertureCorrection_B','Flux_B','m_inst_B','Mag_B',
                              'ClusterMember','sep_arcsec',
                              'cluster_pm_center_pmra','cluster_pm_center_pmdec',
                              'pm_radius_used_masyr','cluster_parallax_center','parallax_window_used_mas']
                     ).to_csv(out_members, index=False, encoding="utf-8-sig")
    print(f"[out] wrote {out_members}")

    # -------- optional overlays using G WCS --------
    try:
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(projection=wcsG)
        vmin = np.nanpercentile(dataG,5); vmax = np.nanpercentile(dataG,99.5)
        ax.imshow(dataG, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        if len(gaia):
            gcoords = SkyCoord(gaia['ra'].values*u.deg, gaia['dec'].values*u.deg)
            gx, gy = wcsG.world_to_pixel(gcoords)
            ax.plot(gx, gy, '+', ms=4, alpha=0.6, transform=ax.get_transform('pixel'))
        ax.set_title("Gaia overlay (diagnostic)")
        diag_png = os.path.join(os.path.dirname(out_png), "gaia_overlay.png")
        plt.savefig(diag_png, dpi=180, bbox_inches="tight"); plt.close()
        print(f"[out] wrote {diag_png}")
    except Exception as e:
        print(f"[warn] Gaia overlay failed: {e}")

    fig = plt.figure(figsize=(10,10))
    ax  = plt.subplot(projection=wcsG)
    vmin = np.nanpercentile(dataG,5); vmax = np.nanpercentile(dataG,99.5)
    ax.imshow(dataG, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    members = df_det[df_det['ClusterMember'] == True]
    for _, r in members.iterrows():
        circ = plt.Circle((r['X'], r['Y']), radius=15, fill=False, edgecolor='red', linewidth=1.2,
                          transform=ax.get_transform('pixel'))
        ax.add_patch(circ)
    ax.set_title("Cluster Members (on G image)")
    plt.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close()
    print(f"[out] wrote {out_png}")
