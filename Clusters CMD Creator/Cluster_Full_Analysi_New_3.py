# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:32:14 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Photometry + CMD with direct BV calibration from APASS DR9 (VizieR, online).
Gaia CSV is optional and used only for membership if you later add it.

Design:
- Detect sources on V image (DAOStarFinder)
- Aperture photometry on V & B at the same centroids
- Convert pixels→RA/DEC via WCS
- Fetch APASS (B,V) around a fixed sky center (HMS/DMS)
- Coarse recenter by medians (handles up to ~degrees shift safely)
- Fine alignment by nearest-neighbour statistics (75″ then 120″)
- Fit ZP + color term vs. (B-V) from APASS; apply to all stars
- Save calibrated table + CMD plot

Requirements (install in the SAME Python environment):
  pip install astroquery photutils astropy scipy pandas matplotlib
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astroquery.vizier import Vizier
from photutils.detection import DAOStarFinder

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================== Hard-coded defaults (editable) ==============================

DEFAULTS = {
    # APASS query center (for catalog only; not from FITS WCS)
    "ra_hms":  "8:51:29",     # H:M:S
    "dec_dms": "+11:49:00",   # D:M:S (sign required)

    # cluster meta
    "cluster_name":        "M67",
    "cluster_distance_pc": 850.0,
    "cluster_type":        "O",   # O=open, G=globular
    "A_V":                 3.33,
    "E_BV":                0.28,

    # files (edit if needed)
    "fits_file_B": r"file:///D:/1 AAA TEMP/Clusters RGB FITS ALL/M 67/M 67-20230123_B_13.fts",
    "fits_file_V": r"D:\1 AAA TEMP\Clusters RGB FITS ALL\M 67\M 67-20230123_G_13.fts",
}

# ============================== Tunables ==============================
# Fine alignment radii (arcsec) – tailored for ~0.75′ offset
R_ALIGN_1_ARCSEC = 75.0      # first pass radius (≈0.75')
R_ALIGN_2_ARCSEC = 120.0     # fallback radius (2')
MIN_ALIGN_PAIRS  = 10        # min pairs to accept a stable median shift

# Photometric calibration match radius (arcsec)
CALIB_MATCH_MAX_ARCSEC = 60.0  # keep tight to avoid false matches

# APASS query radius (arcmin)
APASS_RADIUS_ARCMIN = 45.0

# Photometry knobs
DAO_FWHM         = 5.0
DAO_THRESH_SIGMA = 4.0
PEAK_SNR_MIN     = 10.0
K_APERTURE       = 1.2
K_ANNULUS_IN     = 2.5
K_ANNULUS_OUT    = 4.0
GAIN_E_PER_ADU   = 1.0
READ_NOISE_E     = 5.0

Vizier.ROW_LIMIT = 50000
APASS_CATALOG = "II/336/apass9"

# ============================== Small utilities ==============================

def _ask(prompt, default, cast=str, upper=False):
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "" or s == "0":
        v = default
    else:
        try:
            v = cast(s)
        except Exception:
            v = default
    if upper and isinstance(v, str):
        return v.upper()
    return v

def _norm_path(p: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    if p.lower().startswith("file:///"):
        from urllib.parse import urlparse, unquote
        u_ = urlparse(p); p = unquote(u_.path)
        if len(p) >= 3 and p[0] == "/" and p[2] == ":":
            p = p[1:]
    return os.path.normpath(p)

def _hms_to_deg(hms: str) -> float:
    h, m, s = [float(x) for x in hms.replace(" ", "").split(":")]
    return (h + m/60.0 + s/3600.0) * 15.0

def _dms_to_deg(dms: str) -> float:
    sgn = -1.0 if dms.strip().startswith("-") else 1.0
    dms_ = dms.replace("+", "").replace("-", "").replace(" ", "")
    d, m, s = [float(x) for x in dms_.split(":")]
    return sgn * (abs(d) + m/60.0 + s/3600.0)

def distance_modulus_from_distance_pc(d_pc: float) -> float:
    return 5.0 * np.log10(max(float(d_pc), 1e-6)) - 5.0

DMOD = None  # set in main()

# ============================== Photometry helpers ==============================

def _cutout(img, x, y, r_out):
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
        std  = float(np.nanstd(vals)) if vals.size > 0 else float(np.nanstd(cut))
        return mean, std
    mean, med, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
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
    # simple variance model
    src_e = max(0.0, np.nansum(ap_vals) * gain_e_per_adu - bkg_mean * gain_e_per_adu * npix_ap)
    bkg_var_e = (bkg_std * gain_e_per_adu)**2
    var_e = max(0.0, src_e) + npix_ap * bkg_var_e + npix_ap * (read_noise_e**2)
    flux_err_adu = (np.sqrt(var_e) / gain_e_per_adu) if var_e > 0 else 0.0
    snr = (flux_ap_adu / flux_err_adu) if flux_err_adu > 0 else 0.0
    return flux_ap_adu, float(flux_err_adu), float(snr), float(bkg_mean), float(bkg_std), int(npix_ap)

def detect_on_image(img: np.ndarray, fwhm_pix=DAO_FWHM, thresh_sigma=DAO_THRESH_SIGMA, peak_snr_min=PEAK_SNR_MIN):
    mean, med, std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=fwhm_pix, threshold=thresh_sigma*std)
    tbl = daofind(img - med)
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame(columns=["X","Y","SNR_est"])
    df = tbl.to_pandas()
    df["SNR_est"] = df["peak"] / max(std, 1e-6)
    df = df[df["SNR_est"] >= peak_snr_min]
    return df.rename(columns={"xcentroid":"X","ycentroid":"Y"})[["X","Y","SNR_est"]].reset_index(drop=True)

def wcs_pixels_to_radec(wcs: WCS, xpix, ypix):
    coords = wcs.pixel_to_world(np.asarray(xpix), np.asarray(ypix))
    return np.asarray(coords.ra.deg), np.asarray(coords.dec.deg)

# ============================== APASS ==============================

def fetch_apass_catalog(ra_deg: float, dec_deg: float, radius_arcmin: float=APASS_RADIUS_ARCMIN) -> pd.DataFrame:
    center = SkyCoord(ra_deg*u.deg, dec_deg*u.deg)
    r = (radius_arcmin * u.arcmin)
    cols = ["RAJ2000","DEJ2000","Bmag","Vmag","e_Bmag","e_Vmag"]
    res = Vizier(columns=cols).query_region(center, radius=r, catalog=APASS_CATALOG)
    if len(res) == 0:
        return pd.DataFrame(columns=["ra","dec","Bmag","Vmag","e_Bmag","e_Vmag"])
    t = res[0].to_pandas()
    df = pd.DataFrame({
        "ra":    pd.to_numeric(t.get("RAJ2000"), errors='coerce'),
        "dec":   pd.to_numeric(t.get("DEJ2000"), errors='coerce'),
        "Bmag":  pd.to_numeric(t.get("Bmag"), errors='coerce'),
        "Vmag":  pd.to_numeric(t.get("Vmag"), errors='coerce'),
        "e_Bmag":pd.to_numeric(t.get("e_Bmag"), errors='coerce'),
        "e_Vmag":pd.to_numeric(t.get("e_Vmag"), errors='coerce')
    })
    df = df[np.isfinite(df["Bmag"]) & np.isfinite(df["Vmag"])]
    return df.reset_index(drop=True)

# ============================== Alignment ==============================

def print_nn_stats(det_df, apass_df, label):
    det = SkyCoord(pd.to_numeric(det_df['RA'],errors='coerce').values*u.deg,
                   pd.to_numeric(det_df['DEC'],errors='coerce').values*u.deg)
    cat = SkyCoord(pd.to_numeric(apass_df['ra'],errors='coerce').values*u.deg,
                   pd.to_numeric(apass_df['dec'],errors='coerce').values*u.deg)
    _, d2d, _ = det.match_to_catalog_sky(cat)
    a = d2d.arcsec   # this is already a numpy array
    a = a[np.isfinite(a)]
    if a.size:
        print(f"[{label}] NN→APASS: min={np.min(a):.1f}\"  "
              f"p50={np.median(a):.1f}\"  "
              f"p95={np.percentile(a,95):.1f}\"  "
              f"n={a.size}")
    else:
        print(f"[{label}] NN→APASS: no finite distances")


def coarse_recenter_wcs_to_apass(det_df, wcsV, ra_c, dec_c):
    """
    Coarse recenter using image WCS center -> APASS center vector (tangent plane).
    This avoids median-based sign mistakes.
    """
    # image center (in pixels)
    ny, nx = wcsV.array_shape
    ra_det_c, dec_det_c = wcsV.pixel_to_world(nx/2, ny/2).ra.deg, wcsV.pixel_to_world(nx/2, ny/2).dec.deg

    c_det  = SkyCoord(ra_det_c*u.deg, dec_det_c*u.deg)
    c_cat  = SkyCoord(float(ra_c)*u.deg, float(dec_c)*u.deg)

    # shift vector in tangent plane centered at image center
    off_cat = c_cat.transform_to(SkyOffsetFrame(origin=c_det))
    dx = float(off_cat.lon.deg*3600.0)   # arcsec to +X
    dy = float(off_cat.lat.deg*3600.0)   # arcsec to +Y

    # apply to all detections
    det_all = SkyCoord(pd.to_numeric(det_df['RA'],errors='coerce').values*u.deg,
                       pd.to_numeric(det_df['DEC'],errors='coerce').values*u.deg)
    off0 = det_all.transform_to(SkyOffsetFrame(origin=c_det))
    x = off0.lon.deg*3600.0 + dx
    y = off0.lat.deg*3600.0 + dy
    r  = np.hypot(x,y)*u.arcsec
    th = np.arctan2(y,x)*u.rad
    new = c_det.directional_offset_by(th, r.to(u.deg))

    out = det_df.copy()
    out['RA']  = new.ra.deg
    out['DEC'] = new.dec.deg
    return out, {"status":"ok","dx_arcsec":dx,"dy_arcsec":dy,
                 "img_ctr":(ra_det_c,dec_det_c)}


def estimate_and_apply_offset_1arcmin(center_ra_deg, center_dec_deg, det_df, apass_df,
                                      r1_arcsec=R_ALIGN_1_ARCSEC,
                                      r2_arcsec=R_ALIGN_2_ARCSEC,
                                      min_pairs=MIN_ALIGN_PAIRS):
    center = SkyCoord(center_ra_deg*u.deg, center_dec_deg*u.deg)
    det  = SkyCoord(pd.to_numeric(det_df["RA"],errors='coerce').values*u.deg,
                    pd.to_numeric(det_df["DEC"],errors='coerce').values*u.deg)
    cat  = SkyCoord(pd.to_numeric(apass_df["ra"],errors='coerce').values*u.deg,
                    pd.to_numeric(apass_df["dec"],errors='coerce').values*u.deg)

    def _one_pass(radius_arcsec):
        idx, d2d, _ = det.match_to_catalog_sky(cat)
        ok = d2d.arcsec <= radius_arcsec
        if ok.sum() == 0:
            return None
        off_det = det[ok].transform_to(SkyOffsetFrame(origin=center))
        off_cat = cat[idx[ok]].transform_to(SkyOffsetFrame(origin=center))
        dx = (off_cat.lon.deg - off_det.lon.deg) * 3600.0
        dy = (off_cat.lat.deg - off_det.lat.deg) * 3600.0
        return {"N": int(ok.sum()), "dx": float(np.nanmedian(dx)), "dy": float(np.nanmedian(dy))}

    d = _one_pass(r1_arcsec)
    if d is None or d["N"] < min_pairs:
        d = _one_pass(r2_arcsec)
    if d is None or d["N"] < max(6, min_pairs//2):
        return det_df, {"status":"fail"}

    off = det.transform_to(SkyOffsetFrame(origin=center))
    x = off.lon.deg*3600.0 + d["dx"]
    y = off.lat.deg*3600.0 + d["dy"]
    r = np.hypot(x,y) * u.arcsec
    th = np.arctan2(y,x) * u.rad
    new = center.directional_offset_by(th, r.to(u.deg))
    det_df = det_df.copy()
    det_df["RA"] = new.ra.deg
    det_df["DEC"] = new.dec.deg
    d["status"] = "ok"
    return det_df, d

# ============================== Calibration ==============================

def robust_linfit(y, x, clip=3.0, maxiter=10):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    a=b=np.nan
    for _ in range(maxiter):
        if mask.sum() < 5:
            break
        X = np.vstack([np.ones(mask.sum()), x[mask]]).T
        a, b = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        resid = y - (a + b*x)
        s = np.nanstd(resid[mask])
        mask = mask & (np.abs(resid) <= clip*s if s>0 else np.isfinite(resid))
    rms = float(np.nanstd((y - (a + b*x))[mask])) if mask.sum()>=2 else np.nan
    return float(a), float(b), mask, rms

def calibrate_band_from_apass(det_df, apass_df, inst_mag_col, std_mag_col,
                              match_arcsec=CALIB_MATCH_MAX_ARCSEC):
    det_coords = SkyCoord(pd.to_numeric(det_df['RA'],errors='coerce').values*u.deg,
                          pd.to_numeric(det_df['DEC'],errors='coerce').values*u.deg)
    cat_coords = SkyCoord(pd.to_numeric(apass_df['ra'],errors='coerce').values*u.deg,
                          pd.to_numeric(apass_df['dec'],errors='coerce').values*u.deg)
    idx, d2d, _ = det_coords.match_to_catalog_sky(cat_coords)
    ok = d2d.arcsec <= match_arcsec
    if ok.sum() < 8:
        return np.nan, np.nan, np.zeros(len(det_df), dtype=bool), np.nan

    df = det_df.loc[ok].copy().reset_index(drop=True)
    cat = apass_df.iloc[idx[ok]].reset_index(drop=True)

    color = (cat['Bmag'] - cat['Vmag']).to_numpy(float)
    tmag = cat[std_mag_col].to_numpy(float)
    m_inst= df[inst_mag_col].to_numpy(float)

    q = np.isfinite(color) & np.isfinite(tmag) & np.isfinite(m_inst)
    if q.sum() < 8:
        return np.nan, np.nan, np.zeros(len(det_df), dtype=bool), np.nan

    y = tmag[q] - m_inst[q]; x = color[q]
    ZP, beta, used, rms = robust_linfit(y, x)
    use_mask = np.zeros(len(det_df), dtype=bool)
    use_mask[np.where(ok)[0][q][used]] = True
    return ZP, beta, use_mask, rms

# ============================== Main ==============================

def main():
    global DMOD
    print("=== Cluster Analysis with APASS BV Calibration (online) ===")

    cluster_name        = _ask("Cluster name", DEFAULTS["cluster_name"], str)
    cluster_distance_pc = _ask("Cluster distance (pc)", DEFAULTS["cluster_distance_pc"], float)
    cluster_type        = _ask("Cluster type [O=open, G=globular]", DEFAULTS["cluster_type"], str, upper=True)
    A_V                 = _ask("Galactic extinction A_V", DEFAULTS["A_V"], float)
    E_BV                = _ask("Galactic reddening E(B-V)", DEFAULTS["E_BV"], float)

    fits_file_B = _norm_path(_ask("Path to B-band FITS image", DEFAULTS["fits_file_B"], str))
    fits_file_V = _norm_path(_ask("Path to V-band/G-band FITS image (treated as V)", DEFAULTS["fits_file_V"], str))

    # Fixed APASS query center (per requirement)
    ra_c  = _hms_to_deg(DEFAULTS["ra_hms"])
    dec_c = _dms_to_deg(DEFAULTS["dec_dms"])
    print(f"APASS center: RA={ra_c:.6f}  DEC={dec_c:.6f}  radius={APASS_RADIUS_ARCMIN:.1f}'")

    DMOD = distance_modulus_from_distance_pc(cluster_distance_pc)

    # --- Read FITS and WCS
    with fits.open(fits_file_V) as h: imgV=h[0].data.astype(float); hdrV=h[0].header
    with fits.open(fits_file_B) as h: imgB=h[0].data.astype(float); hdrB=h[0].header
    wcsV = WCS(hdrV)

    # --- Detect on V
    det = detect_on_image(imgV)
    if len(det) == 0:
        print("No detections on V image."); return

    # --- Photometric apertures (from FWHM proxy)
    FWHM_est = DAO_FWHM
    r_ap  = K_APERTURE   * FWHM_est
    r_in  = K_ANNULUS_IN * FWHM_est
    r_out = K_ANNULUS_OUT* FWHM_est

    # --- Photometry on V & B at the same centroids
    fluxV, fluxB = [], []
    for x, y in det[['X','Y']].to_numpy():
        fV, *_ = aperture_photometry_single_fast(imgV, x, y, r_ap, r_in, r_out, GAIN_E_PER_ADU, READ_NOISE_E)
        fB, *_ = aperture_photometry_single_fast(imgB, x, y, r_ap, r_in, r_out, GAIN_E_PER_ADU, READ_NOISE_E)
        fluxV.append(fV); fluxB.append(fB)
    det['flux_V'] = np.array(fluxV, float)
    det['flux_B'] = np.array(fluxB, float)

    # --- Instrumental magnitudes (normalize by EXPTIME if present)
    def _get_exptime(h):
        for k in ('EXPTIME','EXPOSURE','EXPT','ITIME'):
            if k in h:
                try:
                    v=float(h[k]); 
                    if np.isfinite(v) and v>0: return v
                except Exception: pass
        return 1.0
    tV=_get_exptime(hdrV); tB=_get_exptime(hdrB)
    with np.errstate(divide='ignore', invalid='ignore'):
        det['m_inst_V'] = -2.5*np.log10(np.maximum(det['flux_V']/tV, 1e-12))
        det['m_inst_B'] = -2.5*np.log10(np.maximum(det['flux_B']/tB, 1e-12))

    # --- Pixels → RA/DEC via WCS
    raV, decV = wcs_pixels_to_radec(wcsV, det['X'], det['Y'])
    det['RA']  = pd.to_numeric(raV,  errors='coerce')
    det['DEC'] = pd.to_numeric(decV, errors='coerce')

    # --- Fetch APASS around fixed center
    apass = fetch_apass_catalog(ra_c, dec_c, radius_arcmin=APASS_RADIUS_ARCMIN)
    if len(apass)==0:
        print("APASS returned zero rows."); return

    # --- Alignment diagnostics + alignment passes
    print_nn_stats(det, apass, "before-align")

    # Coarse: WCS center -> APASS center
    det, coarse = coarse_recenter_wcs_to_apass(det, wcsV, ra_c, dec_c)
    print("Coarse (WCS→APASS):", coarse)
    print_nn_stats(det, apass, "after-coarse")
    
    det, diag = estimate_and_apply_offset_1arcmin(
        ra_c, dec_c, det, apass,
        r1_arcsec=45.0,   # 1′
        r2_arcsec=120.0,  # 2′
        min_pairs=10
    )
    print("Fine offset diagnostic:", diag)
    print_nn_stats(det, apass, "after-fine")


    det, diag = estimate_and_apply_offset_1arcmin(ra_c, dec_c, det, apass,
                                                  r1_arcsec=R_ALIGN_1_ARCSEC,
                                                  r2_arcsec=R_ALIGN_2_ARCSEC,
                                                  min_pairs=MIN_ALIGN_PAIRS)
    print("Fine offset diagnostic:", diag)
    print_nn_stats(det, apass, "after-fine")

    # --- Photometric calibration (APASS)
    ZP_V, beta_V, _, rms_V = calibrate_band_from_apass(det, apass, 'm_inst_V', 'Vmag',
                                                       match_arcsec=CALIB_MATCH_MAX_ARCSEC)
    ZP_B, beta_B, _, rms_B = calibrate_band_from_apass(det, apass, 'm_inst_B', 'Bmag',
                                                       match_arcsec=CALIB_MATCH_MAX_ARCSEC)
    print(f"V: ZP={ZP_V:.3f}, beta(B-V)={beta_V:.3f}, rms={rms_V:.3f}")
    print(f"B: ZP={ZP_B:.3f}, beta(B-V)={beta_B:.3f}, rms={rms_B:.3f}")

    if not np.isfinite(ZP_V) or not np.isfinite(ZP_B):
        print("Calibration failed (no finite ZP). Check alignment diagnostics above.")
        # still save raw table for inspection
    # Apply calibration (one-pass color-term using BV_0 proxy)
    det['V_cal_0'] = det['m_inst_V'] + (ZP_V if np.isfinite(ZP_V) else 0.0)
    det['B_cal_0'] = det['m_inst_B'] + (ZP_B if np.isfinite(ZP_B) else 0.0)
    det['BV_0']    = det['B_cal_0'] - det['V_cal_0']

    det['V_mag'] = det['m_inst_V'] + (ZP_V if np.isfinite(ZP_V) else 0.0) + ((beta_V if np.isfinite(beta_V) else 0.0) * det['BV_0'])
    det['B_mag'] = det['m_inst_B'] + (ZP_B if np.isfinite(ZP_B) else 0.0) + ((beta_B if np.isfinite(beta_B) else 0.0) * det['BV_0'])
    det['B_minus_V'] = det['B_mag'] - det['V_mag']

    # --- Extinction corrections + absolute magnitude
    det['BV0'] = det['B_minus_V'] - E_BV
    det['V0']  = det['V_mag'] - A_V
    det['M_V'] = det['V_mag'] - DMOD - A_V

    # --- Save outputs
    out_dir = os.path.dirname(fits_file_V) if os.path.dirname(fits_file_V) else os.getcwd()
    out_csv  = os.path.join(out_dir, "photometry_with_membership.csv")
    out_plot = os.path.join(out_dir, "cluster_cmd_apass.png")
    det.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # --- Plot CMD (apparent)
    plt.figure(figsize=(6,7))
    m = np.isfinite(det['B_minus_V']) & np.isfinite(det['V_mag'])
    plt.scatter(det.loc[m,'B_minus_V'], det.loc[m,'V_mag'], s=6, alpha=0.6)
    plt.gca().invert_yaxis(); plt.xlabel('B−V'); plt.ylabel('V')
    plt.title(f'{cluster_name} — CMD (APASS-calibrated)')
    plt.tight_layout(); plt.savefig(out_plot, dpi=160)
    print(f"Saved plot: {out_plot}")

if __name__=="__main__":
    main()
