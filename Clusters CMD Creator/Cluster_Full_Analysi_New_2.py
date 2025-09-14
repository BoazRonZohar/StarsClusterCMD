# -*- coding: utf-8 -*-
"""
Photometry + CMD with direct BV calibration from APASS DR9 (VizieR, online).
Gaia CSV is optional and used only for membership matching if provided.
No Gaia→BV transformations are used for photometric calibration.

Install in the SAME Python env:
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

# ============================== Defaults (editable) ==============================

DEFAULTS = {
    "ra_hms":  "8:51:29",      # APASS query center (H:M:S)
    "dec_dms": "+11:49:00",    # APASS query center (±D:M:S)

    "cluster_name":        "M67",
    "cluster_distance_pc": 850.0,
    "cluster_type":        "O",
    "A_V":                 3.33,
    "E_BV":                0.28,

    "fits_file_B": r"file:///D:/1 AAA TEMP/Clusters RGB FITS ALL/M 67/M 67-20230123_B_13.fts",
    "fits_file_V": r"D:\1 AAA TEMP\Clusters RGB FITS ALL\M 67\M 67-20230123_G_13.fts",
    "gaia_csv":    r"0",  # "0" to skip
}

# ============================== Helpers ==============================

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

# ============================== Tunables ==============================

DAO_FWHM         = 5.0
DAO_THRESH_SIGMA = 4.0
PEAK_SNR_MIN     = 10.0
K_APERTURE       = 1.2
K_ANNULUS_IN     = 2.5
K_ANNULUS_OUT    = 4.0
GAIN_E_PER_ADU   = 1.0
READ_NOISE_E     = 5.0

CALIB_MATCH_MAX_ARCSEC  = 60.0
APASS_RADIUS_ARCMIN     = 45.0
R_ALIGN_1_ARCSEC        = 120.0
R_ALIGN_2_ARCSEC        = 120.0
MIN_ALIGN_PAIRS         = 10

Vizier.ROW_LIMIT = 50000
APASS_CATALOG = "II/336/apass9"


# ============================== Photometry ==============================

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
    try:
        res = Vizier(columns=cols).query_region(center, radius=r, catalog=APASS_CATALOG)
    except Exception as e:
        print(f"[APASS] Query failed: {e}")
        return pd.DataFrame(columns=["ra","dec","Bmag","Vmag","e_Bmag","e_Vmag"])
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
    df = df[np.isfinite(df["ra"]) & np.isfinite(df["dec"]) & np.isfinite(df["Bmag"]) & np.isfinite(df["Vmag"])]
    return df.reset_index(drop=True)

# ============================== Alignment (≤1′, fallback 2′) ==============================

def estimate_and_apply_offset_1arcmin(center_ra_deg, center_dec_deg, det_df, apass_df,
                                      r1_arcsec=R_ALIGN_1_ARCSEC,
                                      r2_arcsec=R_ALIGN_2_ARCSEC,
                                      min_pairs=MIN_ALIGN_PAIRS):
    center = SkyCoord(float(center_ra_deg)*u.deg, float(center_dec_deg)*u.deg)

    ra_det  = pd.to_numeric(det_df["RA"], errors="coerce").values
    dec_det = pd.to_numeric(det_df["DEC"], errors="coerce").values
    ra_cat  = pd.to_numeric(apass_df["ra"], errors="coerce").values
    dec_cat = pd.to_numeric(apass_df["dec"], errors="coerce").values

    det  = SkyCoord(ra_det*u.deg,  dec_det*u.deg)
    cat  = SkyCoord(ra_cat*u.deg,  dec_cat*u.deg)

    def _one_pass(radius_arcsec):
        idx, d2d, _ = det.match_to_catalog_sky(cat)
        ok = d2d.arcsec <= radius_arcsec
        if ok.sum() == 0:
            return None
        off_det = det[ok].transform_to(SkyOffsetFrame(origin=center))
        off_cat = cat[idx[ok]].transform_to(SkyOffsetFrame(origin=center))
        dx = (off_cat.lon.deg - off_det.lon.deg) * 3600.0
        dy = (off_cat.lat.deg - off_det.lat.deg) * 3600.0
        return {"N": int(ok.sum()),
                "dx": float(np.nanmedian(dx)),
                "dy": float(np.nanmedian(dy)),
                "min": float(np.nanmin(d2d.arcsec)),
                "med": float(np.nanmedian(d2d.arcsec)),
                "rad": float(radius_arcsec)}

    d = _one_pass(r1_arcsec)
    if (d is None) or (d["N"] < min_pairs):
        d2 = _one_pass(r2_arcsec)
        if (d2 is not None) and (d2["N"] >= max(10, min_pairs//2)):
            d = d2

    if (d is None) or (d["N"] < 10):
        return det_df, {"status":"fail"}

    off = det.transform_to(SkyOffsetFrame(origin=center))
    x = off.lon.deg*3600.0 + d["dx"]
    y = off.lat.deg*3600.0 + d["dy"]
    r = np.hypot(x,y) * u.arcsec
    th = np.arctan2(y,x) * u.rad
    new = center.directional_offset_by(th, r.to(u.deg))
    det_df = det_df.copy()
    det_df["RA"]  = new.ra.deg
    det_df["DEC"] = new.dec.deg
    d["status"] = "ok"
    return det_df, d

# ============================== Calibration ==============================

def robust_linfit(y, x, clip=3.0, maxiter=10):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    a=b=np.nan
    for _ in range(maxiter):
        if mask.sum() < 5: break
        X = np.vstack([np.ones(mask.sum()), x[mask]]).T
        a, b = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        resid = y - (a + b*x)
        s = np.nanstd(resid[mask])
        new_mask = mask & (np.abs(resid) <= clip*s if s>0 else np.isfinite(resid))
        if new_mask.sum() == mask.sum(): break
        mask = new_mask
    rms = float(np.nanstd((y - (a + b*x))[mask])) if mask.sum()>=2 else np.nan
    return float(a), float(b), mask, rms

def calibrate_band_from_apass(det_df, apass_df, inst_mag_col, std_mag_col,
                              match_arcsec=CALIB_MATCH_MAX_ARCSEC, mag_range=(10, 18)):
    # Force numeric RA/DEC to avoid Angle unit errors
    ra_det  = pd.to_numeric(det_df['RA'], errors="coerce").values
    dec_det = pd.to_numeric(det_df['DEC'], errors="coerce").values
    ra_cat  = pd.to_numeric(apass_df['ra'], errors="coerce").values
    dec_cat = pd.to_numeric(apass_df['dec'], errors="coerce").values

    det_coords = SkyCoord(ra_det*u.deg, dec_det*u.deg)
    cat_coords = SkyCoord(ra_cat*u.deg, dec_cat*u.deg)

    idx, d2d, _ = det_coords.match_to_catalog_sky(cat_coords)
    ok = d2d.arcsec <= match_arcsec
    if ok.sum() < 8:
        return np.nan, np.nan, np.zeros(len(det_df), dtype=bool), np.nan

    df  = det_df.loc[ok].copy().reset_index(drop=True)
    cat = apass_df.iloc[idx[ok]].reset_index(drop=True)

    color = (pd.to_numeric(cat['Bmag'], errors='coerce') - pd.to_numeric(cat['Vmag'], errors='coerce')).to_numpy(float)
    tmag  = pd.to_numeric(cat[std_mag_col], errors='coerce').to_numpy(float)
    m_inst= pd.to_numeric(df[inst_mag_col], errors='coerce').to_numpy(float)

    q = np.isfinite(color) & np.isfinite(tmag) & np.isfinite(m_inst)
    if mag_range is not None:
        lo, hi = float(mag_range[0]), float(mag_range[1])
        q &= (tmag >= lo) & (tmag <= hi)
    if q.sum() < 8:
        return np.nan, np.nan, np.zeros(len(det_df), dtype=bool), np.nan

    ZP, beta, used, rms = robust_linfit(tmag[q] - m_inst[q], color[q], clip=3.0, maxiter=10)

    use_mask = np.zeros(len(det_df), dtype=bool)
    use_mask[np.where(ok)[0][q][used]] = True
    return ZP, beta, use_mask, rms

# ============================== Main ==============================

def main():
    print("=== Cluster Analysis with APASS BV Calibration (online) ===")

    cluster_name        = _ask("Cluster name", DEFAULTS["cluster_name"], str)
    cluster_distance_pc = _ask("Cluster distance (pc)", DEFAULTS["cluster_distance_pc"], float)
    cluster_type        = _ask("Cluster type [O=open, G=globular]", DEFAULTS["cluster_type"], str, upper=True)
    A_V                 = _ask("Galactic extinction A_V", DEFAULTS["A_V"], float)
    E_BV                = _ask("Galactic reddening E(B-V)", DEFAULTS["E_BV"], float)

    fits_file_B_in = _ask("Path to B-band FITS image", DEFAULTS["fits_file_B"], str)
    fits_file_V_in = _ask("Path to V-band/G-band FITS image (treated as V)", DEFAULTS["fits_file_V"], str)

    fits_file_B = _norm_path(fits_file_B_in)
    fits_file_V = _norm_path(fits_file_V_in)

    out_dir  = os.path.dirname(fits_file_V) if os.path.dirname(fits_file_V) else os.getcwd()
    out_csv  = os.path.join(out_dir, "photometry_with_membership.csv")
    out_plot = os.path.join(out_dir, "cluster_cmd_apass.png")
    os.makedirs(out_dir, exist_ok=True)

    # APASS query center from hard-coded HMS/DMS
    ra_c = _hms_to_deg(DEFAULTS["ra_hms"])
    dec_c = _dms_to_deg(DEFAULTS["dec_dms"])
    print(f"APASS center: RA={ra_c:.6f}  DEC={dec_c:.6f}  radius={APASS_RADIUS_ARCMIN:.1f}'")

    # Read FITS
    assert os.path.exists(fits_file_V), f"V FITS not found: {fits_file_V}"
    assert os.path.exists(fits_file_B), f"B FITS not found: {fits_file_B}"
    with fits.open(fits_file_V) as h: imgV = h[0].data.astype(float); hdrV = h[0].header
    with fits.open(fits_file_B) as h: imgB = h[0].data.astype(float); hdrB = h[0].header
    wcsV = WCS(hdrV)

    # Detect on V
    det = detect_on_image(imgV, fwhm_pix=DAO_FWHM, thresh_sigma=DAO_THRESH_SIGMA, peak_snr_min=PEAK_SNR_MIN)
    if len(det) == 0:
        print("No detections on V image."); return

    # Convert to RA/DEC via WCS (force float dtype)
    raV, decV = wcs_pixels_to_radec(wcsV, det['X'], det['Y'])
    det['RA']  = pd.to_numeric(raV,  errors='coerce')
    det['DEC'] = pd.to_numeric(decV, errors='coerce')

    # APASS catalog around fixed center
    apass = fetch_apass_catalog(ra_c, dec_c, radius_arcmin=APASS_RADIUS_ARCMIN)
    if len(apass) == 0:
        print("APASS returned zero rows."); return

    # Estimate bulk offset ≤1′ (fallback 2′) and correct RA/DEC
    det, diag = estimate_and_apply_offset_1arcmin(ra_c, dec_c, det, apass,
                                              r1_arcsec=R_ALIGN_1_ARCSEC,
                                              r2_arcsec=R_ALIGN_2_ARCSEC,
                                              min_pairs=MIN_ALIGN_PAIRS)
    print("Fine offset diagnostic:", diag)

    # Aperture photometry (use same centroids on both images)
    FWHM_est = DAO_FWHM
    r_ap  = K_APERTURE    * FWHM_est
    r_in  = K_ANNULUS_IN  * FWHM_est
    r_out = K_ANNULUS_OUT * FWHM_est

    fluxV, fluxB = [], []
    for x, y in det[['X','Y']].to_numpy():
        fV, *_ = aperture_photometry_single_fast(imgV, x, y, r_ap, r_in, r_out, GAIN_E_PER_ADU, READ_NOISE_E)
        fB, *_ = aperture_photometry_single_fast(imgB, x, y, r_ap, r_in, r_out, GAIN_E_PER_ADU, READ_NOISE_E)
        fluxV.append(fV); fluxB.append(fB)
    det['flux_V'] = np.array(fluxV, float)
    det['flux_B'] = np.array(fluxB, float)

    # Instrumental magnitudes (normalize by exposure time if exists)
    def _exptime(h):
        for k in ('EXPTIME','EXPOSURE','ITIME','EXPT'):
            if k in h:
                try:
                    v = float(h[k]); 
                    if np.isfinite(v) and v>0: return v
                except Exception: 
                    pass
        return 1.0
    tV = _exptime(hdrV); tB = _exptime(hdrB)
    with np.errstate(divide='ignore', invalid='ignore'):
        det['m_inst_V'] = -2.5*np.log10(np.maximum(det['flux_V'], 1e-12)/tV)
        det['m_inst_B'] = -2.5*np.log10(np.maximum(det['flux_B'], 1e-12)/tB)

    # Calibrate from APASS (robust ZP + color-term)
    ZP_V,beta_V,_,rms_V = calibrate_band_from_apass(det, apass, 'm_inst_V', 'Vmag',
                                                    match_arcsec=CALIB_MATCH_MAX_ARCSEC, mag_range=(10,17.5))
    ZP_B,beta_B,_,rms_B = calibrate_band_from_apass(det, apass, 'm_inst_B', 'Bmag',
                                                    match_arcsec=CALIB_MATCH_MAX_ARCSEC, mag_range=(10,18.0))
    print(f"V: ZP={ZP_V:.3f}, beta(B-V)={beta_V:.3f}, rms={rms_V if np.isfinite(rms_V) else np.nan:.3f}")
    print(f"B: ZP={ZP_B:.3f}, beta(B-V)={beta_B:.3f}, rms={rms_B if np.isfinite(rms_B) else np.nan:.3f}")

    # Apply calibration (single iteration using BV_0)
    det['V_cal_0'] = det['m_inst_V'] + (ZP_V if np.isfinite(ZP_V) else 0.0)
    det['B_cal_0'] = det['m_inst_B'] + (ZP_B if np.isfinite(ZP_B) else 0.0)
    det['BV_0']    = det['B_cal_0'] - det['V_cal_0']
    det['V_mag']   = det['m_inst_V'] + ZP_V + (beta_V * det['BV_0'] if np.isfinite(beta_V) else 0.0)
    det['B_mag']   = det['m_inst_B'] + ZP_B + (beta_B * det['BV_0'] if np.isfinite(beta_B) else 0.0)
    det['B_minus_V'] = det['B_mag'] - det['V_mag']

    # Extinction-corrected and absolute
    DMOD = distance_modulus_from_distance_pc(cluster_distance_pc)
    det['BV0'] = det['B_minus_V'] - E_BV
    det['V0']  = det['V_mag'] - A_V
    det['M_V'] = det['V_mag'] - DMOD - A_V

    # Save
    det.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plot CMD (apparent)
    m = np.isfinite(det['B_minus_V']) & np.isfinite(det['V_mag'])
    plt.figure(figsize=(6,7))
    plt.scatter(det.loc[m,'B_minus_V'], det.loc[m,'V_mag'], s=6, alpha=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel('B − V'); plt.ylabel('V')
    plt.title(f'{cluster_name} — CMD (APASS calibrated)')
    plt.tight_layout(); plt.savefig(out_plot, dpi=160)
    print(f"Saved plot: {out_plot}")

if __name__ == "__main__":
    main()
