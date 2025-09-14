"""
Photometry + CMD with direct BV calibration from APASS DR9 (via VizieR, online).
Gaia is optional and used only for astrometry/membership if a CSV is provided.
No Gaia→BV color transformations are used for photometric calibration.

Install (same Python env as Spyder):
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
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================== Hard-coded defaults (editable) ==============================

DEFAULTS = {
    # sky center for APASS query (used STRICTLY for catalog query, not from FITS WCS)
    "ra_hms":  "8:51:29",     # H:M:S
    "dec_dms": "+11:49:00",   # D:M:S (sign required)

    # cluster meta
    "cluster_name":        "M67",
    "cluster_distance_pc": 850.0,
    "cluster_type":        "O",      # O=open, G=globular
    "A_V":                 3.33,
    "E_BV":                0.28,

    # file paths
    "fits_file_B": r"file:///D:/1 AAA TEMP/Clusters RGB FITS ALL/M 67/M 67-20230123_B_13.fts",
    "fits_file_V": r"D:\1 AAA TEMP\Clusters RGB FITS ALL\M 67\M 67-20230123_G_13.fts",
    "gaia_csv":    r"D:\1 AAA TEMP\Clusters RGB FITS ALL\M 67\1757251169166O-result.csv",  # put "0" to skip
}

# ============================== User I/O helpers ==============================

def _ask(prompt, default, cast=str, upper=False):
    """Prompt with default; Enter keeps default. Returns cast(value)."""
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
    """Normalize a path: strip quotes and handle file:/// URIs."""
    p = str(p).strip().strip('"').strip("'")
    if p.lower().startswith("file:///"):
        from urllib.parse import urlparse, unquote
        u_ = urlparse(p); p = unquote(u_.path)
        if len(p) >= 3 and p[0] == "/" and p[2] == ":":
            p = p[1:]
    return os.path.normpath(p)

def _hms_to_deg(hms: str) -> float:
    """Convert 'H:M:S' to decimal degrees."""
    h, m, s = [float(x) for x in hms.replace(" ", "").split(":")]
    return (h + m/60.0 + s/3600.0) * 15.0

def _dms_to_deg(dms: str) -> float:
    """Convert '±D:M:S' to decimal degrees."""
    sgn = -1.0 if dms.strip().startswith("-") else 1.0
    dms_ = dms.replace("+", "").replace("-", "").replace(" ", "")
    d, m, s = [float(x) for x in dms_.split(":")]
    return sgn * (abs(d) + m/60.0 + s/3600.0)

# ============================== Parameters ==============================

# Photometry
DAO_FWHM         = 5.0
DAO_THRESH_SIGMA = 4.0
PEAK_SNR_MIN     = 10.0
K_APERTURE       = 1.2
K_ANNULUS_IN     = 2.5
K_ANNULUS_OUT    = 4.0
GAIN_E_PER_ADU   = 1.0
READ_NOISE_E     = 5.0

# Matching / membership
MATCH_MAX_ARCSEC        = 6.0
CALIB_MATCH_MAX_ARCSEC  = 60.0   # generous to tolerate modest WCS offsets
APASS_RADIUS_ARCMIN     = 90.0   # wide cone to ensure standards
PM_SIGMA_FLOOR          = 1.0
PM_ERR_SIGMA            = 5.0
PARALLAX_MIN_HALF       = 0.10
PARALLAX_MAX_HALF       = 1.00
PARALLAX_FRAC           = 20.0   # percent of parallax center
PARALLAX_ERR_SIGMA      = 4.0
SPATIAL_CORE_FRAC       = 0.15
ALIGN_SEARCH_ARCSEC     = 300.0
ALIGN_MIN_PAIRS         = 20

Vizier.ROW_LIMIT = 50000
APASS_CATALOG = "II/336/apass9"  # DR9

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
    # approximate uncertainty
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

def wcs_pixels_to_radec(wcs: WCS, xpix, ypix, hdr):
    """
    Convert pixel coords to RA/Dec (deg), honoring IRAF/PixInsight LTM/LTV if present.
    Uses: x_wcs = LTM1_1 * xpix + LTV1 ; y_wcs = LTM2_2 * ypix + LTV2
    """
    x = np.asarray(xpix, float)
    y = np.asarray(ypix, float)

    ltm11 = float(hdr.get('LTM1_1', 1.0))
    ltm22 = float(hdr.get('LTM2_2', 1.0))
    ltv1  = float(hdr.get('LTV1', 0.0))
    ltv2  = float(hdr.get('LTV2', 0.0))

    # Apply IRAF linear transform if present
    if (ltm11 != 1.0) or (ltm22 != 1.0) or (ltv1 != 0.0) or (ltv2 != 0.0):
        # optional: print once for diagnostics
        print(f"LTM/LTV applied: LTM1_1={ltm11} LTM2_2={ltm22} LTV1={ltv1} LTV2={ltv2}")
        x = ltm11 * x + ltv1
        y = ltm22 * y + ltv2

    coords = wcs.pixel_to_world(x, y)
    return np.asarray(coords.ra.deg), np.asarray(coords.dec.deg)

# ============================== APASS ==============================

def fetch_apass_catalog(ra_deg: float, dec_deg: float, radius_arcmin: float=APASS_RADIUS_ARCMIN) -> pd.DataFrame:
    """Fetch APASS DR9 around (ra,dec) within radius (arcmin)."""
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
    df = df[np.isfinite(df["Bmag"]) & np.isfinite(df["Vmag"])].reset_index(drop=True)
    return df

# ============================== Calibration helpers ==============================

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

def calibrate_band_from_apass(det_df: pd.DataFrame, apass_df: pd.DataFrame,
                              inst_mag_col: str,
                              std_mag_col: str,
                              match_arcsec: float=CALIB_MATCH_MAX_ARCSEC,
                              mag_range=(10, 17.5)):
    """
    Calibrate B or V against APASS:
      m_std - m_inst = ZP + beta*(B-V)_std
    """
    det_coords = SkyCoord(det_df['RA'].values*u.deg, det_df['DEC'].values*u.deg)
    cat_coords = SkyCoord(apass_df['ra'].values*u.deg, apass_df['dec'].values*u.deg)
    idx, d2d, _ = det_coords.match_to_catalog_sky(cat_coords)
    ok = d2d.arcsec <= match_arcsec
    if ok.sum() < 8:
        return np.nan, np.nan, np.zeros(len(det_df), dtype=bool), np.nan

    df  = det_df.loc[ok].copy().reset_index(drop=True)
    cat = apass_df.iloc[idx[ok]].reset_index(drop=True)

    color = (cat['Bmag'] - cat['Vmag']).to_numpy(float)
    tmag  = cat[std_mag_col].to_numpy(float)
    m_inst= pd.to_numeric(df[inst_mag_col], errors='coerce').to_numpy(float)

    q = np.isfinite(color) & np.isfinite(tmag) & np.isfinite(m_inst)
    if mag_range is not None:
        lo, hi = float(mag_range[0]), float(mag_range[1])
        q &= (tmag >= lo) & (tmag <= hi)

    if q.sum() >= 10:
        mi = m_inst[q]
        qmi = (mi > np.nanpercentile(mi, 5)) & (mi < np.nanpercentile(mi, 95))
        mask = np.zeros_like(q, dtype=bool)
        mask[np.where(q)[0][qmi]] = True
        q = mask

    if q.sum() < 8:
        return np.nan, np.nan, np.zeros(len(det_df), dtype=bool), np.nan

    y = tmag[q] - m_inst[q]
    x = color[q]
    ZP, beta, used, rms = robust_linfit(y, x, clip=3.0, maxiter=10)

    use_mask = np.zeros(len(det_df), dtype=bool)
    use_mask[np.where(ok)[0][q][used]] = True
    return ZP, beta, use_mask, rms

# ============================== Main ==============================

def main():
    print("=== Cluster Analysis with APASS BV Calibration ===")

    # Defaults shown as prompts; Enter keeps them.
    cluster_name        = _ask("Cluster name", DEFAULTS["cluster_name"], str)
    cluster_distance_pc = _ask("Cluster distance (pc)", DEFAULTS["cluster_distance_pc"], float)
    cluster_type        = _ask("Cluster type [O=open, G=globular]", DEFAULTS["cluster_type"], str, upper=True)
    A_V                 = _ask("Galactic extinction A_V", DEFAULTS["A_V"], float)
    E_BV                = _ask("Galactic reddening E(B-V)", DEFAULTS["E_BV"], float)

    fits_file_B_in = _ask("Path to B-band FITS image", DEFAULTS["fits_file_B"], str)
    fits_file_V_in = _ask("Path to V-band/G-band FITS image (treated as V)", DEFAULTS["fits_file_V"], str)
    gaia_csv_in    = _ask("Path to Gaia CSV (for membership only) — or 0 to skip", DEFAULTS["gaia_csv"], str)

    # Normalize paths first
    fits_file_B = _norm_path(fits_file_B_in)
    fits_file_V = _norm_path(fits_file_V_in)
    gaia_csv    = "0" if gaia_csv_in.strip() == "0" else _norm_path(gaia_csv_in)

    # Output paths in same folder as V FITS
    out_dir  = os.path.dirname(fits_file_V) if os.path.dirname(fits_file_V) else os.getcwd()
    out_csv  = os.path.join(out_dir, "photometry_with_membership.csv")
    out_memb = os.path.join(out_dir, "matched_with_membership.csv")
    out_plot = os.path.join(out_dir, "cluster_cmd_apass.png")

    # Convert hard-coded RA/Dec to degrees for APASS query center
    ra_c = _hms_to_deg(DEFAULTS["ra_hms"])
    dec_c = _dms_to_deg(DEFAULTS["dec_dms"])
    print(f"APASS query center (hard-coded): RA={ra_c:.6f} deg  DEC={dec_c:.6f} deg  radius={APASS_RADIUS_ARCMIN} arcmin")

    # Sanity: FITS existence
    assert os.path.exists(fits_file_V), f"V FITS not found: {fits_file_V}"
    assert os.path.exists(fits_file_B), f"B FITS not found: {fits_file_B}"

    # Read FITS
    with fits.open(fits_file_V) as hdul:
        imgV = hdul[0].data.astype(float)
        hdrV = hdul[0].header
    with fits.open(fits_file_B) as hdul:
        imgB = hdul[0].data.astype(float)
        hdrB = hdul[0].header

    # WCS for RA/DEC of detections
    wcsV = WCS(hdrV)

    # Detect sources on V image
    det = detect_on_image(imgV, fwhm_pix=DAO_FWHM, thresh_sigma=DAO_THRESH_SIGMA, peak_snr_min=PEAK_SNR_MIN)
    if len(det) == 0:
        print("No detections found on V image.")
        return

    # Pixel → sky for detections (decimal degrees)
    raV, decV = wcs_pixels_to_radec(wcsV, det['X'].values, det['Y'].values, hdrV)
    det['RA']  = raV
    det['DEC'] = decV

    print(det[['X','Y','RA','DEC']].head())

    # Aperture photometry on both bands at same centroids
    FWHM_est = DAO_FWHM
    r_ap  = K_APERTURE   * FWHM_est
    r_in  = K_ANNULUS_IN * FWHM_est
    r_out = K_ANNULUS_OUT* FWHM_est

    fluxV, fluxB = [], []
    for x, y in det[['X','Y']].to_numpy():
        fV, _, _, _, _, _ = aperture_photometry_single_fast(imgV, x, y, r_ap, r_in, r_out, GAIN_E_PER_ADU, READ_NOISE_E)
        fB, _, _, _, _, _ = aperture_photometry_single_fast(imgB, x, y, r_ap, r_in, r_out, GAIN_E_PER_ADU, READ_NOISE_E)
        fluxV.append(fV); fluxB.append(fB)
    det['flux_V'] = np.array(fluxV, float)
    det['flux_B'] = np.array(fluxB, float)

    # Instrumental magnitudes normalized by EXPTIME
    def _get_exptime(h):
        for k in ('EXPTIME','EXPOSURE','EXPT','ITIME'):
            if k in h:
                try:
                    v=float(h[k])
                    if np.isfinite(v) and v>0: return v
                except Exception:
                    pass
        return 1.0
    tV=_get_exptime(hdrV); tB=_get_exptime(hdrB)
    with np.errstate(divide='ignore', invalid='ignore'):
        det['m_inst_V'] = -2.5*np.log10(np.maximum(det['flux_V'], 1e-12)/tV)
        det['m_inst_B'] = -2.5*np.log10(np.maximum(det['flux_B'], 1e-12)/tB)

    # Fetch APASS using the hard-coded sky center
    print("Querying APASS DR9 from VizieR …")
    apass = fetch_apass_catalog(ra_c, dec_c, radius_arcmin=APASS_RADIUS_ARCMIN)
    print(f"APASS stars fetched: {len(apass)}")
    
    det_coords = SkyCoord(det['RA'].values*u.deg, det['DEC'].values*u.deg)
    cat_coords = SkyCoord(apass['ra'].values*u.deg, apass['dec'].values*u.deg)
    idx, d2d, _ = det_coords.match_to_catalog_sky(cat_coords)
    print("Min distance [arcsec]:", np.nanmin(d2d.arcsec))
    print("Median distance [arcsec]:", np.nanmedian(d2d.arcsec))

    if len(apass) < 8:
        print("APASS returned too few stars — aborting calibration.")
        return

    # Calibrate V and B
    ZP_V, beta_V, used_V, rms_V = calibrate_band_from_apass(
        det, apass, inst_mag_col='m_inst_V', std_mag_col='Vmag',
        match_arcsec=CALIB_MATCH_MAX_ARCSEC, mag_range=(10,17.5)
    )
    ZP_B, beta_B, used_B, rms_B = calibrate_band_from_apass(
        det, apass, inst_mag_col='m_inst_B', std_mag_col='Bmag',
        match_arcsec=CALIB_MATCH_MAX_ARCSEC, mag_range=(10,18.0)
    )

    print(f"V-calibration: ZP={ZP_V:.3f}  beta(B-V)={beta_V:.3f}  rms={rms_V:.03f}  N={int(used_V.sum())}")
    print(f"B-calibration: ZP={ZP_B:.3f}  beta(B-V)={beta_B:.3f}  rms={rms_B:.03f}  N={int(used_B.sum())}")

    if not np.isfinite(ZP_V) or not np.isfinite(ZP_B):
        print("Calibration failed (no finite ZP). Check WCS or parameters.")
        return

    # Apply calibration (single iteration using BV_0 as proxy color)
    det['V_cal_0'] = det['m_inst_V'] + ZP_V
    det['B_cal_0'] = det['m_inst_B'] + ZP_B
    det['BV_0']    = det['B_cal_0'] - det['V_cal_0']

    det['V_mag']     = det['m_inst_V'] + ZP_V + (beta_V * det['BV_0'] if np.isfinite(beta_V) else 0.0)
    det['B_mag']     = det['m_inst_B'] + ZP_B + (beta_B * det['BV_0'] if np.isfinite(beta_B) else 0.0)
    det['B_minus_V'] = det['B_mag'] - det['V_mag']

    # Optional: Gaia membership
    det['ClusterMember'] = False
    if gaia_csv != "0" and os.path.exists(gaia_csv):
        gaia = pd.read_csv(gaia_csv)
        cols_lower = {c.lower(): c for c in gaia.columns}
        for want in ["ra","dec","parallax","parallax_error","pmra","pmra_error","pmdec","pmdec_error"]:
            if want in cols_lower:
                gaia.rename(columns={cols_lower[want]: want}, inplace=True)
        if all(k in gaia.columns for k in ["ra","dec"]):
            gaia_coords = SkyCoord(gaia['ra'].values*u.deg, gaia['dec'].values*u.deg)
            det_coords  = SkyCoord(det['RA'].values*u.deg,  det['DEC'].values*u.deg)
            idx, d2d, _ = det_coords.match_to_catalog_sky(gaia_coords)
            det['gaia_index'] = idx
            det['sep_arcsec'] = d2d.arcsec
            det['matched']    = det['sep_arcsec'] <= MATCH_MAX_ARCSEC

            m = det['matched']
            if m.sum() > 10 and all(k in gaia.columns for k in ["parallax","pmra","pmdec"]):
                gsub = gaia.iloc[det.loc[m,'gaia_index'].astype(int)]
                par = pd.to_numeric(gsub.get('parallax'), errors='coerce').to_numpy(float)
                pmr = pd.to_numeric(gsub.get('pmra'),     errors='coerce').to_numpy(float)
                pmd = pd.to_numeric(gsub.get('pmdec'),    errors='coerce').to_numpy(float)
                def _mode(arr):
                    arr = arr[np.isfinite(arr)]
                    if arr.size==0: return np.nan, np.nan
                    lo, hi = np.nanpercentile(arr, (5,95))
                    H, e = np.histogram(arr[(arr>=lo)&(arr<=hi)], bins=60)
                    j = np.argmax(H); cen = 0.5*(e[j]+e[j+1])
                    sig = 1.4826*np.nanmedian(np.abs(arr - np.nanmedian(arr)))
                    return float(cen), float(max(sig, 1e-3))
                par0, spar = _mode(par)
                pmr0, srm  = _mode(pmr)
                pmd0, sdm  = _mode(pmd)
                mem = (
                    np.isfinite(par) & (np.abs(par - par0) <= max(PARALLAX_MIN_HALF, min(PARALLAX_MAX_HALF, PARALLAX_FRAC*par0/100.0, 3*spar))) &
                    np.isfinite(pmr) & (np.abs(pmr - pmr0) <= max(PM_SIGMA_FLOOR, 4*srm)) &
                    np.isfinite(pmd) & (np.abs(pmd - pmd0) <= max(PM_SIGMA_FLOOR, 4*sdm))
                )
                det.loc[m, 'ClusterMember'] = mem
        else:
            det['matched'] = False
            det['sep_arcsec'] = np.nan
            det['gaia_index'] = -1
    else:
        det['matched'] = False
        det['sep_arcsec'] = np.nan
        det['gaia_index'] = -1

    # Reddening/extinction views
    DMOD = 5.0 * np.log10(max(cluster_distance_pc, 1e-6)) - 5.0
    det['BV0'] = det['B_minus_V'] - E_BV
    det['V0']  = det['V_mag'] - A_V
    det['M_V'] = det['V_mag'] - DMOD - A_V

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    det.to_csv(out_csv, index=False)
    det[['RA','DEC','V_mag','B_mag','B_minus_V','BV0','V0','M_V','ClusterMember']].to_csv(out_memb, index=False)
    print(f"Saved photometry: {out_csv}\nSaved members table: {out_memb}")

    # Plot CMD
    plt.figure(figsize=(6,7))
    m = np.isfinite(det['B_minus_V']) & np.isfinite(det['V_mag'])
    plt.scatter(det.loc[m & ~det['ClusterMember'], 'B_minus_V'], det.loc[m & ~det['ClusterMember'], 'V_mag'],
                s=6, alpha=0.4, label='Field')
    plt.scatter(det.loc[m & det['ClusterMember'],  'B_minus_V'], det.loc[m & det['ClusterMember'],  'V_mag'],
                s=9, alpha=0.9, label='Members')
    plt.gca().invert_yaxis()
    plt.xlabel('B−V'); plt.ylabel('V')
    plt.title(f'{cluster_name} — CMD (APASS-calibrated)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=160)
    print(f"Saved CMD plot: {out_plot}")

if __name__ == "__main__":
    main()
