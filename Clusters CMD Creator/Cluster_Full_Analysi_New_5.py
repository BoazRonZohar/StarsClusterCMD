# -*- coding: utf-8 -*-
"""
Cluster photometry + CMD with APASS BV calibration.
Two modes available:
  1. Online query from Vizier (APASS DR9).
  2. Local file provided by the user (CSV/TSV downloaded from Vizier).

Requirements:
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

# ============================== Defaults ==============================

DEFAULTS = {
    "ra_hms":  "8:51:29",
    "dec_dms": "+11:49:00",
    "cluster_name": "M67",
    "cluster_distance_pc": 850.0,
    "cluster_type": "O",
    "A_V": 3.33,
    "E_BV": 0.28,
    "fits_file_B": r"file:///D:/1 AAA TEMP/Clusters RGB FITS ALL/M 67/M 67-20230123_B_13.fts",
    "fits_file_V": r"D:\1 AAA TEMP\Clusters RGB FITS ALL\M 67\M 67-20230123_G_13.fts",
    "apass_local_file": r"D:\1 AAA TEMP\Clusters RGB FITS ALL\M 67\M67 Mag Cat Clean.xlsx"  # optional
}

DAO_FWHM = 5.0
DAO_THRESH_SIGMA = 4.0
PEAK_SNR_MIN = 10.0

Vizier.ROW_LIMIT = 50000
APASS_CATALOG = "II/336/apass9"

# ============================== Helpers ==============================

def _ask(prompt, default, cast=str, upper=False):
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "" or s == "0":
        v = default
    else:
        try: v = cast(s)
        except: v = default
    if upper and isinstance(v,str):
        return v.upper()
    return v

def _norm_path(p: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    if p.lower().startswith("file:///"):
        from urllib.parse import urlparse, unquote
        u_ = urlparse(p); p = unquote(u_.path)
        if len(p)>=3 and p[0]=="/" and p[2]==":":
            p=p[1:]
    return os.path.normpath(p)

def _hms_to_deg(hms: str) -> float:
    h,m,s = [float(x) for x in hms.replace(" ","").split(":")]
    return (h+m/60+s/3600)*15.0

def _dms_to_deg(dms: str) -> float:
    sgn=-1 if dms.strip().startswith("-") else 1
    dms_=dms.replace("+","").replace("-","").replace(" ","")
    d,m,s=[float(x) for x in dms_.split(":")]
    return sgn*(abs(d)+m/60+s/3600)

# ============================== Detection ==============================

def detect_on_image(img, fwhm_pix=DAO_FWHM, thresh_sigma=DAO_THRESH_SIGMA, peak_snr_min=PEAK_SNR_MIN):
    mean, med, std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=fwhm_pix, threshold=thresh_sigma*std)
    tbl = daofind(img - med)
    if tbl is None or len(tbl)==0:
        return pd.DataFrame(columns=["X","Y","SNR_est"])
    df = tbl.to_pandas()
    df["SNR_est"]=df["peak"]/max(std,1e-6)
    df=df[df["SNR_est"]>=peak_snr_min]
    return df.rename(columns={"xcentroid":"X","ycentroid":"Y"})[["X","Y","SNR_est"]].reset_index(drop=True)

def wcs_pixels_to_radec(wcs: WCS, xpix, ypix):
    coords=wcs.pixel_to_world(np.asarray(xpix), np.asarray(ypix))
    return np.asarray(coords.ra.deg), np.asarray(coords.dec.deg)

# ============================== APASS ==============================

def fetch_apass_online(ra_deg, dec_deg, radius_arcmin=45.0) -> pd.DataFrame:
    """Query Vizier online for APASS DR9."""
    center=SkyCoord(ra_deg*u.deg, dec_deg*u.deg)
    cols=["RAJ2000","DEJ2000","Bmag","Vmag","e_Bmag","e_Vmag"]
    res=Vizier(columns=cols).query_region(center, radius=radius_arcmin*u.arcmin, catalog=APASS_CATALOG)
    if len(res)==0:
        return pd.DataFrame(columns=["ra","dec","Bmag","Vmag","e_Bmag","e_Vmag"])
    t=res[0].to_pandas()
    df=pd.DataFrame({
        "ra":pd.to_numeric(t.get("RAJ2000"),errors='coerce'),
        "dec":pd.to_numeric(t.get("DEJ2000"),errors='coerce'),
        "Bmag":pd.to_numeric(t.get("Bmag"),errors='coerce'),
        "Vmag":pd.to_numeric(t.get("Vmag"),errors='coerce'),
        "e_Bmag":pd.to_numeric(t.get("e_Bmag"),errors='coerce'),
        "e_Vmag":pd.to_numeric(t.get("e_Vmag"),errors='coerce')
    })
    return df.dropna().reset_index(drop=True)

def load_apass_excel(path):
    df = pd.read_excel(path)

    # Rename columns from your Excel file
    rename_map = {
        "RAJ2000": "ra",
        "DEJ2000": "dec",
        "Bmag": "Bmag",
        "Vmag": "Vmag",
        "B-V": "B_V"  # אם אתה רוצה גם את זה
    }
    df = df.rename(columns=rename_map)

    # Force numeric conversion, drop rows with invalid values
    for col in ["ra", "dec", "Bmag", "Vmag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ra", "dec", "Bmag", "Vmag"])

    print(f"Loaded {len(df)} stars from Excel (cleaned)")
    return df





# ============================== Calibration ==============================

def calibrate_band(det_df, apass_df, inst_col, std_col, match_arcsec=60.0):
    """
    Calibrate instrumental magnitudes against APASS catalog.

    Parameters
    ----------
    det_df : DataFrame
        Detected sources with RA, DEC, and instrumental magnitudes.
    apass_df : DataFrame
        Catalog sources with ra, dec, and standard magnitudes.
    inst_col : str
        Name of column in det_df with instrumental magnitudes.
    std_col : str
        Name of column in apass_df with standard magnitudes.
    match_arcsec : float
        Matching radius in arcseconds.

    Returns
    -------
    ZP : float
        Zero point of calibration.
    beta : float
        Color term slope (if applicable).
    mask : ndarray
        Boolean mask of matched stars used in calibration.
    rms : float
        Root mean square of residuals.
    """

    # --- Force numeric RA/DEC in both dataframes ---
    det_df['RA']  = pd.to_numeric(det_df['RA'], errors='coerce')
    det_df['DEC'] = pd.to_numeric(det_df['DEC'], errors='coerce')
    apass_df['ra']  = pd.to_numeric(apass_df['ra'], errors='coerce')
    apass_df['dec'] = pd.to_numeric(apass_df['dec'], errors='coerce')

    # Drop rows with missing coordinates
    det_df = det_df.dropna(subset=['RA', 'DEC'])
    apass_df = apass_df.dropna(subset=['ra', 'dec'])

    # --- Build SkyCoord objects ---
    det_coords = SkyCoord(det_df['RA'].values * u.deg,
                          det_df['DEC'].values * u.deg)
    cat_coords = SkyCoord(apass_df['ra'].values * u.deg,
                          apass_df['dec'].values * u.deg)

    # --- Cross-match ---
    idx, d2d, _ = det_coords.match_to_catalog_sky(cat_coords)
    sep_constraint = d2d.arcsec < match_arcsec
    matched_det = det_df[sep_constraint].reset_index(drop=True)
    matched_cat = apass_df.iloc[idx[sep_constraint]].reset_index(drop=True)

    if matched_det.empty or matched_cat.empty:
        print(f"No matches found within {match_arcsec}\" for {inst_col} vs {std_col}")
        return np.nan, np.nan, None, np.nan

    # --- Calibration fit ---
    x = matched_det[inst_col].values
    y = matched_cat[std_col].values

    # linear regression: std = inst + ZP (+ color term optional)
    A = np.vstack([np.ones_like(x), x]).T
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    ZP, beta = coeff[0], coeff[1] - 1.0  # adjust slope relative to unity

    residuals = y - (x + ZP)
    rms = np.std(residuals)

    mask = sep_constraint
    return ZP, beta, mask, rms


# ============================== Main ==============================

def main():
    print("=== Cluster Analysis with APASS BV Calibration (online/local) ===")

    cluster_name=_ask("Cluster name", DEFAULTS["cluster_name"], str)
    fits_file_B=_norm_path(_ask("Path to B-band FITS image", DEFAULTS["fits_file_B"], str))
    fits_file_V=_norm_path(_ask("Path to V-band FITS image", DEFAULTS["fits_file_V"], str))

    ra_c=_hms_to_deg(DEFAULTS["ra_hms"])
    dec_c=_dms_to_deg(DEFAULTS["dec_dms"])

    # Load images
    with fits.open(fits_file_V) as h: imgV=h[0].data.astype(float); hdrV=h[0].header
    with fits.open(fits_file_B) as h: imgB=h[0].data.astype(float); hdrB=h[0].header
    wcsV=WCS(hdrV)

    # Detect sources
    det=detect_on_image(imgV)
    raV,decV=wcs_pixels_to_radec(wcsV,det['X'],det['Y'])
    det['RA']=raV; det['DEC']=decV
    det['m_inst_V']=-2.5*np.log10(np.maximum(1e-12,det['SNR_est']))
    det['m_inst_B']=-2.5*np.log10(np.maximum(1e-12,det['SNR_est']))

    # Choose APASS source: always local Excel file
    choice = _ask("Use local APASS file? (Y/N)", "N", str, upper=True).strip().upper()
    print("DEBUG choice =", repr(choice))
    
    if choice == "Y":
        apass = load_apass_excel(DEFAULTS["apass_local_file"])
    else:
        print("Online APASS fetch disabled. Exiting.")
        return


    if len(apass)==0:
        print("No APASS stars found. Exiting."); return

    # Calibration
    ZP_V,beta_V,maskV,rmsV=calibrate_band(det,apass,'m_inst_V','Vmag')
    ZP_B,beta_B,maskB,rmsB=calibrate_band(det,apass,'m_inst_B','Bmag')
    print(f"V: ZP={ZP_V:.3f}, beta={beta_V:.3f}, rms={rmsV:.3f}")
    print(f"B: ZP={ZP_B:.3f}, beta={beta_B:.3f}, rms={rmsB:.3f}")

    # Apply calibration
    det['V_cal']=det['m_inst_V']+ZP_V+beta_V*((det['m_inst_B']-det['m_inst_V']))
    det['B_cal']=det['m_inst_B']+ZP_B+beta_B*((det['m_inst_B']-det['m_inst_V']))
    det['B-V']=det['B_cal']-det['V_cal']

    # Plot CMD
    plt.scatter(det['B-V'],det['V_cal'],s=5,c='blue')
    plt.gca().invert_yaxis()
    plt.xlabel("B-V")
    plt.ylabel("V")
    plt.title(f"{cluster_name} CMD (APASS-calibrated)")
    plt.show()

if __name__=="__main__":
    main()
