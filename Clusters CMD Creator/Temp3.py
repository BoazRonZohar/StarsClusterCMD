# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 14:17:31 2025

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Cluster photometry pipeline (final version):
- Keeps interactive user input
- Uses per-star FWHM (from base file)
- Each star gets its own aperture radius = 1.2 * FWHM
- Outputs only X, Y, FWHM, Aperture_Radius, Flux_B, Flux_G
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

# ------------------- Interactive user input -------------------
def _ask(prompt, default, cast=str, upper=False):
    s = input(f"{prompt} [{default}]: ").strip()
    if s == "" or s == "0":
        return default
    try:
        v = cast(s)
        return v.upper() if (upper and isinstance(v, str)) else v
    except Exception:
        return default

print("=== Cluster Photometry Interactive Input ===")
cluster_name     = _ask("Cluster name", "UnknownCluster", str)
cluster_type     = _ask("Cluster type [O=open, G=globular]", "O", str, upper=True)
fits_file_B      = _ask("Path to B-band FITS image", r"D:\example_B.fts", str)
fits_file_G      = _ask("Path to G-band FITS image", r"D:\example_G.fts", str)

def _norm_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    return os.path.normpath(p)

fits_file_B = _norm_path(fits_file_B)
fits_file_G = _norm_path(fits_file_G)

# ---------- Background subtraction (new block) ----------
def subtract_background_and_save(path):
    data, hdr = fits.getdata(path, header=True)
    median_val = np.nanmedian(data)
    data_sub = data - median_val
    out_path = os.path.splitext(path)[0] + "_bgsub.fits"
    fits.writeto(out_path, data_sub, hdr, overwrite=True)
    print(f"[bgsub] wrote {out_path} (median={median_val:.3f})")
    return out_path

fits_file_B = subtract_background_and_save(fits_file_B)
fits_file_G = subtract_background_and_save(fits_file_G)
# --------------------------------------------------------

_outdir = os.path.dirname(fits_file_G) if os.path.dirname(fits_file_G) else os.getcwd()
out_csv = os.path.join(_outdir, "fluxes_XY_FWHM_Ap.csv")


# ------------------- Photometry config -------------------
DETECTION_SIGMA = 5.0   # detection threshold in units of background sigma (controls how many stars are detected)
FWHM_HINT = 3.0         # fallback FWHM in pixels (used only if per-star FWHM estimation fails)
K_APERTURE = 2       # aperture radius scaling factor relative to each star's FWHM
K_ANNULUS_IN = 2      # inner radius of background annulus, in units of FWHM
K_ANNULUS_OUT = 4     # outer radius of background annulus, in units of FWHM
GAIN_E_PER_ADU = 1.0    # CCD gain: electrons per ADU (used for noise estimation if needed)
READ_NOISE_E = 5.0      # detector read noise in electrons (used for noise estimation if needed)


# ------------------- Core photometry helpers -------------------
def estimate_fwhm_moments(img, x, y, box=15, r_bg_in=8, r_bg_out=12):
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

def aperture_photometry_single_fast(img, x, y, r_ap, r_in, r_out):
    cut, xx, yy = _cutout(img, x, y, r_out)
    r = np.hypot(yy - y, xx - x)
    ap = (r <= r_ap)
    bkg_mean, bkg_std = local_background_cutout(img, x, y, r_in, r_out)
    ap_vals = cut[ap].astype(float)
    npix_ap = ap_vals.size
    flux_ap_adu = float(np.nansum(ap_vals) - bkg_mean * npix_ap)
    return flux_ap_adu

# ------------------- Main photometry routine -------------------
def run_photometry(filename, band):
    data = fits.getdata(filename, ext=0).astype(float)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=FWHM_HINT, threshold=DETECTION_SIGMA * std)
    tbl = daofind(data - median)
    if tbl is None or len(tbl) == 0:
        raise RuntimeError(f"No sources detected in {band}")
    tbl = tbl[tbl['peak'] > 10 * std]
    positions = [(float(r['xcentroid']), float(r['ycentroid'])) for r in tbl]

    rows = []
    for (x, y) in positions:
        fwhm_star = estimate_fwhm_moments(data, x, y, box=15, r_bg_in=8, r_bg_out=12)
        if not np.isfinite(fwhm_star) or fwhm_star <= 0:
            continue
        r_ap = K_APERTURE * fwhm_star
        r_in = K_ANNULUS_IN * fwhm_star
        r_out = K_ANNULUS_OUT * fwhm_star
        flux = aperture_photometry_single_fast(data, x, y, r_ap, r_in, r_out)
        rows.append([x, y, fwhm_star, r_ap, flux])

    df = pd.DataFrame(rows, columns=["X", "Y", "FWHM", "Aperture_Radius", f"Flux_{band}"])
    return df

# ------------------- Run -------------------
if __name__ == "__main__":
    dfG = run_photometry(fits_file_G, "G")
    dfB = run_photometry(fits_file_B, "B")

    # join on nearest pixel
    dfG["X_int"] = dfG["X"].round().astype(int)
    dfG["Y_int"] = dfG["Y"].round().astype(int)
    dfB["X_int"] = dfB["X"].round().astype(int)
    dfB["Y_int"] = dfB["Y"].round().astype(int)

    df = pd.merge(dfG[["X_int","Y_int","X","Y","FWHM","Aperture_Radius","Flux_G"]],
                  dfB[["X_int","Y_int","Flux_B"]],
                  on=["X_int","Y_int"], how="inner")

    df = df[["X","Y","FWHM","Aperture_Radius","Flux_B","Flux_G"]]

    df.to_csv(out_csv, index=False)
    print(f"[out] wrote {out_csv}")
