# -*- coding: utf-8 -*-
"""
Optimized version: avoids full-image np.mgrid per source.
Uses small cutouts around each star (size ~= ceil(r_out)+2).
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import math

# -----------------------------
# Keep user's path EXACT
# -----------------------------
#file_path = r"D:\1 AAA TEMP\Clusters RGB fits\New\NGC 1647\ngc 1647-20250129_G_21.fts"
file_path = "D:\1 AAA TEMP\Clusters RGB FITS ALL\M 67\M 67-20230123_B_13.fts"

# -----------------------------
# Config
# -----------------------------
DETECTION_SIGMA = 5.0
FWHM_HINT = 3.0
K_APERTURE = 1.2
K_ANNULUS_IN = 2.5
K_ANNULUS_OUT = 4.0
GAIN_E_PER_ADU = 1.0
READ_NOISE_E = 5.0

# -----------------------------
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

# -----------------------------
# Efficient local photometry using a cutout around the source
# -----------------------------
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

def compute_aperture_correction(img, positions, fwhm_field,
                                gain=GAIN_E_PER_ADU, read_noise=READ_NOISE_E):
    if len(positions) == 0:
        return 1.0, None, None
    radii = np.array([0.7, 1.0, 1.2, 1.4, 2.0, 3.0]) * fwhm_field
    r_in = K_ANNULUS_IN * fwhm_field
    r_out = K_ANNULUS_OUT * fwhm_field

    pos = np.array(positions, float)
    keep = []
    for i, (x, y) in enumerate(pos):
        d = np.hypot(pos[:,0]-x, pos[:,1]-y)
        d[i] = np.inf
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
    corr = 1.0 if not np.isfinite(flux_small) or flux_small <= 0 or not np.isfinite(flux_large) or flux_large <= 0 else float(flux_large / flux_small)
    return corr, radii, median_curve

def process_fits(filename, band):
    data = fits.getdata(filename, ext=0).astype(float)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)

    daofind = DAOStarFinder(fwhm=FWHM_HINT, threshold=DETECTION_SIGMA * std)
    tbl = daofind(data - median)
    if tbl is None or len(tbl) == 0:
        print("No sources detected with current parameters.")
        return pd.DataFrame(columns=[
            "X","Y","FWHM","Aperture Radius","Flux_Aperture","Flux_Err","SNR",
            "Band","Annulus Inner Radius","Annulus Outer Radius","Npix_Aperture",
            "Bkg_Mean","Bkg_Std","ApertureCorrection","Flux",
            "FWHM_Field","Aperture_Radius_Used","Annulus_In_Used","Annulus_Out_Used"
        ])

    tbl = tbl[tbl['peak'] > 10 * std]
    positions = [(float(r['xcentroid']), float(r['ycentroid'])) for r in tbl]

    # Estimate median FWHM on a small subset for speed
    sample = positions[:min(30, len(positions))]
    fwhms = []
    for (x, y) in sample:
        f = estimate_fwhm_moments(data, x, y, box=15, r_bg_in=8, r_bg_out=12)
        if np.isfinite(f) and f > 0:
            fwhms.append(float(f))
    fwhm_field = float(np.nanmedian(fwhms)) if len(fwhms) else FWHM_HINT

    r_ap = K_APERTURE * fwhm_field
    r_in = K_ANNULUS_IN * fwhm_field
    r_out = K_ANNULUS_OUT * fwhm_field

    rows = []
    # Compute per-star FWHM only if needed; otherwise use field FWHM in output
    for (x, y) in positions:
        flux_ap, flux_err, snr, bkg_mean, bkg_std, npix_ap = aperture_photometry_single_fast(
            data, x, y, r_ap, r_in, r_out, GAIN_E_PER_ADU, READ_NOISE_E
        )
        rows.append([x, y, np.nan, r_ap, flux_ap, flux_err, snr, band, r_in, r_out, npix_ap, bkg_mean, bkg_std])

    df = pd.DataFrame(rows, columns=[
        "X", "Y", "FWHM", "Aperture Radius",
        "Flux_Aperture", "Flux_Err", "SNR",
        "Band", "Annulus Inner Radius", "Annulus Outer Radius",
        "Npix_Aperture", "Bkg_Mean", "Bkg_Std"
    ])

    corr, radii, median_curve = compute_aperture_correction(data, positions, fwhm_field,
                                                            gain=GAIN_E_PER_ADU, read_noise=READ_NOISE_E)
    df["ApertureCorrection"] = corr
    df["Flux"] = df["Flux_Aperture"] * corr
    df["FWHM_Field"] = fwhm_field
    df["Aperture_Radius_Used"] = r_ap
    df["Annulus_In_Used"] = r_in
    df["Annulus_Out_Used"] = r_out
    return df

if __name__ == "__main__":
    df = process_fits(file_path, file_path)
    end = "photometry_results"
    index = file_path
    csv_filename = f"{index}_{end}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")
