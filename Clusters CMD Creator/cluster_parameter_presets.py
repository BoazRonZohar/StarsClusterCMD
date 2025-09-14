# ============================================
# Parameter presets for different cluster types
# ============================================

# --------- Sparse / nearby open cluster (≈0.2–1 kpc, low crowding) ---------
DAO_FWHM           = 5.0
DAO_THRESH_SIGMA   = 3.5
PEAK_SNR_MIN       = 8.0
FWHM_BOX_SIZE      = 24

K_APERTURE         = 1.2
K_ANNULUS_IN       = 2.5
K_ANNULUS_OUT      = 4.0

GAIN_E_PER_ADU     = 1.0
READ_NOISE_E       = 5.0

CLUSTER_DISTANCE_PC = 700
AG_EXTINCTION       = 0.0
EBG_REDDENING       = 0.0

MATCH_MAX_ARCSEC   = 3.0
PARALLAX_MIN_HALF  = 0.10
PARALLAX_MAX_HALF  = 2.00
PARALLAX_FRAC      = 1.00
PARALLAX_ERR_SIGMA = 4.0

PM_SIGMA_FLOOR     = 0.8
PM_ERR_SIGMA       = 4.0
PM_CHI2            = 25.0

SPATIAL_CORE_FRAC  = 0.35

CALIB_MATCH_MAX_ARCSEC = 5
CALIB_G_MAG_RANGE      = (9.0, 19.0)
CALIB_BP_MAG_RANGE     = (9.0, 19.5)

# --------- Moderately distant open cluster (≈1–4 kpc, moderate crowding) ---------
DAO_FWHM           = 5.0
DAO_THRESH_SIGMA   = 4.0
PEAK_SNR_MIN       = 10.0
FWHM_BOX_SIZE      = 24

K_APERTURE         = 1.2
K_ANNULUS_IN       = 2.5
K_ANNULUS_OUT      = 4.0

GAIN_E_PER_ADU     = 1.0
READ_NOISE_E       = 5.0

CLUSTER_DISTANCE_PC = 2500
AG_EXTINCTION       = 0.0
EBG_REDDENING       = 0.0

MATCH_MAX_ARCSEC   = 2.0
PARALLAX_MIN_HALF  = 0.08
PARALLAX_MAX_HALF  = 3.00
PARALLAX_FRAC      = 10.0
PARALLAX_ERR_SIGMA = 6.0

PM_SIGMA_FLOOR     = 1.2
PM_ERR_SIGMA       = 6.0
PM_CHI2            = 40.0

SPATIAL_CORE_FRAC  = 0.25

CALIB_MATCH_MAX_ARCSEC = 8
CALIB_G_MAG_RANGE      = (10.0, 19.0)
CALIB_BP_MAG_RANGE     = (10.0, 19.5)

# --------- Globular cluster (very distant, very crowded, e.g. M13) ---------
DAO_FWHM           = 5.0
DAO_THRESH_SIGMA   = 5.0
PEAK_SNR_MIN       = 15.0
FWHM_BOX_SIZE      = 24

K_APERTURE         = 1.2
K_ANNULUS_IN       = 2.5
K_ANNULUS_OUT      = 4.0

GAIN_E_PER_ADU     = 1.0
READ_NOISE_E       = 5.0

CLUSTER_DISTANCE_PC = 7000
AG_EXTINCTION       = 0.0
EBG_REDDENING       = 0.0

MATCH_MAX_ARCSEC   = 1.0
PARALLAX_MIN_HALF  = 0.05
PARALLAX_MAX_HALF  = 5.00
PARALLAX_FRAC      = 100.0
PARALLAX_ERR_SIGMA = 6.0

PM_SIGMA_FLOOR     = 1.5
PM_ERR_SIGMA       = 8.0
PM_CHI2            = 50.0

SPATIAL_CORE_FRAC  = 0.30

CALIB_MATCH_MAX_ARCSEC = 10
CALIB_G_MAG_RANGE      = (10.0, 19.0)
CALIB_BP_MAG_RANGE     = (10.0, 19.5)


Here are the practical rules in plain English, using your parameter names:

* **Sparse / nearby open clusters**

  * `MATCH_MAX_ARCSEC`: allow larger (≈2–3″) because field is not crowded.
  * `PARALLAX_MIN_HALF`, `PARALLAX_MAX_HALF`: keep small (≈0.1–2 mas) since parallax signal is strong.
  * `PM_SIGMA_FLOOR`, `PM_ERR_SIGMA`, `PM_CHI2`: keep tighter (≈0.8–1.0 mas/yr, χ² ≤ 25–30) because Gaia astrometry is precise nearby.

* **Moderately distant open clusters**

  * `MATCH_MAX_ARCSEC`: use smaller (≈2″) to avoid mismatches.
  * `PARALLAX_MIN_HALF`, `PARALLAX_MAX_HALF`: expand (≈0.1–3 mas) because parallax errors are larger.
  * `PARALLAX_FRAC`: increase (≈10) so the window does not collapse around small parallaxes.
  * `PM_SIGMA_FLOOR`, `PM_ERR_SIGMA`, `PM_CHI2`: relax (≈1.2 mas/yr, σ ≈6 mas/yr, χ² ≤ 40).

* **Globular clusters (very distant, very crowded)**

  * `MATCH_MAX_ARCSEC`: keep very small (≈1″) to avoid false matches in crowded cores.
  * `PARALLAX_MIN_HALF`, `PARALLAX_MAX_HALF`: broaden strongly (≈0.05–5 mas).
  * `PARALLAX_FRAC`: set very high (≈100) so the relative cap does not choke around π ≈ 0.1 mas.
  * `PARALLAX_ERR_SIGMA`: use ≥6 to tolerate large fractional parallax errors.
  * `PM_SIGMA_FLOOR`, `PM_ERR_SIGMA`, `PM_CHI2`: broaden further (≈1.5 mas/yr floor, σ ≈8 mas/yr, χ² ≤ 50).
  * `SPATIAL_CORE_FRAC`: increase (≈0.25–0.30) to base statistics on an outer annulus, since Gaia misses the very core.

* **Always adjust**

  * `CLUSTER_DISTANCE_PC`: set to the literature value of the cluster distance.
  * `AG_EXTINCTION`, `EBG_REDDENING`: set to values appropriate for the field (from extinction maps or literature).

This way you adapt the same code to both open and globular clusters without rewriting it.

