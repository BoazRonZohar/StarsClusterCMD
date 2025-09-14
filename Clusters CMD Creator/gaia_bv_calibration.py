import numpy as np
from dataclasses import dataclass

@dataclass
class Poly:
    coefs: tuple
    def __call__(self, x):
        return sum(c * x**i for i, c in enumerate(self.coefs))

@dataclass
class BVTransformCoefs:
    PV: Poly   # V - G as function of BP-RP
    PBV: Poly  # B - V as function of BP-RP

    @staticmethod
    def defaults():
        return BVTransformCoefs(
            # Y = V - G ; X = BP-RP ; Validity = All
            PV=Poly((
                0.02458722322568100,
                0.05385533955796798,
                0.2187428599449099,
                -0.1871099675498201,
                0.01678496838235600,
                0.04153625834745960,
                -0.056037228505745e0,
                0.01633249830098700,
                -0.00154217904703490,
                0.0
            )),
            # Y = B - V ; X = BP-RP ; Validity = Giants
            PBV=Poly((
                0.92062047359143025,
                -2.730139824259589974,
                4.81829140057860972,
                -2.79131045455304998,
                0.85320331642618796,
                -0.12900403100039102
            ))
        )

def gaia_to_BV(df_gaia, coefs: BVTransformCoefs):
    """
    Compute Johnson B,V magnitudes from Gaia DR3 photometry.
    Input df_gaia must contain:
      'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'
    """
    bp_rp = df_gaia['phot_bp_mean_mag'] - df_gaia['phot_rp_mean_mag']
    G = df_gaia['phot_g_mean_mag'].to_numpy()

    # Transformations
    V_minus_G = coefs.PV(bp_rp)
    V = G + V_minus_G
    B_minus_V = coefs.PBV(bp_rp)
    B = V + B_minus_V

    return {
        'V_std': V,
        'B_std': B,
        'B_minus_V': B_minus_V
    }
