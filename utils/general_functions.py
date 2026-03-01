import numpy as np
import itertools
from typing import List

def complex_to_mag_db_ph_deg(
    gain_complex: List[float]
) -> List[float]:
    """
    Converts a complex-valued gain (e.g., frequency response) into its magnitude in decibels (dB)
    and phase in degrees.

    The magnitude is computed as 20·log₁₀(|gain_complex|), which is standard for voltage or field
    quantities. The phase is computed using the unwrapped angle of the complex input to avoid
    discontinuities (±180° jumps), then converted from radians to degrees.

    Parameters
    ----------
    gain_complex : array-like or scalar of complex numbers
        Complex gain values (e.g., from a transfer function or FFT output).

    Returns
    -------
    list
        A list containing two elements:
        - mag_db : ndarray or float
            Magnitude in decibels (dB).
        - ph_deg : ndarray or float
            Unwrapped phase in degrees.

    Notes
    -----
    - Uses `np.unwrap` to ensure smooth phase transitions—important for plotting or further analysis.
    - Input can be a scalar complex number or an array (e.g., frequency response vector).
    - Assumes `np` (NumPy) is imported.

    Example
    -------
    >>> import numpy as np
    >>> H = 1 + 1j
    >>> mag_db, ph_deg = complex_to_mag_db_ph_deg(H)
    >>> print(f"{mag_db:.2f} dB, {ph_deg:.2f}°")
    3.01 dB, 45.00°
    """
    
    # Magnitude and angle.
    mag_db = 20*np.log10(np.abs(gain_complex))
    ph_deg = np.rad2deg(np.unwrap(np.angle(gain_complex)))

    return [mag_db, ph_deg]

def transfer_function(
  freq: List[float],
  zero_poles: int,
  poles: List[float],
  zeros: List[float]
) -> List[float]:
        
    gain_complex = 1.0 / (1j*2*np.pi*freq)**zero_poles
        
    for zero in zeros:
        gain_complex *= 1.0 + 1j*freq/zero
    
    for pole in poles:
        gain_complex /= 1.0 + 1j*freq/pole
    
    return gain_complex


def generate_masks(masks, configer):
    
    N = configer["length"]
    N_ZERO_POLE_MAX = configer["Nzp_max"]
    N_LEFT_POLE_MAX = configer["Nlp_max"]
    N_RIGHT_POLE_MAX = configer["Nrp_max"]
    N_LEFT_ZERO_MAX = configer["Nlz_max"]
    N_RIGHT_ZERO_MAX = configer["Nrz_max"]
    
    param_ranges = [
        range(N_ZERO_POLE_MAX+1),
        range(N_LEFT_POLE_MAX+1),
        range(N_RIGHT_POLE_MAX+1),
        range(N_LEFT_ZERO_MAX+1),
        range(N_RIGHT_ZERO_MAX+1),
        range(int(configer["size"]))
    ]

    for nzp, nlp, nrp, nlz, nrz, n in itertools.product(*param_ranges):
        
        key = f"{nzp}zp{nlp}lp{nrp}rp{nlz}lz{nrz}rz_{n:03}"
        
        masks[key] = {
            "zero_poles": int(nzp),
            
            "left_poles": np.random.choice(
                N, size=nlp, replace=False).tolist(),

            "right_poles": np.random.choice(
                N, size=nrp, replace=False).tolist(),
            
            "left_zeros": np.random.choice(
                N, size=nlz, replace=False).tolist(),

            "right_zeros": np.random.choice(
                N, size=nrz, replace=False).tolist()
        }

    return masks


def generate_freq_zeros_poles(mask, configer):

    N = configer["length"]
    F_MIN_RANGE = configer["fmin"]
    F_MAX_RANGE = configer["fmax"]
    
    freq_lim = np.random.uniform(
        low=[F_MIN_RANGE[0], F_MAX_RANGE[0]],
        high=[F_MIN_RANGE[-1], F_MAX_RANGE[-1]]
    )
    
    freq = np.linspace(freq_lim[0], freq_lim[-1], N)
    delta_f = freq[1] - freq[0]
    
    poles = []
    poles.extend(  freq_lim[0] + np.array(mask["left_poles"]) * delta_f)
    poles.extend(-(freq_lim[0] + np.array(mask["right_poles"]) * delta_f))
    
    zeros = []
    zeros.extend(  freq_lim[0] + np.array(mask["left_zeros"]) * delta_f)
    zeros.extend(-(freq_lim[0] + np.array(mask["right_zeros"]) * delta_f))
        
    return freq, zeros, poles


def positions_to_mask(positions, total_bits):
    """Convert list of bit positions to an integer mask."""
    mask = [0] * total_bits
    for pos in positions:
        mask[pos] = 1
    return mask