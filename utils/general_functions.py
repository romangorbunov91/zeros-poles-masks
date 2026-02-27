import numpy as np
from typing import List

def complex_to_mag_db_ph_deg(gain_complex):
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
  N: int,
  freq_lim: List[float],
  zero_poles: int,
  poles: List[float],
  zeros: List[float]
) -> List[float]:
    freq = np.linspace(freq_lim[0], freq_lim[-1], N)
    
    gain_complex = 1.0 / (1j*2*np.pi*freq)**zero_poles
    
    for pole in poles:
        gain_complex *= 1.0 / (1.0 + 1j*freq/pole)
        
    for zero in zeros:
        gain_complex *= 1.0 + 1j*freq/zero
    
    return gain_complex