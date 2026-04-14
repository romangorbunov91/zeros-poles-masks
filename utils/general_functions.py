import numpy as np
import itertools
from typing import List, Dict, Any


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


def generate_masks(masks: Dict[str, Any], configer: Dict[str, Any]) -> Dict[str, Any]:
    
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

    clearance_dist = configer["clearance_dist"]
    rng = np.random.default_rng(configer["seed"])
    
    if clearance_dist > 0:
        
    else:
        for nzp, nlp, nrp, nlz, nrz, n in itertools.product(*param_ranges):
            
            key = f"{nzp}zp{nlp}lp{nrp}rp{nlz}lz{nrz}rz_{n:03}"
            
            masks[key] = {
                "zero_poles": int(nzp),
                
                "left_poles": rng.choice(
                    N, size=nlp, replace=False).tolist(),

                "right_poles": rng.choice(
                    N, size=nrp, replace=False).tolist(),
                
                "left_zeros": rng.choice(
                    N, size=nlz, replace=False).tolist(),

                "right_zeros": rng.choice(
                    N, size=nrz, replace=False).tolist()
            }

    return masks


def calculate_freq_zeros_poles(mask, configer):

    N = configer["length"]
    F_MIN_RANGE = configer["fmin"]
    F_MAX_RANGE = configer["fmax"]
    
    freq_lim = np.random.uniform(
        low=[F_MIN_RANGE[0], F_MAX_RANGE[0]],
        high=[F_MIN_RANGE[-1], F_MAX_RANGE[-1]]
    )
    '''
    freq = np.linspace(freq_lim[0], freq_lim[-1], N)
    delta_f = freq[1] - freq[0]
    
    poles = []
    poles.extend(  freq_lim[0] + np.array(mask["left_poles"]) * delta_f)
    poles.extend(-(freq_lim[0] + np.array(mask["right_poles"]) * delta_f))
    
    zeros = []
    zeros.extend(  freq_lim[0] + np.array(mask["left_zeros"]) * delta_f)
    zeros.extend(-(freq_lim[0] + np.array(mask["right_zeros"]) * delta_f))
    '''
    
    freq = np.logspace(np.log10(freq_lim[0]), np.log10(freq_lim[-1]), N)
    
    poles = []
    poles.extend( freq[mask["left_poles"]])
    poles.extend(-freq[mask["right_poles"]])
    
    zeros = []
    zeros.extend( freq[mask["left_zeros"]])
    zeros.extend(-freq[mask["right_zeros"]])
     
    return freq, zeros, poles