import numpy as np
import itertools
import json
from pathlib import Path

from utils.settings import var
from utils.general_functions import transfer_function, complex_to_vect

config_dir = Path("./config/")

if __name__ == "__main__":
    

    config_path = config_dir / "config.json"
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path, "r") as f:
        configer = json.load(f)
    
    N = configer["length"]
    F_MIN_RANGE = configer["fmin"]
    F_MAX_RANGE = configer["fmax"]
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
    masks = {}

    for nzp, nlp, nrp, nlz, nrz, n in itertools.product(*param_ranges):
        
        key = f"{nzp}zp{nlp}lp{nrp}rp{nlz}lz{nrz}rz_{n:03}"
        
        masks[key] = {
            "zero_poles": int(nzp),
            
            "left_poles": np.random.choice(
                N, size=nlp, replace=False),

            "right_poles": np.random.choice(
                N, size=nrp, replace=False),
            
            "left_zeros": np.random.choice(
                N, size=nlz, replace=False),

            "right_zeros": np.random.choice(
                N, size=nrz, replace=False)
        }
        
        freq_lim = np.random.uniform(
            low=[F_MIN_RANGE[0], F_MAX_RANGE[0]],
            high=[F_MIN_RANGE[-1], F_MAX_RANGE[-1]]
        )
        
        freq = np.linspace(freq_lim[0], freq_lim[-1], N)
        delta_f = freq[1] - freq[0]
        
        poles = []
        poles.extend(  freq_lim[0] + masks[key]["left_poles"] * delta_f)
        poles.extend(-(freq_lim[0] + masks[key]["right_poles"] * delta_f))
        
        zeros = []
        zeros.extend(  freq_lim[0] + masks[key]["left_zeros"] * delta_f)
        zeros.extend(-(freq_lim[0] + masks[key]["right_zeros"] * delta_f))
        
        gain_complex = transfer_function(
            freq=freq,
            zero_poles=masks[key]["zero_poles"],
            poles=poles,
            zeros=zeros)
        
        out = []
        out.extend(freq)
        out.extend(complex_to_vect(gain_complex))
    print(out[1])