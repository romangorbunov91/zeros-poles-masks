import numpy as np
import itertools
import json
from pathlib import Path

from utils.settings import var

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
    samples = {}

    for nzp, nlp, nrp, nlz, nrz, n in itertools.product(*param_ranges):
        
        key = f"{nzp}zp{nlp}lp{nrp}rp{nlz}lz{nrz}rz_{n:03}"
        
        data = {
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
        samples[key] = data
        
        freq_lim = np.random.uniform(
            low=[F_MIN_RANGE[0], F_MAX_RANGE[0]],
            high=[F_MIN_RANGE[-1], F_MAX_RANGE[-1]]
        )
        delta_f = (freq_lim[-1] - freq_lim[0]) / N