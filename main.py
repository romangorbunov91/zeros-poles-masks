import numpy as np
import json
from pathlib import Path

from utils.settings import var
from utils.general_functions import transfer_function, generate_masks, generate_freq_zeros_poles

config_dir = Path("./config/")

if __name__ == "__main__":
    
    config_path = config_dir / "config.json"
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path, "r") as f:
        configer = json.load(f)
    
    dataset_dir = Path('./dataset/')
    output_data_dir = dataset_dir / configer["type"]
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    masks = generate_masks(masks={}, configer=configer)

    # save masks as json.
    with open(dataset_dir / (configer["type"] + '_masks.json'), "w") as f:
        json.dump(masks, f, indent=4)
    
    for key, mask in masks.items():
        freq, zeros, poles = generate_freq_zeros_poles(mask, configer)
        
        gain_complex = transfer_function(
            freq=freq,
            zero_poles=mask["zero_poles"],
            poles=poles,
            zeros=zeros)
        
        data = np.column_stack((freq, np.real(gain_complex), np.imag(gain_complex)))

        np.savetxt(output_data_dir / f"{key}.csv", data, delimiter=',', header='Frequency (Hz), Gain (Real), Gain (Imag')