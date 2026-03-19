import numpy as np
import json
from pathlib import Path

from utils.general_functions import transfer_function, generate_masks, calculate_freq_zeros_poles

config_dir = Path("./config/")
dataset_dir = Path('./dataset/')

if __name__ == "__main__":
    
    # Load configuration.
    config_path = config_dir / "config.json"
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path, "r") as f:
        configer = json.load(f)
    
    # Create output directory.
    output_data_dir = dataset_dir / configer['split']
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate masks.
    masks = generate_masks(masks={}, configer=configer)

    # Save masks as json.
    with open(dataset_dir / (configer['split'] + '_masks.json'), "w") as f:
        json.dump(masks, f, indent=4)
    
    # Generate zeros-poles, then complete frequency response.
    for key, mask in masks.items():
        freq, zeros, poles = calculate_freq_zeros_poles(mask, configer)
        
        gain_complex = transfer_function(
            freq=freq,
            zero_poles=mask["zero_poles"],
            poles=poles,
            zeros=zeros)
        
        gain = configer["gain"]
        gain_complex *= (gain[0] + np.random.rand() * (gain[1] - gain[0]))
        
        data = np.column_stack((freq, np.real(gain_complex), np.imag(gain_complex)))

        np.savetxt(
            output_data_dir / f"{key}.csv",
            data,
            delimiter=',',
            header='Frequency (Hz), Gain (Real), Gain (Imag)',
            comments=''
        )