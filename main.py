import numpy as np
import json
from pathlib import Path

from utils.general_functions import transfer_function, generate_masks, calculate_freq_zeros_poles

config_dir = Path("./config/")
dataset_dir = Path('./zeros-poles-dataset/')


if __name__ == "__main__":
    
    # Load configuration.
    config_path = config_dir / "config.json"
    assert config_path.exists(), f"Config not found: {config_path}"
    with open(config_path, "r") as f:
        configer = json.load(f)
    
    # Create output directory.
    output_data_dir = dataset_dir / configer['split']
    output_data_dir.mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(configer["seed"])
    
    # Generate masks.
    masks = generate_masks(
        masks={},
        configer=configer,
        rng=rng
        )

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
            zeros=zeros
            )
        
        magnitude = 20*np.log10(np.abs(gain_complex))
        phase = np.unwrap(np.angle(gain_complex))

        configer.rng = rng
        
        
        magnitude.T
        phase.T
        data = ???(
            configer=configer,
            freq=freq,
            magnitude=magnitude,
            phase=phase
            )

        np.savetxt(
            output_data_dir / f"{key}.csv",
            data,
            delimiter=',',
            header='Frequency (Hz), Gain (dB), Phase (rad)',
            comments=''
            )