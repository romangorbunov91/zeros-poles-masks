import numpy as np
import json
from pathlib import Path

from utils.general_functions import transfer_function, generate_masks, calculate_freq_zeros_poles

config_dir = Path("./config/")
dataset_dir = Path('./zeros-poles-dataset/')

def _augmentations_(
        configer,
        freq: np.ndarray,
        magnitude: np.ndarray,
        phase: np.ndarray
        ) -> np.ndarray:
        
        """
        Args:
            freq: 1D float array of frequencies.
            magnitude: 1D float array of magnitudes.
            phase: 1D float array of phases.
        Returns:
            np.ndarray of shape (2, Length) with dtype float.
        """
        
        rng = configer.rng
        
        gain = configer["gain"]
        delay = configer["delay"]
        noise_level = configer["noise_level"]
        noise_reduce = configer["noise_reduce"]

        # Random gain (magnitude only).
        if any(x != 1.0 for x in gain):
            magnitude += 20*np.log10(gain[0] + rng.random() * (gain[1] - gain[0]))
        
        # Random time-delay (phase only).
        if max(delay) > 0.0:
            phase -= 2 * np.pi * freq * (delay[0] + rng.random() * (delay[1] - delay[0]))

        # Combine before noising.
        data = np.vstack([magnitude, phase])

        # Random Noise.
        if max(noise_level) > 0.0:
            noise = rng.standard_normal(size=data.shape)
            noise_mask = (rng.random(size=noise.shape) < (0.5 ** noise_reduce)).astype(noise.dtype)
            data_std = np.std(data, axis=-1, keepdims=True)
            scale = noise_level[0] + rng.random() * (noise_level[1] - noise_level[0])
            data += noise * noise_mask * data_std * scale
        
        return data

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
        
        data = _augmentations_(
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