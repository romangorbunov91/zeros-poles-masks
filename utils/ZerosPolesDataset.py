import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

def positions_to_mask(
    positions: List[int],
    total_bits: int,
    halfwindow: int=0
    ) -> List[int]:
    
    # Convert list of bit positions to an integer mask.
    
    mask = [0] * total_bits
    for pos in positions:

        start = max(0, pos - halfwindow)
        end = min(total_bits, pos + halfwindow + 1)

        mask[start:end] = [1] * (end - start)
    
    return mask


@dataclass
class TransformsConfig:
    gain: List[float] = field(default_factory=lambda: [1.0, 1.0])
    time_delay: List[float] = field(default_factory=lambda: [0.0, 0.0])
    noise_level: List[float] = field(default_factory=lambda: [0.0, 0.0])
    noise_reduce: int = 0
    

class ZerosPolesDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str,
        mask_halfwindow: int = 0,
        samples: Optional[List] = None,
        transforms: Optional[TransformsConfig] = None,
        rng: Optional[np.random.Generator] = None
        ):
        
        super().__init__()
        
        self.dataset_path = Path(dataset_dir) / split
        
        mask_path = Path(dataset_dir) / (split + "_masks.json")
        assert mask_path.exists(), f"Mask not found: {mask_path}"
        with open(mask_path, "r") as f:
            self.masks = json.load(f)
        
        self.mask_halfwindow = mask_halfwindow
        
        # Read all samples.
        if samples is None:
            self.samples = list(self.masks.keys())
        else:
            self.samples = samples
        
        self.transforms = transforms
        
        if rng is None:
            self.rng = np.random.default_rng()
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def _augmentations_(self,
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
        
        rng = self.rng
        
        gain = self.transforms.gain
        delay = self.transforms.delay
        noise_level = self.transforms.noise_level
        noise_reduce = self.transforms.noise_reduce

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        sample_id = self.samples[idx]
        sample_path = self.dataset_path / f"{sample_id}.csv"

        if not sample_path.exists():
            raise FileNotFoundError(f"File not found: {sample_path}")
        
        file_data_np = (np.loadtxt(self.dataset_path / f"{sample_id}.csv", delimiter=',', skiprows=1)).T
        
        freq = file_data_np[0, :]
        magnitude = file_data_np[1, :]
        phase = file_data_np[2, :]

        mask_dict = self.masks[sample_id]
        masks_list = []
        for key, positions in mask_dict.items():
            if key == 'zero_poles':
                continue
            masks_list.append(
                positions_to_mask(
                        positions=positions,
                        total_bits=len(freq),
                        halfwindow=self.mask_halfwindow
                    ),
                )
        masks_np = np.vstack(masks_list)

        if self.transforms is None:
            data_np = np.vstack([magnitude, phase])
        else:
            data_np = self._augmentations_(
                freq=freq,
                magnitude=magnitude,
                phase=phase
                )
        
        data_diff1_np = np.gradient(data_np, axis=1)
        data_diff2_np = np.gradient(data_diff1_np, axis=1)
        
        # Convert numpy to tensor.
        data_tensor = torch.from_numpy(np.vstack([data_np, data_diff1_np, data_diff2_np])).float()
        masks_tensor = torch.from_numpy(masks_np).float()
        freq_tensor = torch.from_numpy(freq).float()
        
        return data_tensor, masks_tensor, freq_tensor