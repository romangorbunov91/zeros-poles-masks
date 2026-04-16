import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

def positions_to_mask(
    positions,
    total_bits: int,
    halfwindow: int=0
    ):
    
    """Convert list of bit positions to an integer mask."""
    
    mask = [0] * total_bits
    for pos in positions:

        start = max(0, pos - halfwindow)
        end = min(total_bits, pos + halfwindow + 1)

        mask[start:end] = [1] * (end - start)
    
    return mask

def linear_idx_to_log_idx(
    positions: List[int],
    total_bits: int,
    fmin: float,
    fmax: float
    ):

    """Map linear spacing index/indices to equivalent log spacing index/indices."""

    positions = np.asarray(positions)
    R = fmax / fmin
    N = total_bits - 1
    return np.round((N * np.log10(1 + positions * (R - 1) / N) / np.log10(R))).astype(int)


@dataclass
class TransformsConfig:
    crop_ratio: List[float] = field(default_factory=lambda: [1.0, 1.0])
    time_delay: List[float] = field(default_factory=lambda: [0.0, 0.0])
    noise_level: List[float] = field(default_factory=lambda: [0.0, 0.0])
    noise_reduce: int = 0
    gain: List[float] = field(default_factory=lambda: [1.0, 1.0])
    

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
        freq,
        magnitude,
        phase,
        masks
        ):
        
        crop_ratio = self.transforms.crop_ratio
        delay = self.transforms.delay
        noise_level = self.transforms.noise_level
        noise_reduce = self.transforms.noise_reduce
        gain = self.transforms.gain
        rng = self.rng
        
        # 1. Crop-Resize Augmentation: both data and masks.
        if min(crop_ratio) < 1.0:

            N = data.shape[-1]          
            
            # Determine random crop length.
            N_crop = int((crop_ratio[0] + rng.random() * (crop_ratio[1] - crop_ratio[0])) * N)

            # Determine random start index.
            start_idx = 0
            if N_crop < N:
                start_idx = rng.integers(0, N - N_crop + 1)
            
            # Slice tensors (keep dimensions for interpolation).
            data_crop = data[:, start_idx:(start_idx + N_crop)]
            masks_crop = masks[:, start_idx:(start_idx + N_crop)]
            
            if N_crop != N:
                # PyTorch's align_corners=False uses half-pixel center alignment:
                # src_coord = (dst_coord + 0.5) * (src_size / dst_size) - 0.5
                scale = N_crop / N
                x_new = (np.arange(N) + 0.5) * scale - 0.5
                x_new = np.clip(x_new, 0, N_crop - 1)
                
                # Vectorized interpolation: flatten leading dims, interp, reshape back
                orig_shape = data_crop.shape
                data_2d = data_crop.reshape(-1, N_crop)
                data_resized = np.interp(x_new, np.arange(N_crop), data_2d)
                data_tensor = data_resized.reshape(orig_shape[:-1] + (N,))
            else:
                data_tensor = data_crop.copy()
            
            masks_remaped = np.zeros_like(masks)
            ch_idxs, idxs = np.where(masks_crop > 0.5)
                
            data_tensor = F.interpolate(data_crop.unsqueeze(0), size=N, mode='linear', align_corners=False).squeeze(0)
            
            masks_tensor_remaped = torch.zeros_like(masks_tensor)
            ch_idxs, idxs = torch.where(masks_crop > 0.5)
            if idxs.numel() > 0:
                # Map coordinates from N_crop space to N space.
                new_idxs = (idxs.float() * N / N_crop).round().long()#.clamp(0, N-1)
                masks_tensor_remaped[ch_idxs, new_idxs] = 1
                
            masks_tensor = masks_tensor_remaped

        # Random gain (magnitude only).
        magnitude += 20*np.log10(gain[0] + rng.random() * (gain[1] - gain[0]))
        
        # Random time-delay (phase only).
        if max(delay) > 0.0:
            phase -= 2 * np.pi * freq * (delay[0] + rng.random() * (delay[1] - delay[0]))

        data = np.vstack([magnitude, phase])
        
        # 4. Random Noise Augmentation (Data only).
        if max(noise_level) > 0.0:
            noise = rng.standard_normal(size=data.shape)
            noise_mask = (rng.random(size=noise.shape) < (0.5 ** noise_reduce)).astype(noise.dtype)
            data_std = np.std(data, axis=-1, keepdims=True)
            scale = noise_level[0] + rng.random() * (noise_level[1] - noise_level[0])
            data += noise * noise_mask * data_std * scale
        
        return data, masks

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
            data_np, masks_np = self._augmentations_(
                        freq=freq
                        magnitude=magnitude,
                        phase=phase,
                        masks=masks_np,
                        )
        
        data_diff1_np = np.gradient(data_np, axis=1)
        data_diff2_np = np.gradient(data_diff1_np, axis=1)
        
        # Convert numpy to tensor.
        data_tensor = torch.from_numpy(np.vstack([data_np, data_diff1_np, data_diff2_np])).float()
        masks_tensor = torch.from_numpy(masks_np).float()
        freq_tensor = torch.from_numpy(freq).float()
        
        return data_tensor, masks_tensor, freq_tensor