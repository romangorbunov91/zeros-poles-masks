import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Union, Tuple, List, Optional

def positions_to_mask(positions, total_bits):
    """Convert list of bit positions to an integer mask."""
    mask = [0] * total_bits
    for pos in positions:
        mask[pos] = 1
    
    return mask


class ZerosPolesDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str,
        samples: Optional[List] = None,
        transforms_flag: bool = False,
        crop_size: float = 0.0,
        noise_level: List[float] = [0.0, 0.0],
        noise_alfa: float = 0.6,
        gain: List[float] = [1.0, 1.0],
        #delay: List[float] = [0.0, 0.0]
    ):
        
        super().__init__()
        
        self.dataset_path = Path(dataset_dir) / split
        
        mask_path = Path(dataset_dir) / (split + "_masks.json")
        assert mask_path.exists(), f"Mask not found: {mask_path}"
        with open(mask_path, "r") as f:
            self.masks = json.load(f)
        
        # Read all samples.
        if samples is None:
            self.samples = list(self.masks.keys())
        else:
            self.samples = samples

        self.transforms_flag = transforms_flag
        self.crop_size = crop_size
        self.noise_level = noise_level
        self.noise_alfa = noise_alfa
        self.gain = gain
        
    def __len__(self) -> int:
        return len(self.samples)

    def _augmentations_(self, data_tensor, masks_tensor):
        
        # 1. Crop-Resize Augmentation
        if max(self.crop_size) > 0.0:
            
            N = data_tensor.shape[-1]          
            
            # Determine random crop length.
            crop_ratio = 1 - (self.crop_size[0] + torch.rand(1).item() * (self.crop_size[1] - self.crop_size[0]))           
            N_crop = int(crop_ratio * N)
            
            # Determine random start index.
            start_idx = 0
            if N_crop < N:
                start_idx = torch.randint(0, N - N_crop + 1, (1,)).item()
            
            # Slice tensors (keep dimensions for interpolation).
            data_crop = data_tensor[:, start_idx:(start_idx + N_crop)]
            masks_crop = masks_tensor[:, start_idx:(start_idx + N_crop)]
            
            # Add batch dimension since torch.nn.functional.interpolate expects (Batch, Channel, Length).
            data_crop = data_crop.unsqueeze(0)
            data_tensor = F.interpolate(data_crop, size=N, mode='linear', align_corners=False)
            data_tensor = data_tensor.squeeze(0) # Remove back.

            masks_tensor_remaped = torch.zeros_like(masks_tensor)
            ch_idxs, idxs = torch.where(masks_crop > 0.5)
            if idxs.numel() > 0:
                # Map coordinates from N_crop space to N space.
                new_idxs = (idxs.float() * N / N_crop).round().long()
                masks_tensor_remaped[ch_idxs, new_idxs] = 1
                
            masks_tensor = masks_tensor_remaped

        freq_tensor = data_tensor[0 ,:]
        data_tensor = data_tensor[1:,:]
        
        # 2. Random Noise Augmentation (Data only).
        if max(self.noise_level) > 0.0:
            noise = torch.randn_like(data_tensor)
            filtered_noise = torch.zeros_like(noise)
            for n in range(1, data_tensor.shape[-1]):
                filtered_noise[:, n] = self.noise_alfa * noise[:, n] + (1 - self.noise_alfa) * filtered_noise[:, n - 1]
                    
            data_tensor += filtered_noise * data_tensor.std(dim=-1, keepdim=True) * (self.noise_level[0] + torch.rand(1).item() * (self.noise_level[1] - self.noise_level[0]))
        
        # 3. Random gain (Data only).
        data_tensor *= (self.gain[0] + torch.rand(1).item() * (self.gain[1] - self.gain[0]))
                    
        return data_tensor, masks_tensor, freq_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        sample_id = self.samples[idx]
        sample_path = self.dataset_path / f"{sample_id}.csv"

        if not sample_path.exists():
            raise FileNotFoundError(f"File not found: {sample_path}")
        
        data_T = np.loadtxt(self.dataset_path / f"{sample_id}.csv", delimiter=',', skiprows=1)
        data = data_T.T
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        mask_dict = self.masks[sample_id]
        masks_list = []
        for key, positions in mask_dict.items():
            if key == 'zero_poles':
                continue
            masks_list.append(positions_to_mask(positions, total_bits=data.shape[-1]))
        masks_tensor = torch.tensor(np.vstack(masks_list), dtype=torch.float32)
        
        if self.transforms_flag:
            data_tensor, masks_tensor, freq_tensor = self._augmentations_(data_tensor, masks_tensor)
        else:
            freq_tensor = data_tensor[0 ,:]
            data_tensor = data_tensor[1:,:]

        return data_tensor, masks_tensor, freq_tensor