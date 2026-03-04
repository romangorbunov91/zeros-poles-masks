import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
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
        samples: Optional[List] = None
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sample_id = self.samples[idx]
        sample_path = self.dataset_path / f"{sample_id}.csv"

        if not sample_path.exists():
            raise FileNotFoundError(f"File not found: {sample_path}")
        
        data = np.loadtxt(self.dataset_path / f"{sample_id}.csv", delimiter=',', skiprows=1)

        mask_dict = self.masks[sample_id]
        
        masks_list = []
        for key, positions in mask_dict.items():
            if key == 'zero_poles':
                continue
            masks_list.append(positions_to_mask(positions, total_bits=len(data)))
        
        # Outputs.
        data_tensor = torch.tensor(data[:,1:], dtype=torch.float32)
        masks_tensor = torch.tensor(np.vstack(masks_list), dtype=torch.float32)
        freq = data[:,0]

        return data_tensor, masks_tensor, freq