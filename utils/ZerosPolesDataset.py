import numpy as np
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from typing import Union, Tuple

from utils.general_functions import positions_to_mask

class ZerosPolesDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        split: str
    ):
        
        super().__init__()
        
        self.dataset_dir = Path(dataset_dir)
        self.dataset_path = self.dataset_dir / split
        
        mask_path = self.dataset_dir / (split + "_masks.json")
        assert mask_path.exists(), f"Mask not found: {mask_path}"
        with open(mask_path, "r") as f:
            self.masks = json.load(f)
        
        self.samples = list(self.masks.keys())
        print(self.samples[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sample_id = self.samples[idx]
        sample_path = self.dataset_path / f"{sample_id}.csv"

        if not sample_path.exists():
            raise FileNotFoundError(f"File not found: {sample_path}")
        
        data = np.loadtxt(self.dataset_path / f"{sample_id}.csv", delimiter=',', skiprows=1)

        freq = data[:,0]
        data_tensor = torch.tensor(data[:,1:], dtype=torch.float32)

        mask_dict = self.masks[sample_id]

        mask_list = []
        for key, positions in mask_dict.items():
            if key == 'zero_poles':
                continue
            mask_list.append(positions_to_mask(positions, total_bits=len(freq)))
        masks = np.vstack(mask_list)

        masks_tensor = torch.tensor(masks, dtype=torch.float32)

        return freq, data_tensor, masks_tensor