import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

class MRIDataset(Dataset):
    """
    A PyTorch Dataset for loading the fastMRI knee dataset.
    Implements the preprocessing pipeline described in the paper [cite: 80-88].
    """
    def __init__(self, h5_files_list, mask_func):
        self.h5_files_list = h5_files_list
        self.mask_func = mask_func

    def __len__(self):
        return len(self.h5_files_list)

    def __getitem__(self, idx):
        file_path = self.h5_files_list[idx]
        
        with h5py.File(file_path, 'r') as hf:
            # 1. K-space Data Loading [cite: 81]
            kspace_complex = hf['kspace'][()]
            
            # 2. Middle Slice Extraction [cite: 82]
            slice_idx = kspace_complex.shape[0] // 2
            kspace_slice = kspace_complex[slice_idx]
            
            # Get max value for normalization [cite: 87]
            max_val = hf.attrs.get('max', 1.0)
            
            # 3. K-space to Tensor Conversion [cite: 83]
            kspace_tensor = T.to_tensor(kspace_slice)

            # --- Ground Truth Image (Fully Sampled) ---
            image_complex = T.ifft2c(kspace_tensor)
            image_abs = T.complex_abs(image_complex)
            
            # --- Undersampled Input Image ---
            # 4. Undersampling [cite: 85]
            mask, _ = self.mask_func(kspace_tensor.shape[:-1], seed=None)
            mask = mask.to(kspace_tensor.device).unsqueeze(-1) # Add complex dim
            
            undersampled_kspace = kspace_tensor * mask
            
            # 5. Image Reconstruction (Zero-filled) [cite: 86]
            undersampled_image_complex = T.ifft2c(undersampled_kspace)
            undersampled_image_abs = T.complex_abs(undersampled_image_complex)
            
            # 6. Normalization [cite: 87]
            target_image = image_abs / max_val
            input_image = undersampled_image_abs / max_val

            # 7. Center Cropping [cite: 88]
            target_image = T.center_crop(target_image, (320, 320))
            input_image = T.center_crop(input_image, (320, 320))

            # Add a channel dimension for the models (C, H, W)
            return input_image.unsqueeze(0), target_image.unsqueeze(0)

def get_mask_func():
    """
    Creates the mask function as described in the paper.
    Center fraction 0.08, acceleration 4. [cite: 85]
    """
    return RandomMaskFunc(
        center_fractions=[0.08],
        accelerations=[4]
    )
