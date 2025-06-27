# =============================================================================
# spec_aug.py
#
# This file defines a custom SpecAugment transform that works on 3-channel
# RGB images, which is necessary because the default PyTorch transform
# expects a single-channel (grayscale) input.
#
# This implementation is based on the principles outlined in the Google Brain
# paper on SpecAugment.
# =============================================================================

import torch
import random

class SpecAugmentRGB(torch.nn.Module):
    """
    Custom SpecAugment for 3-channel (RGB) spectrogram images.

    Args:
        freq_mask_param (int): Maximum possible width of the frequency mask.
        time_mask_param (int): Maximum possible width of the time mask.
        num_masks (int): The number of masks to apply for both time and frequency.
    """
    def __init__(self, freq_mask_param=24, time_mask_param=48, num_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the SpecAugment transformations.

        Args:
            x (torch.Tensor): Input tensor of shape (C, H, W)
                              where C=3 for RGB.

        Returns:
            torch.Tensor: Augmented tensor.
        """
        # Clone the tensor to avoid modifying the original in-place
        augmented_spec = x.clone()
        _, height, width = augmented_spec.shape

        # Apply frequency masking
        for _ in range(self.num_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, height - f)
            # Apply the same mask across all 3 channels
            augmented_spec[:, f0:f0+f, :] = 0

        # Apply time masking
        for _ in range(self.num_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, width - t)
            # Apply the same mask across all 3 channels
            augmented_spec[:, :, t0:t0+t] = 0

        return augmented_spec

    def __repr__(self):
        return self.__class__.__name__ + f'(F={self.freq_mask_param}, T={self.time_mask_param}, N={self.num_masks})'
