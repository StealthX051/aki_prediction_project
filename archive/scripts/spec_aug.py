# spec_aug.py  • drop-in replacement
import random, torch
from torchvision.transforms import functional as F
from PIL import Image

class SpecAugmentRGB(torch.nn.Module):
    def __init__(self, freq_mask_param=24, time_mask_param=48, num_masks=2):
        super().__init__()
        self.F, self.T, self.N = freq_mask_param, time_mask_param, num_masks

    def _mask(self, x, dim, width):
        if width < 1: return x
        start = random.randint(0, x.size(dim) - width)
        sl = [slice(None)] * x.ndim
        sl[dim] = slice(start, start + width)
        x[tuple(sl)] = 0.0
        return x

    def forward(self, img):
        # ── 1 ensure tensor in 0-1 ──────────────────────────────────────────────
        if isinstance(img, Image.Image):
            x = F.pil_to_tensor(img).float() / 255.0          # [C,H,W]
        elif isinstance(img, torch.Tensor):
            x = img.float().clone()
        else:
            raise TypeError(f"Unsupported type {type(img)}")

        # ── 2 apply N masks along freq (H) and time (W) ─────────────────────────
        for _ in range(self.N):
            x = self._mask(x, 1, random.randint(0, self.F))   # freq
            x = self._mask(x, 2, random.randint(0, self.T))   # time

        # ── 3 return PIL so the rest of the pipeline is unchanged ──────────────
        return F.to_pil_image(torch.clamp(x, 0, 1))

    def __repr__(self):
        return f"{self.__class__.__name__}(F={self.F}, T={self.T}, N={self.N})"
