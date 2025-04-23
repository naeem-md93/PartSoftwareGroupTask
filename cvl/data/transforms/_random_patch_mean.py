import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor


class RandomPatchMean(nn.Module):
    """
    Replace a random patch of size (h, w) in a Tensor image with the scalar mean
    of the entire image.

    Args:
        patch_size (int or tuple): If int, uses square patch of (patch_size, patch_size).
    """
    def __init__(self, patch_size):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_h, self.patch_w = patch_size

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Float Tensor image of shape (C, H, W).
        Returns:
            Tensor: New Tensor with one patch replaced.
        """
        if not torch.is_tensor(img):
            raise TypeError("RandomPatchMean expects a torch.Tensor as input.")

        C, H, W = img.shape
        if self.patch_h > H or self.patch_w > W:
            raise ValueError(f"Patch size {self.patch_h}×{self.patch_w} "
                             f"exceeds image size {H}×{W}.")

        # Clone so we don’t modify input in-place
        out = img.clone()

        # Global scalar mean
        mean_val = out.mean()

        # Random top-left corner
        top  = torch.randint(0, H - self.patch_h + 1, ()).item()
        left = torch.randint(0, W - self.patch_w + 1, ()).item()

        # Fill patch
        out[:, top : top + self.patch_h,
               left: left + self.patch_w] = mean_val

        return out

if __name__ == "__main__":
    x = torch.randn(3, 34, 34)
    trans = RandomPatchMean(8)
    y = trans(x).moveaxis(0, -1)
    plt.matshow(y)
    plt.show()
