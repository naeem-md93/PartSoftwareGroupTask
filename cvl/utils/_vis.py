import os.path as osp
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor

from ._io import mkdir


def plot_image(n: str, p: np.ndarray, path: str) -> None:

    mkdir(path)

    plt.imshow(p)
    plt.title(f"{n}")
    plt.savefig(osp.join(path, f"{n}.png"))
    plt.close()

def plot_histogram(n: str, p: Tensor, path: str) -> None:

    mkdir(path)

    plt.hist(p.detach().cpu().numpy().flatten(), bins=100)
    plt.title(f"Histogram of {n} - # P: {p.numel()}")
    plt.savefig(osp.join(path, f"{n}.png"))
    plt.close()


def plot_heatmap(n: str, p: Tensor, path: str) -> None:

    mkdir(path)

    plt.matshow(p.detach().cpu().numpy())
    plt.colorbar()
    plt.title(f"Heatmap of {n} - # P: {p.numel()}")
    plt.savefig(osp.join(path, f"{n}.png"))
    plt.close()

def plot_cam(n: str, p: np.ndarray, path: str) -> None:

    mkdir(path)

    plt.matshow(p)
    plt.colorbar()
    plt.title(f"Heatmap of {n}")
    plt.savefig(osp.join(path, f"{n}.png"))
    plt.close()

