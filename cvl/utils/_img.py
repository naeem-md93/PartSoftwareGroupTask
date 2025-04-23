from typing import Callable, Optional
import cv2
import os.path as osp
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import Pool

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.transforms import v2

from ._vis import (
    plot_image,
    plot_histogram,
    plot_heatmap,
    plot_cam
)


def imagenet_unnormalizer(images: Tensor) -> Tensor:
    trans = v2.Compose([
        v2.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        v2.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ])

    return trans(images)


def multiprocess_plot(d):
    func_name = d[0]
    func_inputs = d[1]

    funcs = {
        "histogram": plot_histogram,
        "heatmap": plot_heatmap,
        "image": plot_image,
        "cam": plot_cam,
    }

    funcs[func_name](*func_inputs)


class GradCAM:
    """
    Compute Grad-CAM for specified target layers in a PyTorch model.

    Args:
        model (torch.nn.Module): The pretrained model.
        target_layers (list of str): Names of the layers to compute Grad-CAM for.
    """
    def __init__(self, model: nn.Module, target_layers: list[str]) -> None:
        self.model = model
        self.model.eval()
        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self.processes = 4
        self._register_hooks()

    def _register_hooks(self) -> None:
        # Register forward and backward hooks on target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(self._save_activation(name))
                module.register_full_backward_hook(self._save_gradient(name))

    def _save_activation(self, name: str) -> Callable:
        def hook(module, input, output):
            # Save the forward activation
            self.activations[name] = output.detach()
        return hook

    def _save_gradient(self, name: str) -> Callable:
        def hook(module, grad_input, grad_output):
            # Save the backward gradient
            self.gradients[name] = grad_output[0].detach()
        return hook

    # @staticmethod
    # def overlay_cam_on_image(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    #     """
    #     Overlay the Grad-CAM heatmap onto the original image.
    #
    #     Args:
    #         image (np.ndarray): Original image as a NumPy array in RGB format, shape (H, W, 3), dtype uint8.
    #         cam (np.ndarray): Normalized CAM map, shape (H, W), values in [0, 1].
    #         alpha (float): Weight of the heatmap overlay (0 transparent, 1 full heatmap).
    #         colormap (int): OpenCV colormap to apply to the heatmap.
    #
    #     Returns:
    #         np.ndarray: Image with heatmap overlay, same shape as input image.
    #     """
    #     # Ensure cam is in 0-255 uint8
    #     heatmap = np.uint8(255 * cam)
    #     # Apply the colormap
    #     heatmap = cv2.applyColorMap(heatmap, colormap)      # BGR heatmap
    #     # Convert heatmap to RGB
    #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    #
    #     # Resize heatmap to match image size if needed
    #     if heatmap.shape[:2] != image.shape[:2]:
    #         heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    #
    #     # Blend heatmap with original image
    #     overlaid = np.uint8(alpha * heatmap + (1 - alpha) * image)
    #     return overlaid

    def plot_images(self, names: list[str], images: np.ndarray, checkpoints_path: str) -> None:

        plot_pool = []

        save_path = osp.join(checkpoints_path, f"plots/samples/")

        for img_idx, img_name in enumerate(names):
            plot_pool.append(("image", (f"image_{img_name}", images[img_idx], osp.join(save_path, img_name))))

        print(f"=> Plotting {len(plot_pool)} images")
        Pool(processes=self.processes).map(multiprocess_plot, plot_pool)

    def plot_weights(self, checkpoints_path: str) -> None:

        plot_pool = []

        save_path = osp.join(checkpoints_path, f"plots/model/weights/")

        for name, params in self.model.named_parameters():

            plot_pool.append(("histogram", (
                f"hist_{name}",
                params,
                save_path
            )))

        print(f"=> Plotting {len(plot_pool)} weights")
        Pool(processes=self.processes).map(multiprocess_plot, plot_pool)

    def plot_activations(self, names: list[str], checkpoints_path: str) -> None:

        plot_pool = []

        for img_idx, img_name in enumerate(names):
            save_path = osp.join(checkpoints_path, f"plots/samples/{img_name}/")

            for layer_name, layer_acts in self.activations.items():

                plot_pool.append(("histogram", (
                    f"hist_act_l_{layer_name}",
                    layer_acts[img_idx],
                    save_path
                )))

                if layer_name == "fc":
                    layer_acts = layer_acts[img_idx].unsqueeze(0).repeat((10, 1))
                else:
                    layer_acts = layer_acts[img_idx].mean(dim=0)

                plot_pool.append(("heatmap", (f"heat_act_l_{layer_name}", layer_acts, save_path)))

        print(f"=> Plotting {len(plot_pool)} activations")
        Pool(processes=self.processes).map(multiprocess_plot, plot_pool)

    def plot_gradients(self, names: list[str], class_label: str, checkpoints_path: str) -> None:

        plot_pool = []

        for img_idx, img_name in enumerate(names):
            save_path = osp.join(checkpoints_path, f"plots/samples/{img_name}/")

            for layer_name, layer_grads in self.gradients.items():

                plot_pool.append(("histogram", (
                    f"hist_grad_wrt_{class_label}_l_{layer_name}",
                    layer_grads[img_idx],
                    save_path
                )))

                if layer_name == "fc":
                    layer_grads = layer_grads[img_idx].unsqueeze(0).repeat((10, 1))
                else:
                    layer_grads = layer_grads[img_idx].mean(dim=0)

                plot_pool.append(("heatmap", (f"heat_grad_wrt_{class_label}_l_{layer_name}", layer_grads, save_path)))

        print(f"=> Plotting {len(plot_pool)} {class_label} gradients")

        Pool(processes=self.processes).map(multiprocess_plot, plot_pool)

    @staticmethod
    def get_cams(acts: Tensor, grads: Tensor, img_size: torch.Size) -> np.ndarray:

        # Global average pooling on gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        weighted_activations = weights * acts  # (1, C, H_l, W_l)
        cams = weighted_activations.sum(dim=1, keepdim=True)  # (1, 1, H_l, W_l)
        cams = F.relu(cams)  # apply ReLU

        # Upsample to input size
        cams = F.interpolate(cams, size=img_size, mode='bilinear', align_corners=False)
        # Normalize cam between 0 and 1
        cams = cams.squeeze().detach().cpu().numpy()
        cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-9)

        return cams

    def plot_cams(self, cams: np.ndarray, names: list[str], class_label: str, layer_name: str, checkpoints_path: str) -> None:

        plot_pool = []

        for img_idx in range(cams.shape[0]):
            save_path = osp.join(checkpoints_path, f"plots/samples/{names[img_idx]}/")
            plot_pool.append(("cam", (
                f"cam_wrt_{class_label}_l_{layer_name}",
                cams[img_idx],
                save_path
            )))

        print(f"=> Plotting {len(plot_pool)} {class_label} cams")

        Pool(processes=self.processes).map(multiprocess_plot, plot_pool)

    def overlay_cams_on_images(
        self,
        images: np.ndarray,
        cams: np.ndarray,
        names: list[str],
        class_label: str,
        layer_name: str,
        checkpoints_path: str,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ):

        image_size = images.shape[1:3]

        plot_pool = []

        for img_idx in range(cams.shape[0]):
            save_path = osp.join(checkpoints_path, f"plots/samples/{names[img_idx]}/")

            heatmap = np.uint8(255 * cams[img_idx])

            # Apply the colormap
            heatmap = cv2.applyColorMap(heatmap, colormap)  # BGR heatmap

            # Convert heatmap to RGB
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Resize heatmap to match image size if needed
            if heatmap.shape[:2] != images.shape[:2]:
                heatmap = cv2.resize(heatmap, image_size, interpolation=cv2.INTER_LINEAR)

            # Blend heatmap with original image
            overlaid = np.uint8(alpha * heatmap + (1 - alpha) * images[img_idx])

            plot_pool.append(("image", (
                f"cam_image_wrt_{class_label}_l_{layer_name}",
                overlaid,
                save_path
            )))

        print(f"=> Plotting {len(plot_pool)} {class_label} cams on images")
        Pool(processes=self.processes).map(multiprocess_plot, plot_pool)

    def run(self, names: list[str], input_tensor: torch.Tensor, class_labels: list[str], checkpoints_path: str) -> None:

        logits, _ = self.model(input_tensor)

        input_tensor = imagenet_unnormalizer(input_tensor.clone().detach().cpu()).moveaxis(1, -1).numpy()
        image_size = input_tensor.shape[1:3]

        # self.plot_weights(checkpoints_path)
        # self.plot_images(names, input_tensor, checkpoints_path)
        # self.plot_activations(names, checkpoints_path)

        for cls_idx in range(logits.shape[1]):

            self.model.zero_grad()
            score = logits[0, cls_idx]
            score.backward(retain_graph=True)

            # self.plot_gradients(names, class_labels[cls_idx], checkpoints_path)

            for layer_name in self.target_layers:
                if layer_name in ("fc", ):
                    continue

                activation = self.activations[layer_name]
                gradient = self.gradients[layer_name]

                cams = self.get_cams(activation, gradient, image_size)

                # self.plot_cams(
                #     cams,
                #     names,
                #     class_labels[cls_idx],
                #     layer_name,
                #     checkpoints_path
                # )

                self.overlay_cams_on_images(
                    input_tensor,
                    cams,
                    names,
                    class_labels[cls_idx],
                    layer_name,
                    checkpoints_path,
                    alpha=0.01
                )