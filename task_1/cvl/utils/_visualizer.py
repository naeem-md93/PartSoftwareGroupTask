import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from sklearn.metrics.pairwise import cosine_similarity


from ._io import mkdir

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.transforms import v2


def unnormalizer(images: Tensor, mean: list[float], std: list[float]) -> Tensor:
    trans = v2.Compose([
        v2.Normalize(mean=[0, 0, 0], std=[1/std[0], 1/std[1], 1/std[2]]),
        v2.Normalize(mean=[-mean[0], -mean[1], -mean[2]], std=[1, 1, 1]),
    ])

    return trans(images)


def plot_image(n: str, p: np.ndarray, path: str) -> None:

    mkdir(path)

    plt.imshow(p)
    plt.title(f"{n}")
    plt.savefig(osp.join(path, f"{n}.png"))
    plt.close()


def plot_heatmap(n: str, p: Tensor, path: str) -> None:

    mkdir(path)

    plt.matshow(p.detach().cpu().numpy(), cmap="seismic")
    plt.colorbar()
    plt.title(f"{n} - # P: {p.numel()}")
    plt.savefig(osp.join(path, f"{n}.png"))
    plt.close()

def plot_cam(n: str, p: np.ndarray, path: str) -> None:

    mkdir(path)

    plt.matshow(p, cmap="seismic")
    plt.colorbar()
    plt.title(f"{n}")
    plt.savefig(osp.join(path, f"{n}.png"))
    plt.close()


def multiprocess_plot(d):
    func_name = d[0]
    func_inputs = d[1]

    funcs = {
        "heatmap": plot_heatmap,
        "image": plot_image,
        "cam": plot_cam,
    }

    funcs[func_name](*func_inputs)


class Visualizer:
    """
    Grad-CAM for visualizing activations, gradients, and CAM of chosen layers using a single forward pass.

    Args:
        model (torch.nn.Module): The neural network model
        target_layers (list of str): List of layer names to hook into
        device (torch.device): Device to run computations on
    """
    def __init__(
        self, 
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        target_layers: list[str],
        class_labels: list[str],
        checkpoints_path: str,
        device: torch.device = None,
    ):
        self.dataloader = dataloader
        self.model = model.eval()
        self.class_labels = class_labels
        self.checkpoints_path = checkpoints_path
        self.device = device
        self.model.to(self.device)
        self.num_process = 8

        self.target_layers = target_layers
        self.activations = {}
        self.gradients = {}
        self._register_hooks()
        self.norm_mean = [ 0.4914, 0.4822, 0.4465 ]
        self.norm_std = [ 0.247, 0.243, 0.261 ]

    @staticmethod
    def _get_module_by_name(module, access_string):
        for name in access_string.split('.'):
            module = getattr(module, name)
        return module

    def _register_hooks(self):
        for layer_name in self.target_layers:
            module = self._get_module_by_name(self.model, layer_name)
            module.register_forward_hook(self._save_activation(layer_name))
            module.register_full_backward_hook(self._save_gradient(layer_name))

    def _save_activation(self, layer_name):
        def hook(module, input, output):
            # store activations for the whole batch
            self.activations[layer_name] = output.detach().cpu()

        return hook

    def _save_gradient(self, layer_name):
        def hook(module, grad_input, grad_output):
            # store gradients for the whole batch
            self.gradients[layer_name] = grad_output[0].detach().cpu()

        return hook

    def _normalize_map(self, cam):
        cam = cam - np.min(cam)
        max_val = np.max(cam)
        if max_val != 0:
            cam = cam / max_val
        return cam

    def _compute_cam(self, acts, grads):
        # acts, grads: (C, H, W)
        weights = np.mean(grads, axis=(1, 2))  # (C,)
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        return self._normalize_map(cam)

    def _plot_weights(self):

        save_path = osp.join(self.checkpoints_path, "plots/weights")
        mkdir(save_path)

        for n, m in self.model.named_modules():

            if isinstance(m, nn.Conv2d):
                t_p = m.weight.clone().detach().cpu().numpy()
                t_p = t_p.reshape(t_p.shape[0], t_p.shape[1], -1)
                t_p = np.moveaxis(t_p, -1, 0)
                t_p = t_p.reshape(t_p.shape[0], -1)
                t_p = np.moveaxis(t_p, -1, 0)
                sim = cosine_similarity(t_p, t_p)

            elif isinstance(m, nn.Linear):
                t_p = m.weight.clone().detach().cpu().numpy()
                t_p = np.moveaxis(t_p, -1, 0)
                sim = cosine_similarity(t_p, t_p)

            else:
                continue

            n_rows, chunk_size = sim.shape[0], 1000
            for start in range(0, n_rows, chunk_size):
                chunk = sim[start: start + chunk_size, :]  # rows start…start+999
                plt.figure()
                plt.matshow(chunk, fignum=False)  # display this block
                plt.title(f"{n}.weight Rows {start}–{min(start + chunk_size - 1, n_rows - 1)}")
                plt.colorbar()
                plt.savefig(
                    osp.join(save_path, f"{n}.weight Rows {start}–{min(start + chunk_size - 1, n_rows - 1)}.png"))
                plt.close()

    def _sample_data(self):

        found_images = []
        found_cats = []
        found_names = []
        found = {True: [], False: []}

        for bn, data in enumerate(self.dataloader):
            if (len(found[True]) == len(self.class_labels)) and (len(found[False]) == len(self.class_labels)):
                break

            imgs = data["image"].to(self.device)
            lbls = data["label"].argmax(dim=-1).detach().cpu().numpy()

            preds = self.model(imgs)
            preds = preds.argmax(dim=-1).detach().cpu().numpy()

            for i in range(len(lbls)):
                currectly_predicted = preds[i] == lbls[i]

                if lbls[i] not in found[currectly_predicted]:
                    found[currectly_predicted].append(lbls[i])
                    found_images.append(imgs[i])
                    found_cats.append(lbls[i])
                    found_names.append(f"{currectly_predicted}_{self.class_labels[lbls[i]]}")

        found_images = torch.stack(found_images, 0)

        return found_images, found_cats, found_names

    def _plot_images(self, names: list[str], images: np.ndarray) -> None:

        plot_pool = []

        save_path = osp.join(self.checkpoints_path, f"plots/samples/")

        for img_idx, img_name in enumerate(names):
            plot_pool.append(("image", (f"image_{img_name}", images[img_idx], osp.join(save_path, img_name))))

        print(f"=> Plotting {len(plot_pool)} images")
        Pool(processes=self.num_process).map(multiprocess_plot, plot_pool)

    def _plot_activations(self, names: list[str]) -> None:

        plot_pool = []

        for img_idx, img_name in enumerate(names):
            save_path = osp.join(self.checkpoints_path, f"plots/samples/{img_name}/acts")

            for layer_name, layer_acts in self.activations.items():

                if layer_name == "fc":
                    acts = layer_acts[img_idx].unsqueeze(0).repeat((10, 1))
                    plot_pool.append(("heatmap", (f"l_{layer_name}", acts, save_path)))

                else:
                    for c in range(layer_acts.shape[1]):
                        acts = layer_acts[img_idx, c]
                        plot_pool.append(("heatmap", (f"l_{layer_name}_c_{c}", acts, save_path)))

        print(f"=> Plotting {len(plot_pool)} activations")
        Pool(processes=self.num_process).map(multiprocess_plot, plot_pool)

    def _plot_gradients(self, names: list[str], class_label: str) -> None:

        plot_pool = []

        for img_idx, img_name in enumerate(names):
            save_path = osp.join(self.checkpoints_path, f"plots/samples/{img_name}/grads/{class_label}")

            for layer_name, layer_grads in self.gradients.items():

                if layer_name == "fc":
                    grads = layer_grads[img_idx].unsqueeze(0).repeat((10, 1))
                    plot_pool.append(("heatmap", (f"wrt_{class_label}_l_{layer_name}", grads, save_path)))

                else:
                    for c in range(layer_grads.shape[1]):
                        grads = layer_grads[img_idx, c]

                        plot_pool.append(("heatmap", (f"wrt_{class_label}_l_{layer_name}_c_{c}", grads, save_path)))

        print(f"=> Plotting {len(plot_pool)} {class_label} gradients")

        Pool(processes=self.num_process).map(multiprocess_plot, plot_pool)

    @staticmethod
    def _get_cams(acts: Tensor, grads: Tensor, img_size: torch.Size) -> tuple[np.ndarray, np.ndarray]:

        # Global average pooling on gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        weighted_activations = weights * acts  # (1, C, H_l, W_l)
        inp_wa = F.interpolate(weighted_activations, size=img_size, mode="bilinear", align_corners=False)
        inp_wa = F.relu(inp_wa).detach().cpu().numpy()
        inp_wa = (inp_wa - inp_wa.min()) / (inp_wa.max() - inp_wa.min() + 1e-9)

        cams = weighted_activations.sum(dim=1, keepdim=True)  # (1, 1, H_l, W_l)
        cams = F.relu(cams)  # apply ReLU
        cams = F.interpolate(cams, size=img_size, mode='bilinear', align_corners=False)
        # Normalize cam between 0 and 1

        cams = cams.squeeze().detach().cpu().numpy()
        cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-9)

        return cams, inp_wa

    def _plot_cams(self, cams: np.ndarray, names: list[str], class_label: str, layer_name: str) -> None:

        plot_pool = []

        for img_idx in range(cams.shape[0]):
            save_path = osp.join(self.checkpoints_path, f"plots/samples/{names[img_idx]}/")
            plot_pool.append(("cam", (
                f"cam_wrt_{class_label}_l_{layer_name}",
                cams[img_idx],
                save_path
            )))

        print(f"=> Plotting {len(plot_pool)} {class_label} cams")

        Pool(processes=self.num_process).map(multiprocess_plot, plot_pool)

    def _plot_channel_cams(self, cams: np.ndarray, names: list[str], class_label: str, layer_name: str) -> None:

        plot_pool = []

        for img_idx in range(cams.shape[0]):
            for c in range(cams.shape[1]):
                save_path = osp.join(self.checkpoints_path, f"plots/samples/{names[img_idx]}/cams")
                plot_pool.append(("cam", (
                    f"cam_wrt_{class_label}_l_{layer_name}_c_{c}",
                    cams[img_idx, c],
                    save_path
                )))

        print(f"=> Plotting {len(plot_pool)} {class_label} cams")

        Pool(processes=self.num_process).map(multiprocess_plot, plot_pool)

    def save_maps(self):
        """
        Compute and save activations, gradients, and CAMs for a batch of images in one forward pass.

        Args:
            img_paths (list of str): List of image file paths
            output_dir (str): Directory to save outputs
            target_class (int or None): Class index to backprop; if None uses predicted class per image
            transform: Optional torchvision transforms for preprocessing
        """
        self._plot_weights()

        images, cats, names = self._sample_data()
        images = images.to(self.device)

        # Single forward pass

        logits = self.model(images)  # (N, num_classes)

        images = unnormalizer(images.detach().cpu(), self.norm_mean, self.norm_std).moveaxis(1, -1).numpy()
        image_size = images.shape[1:3]

        self._plot_images(names, images)

        self._plot_activations(names)

        for cls_idx in range(logits.shape[1]):

            self.model.zero_grad()
            score = logits[:, cls_idx]
            score.backward(retain_graph=True, gradient=torch.ones_like(score))

            self._plot_gradients(names, self.class_labels[cls_idx])

            for layer_name in self.target_layers:
                if layer_name in ("fc",):
                    continue

                activation = self.activations[layer_name]
                gradient = self.gradients[layer_name]

                cams, ch_cams = self._get_cams(activation, gradient, image_size)

                self._plot_cams(
                    cams,
                    names,
                    self.class_labels[cls_idx],
                    layer_name
                )

                self._plot_channel_cams(
                    ch_cams,
                    names,
                    self.class_labels[cls_idx],
                    layer_name
                )
