"""Target attack on detector model with patch."""
from __future__ import division

import pathlib
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from patch_generation import NPSCalculator, PatchApplier, PatchTransformer, SmoothTV
from PIL import Image
from tqdm import tqdm


class PacthDetectAttack:
    """Main class with patch attack for detector."""

    def __init__(
        self,
        patch_size: tuple[int],
        target_cls: int,
        scale_factor: float = 0.22,
        delta: float = 0.1,
        gamma: float = 0.1,
        augment: bool = True,
        lr: float = 0.02,
        slide_patch: Optional[float] = None,
        highlight: float = 0.2,
        use_perspective: bool = False,
        use_tps: bool = False,
        patch: Optional[np.array] = None,
        device: str = "cpu",
        nonprintability_file: Optional[list] = None,
    ) -> None:
        """Init target patch attack."""
        self.patch_size = patch_size
        self.delta = delta
        self.gamma = gamma
        self.augment = augment
        self.lr = lr
        self.device = device
        self.target_cls = target_cls
        self.scale_factor = scale_factor
        self.slide_patch = slide_patch
        self.highlight = highlight
        self.use_tps = use_tps
        self.use_perspective = use_perspective

        if patch is None:
            self.adv_patch_cpu = torch.rand(3, self.patch_size[0], self.patch_size[1])  # torch.rand(3, 250, 150)
        else:
            image = cv2.resize(patch, self.patch_size)
            self.adv_patch_cpu = torch.tensor(image).transpose(0, 2) / 255.0

        self.adv_patch_cpu.requires_grad_(True)

        self.optimizer = torch.optim.Adam([self.adv_patch_cpu], lr=self.lr, amsgrad=True)

        def scheduler_factory(x):
            return optim.lr_scheduler.ReduceLROnPlateau(x, "min", patience=50)

        self.scheduler = scheduler_factory(self.optimizer)

        self.patch_applier = PatchApplier().to(self.device)
        self.patch_transformer = PatchTransformer(augment=self.augment, slide_patch=self.slide_patch, highlight=self.highlight).to(
            self.device
        )

        self.arguments = {}
        self.arguments["iteration"] = 0

        self.stv = SmoothTV().to(self.device)
        self.nps_calculator = NPSCalculator(nonprintability_file, (self.patch_size[0], self.patch_size[1])).to(self.device)

    def attack_step(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        start_iter: int,
        special_loss: bool = False,
    ) -> tuple[list, float]:
        """Attack step for training."""
        total_loss = 0
        list_loss = []

        torch.cuda.empty_cache()

        for iteration, (images, targets, _, _, _, _) in tqdm(enumerate(dataloader, start_iter)):
            if targets[targets != 0].shape[0] == 0:
                continue

            rcnn_targets = []

            twice_target = {"rcnn": None, "yolo": None}

            twice_target["yolo"] = targets.to(self.device)

            twice_target["rcnn"] = rcnn_targets

            if any(len(target) < 1 for target in targets):
                continue

            iteration = iteration + 1
            self.arguments["iteration"] = iteration

            imgs = images.to(self.device)
            _, _, height, width = imgs.shape

            lab_batch_targets = []

            for j in range(targets.shape[0]):
                con = targets[j][:, 0] == self.target_cls
                lab_batch_targets.append(targets[j][con])
            lab_batch_targets = torch.stack(lab_batch_targets)

            if lab_batch_targets.nelement() == 0:
                continue

            adv_batch = self.patch_transformer(
                adv_patch=self.adv_patch_cpu.to(self.device),
                targets=lab_batch_targets.to(self.device),
                height=height,
                width=width,
                scale_factor=self.scale_factor,
                cls_label=self.target_cls,
                use_perspective=self.use_perspective,
                use_tps=self.use_tps,
                device=self.device,
            )

            adv_batch = adv_batch.mul_(255)

            imgs = self.patch_applier(imgs * 255, adv_batch.to(self.device))
            imgs = imgs / 255.0

            _, _, _, mean_objectness_dark = model.get_losses(
                batched_inputs=imgs,
                targets=twice_target,
                target_cls=self.target_cls,
                special_loss=special_loss,
                device=self.device,
            )

            losses = (
                mean_objectness_dark
                + self.gamma * self.stv(self.adv_patch_cpu.to(self.device))
                + self.delta * self.nps_calculator(self.adv_patch_cpu.to(self.device))
            )
            print(
                "losses",
                losses,
                mean_objectness_dark,
                self.gamma * self.stv(self.adv_patch_cpu.to(self.device)),
                self.delta * self.nps_calculator(self.adv_patch_cpu.to(self.device)),
            )
            list_loss.append(losses)
            total_loss += losses

            losses.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.adv_patch_cpu.data.clamp_(0, 1)
        return list_loss, total_loss / len(dataloader)

    def attack_train(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        num_iter: int,
        exp_name: str = "patch_detector_result",
        output_path: str = "test_patch",
        start_iter: int = 0,
        special_loss: bool = False,
    ) -> list[float]:
        """Attack loop for training patch."""
        torch.cuda.empty_cache()
        model.to(self.device)

        list_loss = []
        mean_epoch_loss = []

        total_out_path = pathlib.Path(output_path) / exp_name

        pathlib.Path(total_out_path).mkdir(parents=True, exist_ok=True)

        for k in range(num_iter):
            epoch_losses, ep_loss = self.attack_step(model, loader, start_iter, special_loss)
            self.scheduler.step(ep_loss)
            mean_epoch_loss.append(ep_loss)
            list_loss += epoch_losses
            print(f"{k} epoch_losses = {epoch_losses}")

            if k % 4 == 0:
                path_name_output = pathlib.Path.joinpath(total_out_path, f"{k}_{exp_name}_bgr.jpg")
                w = self.adv_patch_cpu.clone().cpu().detach().numpy().transpose(1, 2, 0) * 255
                cv2.imwrite(str(path_name_output), cv2.cvtColor(w, cv2.COLOR_RGB2BGR))

                path_name_output = pathlib.Path.joinpath(total_out_path, f"{k}_{exp_name}.jpg")
                w = self.adv_patch_cpu.clone().cpu().detach().numpy().transpose(1, 2, 0) * 255
                cv2.imwrite(str(path_name_output), w)
            torch.cuda.empty_cache()

        plt.plot(list(range(len(list_loss))), [e.cpu().detach().numpy() for e in list_loss])
        plt.savefig(str(pathlib.Path.joinpath(total_out_path, f"loss_func_nms_{exp_name}.png")))

        path_name_output = pathlib.Path.joinpath(total_out_path, f"{k}_{exp_name}_bgr.jpg")
        w = self.adv_patch_cpu.clone().cpu().detach().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(str(path_name_output), cv2.cvtColor(w, cv2.COLOR_RGB2BGR))

        path_name_output = pathlib.Path.joinpath(total_out_path, f"{k}_{exp_name}.jpg")
        w = self.adv_patch_cpu.clone().cpu().detach().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(str(path_name_output), w)

        return list_loss

    def attack_apply(self, dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool) -> tuple[np.array, list[int]]:
        """Applay patch for dataset."""
        loader = dataset.get_custom_loader(batch_size, shuffle)

        adversarial_images = []
        id_s = []

        for data in loader:
            for i in range(data[0].shape[0]):
                images = data[0][i]
                targets = data[1][i]
                id_ = data[3][i]

                imgs = images.to(self.device)
                _, height, width = imgs.shape

                lab_batch_targets = []
                con = torch.logical_and((targets[:, 0] == self.target_cls), (targets[:, 1:].sum(1) != 0))
                lab_batch_targets.append(targets[con])

                lab_batch_targets = torch.stack(lab_batch_targets)

                if lab_batch_targets.nelement() == 0:
                    continue

                adv_batch = self.patch_transformer(
                    self.adv_patch_cpu.to(self.device),
                    lab_batch_targets.to(self.device),
                    height,
                    width,
                    scale_factor=self.scale_factor,
                    cls_label=self.target_cls,
                    device=self.device,
                )

                adv_batch = adv_batch.mul_(255)

                adv_img = self.patch_applier(imgs.to(self.device) * 255, adv_batch.to(self.device))
                img_cp = adv_img[0].cpu().detach().numpy().transpose(1, 2, 0)
                h, w, nh, nw, dx, dy = data[2][i]
                adv_img = Image.fromarray((img_cp[dy : nh + dy, dx : nw + dx]).astype("uint8"), mode="RGB").resize((w, h))
                # adv_img = torch.tensor(np.array(adv_img).transpose(0,1,2) / 255)

                adversarial_images.append(adv_img)
                id_s.append(id_)
        # adversarial_images = np.stack(adversarial_images)

        return adversarial_images, id_s
