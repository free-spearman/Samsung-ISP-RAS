"""Generation and transformation of adversarial patch."""
import math
import random
from typing import Optional, Union

import numpy as np
import thinplate as tps
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_perspective
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size: int = 3, stride: int = 1, padding: int = 0, same: bool = False) -> None:
        """Init medianpool2d layer."""
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        """Calculate padding."""
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def set_bn_eval(m: torch.nn.Module) -> None:
    """Set eval mode for batchnorm layers."""
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Total variation loss
    """

    def __init__(self) -> None:
        """Init total variation model."""
        super().__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        tvcomp1 = adv_patch[:, :, 1:] - adv_patch[:, :, :-1]
        tvcomp1 = (tvcomp1 * tvcomp1 + 0.01).sqrt().sum()
        tvcomp2 = adv_patch[:, 1:, :] - adv_patch[:, :-1, :]
        tvcomp2 = (tvcomp2 * tvcomp2 + 0.01).sqrt().sum()
        tv = tvcomp1 + tvcomp2
        return tv


class SmoothTV(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Smooth total variation loss
    """

    def __init__(self) -> None:
        """Init smooth total variation model."""
        super().__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        tvcomp1 = adv_patch[:, :, 1:] - adv_patch[:, :, :-1]
        tvcomp1 = (tvcomp1 * tvcomp1).sum()
        tvcomp2 = adv_patch[:, 1:, :] - adv_patch[:, :-1, :]
        tvcomp2 = (tvcomp2 * tvcomp2).sum()
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches.

    Generate patches with different augmentations
    """

    def __init__(self, augment: bool = True, slide_patch: Optional[float] = None, highlight: float = 0.1) -> None:
        """Init patch transformer for patch."""
        super().__init__()
        self.augment = augment
        self.min_contrast = 0.8 if augment else 1
        self.max_contrast = 1.2 if augment else 1
        self.min_brightness = -0.1 if augment else 0
        self.max_brightness = 0.1 if augment else 0
        self.noise_factor = 0.0  # 0.0
        self.minangle = -5 / 180 * math.pi if augment else 0
        self.maxangle = 5 / 180 * math.pi if augment else 0
        self.medianpooler = MedianPool2d(7, same=True)
        self.distortion_max = 0.1 if augment else 0
        self.sliding = -0.05 if augment else 0  # вроде его надо поменять для смещения
        self.slide_patch = slide_patch
        self.highlight = highlight

    def get_tps_thetas(self, num_images: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get angle theta for tps."""
        c_src = np.array(
            [
                [0.0, 0.0],
                [1.0, 0],
                [1, 1],
                [0, 1],
                [0.2, 0.3],
                [0.6, 0.7],
            ]
        )

        theta_list = []
        dst_list = []
        for _ in range(num_images):
            c_dst = c_src + np.random.uniform(-1, 1, (6, 2)) / 20

            theta = tps.tps_theta_from_points(c_src, c_dst)
            theta_list.append(torch.from_numpy(theta).unsqueeze(0))
            dst_list.append(torch.from_numpy(c_dst).unsqueeze(0))

        theta = torch.cat(theta_list, dim=0).float()
        dst = torch.cat(dst_list, dim=0).float()
        return theta, dst

    def get_perspective_params(self, width: int, height: int, distortion_scale: float) -> tuple[list[float], list[int]]:
        """Get points for perspective transformation."""
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (
            random.randint(0, int(distortion_scale * half_width)),
            random.randint(0, int(distortion_scale * half_height)),
        )
        topright = (
            random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
            random.randint(0, int(distortion_scale * half_height)),
        )
        botright = (
            random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
            random.randint(height - int(distortion_scale * half_height) - 1, height - 1),
        )
        botleft = (
            random.randint(0, int(distortion_scale * half_width)),
            random.randint(height - int(distortion_scale * half_height) - 1, height - 1),
        )
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def forward(
        self,
        adv_patch: torch.Tensor,
        targets: torch.Tensor,
        height: int,
        width: int,
        do_rotate: bool = True,
        scale_factor: float = 0.25,
        cls_label: int = 1,
        use_perspective: bool = False,
        use_tps: bool = False,
        device: str = "cpu",
        return_mask: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        patch_size = adv_patch.size(-1)

        # Determine size of padding
        pad_width = (width - adv_patch.size(-1)) / 2
        pad_height = (height - adv_patch.size(-2)) / 2

        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(targets[:, :, 1:].size(0), targets[:, :, 1:].size(1), -1, -1, -1)
        batch_size = torch.Size((targets[:, :, 1:].size(0), targets[:, :, 1:].size(1)))
        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        # Create random brightness tensor
        brightness = torch.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

        # Create random noise tensor
        noise = torch.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch.to(device) * contrast.to(device) + brightness.to(device) + noise.to(device)
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        if int(np.random.rand() * 100) % 5 == 1:
            # Adding highlight effects
            for k, _ in enumerate(adv_batch):
                determine = int(np.random.rand() * 100) % 2
                if determine == 1:
                    b = np.random.normal(0.05, self.highlight, (adv_batch.shape[3], adv_batch.shape[4]))
                    bb = np.repeat(np.expand_dims(b, 0), 3, 0)
                    if int(np.random.rand() * 100) % 2 == 1:
                        bb = np.sort(np.sort(bb, 1), 0)
                    else:
                        bb = np.sort(np.sort(bb, 0), 1)
                    if int(np.random.rand() * 100) % 2 == 1:
                        bb = bb[::-1]
                    if int(np.random.rand() * 100) % 2 == 1:
                        bb = bb[:, ::-1]
                    bb = np.expand_dims(bb, 0)
                    bb = torch.tensor(bb.copy()).to(device)
                    adv_batch[k] = adv_batch[k] + bb

        # perspective transformations
        dis_scale = targets[:, :, 1:].size(0) * targets[:, :, 1:].size(1)
        distortion = torch.empty(dis_scale).uniform_(0, self.distortion_max)

        adv_height = adv_batch.size(-1)
        adv_width = adv_batch.size(-2)

        # tps transformation
        if self.augment and use_tps:
            theta, dst = self.get_tps_thetas(dis_scale)
            img = adv_batch.clone()
            img = img.view(-1, 3, adv_width, adv_height)

            grid = tps.torch.tps_grid(
                theta.to(device),
                dst.to(device),
                (img.size(0), 1, adv_width, adv_height),
            )
            adv_batch = F.grid_sample(img, grid.to(device), padding_mode="border")
            adv_batch = adv_batch.view(
                targets[:, :, 1:].size(0),
                targets[:, :, 1:].size(1),
                3,
                adv_width,
                adv_height,
            )

        # perpective transformations with random distortion scales
        if use_perspective:
            start_end = [
                torch.tensor(
                    self.get_perspective_params(adv_width, adv_height, x),
                    dtype=torch.float,
                ).unsqueeze(0)
                for x in distortion
            ]
            start_end = torch.cat(start_end, 0)
            start_points = start_end[:, 0, :, :].squeeze()
            end_points = start_end[:, 1, :, :].squeeze()

            if dis_scale == 1:
                start_points = start_points.unsqueeze(0)
                end_points = end_points.unsqueeze(0)
            try:
                prespective_trans = get_perspective_transform(start_points, end_points)
                img = adv_batch.clone()
                img = img.view(-1, 3, adv_width, adv_height)
                adv_batch = warp_perspective(img, prespective_trans.to(device), dsize=(adv_width, adv_height))
                adv_batch = adv_batch.view(
                    targets[:, :, 1:].size(0),
                    targets[:, :, 1:].size(1),
                    3,
                    adv_width,
                    adv_height,
                )

            except Exception:
                print("Exception: problem with perspective transformation.")

        cls_ids = torch.narrow(targets[:, :, :1], 2, 0, 1)
        cls_mask = 1.0 * (cls_ids.expand(-1, -1, 3) != cls_label)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.FloatTensor(cls_mask.size()).fill_(1).to(device) - cls_mask.float().to(device)

        if self.slide_patch is None:
            mypad = nn.ConstantPad2d(
                (
                    int(pad_width + 0.5),
                    int(pad_width),
                    int(pad_height + 0.5),
                    int(pad_height),
                ),
                0,
            )
        else:
            mypad = nn.ConstantPad2d(
                (
                    int(pad_width + 0.5),
                    int(pad_width),
                    int(pad_height + 0.5) + self.slide_patch,
                    int(pad_height) - self.slide_patch,
                ),
                0,
            )
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = targets[:, :, 1:].size(0) * targets[:, :, 1:].size(1)
        if do_rotate:
            angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.FloatTensor(anglesize).fill_(0)
        angle = angle.to(device)

        # Resizes and rotates
        batch_size = torch.Size((targets[:, :, 1:].size(0), targets[:, :, 1:].size(1)))

        lab_batch_scaled = torch.zeros(targets[:, :, 1:].size())
        lab_batch_scaled[:, :, 0] = targets[:, :, 1] * width
        lab_batch_scaled[:, :, 1] = targets[:, :, 2] * height
        lab_batch_scaled[:, :, 2] = targets[:, :, 3] * width
        lab_batch_scaled[:, :, 3] = targets[:, :, 4] * height

        # roughly estimate the size and compute the scale
        target_size = scale_factor * torch.sqrt((lab_batch_scaled[:, :, 2]) ** 2 + (lab_batch_scaled[:, :, 3]) ** 2)

        target_x = targets[:, :, 1].view(np.prod(batch_size))  # (batch_size, num_objects)
        target_y = targets[:, :, 2].view(np.prod(batch_size))

        # shift a bit from the center
        targetoff_y = targets[:, :, 4].view(np.prod(batch_size))

        off_y = torch.FloatTensor(targetoff_y.size()).uniform_(self.sliding, 0)
        off_y = targetoff_y.to(device) * off_y.to(device)
        target_y = target_y + off_y

        scale = target_size / patch_size
        scale = scale.view(anglesize)
        scale = scale.to(device)

        s = adv_batch.size()

        adv_batch = adv_batch.view(-1, 3, height, width)
        msk_batch = msk_batch.view_as(adv_batch)

        tx = ((-target_x + 0.5) * 2).to(device)
        ty = ((-target_y + 0.5) * 2).to(device)
        sin = torch.sin(angle).to(device)
        cos = torch.cos(angle).to(device)

        # Theta = rotation,rescale matrix
        theta = torch.FloatTensor(anglesize, 2, 3).fill_(0).to(device)

        if not self.augment:
            return adv_batch.view(s[0], s[1], s[2], s[3], s[4])

        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
        theta = theta.to(device)

        grid = F.affine_grid(theta, adv_batch.shape, align_corners=True)
        adv_batch = F.grid_sample(adv_batch, grid)
        msk_batch = F.grid_sample(msk_batch, grid)

        adv_batch = adv_batch.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0], s[1], s[2], s[3], s[4])
        adv_batch = adv_batch * msk_batch

        if return_mask:
            return adv_batch, msk_batch
        return adv_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """

    def __init__(self) -> None:
        """Init patch applier."""
        super().__init__()

    def forward(self, img_batch: torch.Tensor, adv_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        advs = torch.unbind(adv_batch, 1)

        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file: list[float], patch_side: tuple[int, int]) -> None:
        """Init NPS calculator."""
        super().__init__()
        self.printability_array = nn.Parameter(
            self.get_printability_array(printability_file, patch_side),
            requires_grad=False,
        )

    def forward(self, adv_patch: torch.Tensor) -> float:
        """Forward pass."""
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = adv_patch - self.printability_array + 0.000001
        color_dist = color_dist**2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_list: list[float], side: tuple[int, int]) -> torch.Tensor:
        """Get array with printability colors."""
        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet

            printability_imgs.append(np.full((side[0], side[1]), blue))
            printability_imgs.append(np.full((side[0], side[1]), green))
            printability_imgs.append(np.full((side[0], side[1]), red))

            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa
