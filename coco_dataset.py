"""COCO dataset module."""
from __future__ import division

import pathlib
from typing import Optional, Union

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

import dataset_utils as utils


class COCODataset(Dataset):
    """COCO dataset class."""

    def __init__(
        self,
        model_type: str = "YOLO",
        data_dir: str = "COCO",
        json_file: str = "instances_train2017.json",
        name: str = "train2017",
        img_size: int = 640,
        augmentation: Optional[dict] = None,
        min_size: int = 1,
        debug: bool = False,
        zeros: bool = True,
        cust: bool = False,
        return_path: bool = False,
    ) -> None:
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.

        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
            zeros (bool): if True, will be used added zeros in name before number
                of image.
            cust (bool): if True, will be used images names form file_name variable,
                not from _id.
            return_path (bool): if True, will return img_file path.

        """
        self.return_path = return_path
        self.data_dir = data_dir
        self.json_file = json_file
        self.model_type = model_type
        self.cust = cust
        self.coco = COCO(pathlib.Path(self.data_dir) / "annotations" / self.json_file)
        self.ids = self.coco.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.zeros = zeros

        self.lrflip = augmentation["LRFLIP"] if augmentation else utils.Augment.LRFLIP
        self.jitter = augmentation["JITTER"] if augmentation else utils.Augment.JITTER
        self.random_placing = augmentation["RANDOM_PLACING"] if augmentation else utils.Augment.RANDOM_PLACING
        self.hue = augmentation["HUE"] if augmentation else utils.Augment.HUE
        self.saturation = augmentation["SATURATION"] if augmentation else utils.Augment.SATURATION
        self.exposure = augmentation["EXPOSURE"] if augmentation else utils.Augment.EXPOSURE
        self.random_distort = augmentation["RANDOM_DISTORT"] if augmentation else utils.Augment.RANDOM_DISTORT

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.ids)

    def __getitem__(
        self, index: int
    ) -> Union[
        tuple[np.array, torch.Tensor, tuple[int, int, int, int, int, int], int, Union[list[int], np.array], np.array, pathlib.Path],
        tuple[np.array, torch.Tensor, tuple[int, int, int, int, int, int], int, Union[list[int], np.array], np.array],
    ]:
        """
        One image / label pair for the given index is picked up \
        and pre-processed.

        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
            pred_labels (torch.Tensor): labels.
            img_copy (numpy.array): image before pre-processe.
            img_file (pathlib.Path): path to image.
        """
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        lrflip = False
        if (np.random.rand() > 0.5) and (self.lrflip is True):
            lrflip = True

        # load image and preprocess
        if not self.zeros:
            if self.cust:
                img_file = pathlib.Path(self.data_dir) / self.name / self.coco.imgs[id_]["file_name"]
            else:
                img_file = pathlib.Path(self.data_dir) / self.name / f"{str(id_)}.jpg"
        else:
            img_file = pathlib.Path(self.data_dir, self.name) / "{:012}.jpg".format(id_)

        img = cv2.imread(str(img_file))

        if self.json_file == "instances_val5k.json" and img is None:
            if not self.zeros:
                img_file = pathlib.Path(self.data_dir) / "train2017" / f"{str(id_)}.jpg"
            else:
                img_file = pathlib.Path(self.data_dir) / "train2017" / "{:012}.jpg".format(id_)
            img = cv2.imread(str(img_file))

        assert img is not None
        img_copy = img.copy()
        img, info_img = utils.preprocess(img, self.img_size, jitter=self.jitter, random_placing=self.random_placing)

        if self.random_distort:
            img = utils.random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255.0, (2, 0, 1))

        if lrflip:
            img = np.flip(img, axis=2).copy()

        # load labels
        labels = []
        for anno in annotations:
            if anno["bbox"][2] > self.min_size and anno["bbox"][3] > self.min_size:
                labels.append([])
                labels[-1].append(self.class_ids.index(anno["category_id"]))
                labels[-1].extend(anno["bbox"])

        padded_labels = np.zeros((self.max_labels, 5))
        pred_labels = [-42]
        if len(labels) > 0:
            labels = np.stack(labels)

            if "YOLO" in self.model_type:
                pred_labels = labels.copy()
                labels = utils.label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[: self.max_labels]] = labels[: self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        if self.return_path:
            return img, padded_labels, info_img, id_, pred_labels, img_copy, img_file
        return img, padded_labels, info_img, id_, pred_labels, img_copy

    def get_revert_class_ids(self) -> dict[int, int]:
        """Get reverted mapping class and ids."""
        coco_class_ids = sorted(self.coco.getCatIds())
        revert_coco_class_ids = {}
        for i, e in enumerate(coco_class_ids):
            revert_coco_class_ids[e] = i
        return revert_coco_class_ids

    @staticmethod
    def collate_fn(
        data: Union[
            tuple[np.array, torch.Tensor, tuple[int, int, int, int, int, int], int, Union[list[int], np.array], np.array, pathlib.Path],
            tuple[np.array, torch.Tensor, tuple[int, int, int, int, int, int], int, Union[list[int], np.array], np.array],
        ]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[torch.Tensor], list[torch.Tensor]]:
        """Collate function needed for dataloader."""
        image_batch = []
        target_batch = []
        info_batch = []
        id_batch = []
        pred_label_batch = []
        img_orig_size_batch = []

        if len(data) > 6:
            images, targets, info_img, id_, pred_labels, img_copy = zip(*data[:6])
        else:
            images, targets, info_img, id_, pred_labels, img_copy = zip(*data)

        for i in range(len(data)):
            image_batch.append(images[i])
            target_batch.append(targets[i])
            info_batch.append(torch.tensor(info_img[i]))
            id_batch.append(id_[i])
            pred_label_batch.append(torch.tensor(pred_labels[i]))
            img_orig_size_batch.append(img_copy[i])

        image_batch = np.stack(image_batch)
        target_batch = torch.stack(target_batch)
        info_batch = torch.stack(info_batch)

        return (
            torch.tensor(image_batch),
            target_batch,
            info_batch,
            id_batch,
            pred_label_batch,
            img_orig_size_batch,
        )

    def get_custom_loader(
        self,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = True,
        collate_fn: callable = collate_fn,
    ) -> torch.utils.data.DataLoader:
        """Get dataloader for coco dataset."""
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
        return dataloader
