"""Utils for dataset."""
from __future__ import division

from typing import Optional, Union

import cv2
import numpy as np
import torch


def nms(
    bbox: np.array,
    thresh: float,
    score: Optional[np.array] = None,
    limit: Optional[int] = None,
) -> np.array:
    r"""
    Suppress bounding boxes according to their IoUs and confidence scores.

    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.
    from: https://github.com/chainer/chainercv
    """
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def postprocess(
    prediction: torch.Tensor,
    num_classes: int,
    conf_thre: float = 0.7,
    nms_thre: float = 0.45,
) -> torch.Tensor:
    """
    Postprocessing of predictions.

    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.

    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.
    Returns:
        output (list of torch tensor):
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1)
        class_pred = class_pred[0]
        conf_mask = (image_pred[:, 4] * class_pred >= conf_thre).squeeze()
        image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get detections with higher confidence scores than the threshold
        ind = (image_pred[:, 5:] * image_pred[:, 4][:, None] >= conf_thre).nonzero()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (
                image_pred[ind[:, 0], :5],
                image_pred[ind[:, 0], 5 + ind[:, 1]].unsqueeze(1),
                ind[:, 1].float().unsqueeze(1),
            ),
            1,
        )
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            nms_in = detections_class.cpu().numpy()
            nms_out_index = nms(nms_in[:, :4], nms_thre, score=nms_in[:, 4] * nms_in[:, 5])
            detections_class = detections_class[nms_out_index]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))

    return output


def label2yolobox(labels: np.ndarray, info_img: tuple, maxsize: int, lrflip: bool) -> np.array:
    """
    Transform coco labels to yolo box labels.

    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag
    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    h, w, nh, nw, dx, dy = info_img
    x1 = labels[:, 1] / w
    y1 = labels[:, 2] / h
    x2 = (labels[:, 1] + labels[:, 3]) / w
    y2 = (labels[:, 2] + labels[:, 4]) / h

    labels = labels.astype("float")
    labels[:, 1] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 2] = (((y1 + y2) / 2) * nh + dy) / maxsize
    labels[:, 3] *= nw / w / maxsize
    labels[:, 4] *= nh / h / maxsize
    if lrflip:
        labels[:, 1] = 1 - labels[:, 1]
    return labels


def yolobox2label(box: list, info_img: tuple) -> list[int]:
    """
    Transform yolo box labels to yxyx box labels.

    Args:
        box (list): box data with the format of [yc, xc, w, h]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    h, w, nh, nw, dx, dy = info_img
    y1, x1, y2, x2 = box
    box_h = ((y2 - y1) / nh) * h
    box_w = ((x2 - x1) / nw) * w
    y1 = ((y1 - dy) / nh) * h
    x1 = ((x1 - dx) / nw) * w
    label = [y1, x1, y1 + box_h, x1 + box_w]
    return label


def preprocess(
    img: np.ndarray, imgsize: int, jitter: float, random_placing: bool = False
) -> Union[np.array, tuple[int, int, int, int, int, int]]:
    """
    Image preprocess for yolo input.

    Pad the shorter side of the image and resize to (imgsize, imgsize)
    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        imgsize (int): target image size after pre-processing
        jitter (float): amplitude of jitter for resizing
        random_placing (bool): if True, place the image at random position
    Returns:
        img (numpy.ndarray): input image whose shape is :math:`(C, imgsize, imgsize)`.
            Values range from 0 to 1.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
    """
    h, w, _ = img.shape

    img = img[:, :, ::-1]
    assert img is not None

    if jitter > 0:
        # add jitter
        dw = jitter * w
        dh = jitter * h
        new_ar = (w + np.random.uniform(low=-dw, high=dw)) / (h + np.random.uniform(low=-dh, high=dh))
    else:
        new_ar = w / h

    if new_ar < 1:
        nh = imgsize
        nw = nh * new_ar
    else:
        nw = imgsize
        nh = nw / new_ar
    nw, nh = int(nw), int(nh)

    if random_placing:
        dx = int(np.random.uniform(imgsize - nw))
        dy = int(np.random.uniform(imgsize - nh))
    else:
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

    img = cv2.resize(img, (nw, nh))
    sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
    sized[dy : dy + nh, dx : dx + nw, :] = img

    info_img = (h, w, nh, nw, dx, dy)
    return sized, info_img


def rand_scale(s: float) -> float:
    """
    Calculate random scaling factor.

    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


def random_distort(img: np.ndarray, hue: float, saturation: float, exposure: float) -> np.array:
    """
    Perform random distortion in the HSV color space.

    Args:
        img (numpy.ndarray): input image whose shape is :math:`(H, W, C)`.
            Values range from 0 to 255.
        hue (float): random distortion parameter.
        saturation (float): random distortion parameter.
        exposure (float): random distortion parameter.
    Returns:
        img (numpy.ndarray)
    """
    dhue = np.random.uniform(low=-hue, high=hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    img_with_hue = img[:, :, 0] + dhue

    if dhue > 0:
        img_with_hue[img_with_hue > 1.0] -= 1.0
    else:
        img_with_hue[img_with_hue < 0.0] += 1.0

    img[:, :, 0] = img_with_hue.copy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = np.asarray(img, dtype=np.float32)

    return img


def get_coco_label_names() -> tuple[np.array, list[int], tuple]:
    """
    COCO label names and correspondence between the model's class index and COCO class index.

    Returns:
        coco_label_names (tuple of str) : all the COCO label names including background class.
        coco_class_ids (list of int) : index of 80 classes that are used in 'instance' annotations
        coco_cls_colors (np.ndarray) : randomly generated color vectors used for box visualization
    """
    coco_label_names = (
        "background",  # class zero
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "street sign",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "hat",
        "backpack",
        "umbrella",
        "shoe",
        "eye glasses",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "plate",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "mirror",
        "dining table",
        "window",
        "desk",
        "toilet",
        "door",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "blender",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )
    coco_class_ids = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]

    coco_cls_colors = np.random.randint(128, 255, size=(80, 3))

    return coco_label_names, coco_class_ids, coco_cls_colors


class Augment:
    """Parameters for augmentation."""

    RANDRESIZE = False  # True
    JITTER = 0  # 0.3
    RANDOM_PLACING = False
    HUE = 0
    SATURATION = 1.5
    EXPOSURE = 1.5
    LRFLIP = False
    RANDOM_DISTORT = False
