"""Utils for yolo model."""

from __future__ import division

import json
import tempfile

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from coco_dataset import COCODataset


def bboxes_iou(bboxes_a: np.array, bboxes_b: np.array, xyxy: bool = True) -> np.array:
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        bboxes_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bboxes_b (array): An array similar to :obj:`bboxes_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
        xyxy (bool): flag, which is True, when all coordinates from bboxes
            mean points of vertexes of boxes, not xywh.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bboxes_a` and :math:`k` th bounding \
        box in :obj:`bboxes_b`.
    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        # bottom right
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def burnin_schedule(i: int, cfg: dict) -> float:
    """Get factor for sheduler."""
    burn_in = cfg["TRAIN"]["BURN_IN"]
    steps = cfg["TRAIN"]["STEPS"]

    if i < burn_in:
        factor = pow(i / burn_in, 4)
    elif i < steps[0]:
        factor = 1.0
    elif i < steps[1]:
        factor = 0.1
    else:
        factor = 0.01
    return factor


def parse_conv_block(m: torch.nn.Sequential, weights: np.ndarray, offset: int, initflag: bool) -> tuple[int, np.array]:
    """
    Init of conv layers with batchnorm.

    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    conv_model = m[0]
    bn_model = m[1]
    param_length = m[1].bias.numel()

    # batchnorm
    for pname in ["bias", "weight", "running_mean", "running_var"]:
        layerparam = getattr(bn_model, pname)

        if initflag:  # yolo initialization - scale to one, bias to zero
            if pname == "weight":
                weights = np.append(weights, np.ones(param_length))
            else:
                weights = np.append(weights, np.zeros(param_length))

        param = torch.from_numpy(weights[offset : offset + param_length]).view_as(layerparam)
        layerparam.data.copy_(param)
        offset += param_length

    param_length = conv_model.weight.numel()

    # conv
    if initflag:  # yolo initialization
        n, c, k, _ = conv_model.weight.shape
        scale = np.sqrt(2 / (k * k * c))
        weights = np.append(weights, scale * np.random.normal(size=param_length))

    param = torch.from_numpy(weights[offset : offset + param_length]).view_as(conv_model.weight)
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights


def parse_yolo_block(m: torch.nn.Sequential, weights: np.ndarray, offset: int, initflag: bool) -> tuple[int, np.array]:
    """
    Init YOLO Layer (one conv with bias).

    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    conv_model = m._modules["conv"]
    param_length = conv_model.bias.numel()

    if initflag:  # yolo initialization - bias to zero
        weights = np.append(weights, np.zeros(param_length))

    param = torch.from_numpy(weights[offset : offset + param_length]).view_as(conv_model.bias)
    conv_model.bias.data.copy_(param)
    offset += param_length

    param_length = conv_model.weight.numel()

    if initflag:  # yolo initialization
        _, c, k, _ = conv_model.weight.shape
        scale = np.sqrt(2 / (k * k * c))
        weights = np.append(weights, scale * np.random.normal(size=param_length))

    param = torch.from_numpy(weights[offset : offset + param_length]).view_as(conv_model.weight)
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights


def parse_yolo_weights(model: torch.nn.Module, weights_path: str) -> None:
    """
    Parse YOLO (darknet) pre-trained weights data onto the pytorch model.

    Args:
        model : pytorch model object
        weights_path (str): path to the YOLO (darknet) pre-trained weights file
    """
    fp = open(weights_path, "rb")

    # skip the header
    _ = np.fromfile(fp, dtype=np.int32, count=5)  # noqa: F841 not used
    # read weights
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()

    offset = 0
    initflag = False  # whole yolo weights : False, darknet weights : True

    for m in model.module_list:
        if m._get_name() == "Sequential":
            # normal conv block
            offset, weights = parse_conv_block(m, weights, offset, initflag)

        elif m._get_name() == "Resblock":
            # residual block
            for modu in m._modules["module_list"]:
                for blk in modu:
                    offset, weights = parse_conv_block(blk, weights, offset, initflag)

        elif m._get_name() == "YOLOLayer":
            # YOLO Layer (one conv with bias) Initialization
            offset, weights = parse_yolo_block(m, weights, offset, initflag)

        initflag = offset >= len(weights)  # the end of the weights file. turn the flag on


def get_ap_metrics(coco_diction: list, dataset_coco: COCODataset, ids: list[int]) -> tuple[float, float]:
    """Calculate average precision."""
    ann_type = ["segm", "bbox", "keypoints"]

    # Evaluate the Dt (detection) json comparing with the ground truth
    if len(coco_diction) > 0:
        coco_gt = dataset_coco
        # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
        _, tmp = tempfile.mkstemp()
        json.dump(coco_diction, open(tmp, "w"))
        coco_dt = coco_gt.loadRes(tmp)
        print(coco_dt)
        coco_eval = COCOeval(dataset_coco, coco_dt, ann_type[1])
        coco_eval.params.imgIds = ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[1]
    else:
        return 0, 0
