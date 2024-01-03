"""Yolo v3 architecture."""
from collections import defaultdict
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.autograd import Variable

import coco_dataset as coco_dataset
import dataset_utils as dataset_utils
import yolo3_utils as yolo3_utils


class YOLOLayer(nn.Module):
    """Detection layer corresponding to yolo_layer.c of darknet."""

    def __init__(self, config_model: dict, layer_no: int, in_ch: int, ignore_thre: float = 0.7) -> None:
        """
        Init Base yolo layer.

        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """
        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8]  # fixed
        self.anchors = config_model["ANCHORS"]
        self.anch_mask = config_model["ANCH_MASK"][layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = config_model["N_CLASSES"]
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)
        self.stride = strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=self.n_anchors * (self.n_classes + 5),
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self, xin: torch.Tensor, labels: Optional[torch.Tensor] = None, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(xin).to(device)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls

        x_shift = torch.tensor(np.broadcast_to(np.arange(fsize, dtype=np.float32), output.shape[:4])).to(device)
        y_shift = torch.tensor(np.broadcast_to(np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4])).to(device)

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = torch.tensor(
            np.broadcast_to(
                np.reshape(masked_anchors[:, 0], (1, self.n_anchors, 1, 1)),
                output.shape[:4],
            )
        ).to(device)
        h_anchors = torch.tensor(
            np.broadcast_to(
                np.reshape(masked_anchors[:, 1], (1, self.n_anchors, 1, 1)),
                output.shape[:4],
            )
        ).to(device)

        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        pred_last_cp = pred.clone()
        pred_last_cp[..., :4] *= self.stride
        pred_last_cp = pred_last_cp.reshape(batchsize, -1, n_ch).data

        if labels is None:  # not training
            pred[..., :4] *= self.stride
            return pred.reshape(batchsize, -1, n_ch).data

        pred = pred[..., :4].data

        # target assignment

        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(device)

        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(device)

        labels = labels.data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros((n, 4))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = yolo3_utils.bboxes_iou(truth_box, self.ref_anchors)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = (best_n_all == self.anch_mask[0]) | (best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2])

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            truth_box = truth_box.to(device)
            pred = pred.to(device)

            pred_ious = yolo3_utils.bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.ignore_thre
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = 1 - pred_best_iou.int()

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 0].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

        # loss calculation

        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        output[..., 2:4] *= tgt_scale

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale

        bceloss = nn.BCELoss(weight=tgt_scale * tgt_scale, size_average=False)  # weighted BCEloss

        loss_xy = bceloss(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
        loss_obj = self.bce_loss(output[..., 4], target[..., 4])
        loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])
        loss_l2 = self.l2_loss(output, target)

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return (
            loss,
            loss_xy,
            loss_wh,
            loss_obj,
            loss_cls,
            loss_l2,
            output,
        )  # output[..., 4]


def add_conv(in_ch: int, out_ch: int, ksize: int, stride: int) -> torch.nn.Sequential:
    """
    Add a conv2d / batchnorm / leaky ReLU block.

    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            bias=False,
        ),
    )
    stage.add_module("batch_norm", nn.BatchNorm2d(out_ch))
    stage.add_module("leaky", nn.LeakyReLU(0.1))
    return stage


class Resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.

    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch: int, nblocks=1, shortcut=True) -> None:
        """Init residual block."""
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch // 2, 1, 1))
            resblock_one.append(add_conv(ch // 2, ch, 3, 1))
            self.module_list.append(resblock_one)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


def create_yolov3_modules(config_model: dict, ignore_thre: float) -> torch.nn.ModuleList:
    """
    Build yolov3 layer modules.

    Args:
        config_model (dict): model configuration.
            See YOLOLayer class for details.
        ignore_thre (float): used in YOLOLayer.
    Returns:
        mlist (ModuleList): YOLOv3 module list.
    """
    # DarkNet53
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
    mlist.append(Resblock(ch=64))
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
    mlist.append(Resblock(ch=128, nblocks=2))
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
    mlist.append(Resblock(ch=256, nblocks=8))  # shortcut 1 from here
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
    mlist.append(Resblock(ch=512, nblocks=8))  # shortcut 2 from here
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
    mlist.append(Resblock(ch=1024, nblocks=4))

    # YOLOv3
    mlist.append(Resblock(ch=1024, nblocks=2, shortcut=False))
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
    # 1st yolo branch
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
    mlist.append(YOLOLayer(config_model, layer_no=0, in_ch=1024, ignore_thre=ignore_thre))

    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    mlist.append(nn.Upsample(scale_factor=2, mode="nearest"))
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    mlist.append(Resblock(ch=512, nblocks=1, shortcut=False))
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    # 2nd yolo branch
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    mlist.append(YOLOLayer(config_model, layer_no=1, in_ch=512, ignore_thre=ignore_thre))

    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
    mlist.append(nn.Upsample(scale_factor=2, mode="nearest"))
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
    mlist.append(Resblock(ch=256, nblocks=2, shortcut=False))
    mlist.append(YOLOLayer(config_model, layer_no=2, in_ch=256, ignore_thre=ignore_thre))

    return mlist


class YOLOv3(nn.Module):
    """
    YOLOv3 model module.

    The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """

    def __init__(self, config_model: dict, ignore_thre: float = 0.7) -> None:
        """
        Init of YOLOv3 class.

        Args:
            config_model (dict): used in YOLOLayer.
            ignore_thre (float): used in YOLOLayer.
        """
        super().__init__()

        if config_model["TYPE"] == "YOLOv3":
            self.module_list = create_yolov3_modules(config_model, ignore_thre)
        else:
            raise Exception("Model name {} is not available".format(config_model["TYPE"]))

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, device: str = "cpu"
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list]]:
        """
        Forward path of YOLOv3.

        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """
        train = targets is not None
        output = []
        pp_objectness = []
        self.loss_dict = defaultdict(float)
        route_layers = []
        for i, module in enumerate(self.module_list):
            # yolo layers
            if i in [14, 22, 28]:
                if train:
                    x, *loss_dict, pp = module(x, targets, device)
                    for name, loss in zip(["xy", "wh", "conf", "cls", "l2"], loss_dict):
                        self.loss_dict[name] += loss
                    pp_objectness.append(pp)
                else:
                    x = module(x)
                output.append(x)
            else:
                x = module(x)

            # route layers
            if i in [6, 8, 12, 20]:
                route_layers.append(x)
            if i == 14:
                x = route_layers[2]
            if i == 22:  # yolo 2nd
                x = route_layers[3]
            if i == 16:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 24:
                x = torch.cat((x, route_layers[0]), 1)
        if train:
            return sum(output), pp_objectness
        else:
            return torch.cat(output, 1)


class YOLOV3Multi(nn.Module):
    """Yolo model with functions."""

    def __init__(self, weights: Optional[str] = None) -> None:
        """Init model."""
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        with open("yolo3_config.yaml", "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            self.cfg = cfg
        model_dark = YOLOv3(cfg["MODEL"], False)
        if weights:
            yolo3_utils.parse_yolo_weights(model_dark, weights)

        self.model = model_dark

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, device: str = "cpu") -> torch.Tensor:
        """Forward pass."""
        out = self.model.forward(x, targets, device)
        self.loss_dict = self.model.loss_dict
        return out

    def train_function(
        self,
        loader: torch.utils.data.DataLoader,
        epochs: int,
        batch_size: int,
    ) -> list:
        """Train model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.train().to(device)
        dtype = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor

        params_dict = dict(self.model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if "conv.weight" in key:
                params += [
                    {
                        "params": value,
                        "weight_decay": self.cfg["TRAIN"]["DECAY"] * batch_size * self.cfg["TRAIN"]["SUBDIVISION"],
                    }
                ]
            else:
                params += [{"params": value, "weight_decay": 0.0}]

        optimizer = optim.SGD(
            params,
            lr=self.cfg["TRAIN"]["LR"] / batch_size / self.cfg["TRAIN"]["SUBDIVISION"],
            momentum=self.cfg["TRAIN"]["MOMENTUM"],
            dampening=0,
            weight_decay=self.cfg["TRAIN"]["DECAY"] * batch_size * self.cfg["TRAIN"]["SUBDIVISION"],
        )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: yolo3_utils.burnin_schedule(x, self.cfg))

        losses = []
        pp = 0
        for _ in range(epochs):
            for imgs_info in loader:
                pp+=1
                if pp > 5:
                    break
                optimizer.zero_grad()
                imgs = Variable(imgs_info[0].type(dtype))
                targets = Variable(imgs_info[1].type(dtype), requires_grad=False)
                loss, _ = self.model(imgs, targets, device)

                losses.append(self.model.loss_dict)

                loss.backward()

                optimizer.step()
                scheduler.step()

        return losses

    def predict(
        self,
        images: np.array,
        confidence: float = 0.9,
        num_classes: int = 8,
        need_eval: bool = False,
        id_images: Optional[List] = None,
    ) -> Union[list, tuple[list, list]]:
        """Get model predictions on one image."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval().to(device)

        coco_class_names, coco_class_ids, _ = dataset_utils.get_coco_label_names()
        colors = np.random.uniform(0, 255, size=(len(coco_class_names), 3))
        font = cv2.FONT_HERSHEY_PLAIN

        process_images = []
        info_images = []

        for image in images:
            image = image.astype("int32")
            sized, info_img = dataset_utils.preprocess(image, 640, 0, random_placing=False)
            info_images.append(info_img)
            im_test = sized.transpose(2, 0, 1) / 255.0
            im_test = im_test.astype("float32")
            process_images.append(im_test)
        process_images = np.stack(process_images)

        with torch.no_grad():
            outputs = self.model(torch.tensor(process_images).to(device).float().clone())
            post_outputs = dataset_utils.postprocess(outputs, num_classes, conf_thre=confidence)

        batch_result_images = []
        batch_result_data_dict = []

        for k in range(len(post_outputs)):
            print(f"k = {k}")
            image = images[k]
            if not (post_outputs[k] is None):
                bboxes = []
                classes = []
                confidences = []
                detect_predict = []
                data_dict = []

                for x1, y1, x2, y2, conf, cls_conf, cls_pred in post_outputs[k]:
                    cls_id = coco_class_ids[int(cls_pred)]
                    print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
                    print("\t+ Label: %s, Conf: %.5f" % (coco_class_names[cls_id], cls_conf.item()))
                    box = dataset_utils.yolobox2label([y1, x1, y2, x2], info_images[k])
                    bboxes.append([e.cpu() for e in box])

                    classes.append(cls_id)
                    detect_predict.append(cls_pred)
                    confidences.append(cls_conf)

                    if need_eval:
                        label = coco_class_ids[int(cls_pred)]
                        box = dataset_utils.yolobox2label((float(y1), float(x1), float(y2), float(x2)), info_img)
                        bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                        score = float(conf.data.item() * cls_conf.data.item())  # object score * class score
                        results_info = {
                            "image_id": id_images[k],
                            "category_id": label,
                            "bbox": bbox,
                            "score": score,
                            "segmentation": [],
                        }  # COCO json format
                        data_dict.append(results_info)

                batch_result_data_dict.append(data_dict)

                for i in range(len(bboxes)):
                    x, y, w, h = bboxes[i]

                    label = str(coco_class_names[classes[i]])
                    confidence = confidences[i]
                    color = colors[classes[i]]
                    start_point = (
                        int(y.cpu().detach().numpy()),
                        int(x.cpu().detach().numpy()),
                    )
                    end_point = (
                        int(h.cpu().detach().numpy()),
                        int(w.cpu().detach().numpy()),
                    )

                    image = cv2.rectangle(image.astype(np.uint8).copy(), start_point, end_point, color, 5)
                    image = cv2.putText(
                        image.astype(np.uint8).copy(),
                        label + " " + str(round(float(confidence), 2)),
                        (
                            int(x.cpu().detach().numpy()),
                            int(y.cpu().detach().numpy()) + 30,
                        ),
                        font,
                        2,
                        color,
                        2,
                    )
            batch_result_images.append(image)
        if need_eval:
            return batch_result_images, batch_result_data_dict
        return batch_result_images

    def coco_evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        confthre: float,
        nmsthre: float,
        dataset_coco: coco_dataset.COCODataset,
        coco_class_ids: list,
    ) -> tuple[float, float]:
        """
        COCO average precision (AP) Evaluation.

        Iterate inference on the test dataset and the results are evaluated by COCO API.

        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        self.model.eval()
        cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        ids = []
        data_dict = []

        for (
            img,
            _,
            info_img,
            id_,
            _,
            _,
        ) in dataloader:
            with torch.no_grad():
                img = Variable(img.type(dtype)).cpu()
                outputs = self.model(img)
                outputs = dataset_utils.postprocess(outputs, 80, confthre, nmsthre)

            for k in range(len(outputs)):
                info_img_k = [float(info) for info in info_img[k].ravel()]
                id_k = int(id_[k])
                ids.append(id_k)

                if outputs[k] is None:
                    continue
                else:
                    batch_outputs = outputs[k].cpu().data

                    for output in batch_outputs:
                        x1 = float(output[0])
                        y1 = float(output[1])
                        x2 = float(output[2])
                        y2 = float(output[3])
                        label = coco_class_ids[int(output[6])]
                        box = dataset_utils.yolobox2label((y1, x1, y2, x2), info_img_k)
                        bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                        score = float(output[4].data.item() * output[5].data.item())  # object score * class score
                        results_info = {
                            "image_id": id_k,
                            "category_id": label,
                            "bbox": bbox,
                            "score": score,
                            "segmentation": [],
                        }  # COCO json format
                        data_dict.append(results_info)

        ap50, ap50_95 = yolo3_utils.get_ap_metrics(coco_diction=data_dict, dataset_coco=dataset_coco, ids=ids)

        return ap50, ap50_95
