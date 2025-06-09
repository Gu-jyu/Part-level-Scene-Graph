"""
Core implementations of detection heads in a single feature scale.
"""

__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = [""]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"


import math
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from detectron2.layers import cat, Conv2d, CNNBlockBase
import fvcore.nn.weight_init as weight_init
from fcsgg.structures import SceneGraph

import fcsgg.utils.centernet_utils as centernet_utils
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock
from fcsgg.layers import get_norm
from fcsgg.modeling.losses import RAFLoss

HEADS_REGISTRY = Registry("HEADS")
HEADS_REGISTRY.__doc__ = """
Registry for heads in a single-stage model.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Head`.
"""

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.01


class GeneralHeads(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        in_features,
        in_channels,
        cls_bias_value,
        ct_loss_weight,
        wh_loss_weight,
        reg_loss_weight,
        hm_loss_func,
        freeze_heads,
        use_gt_box,
        use_gt_object_label,
        output_strides,
    ):
        super().__init__()
        # fmt: off
        self.num_classes     = num_classes
        self.in_features     = in_features
        self.in_channels     = in_channels
        self.cls_bias_value  = cls_bias_value
        self.ct_loss_weight  = ct_loss_weight
        self.wh_loss_weight  = wh_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.freeze_heads    = freeze_heads
        self.hm_loss_func    = getattr(centernet_utils, hm_loss_func)
        self.use_gt_box      = use_gt_box
        self.use_gt_object_label = use_gt_object_label
        self.output_strides  = output_strides
        self.single_scale = len(self.output_strides) == 1
        # fmt: on

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[SceneGraph]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        The forward function will return a dict of loss {name: loss_val}
        """
        raise NotImplementedError()

    def loss(
        self, predictions, targets, after_activation=True
    ) -> Dict[str, torch.Tensor]:
        """
        The basic loss of centerNet heads, namely center heatmaps, size heatmaps, and offsets heatmaps.
        """
        pred_score = predictions["cls"]
        if not after_activation:
            pred_score = torch.sigmoid(pred_score)

        num_instances = [len(x.gt_wh) for x in targets]

        # multi-use index, dim0 - image index in B, dim1 - class index in center maps, dim2 - spatial loc of center
        index = [
            torch.stack(
                (
                    torch.ones(
                        num_instances[i], dtype=torch.long, device=x.gt_index.device
                    )
                    * i,
                    # x.gt_classes,
                    x.gt_index,
                )
            )
            for i, x in enumerate(targets)
        ]

        index = cat(index, dim=1)

        loss_cls = self.hm_loss_func(pred_score, targets)

        gt_wh = cat([x.gt_wh for x in targets], dim=0)
        gt_reg = cat([x.gt_reg for x in targets], dim=0)

        # if regression target at the same location, choose a random object
        # filtered_index, ori_inds = torch.unique(index, dim=1, return_inverse=True)
        # if filtered_index.numel() != 0:
        #     gt_wh = gt_wh[ori_inds[:filtered_index.size(1)]]
        #     gt_reg = gt_reg[ori_inds[:filtered_index.size(1)]]
        # index = filtered_index
        # width and height loss, better version
        loss_wh = centernet_utils.reg_l1_loss(predictions["wh"], index, gt_wh)

        # regression loss
        loss_reg = centernet_utils.reg_l1_loss(predictions["reg"], index, gt_reg)

        loss_cls *= self.ct_loss_weight
        loss_wh *= self.wh_loss_weight
        loss_reg *= self.reg_loss_weight

        loss = {
            "loss_cls": loss_cls,
            "loss_box_wh": loss_wh,
            "loss_center_reg": loss_reg,
        }
        # print(loss)
        return loss

    def freeze(self):
        if len(self.freeze_heads) > 0:
            for head in self.freeze_heads:
                if hasattr(self, head):
                    getattr(self, head).freeze()
        return self

    @classmethod
    def from_config(cls, cfg, input_shape):
        in_features = cfg.MODEL.HEADS.IN_FEATURES
        in_channels = [input_shape[k].channels for k in in_features]
        return {
            "num_classes": cfg.MODEL.HEADS.NUM_CLASSES,
            "in_features": in_features,
            "in_channels": in_channels,
            "cls_bias_value": cfg.MODEL.HEADS.CLS_BIAS_VALUE,
            "ct_loss_weight": cfg.MODEL.HEADS.LOSS.CT_WEIGHT,
            "wh_loss_weight": cfg.MODEL.HEADS.LOSS.WH_WEIGHT,
            "reg_loss_weight": cfg.MODEL.HEADS.LOSS.REG_WEIGHT,
            "freeze_heads": cfg.MODEL.HEADS.FREEZE,
            "hm_loss_func": cfg.MODEL.HEADS.LOSS.HEATMAP_LOSS_TYPE,
            "use_gt_box": cfg.RELATION.USE_GT_BOX,
            "use_gt_object_label": cfg.RELATION.USE_GT_OBJECT_LABEL,
            "output_strides": cfg.MODEL.HEADS.OUTPUT_STRIDES,
        }


class SingleHead(CNNBlockBase):
    def __init__(
        self,
        in_channel,
        out_channel,
        stride,
        conv_dims,
        dilation=1,
        bias_fill=False,
        bias_value=0,
        conv_norm="",
        kernel_size=3,
        deformable_on=False,
        deformable_first=True,
        bottleneck_channels=64,
        activation=None,
        down_ratio=1,
        up_sample=True,
        split_pred=False,
        add_coord=False,
    ):
        super(SingleHead, self).__init__(in_channel, out_channel, stride)
        self.activation = activation
        self.conv_norm_relus = []
        self.down_ratio = down_ratio
        self.up_sample = up_sample
        self.split_pred = split_pred
        cur_channels = in_channel
        cur_channels = cur_channels + 2 if add_coord else cur_channels
        deformable_idx = 0 if deformable_first else len(conv_dims) - 1
        for k, conv_dim in enumerate(conv_dims):
            stride = 2 if self.down_ratio > 1 and k < math.log2(self.down_ratio) else 1
            # if deformable_on and k == deformable_idx:
            if deformable_on:
                conv = DeformBottleneckBlock(
                    cur_channels,
                    conv_dim,
                    bottleneck_channels=conv_dim // 2,
                    stride=stride,
                    # stride_in_1x1 = stride == 1,
                    norm=conv_norm,
                    deform_modulated=True,
                    deform_num_groups=1,
                    dilation=1,
                )
            # elif add_coord and k == 0:
            #     conv = CoordConv(cur_channels, conv_dim,
            #                   kernel_size=kernel_size,
            #                   stride=stride,
            #                   padding=(kernel_size * dilation - 1) // 2,
            #                   dilation=dilation,
            #                   bias=not conv_norm,
            #                   activation=F.relu,
            #                   norm=get_norm(conv_norm, conv_dim, momentum=BN_MOMENTUM))
            else:
                conv = Conv2d(
                    cur_channels,
                    conv_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size * dilation - 1) // 2,
                    dilation=dilation,
                    bias=not conv_norm,
                    activation=F.relu,
                    norm=get_norm(conv_norm, conv_dim, momentum=BN_MOMENTUM),
                )
            self.add_module("head_fcn{}".format(k), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        if self.split_pred:
            self.out_conv_x = Conv2d(cur_channels, out_channel // 2, kernel_size=1)
            self.out_conv_y = Conv2d(cur_channels, out_channel // 2, kernel_size=1)
            nn.init.xavier_normal_(self.out_conv_x.weight)
            self.out_conv_x.bias.data.fill_(bias_value)
            nn.init.xavier_normal_(self.out_conv_y.weight)
            self.out_conv_y.bias.data.fill_(bias_value)
        else:
            self.out_conv = Conv2d(cur_channels, out_channel, kernel_size=1)
            # initialization for output layer
            nn.init.xavier_normal_(self.out_conv.weight)
            self.out_conv.bias.data.fill_(bias_value)
        if not deformable_on:
            for layer in self.conv_norm_relus:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if self.down_ratio > 1 and self.up_sample:
            x = F.interpolate(
                x, scale_factor=self.down_ratio, mode="nearest", align_corners=None
            )
        if self.split_pred:
            x_0 = self.out_conv_x(x)
            x_1 = self.out_conv_y(x)
            x = torch.stack((x_0, x_1), dim=2)
        else:
            x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


@HEADS_REGISTRY.register()
class CenternetHeads(GeneralHeads):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        norm = cfg.MODEL.HEADS.NORM
        # we only use shared head
        shared = len(self.in_features) > 1
        conv_dims = [conv_dim] * num_conv
        # does not influence anything
        self.output_stride = self.output_strides[0]
        add_coord = cfg.MODEL.HEADS.RAF.ADD_COORD

        if not shared:
            assert len(self.in_features) == 1 and len(self.in_channels) == 1

        in_channels = self.in_channels[0]
        self.cls_head = SingleHead(
            in_channels,
            self.num_classes,
            self.output_stride,
            conv_dims,
            conv_norm=norm,
            bias_fill=True,
            bias_value=self.cls_bias_value,
            activation=torch.sigmoid_,
            add_coord=add_coord,
        )
        self.wh_head = SingleHead(
            in_channels,
            2,
            self.output_stride,
            conv_dims,
            conv_norm=norm,
            activation=None,
            add_coord=add_coord,
        )
        self.reg_head = SingleHead(
            in_channels,
            2,
            self.output_stride,
            conv_dims,
            conv_norm=norm,
            activation=None,
            add_coord=add_coord,
        )

    def forward(
        self,
        features: Union[Dict[str, torch.Tensor], torch.Tensor],
        targets: Optional[List[SceneGraph]] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        if self.training:
            assert targets
            if self.single_scale:
                targets = [target[0] for target in targets]

        # e.g. {"p2": tensor1, "p3": tensor2}
        if isinstance(features, Dict):
            features = [features[f] for f in self.in_features][0]

        cls = self.cls_head(features)
        wh = self.wh_head(features)
        reg = self.reg_head(features)
        preds = {"cls": cls, "wh": wh, "reg": reg}
        if self.training:
            losses.update(self.loss(preds, targets))

        return losses, preds


@HEADS_REGISTRY.register()
class CenternetRelationHeads(CenternetHeads):
    """
    Extended heads with relation affinity field prediction head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.relation_on = cfg.RELATION.RELATION_ON
        self.num_predicates = cfg.MODEL.HEADS.NUM_PREDICATES
        self.raf_loss_weight = cfg.MODEL.HEADS.LOSS.RAF_WEIGHT
        self.raf_dilation = cfg.MODEL.HEADS.RAF.DILATION
        self.raf_type = cfg.INPUT.RAF_TYPE
        deformable_on = cfg.MODEL.HEADS.RAF.LAST_DEFORM_ON
        num_groups = cfg.MODEL.HEADS.RAF.NUM_GROUPS
        width_per_group = cfg.MODEL.HEADS.RAF.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        conv_dim = cfg.MODEL.HEADS.CONV_DIM
        num_conv = cfg.MODEL.HEADS.NUM_CONV
        conv_norm = cfg.MODEL.HEADS.NORM
        conv_dims = [conv_dim] * num_conv
        kernel_size = cfg.MODEL.HEADS.RAF.KERNEL_SIZE
        in_channels = self.in_channels[0]
        self.down_ratio = cfg.MODEL.HEADS.RAF.DOWN_SAMPLE_RATIO
        raf_conv_dims = [cfg.MODEL.HEADS.RAF.CONV_DIM] * cfg.MODEL.HEADS.RAF.NUM_CONV
        # raf_conv_dims = cfg.MODEL.HEADS.RAF.CONV_DIMS
        split_pred = cfg.MODEL.HEADS.RAF.SPLIT
        add_coord = cfg.MODEL.HEADS.RAF.ADD_COORD
        if self.relation_on:
            raf_activation_func = (
                torch.sigmoid_ if self.raf_type == "point" else torch.tanh_
            )
            self.raf_head = SingleHead(
                in_channels,
                2 * self.num_predicates,
                self.output_stride,
                raf_conv_dims,
                kernel_size=kernel_size,
                conv_norm=conv_norm,
                bias_fill=True,
                deformable_on=deformable_on,
                dilation=self.raf_dilation,
                bottleneck_channels=bottleneck_channels,
                activation=raf_activation_func,
                bias_value=self.cls_bias_value if self.raf_type == "point" else 0,
                down_ratio=self.down_ratio,
                split_pred=split_pred,
                add_coord=add_coord,
            )
            if self.training:
                self.raf_loss_evaluator = RAFLoss(cfg)
                # self.rel_loss_evaluator = RelationLoss()

    def forward(
        self, features: Dict[str, torch.Tensor], targets: Optional[List[Dict]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # if self.training:
        # only take one scale
        # targets = [target[0] for target in targets]
        losses, preds = super().forward(features, targets)
        if self.relation_on:
            if isinstance(features, Dict):
                features = [features[f] for f in self.in_features][0]
            if self.training and self.single_scale:
                targets = [target[0] for target in targets]
            rafs = self.raf_head(features)
            preds.update({"raf": rafs})
            if self.training:
                loss_raf = self.raf_loss_evaluator(rafs, targets)
                loss_raf *= self.raf_loss_weight
                losses.update({"loss_raf": loss_raf})
                # loss_rel = self.rel_loss_evaluator(rafs, preds['cls'], targets)
                # losses.update({'loss_rel': loss_rel})
        return losses, preds


def build_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.HEADS.NAME
    return HEADS_REGISTRY.get(name)(cfg, input_shape).freeze()
