
import math
import inspect
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.layers import cat, Conv2d, CNNBlockBase
from detectron2.layers.batch_norm import FrozenBatchNorm2d
import fvcore.nn.weight_init as weight_init

HEADS_REGISTRY = Registry("HEADS")
HEADS_REGISTRY.__doc__ = """
Registry for heads in a single-stage model.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Head`.
"""

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.01

class TestHeads(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    @configurable
    def __init__(self, *, num_classes, in_features, in_channels,
                 cls_bias_value, ct_loss_weight, wh_loss_weight, reg_loss_weight,
                 hm_loss_func, freeze_heads, use_gt_box, use_gt_object_label,
                 output_strides):
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


    def loss(self, predictions, targets, after_activation=True) -> Dict[str, torch.Tensor]:
        """
        The basic loss of centerNet heads, namely center heatmaps, size heatmaps, and offsets heatmaps.
        """
        pred_score = predictions['cls']
        if not after_activation:
            pred_score = torch.sigmoid(pred_score)

        num_instances = [len(x.gt_wh) for x in targets]

        # multi-use index, dim0 - image index in B, dim1 - class index in center maps, dim2 - spatial loc of center
        index = [torch.stack((torch.ones(num_instances[i], dtype=torch.long, device=x.gt_index.device) * i,
                              # x.gt_classes,
                              x.gt_index))
                 for i, x in enumerate(targets)]

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
        loss_wh = centernet_utils.reg_l1_loss(predictions['wh'], index, gt_wh)

        # regression loss
        loss_reg = centernet_utils.reg_l1_loss(predictions['reg'], index, gt_reg)

        loss_cls *= self.ct_loss_weight
        loss_wh  *= self.wh_loss_weight
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
            "output_strides": cfg.MODEL.HEADS.OUTPUT_STRIDES
        }
    

def build_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.HEADS.NAME
    return HEADS_REGISTRY.get(name)(cfg, input_shape).freeze()