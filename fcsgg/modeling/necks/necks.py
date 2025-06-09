"""
Implementations of necks (intermediate branches transforming backbone features).

"""

__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import inspect
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import (
    ShapeSpec,
    CNNBlockBase,
    ConvTranspose2d,
    FrozenBatchNorm2d,
)
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import DeformBottleneckBlock
from detectron2.layers.deform_conv import DeformConv, ModulatedDeformConv
from detectron2.layers import ShapeSpec, Conv2d, get_norm

from fcsgg.layers import add_coords


NECKS_REGISTRY = Registry("NECKS")
NECKS_REGISTRY.__doc__ = """
Registry for neck modules in a single-stage model (or even for generalized R-CNN model).
A neck takes feature maps and applies transformation of the feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Neck`.
"""

logger = logging.getLogger(__name__)


class Neck(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.freeze_necks = cfg.MODEL.NECKS.FREEZE

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    def freeze(self):
        if self.freeze_necks:
            # freeze all weights of a neck
            for p in self.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


@NECKS_REGISTRY.register()
class ConcatNeck(Neck):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.in_features = cfg.MODEL.NECKS.IN_FEATURES
        self.in_channels = [input_shape[k].channels for k in self.in_features]
        self.num_groups = cfg.MODEL.NECKS.NUM_GROUPS
        self.deform_modulated = cfg.MODEL.NECKS.DEFORM_MODULATED
        self.deform_num_groups = cfg.MODEL.NECKS.DEFORM_NUM_GROUPS
        self.width_per_group = cfg.MODEL.NECKS.WIDTH_PER_GROUP
        self.bottleneck_channels = self.num_groups * self.width_per_group
        self.down_ratio = cfg.MODEL.HEADS.OUTPUT_STRIDE
        self.norm = cfg.MODEL.NECKS.NORM
        self.fuse_method = "cat"
        self._out_features = [self.fuse_method]
        self._out_feature_channels = {self.fuse_method: sum(self.in_channels)}
        self._out_feature_strides = {self.fuse_method: self.down_ratio}
        upsample_mode = cfg.MODEL.NECKS.UPSAMPLE_MODE
        self.upsample_mode = (
            {"mode": upsample_mode, "align_corners": None}
            if upsample_mode == "nearest"
            else {"mode": upsample_mode, "align_corners": True}
        )
        # for coordconv
        self.add_coord = cfg.MODEL.HEADS.RAF.ADD_COORD

    def forward(self, features: Dict[str, torch.Tensor]):
        # features of different scale, first is the largest
        features = [features[f] for f in self.in_features]
        outputs = [features[0]]
        target_size = features[0].size()[2:]
        for i in range(1, len(features)):
            outputs.append(
                F.interpolate(
                    features[i],
                    size=target_size,
                    mode=self.upsample_mode["mode"],
                    align_corners=self.upsample_mode["align_corners"],
                )
            )

        outputs = torch.cat(outputs, dim=1)
        if self.add_coord:
            outputs = add_coords(outputs)
        return dict(zip(self._out_features, [outputs]))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


def build_necks(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    use_neck = cfg.MODEL.NECKS.ENABLED
    if use_neck:
        name = cfg.MODEL.NECKS.NAME
        return NECKS_REGISTRY.get(name)(cfg, input_shape).freeze()
    else:
        return None
