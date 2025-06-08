
import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from heads import build_heads
from necks import build_necks

@META_ARCH_REGISTRY.register()
class TestModel(nn.Module):
    """
    Generalized one stage detector. Any models that contains the following 2 components:
    1. Per-image feature extraction (aka backbone)
    2 (optional). Neck modules
    3. Prediction head(s)
    """
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        heads: nn.Module,
        necks: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        # hack the size size_divisibility
        self.backbone._size_divisibility = 32
        self.necks = necks
        self.heads = heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"


    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "scene_graph" in batched_inputs[0]:
            gt_scene_graph = [x["scene_graph"].to(self.device) for x in batched_inputs]
        else:
            gt_scene_graph = None

        features = self.backbone(images.tensor)
        if self.necks:
            features = self.necks(features)

        detector_losses = self.heads(features, gt_scene_graph)

        losses = {}
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        if self.necks:
            features = self.necks(features)

        if detected_instances is None:

            _, results = self.heads(features)
        else:
            results = None
            # detected_instances = [x.to(self.device) for x in detected_instances]
            # results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # the default results should be List[Instances], each Instances
        # should at least have fields of pred_boxes, scores and pred_classes.
        if do_postprocess:
            return self._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    
    @classmethod
    def from_config(cls, cfg):
        """
        Create a OneStageDetector from config.
        """
        backbone = build_backbone(cfg)
        necks = build_necks(cfg, backbone.output_shape())
        heads = build_heads(cfg, backbone.output_shape())

        pixel_mean = cfg.MODEL.PIXEL_MEAN
        pixel_std = cfg.MODEL.PIXEL_STD
        input_format = cfg.INPUT.FORMAT
        vis_period = cfg.VIS_PERIOD

        return {
            "backbone": backbone,
            "necks": necks,
            "heads": heads,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            "input_format": input_format,
            "vis_period": vis_period,
        }
