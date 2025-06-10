# Model definition for CenterNet
import torch.nn as nn
from typing import Optional  # Import Optional
from .backbone import get_resnet50_backbone
from .head import get_head_module  # Import the factory function
from .loss import CenterNetLoss  # Import the loss function
from .fpn import FPN  # Import FPN module


class CenterNet(nn.Module):
    def __init__(
        self,
        num_classes,
        head_type="conv",  # Type of head to use ('conv', 'deconv', etc.)
        head_options: Optional[dict] = None,  # Changed type hint
        pretrained_backbone=True,
        compute_loss=False,  # New flag to enable loss computation
        loss_weights: Optional[dict] = None,  # Weights for loss components
        # FPN specific arguments
        use_fpn=False,
        fpn_out_channels=256,
        fpn_extra_blocks=False,
        # Which FPN level to use for heads if FPN is enabled
        # For CenterNet, typically a single high-resolution feature map is used.
        # This could be P3, P4, or P5 depending on the desired output stride.
        # If None, the highest resolution FPN output (P3) will be used.
        fpn_head_level: Optional[str] = "P3",
    ):
        super(CenterNet, self).__init__()

        self.backbone = get_resnet50_backbone(pretrained=pretrained_backbone)

        self.use_fpn = use_fpn
        if self.use_fpn:
            # FPN will take C3, C4, C5 features from the backbone
            # The in_channels_list should match the output channels of C3, C4, C5 from ResNet50
            self.fpn = FPN(
                in_channels_list=[512, 1024, 2048],  # Corresponding to C3, C4, C5
                out_channels=fpn_out_channels,
                extra_blocks=fpn_extra_blocks,
            )
            # The output channels from FPN levels will be fpn_out_channels
            self.neck_out_channels = fpn_out_channels
            # If a specific FPN level is chosen for heads, ensure it exists.
            if fpn_head_level not in ["P3", "P4", "P5"] + (
                ["P6", "P7"] if fpn_extra_blocks else []
            ):
                raise ValueError(f"Invalid fpn_head_level: {fpn_head_level}")
            self.fpn_head_level = fpn_head_level
        else:
            # If no FPN, the heads directly take the output of the backbone (ResNet50's last layer)
            self.neck_out_channels = self.backbone.out_channels[
                "C5"
            ]  # Use the highest level C5 output if no FPN
            # Note: If not using FPN, the backbone should be configured to return only the final feature map,
            # or the model should select it appropriately. Current backbone returns a dict.
            # For simplicity, if use_fpn is False, we assume the model implicitly uses the last output of the backbone.
            # This might require adjusting `get_resnet50_backbone` if a single tensor output is strictly needed.
            # For now, we will just use C5 as the input to heads if FPN is not used.
            self.fpn_head_level = "C5"  # Indicate that C5 is used if no FPN

        # Configuration for the output channels of each specific task head
        self.task_head_outputs_config = {
            "hm": num_classes,  # Heatmap
            "wh": 2,  # Width/Height
            "reg": 2,  # Regression offset
        }

        _head_options = head_options if head_options is not None else {}

        # Set a default for 'head_channels' (intermediate channels in head layers) if not in _head_options
        # This is used by both ConvolutionalHead and DeconvolutionalHead in head.py as their 'head_channels' argument.
        _head_options.setdefault("head_channels", 64)

        # Set a default for 'num_conv_layers' if head_type is 'conv' and not specified in head_options
        if head_type == "conv":
            _head_options.setdefault("num_conv_layers", 1)

        self.heads = nn.ModuleDict()  # Use ModuleDict to store head modules
        for head_name, task_out_channels in self.task_head_outputs_config.items():
            # Prepare arguments for the chosen head type
            current_head_args = {
                "in_channels": self.neck_out_channels,  # Heads now take input from FPN (or backbone directly if no FPN)
                "out_channels": task_out_channels,  # This is the final output channels for the specific task
            }
            current_head_args.update(_head_options)

            self.heads[head_name] = get_head_module(
                head_type=head_type, **current_head_args
            )

        self.compute_loss = compute_loss
        if self.compute_loss:
            _loss_weights = loss_weights if loss_weights is not None else {}
            self.criterion = CenterNetLoss(
                hm_weight=_loss_weights.get("hm_weight", 1),
                wh_weight=_loss_weights.get("wh_weight", 0.1),
                reg_weight=_loss_weights.get("reg_weight", 1),
            )

    def forward(
        self, x, batch=None
    ):  # Add batch as an optional argument for loss computation
        features = self.backbone(x)  # features is now a dict of C3, C4, C5

        if self.use_fpn:
            fpn_features = self.fpn(
                features
            )  # fpn_features is a dict of P3, P4, P5 (and P6, P7 if enabled)
            # Select the desired FPN level for the heads
            features_for_heads = fpn_features[self.fpn_head_level]
        else:
            # If no FPN, use the C5 feature map from the backbone directly
            features_for_heads = features["C5"]

        outputs = {}
        for head_name, head_module in self.heads.items():
            outputs[head_name] = head_module(features_for_heads)

        if self.compute_loss and batch is not None:
            total_loss, loss_stats = self.criterion(outputs, batch)
            return outputs, total_loss, loss_stats

        return outputs


# Example usage:
# if __name__ == '__main__':
#     import torch # Make sure torch is imported for the example
#     num_classes = 80 # Example: COCO dataset
#
#     print("--- Model with FPN (P3 level for heads) ---")
#     # FPN with P3 level used for heads. P3 has 256 channels (fpn_out_channels).
#     # Input 512x512 -> C3 is 64x64, P3 is 64x64. So heads will operate on 64x64 feature maps.
#     model_fpn = CenterNet(num_classes=num_classes, use_fpn=True, fpn_head_level='P3', compute_loss=True)
#     dummy_input = torch.randn(1, 3, 512, 512) # Example input
#     dummy_batch = {
#         'hm': torch.randn(1, num_classes, 64, 64), # HM size should match P3 size
#         'wh': torch.randn(2, 2),
#         'reg': torch.randn(2, 2),
#         'ind': torch.tensor([100, 200], dtype=torch.long),
#         'reg_mask': torch.ones(2, dtype=torch.bool)
#     }
#     outputs_fpn, total_loss_fpn, loss_stats_fpn = model_fpn(dummy_input, dummy_batch)
#     for head_name, head_output in outputs_fpn.items():
#         print(f"  {head_name} output shape: {head_output.shape}")
#     print(f"  Total Loss (FPN): {total_loss_fpn.item()}")
#     print(f"  Loss Stats (FPN): {loss_stats_fpn}")
#
#     print("\n--- Model without FPN (C5 level for heads) ---")
#     # Without FPN, heads operate directly on C5 (2048 channels, 16x16 for 512x512 input)
#     model_no_fpn = CenterNet(num_classes=num_classes, use_fpn=False, compute_loss=True)
#     dummy_batch_no_fpn = {
#         'hm': torch.randn(1, num_classes, 16, 16), # HM size should match C5 size
#         'wh': torch.randn(2, 2),
#         'reg': torch.randn(2, 2),
#         'ind': torch.tensor([10, 20], dtype=torch.long),
#         'reg_mask': torch.ones(2, dtype=torch.bool)
#     }
#     outputs_no_fpn, total_loss_no_fpn, loss_stats_no_fpn = model_no_fpn(dummy_input, dummy_batch_no_fpn)
#     for head_name, head_output in outputs_no_fpn.items():
#         print(f"  {head_name} output shape: {head_output.shape}")
#     print(f"  Total Loss (No FPN): {total_loss_no_fpn.item()}")
#     print(f"  Loss Stats (No FPN): {loss_stats_no_fpn}")
#
#     print("\n--- Model with FPN and P4 level for heads ---")
#     model_fpn_p4 = CenterNet(num_classes=num_classes, use_fpn=True, fpn_head_level='P4', compute_loss=True)
#     dummy_batch_p4 = {
#         'hm': torch.randn(1, num_classes, 32, 32), # HM size should match P4 size
#         'wh': torch.randn(2, 2),
#         'reg': torch.randn(2, 2),
#         'ind': torch.tensor([50, 150], dtype=torch.long),
#         'reg_mask': torch.ones(2, dtype=torch.bool)
#     }
#     outputs_fpn_p4, total_loss_fpn_p4, loss_stats_fpn_p4 = model_fpn_p4(dummy_input, dummy_batch_p4)
#     for head_name, head_output in outputs_fpn_p4.items():
#         print(f"  {head_name} output shape: {head_output.shape}")
#     print(f"  Total Loss (FPN P4): {total_loss_fpn_p4.item()}")
#     print(f"  Loss Stats (FPN P4): {loss_stats_fpn_p4}")
