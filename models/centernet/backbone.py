# Backbone networks for CenterNet
import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter

def get_resnet50_backbone(pretrained=True, return_layers=None):
    """
    Loads a pre-trained ResNet50 model and returns intermediate layers as specified.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        return_layers (dict): A dictionary specifying which layers to return.
                              Keys are the original layer names (e.g., 'layer2', 'layer3', 'layer4'),
                              values are the new names for the output dictionary (e.g., 'C3', 'C4', 'C5').
                              If None, default layers for FPN-like structures are used.
    """
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)

    # Default layers for FPN (C3, C4, C5)
    if return_layers is None:
        return_layers = {'layer2': 'C3', 'layer3': 'C4', 'layer4': 'C5'}

    # Use IntermediateLayerGetter to extract features from specified intermediate layers
    backbone = IntermediateLayerGetter(model, return_layers=return_layers)

    # Determine output channels for each returned layer
    # These are standard output channels for ResNet50 layers
    # layer2 (C3): 512 channels
    # layer3 (C4): 1024 channels
    # layer4 (C5): 2048 channels
    backbone.out_channels = {
        'C3': 512,
        'C4': 1024,
        'C5': 2048
    }
    return backbone

# Example of how to use it:
# if __name__ == '__main__':
#     # Example for FPN: get C3, C4, C5 features
#     backbone_fpn = get_resnet50_backbone(pretrained=True)
#     print(backbone_fpn)
#     dummy_input = torch.randn(1, 3, 512, 512) # Example input
#     output_features = backbone_fpn(dummy_input)
#     for name, feature in output_features.items():
#         print(f"{name} feature shape: {feature.shape}")
#     # Expected output shapes for 512x512 input:
#     # C3 feature shape: torch.Size([1, 512, 64, 64]) (512 / 8 = 64)
#     # C4 feature shape: torch.Size([1, 1024, 32, 32]) (512 / 16 = 32)
#     # C5 feature shape: torch.Size([1, 2048, 16, 16]) (512 / 32 = 16)
