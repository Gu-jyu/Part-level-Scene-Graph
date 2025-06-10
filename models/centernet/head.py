\
import torch.nn as nn

class ConvolutionalHead(nn.Module):
    def __init__(self, in_channels, head_channels, out_channels, num_conv_layers=1):
        """
        A simple convolutional head.
        
        Args:
            in_channels (int): Number of input channels (e.g., from the backbone).
            head_channels (int): Number of channels in the intermediate convolutional layers of the head.
            out_channels (int): Number of output channels for this specific head task.
            num_conv_layers (int): Number of intermediate convolutional layers before the final output layer.
                                   If 0, a single conv layer maps directly from in_channels to out_channels.
        """
        super(ConvolutionalHead, self).__init__()
        layers = []
        current_channels = in_channels

        if num_conv_layers > 0:
            for _ in range(num_conv_layers):
                layers.append(nn.Conv2d(current_channels, head_channels, kernel_size=3, padding=1, bias=True))
                layers.append(nn.ReLU(inplace=True))
                current_channels = head_channels
            
            # Final layer to produce the output channels
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True))
        else:
            # Direct mapping if no intermediate layers
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True))
            
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)

class DeconvolutionalHead(nn.Module):
    def __init__(self, in_channels, head_channels, out_channels, num_deconv_layers=1, deconv_kernel_size=4, deconv_stride=2, deconv_padding=1):
        """
        A conceptual deconvolutional head for upsampling.
        Actual implementation would need careful design of deconv layers.
        """
        super(DeconvolutionalHead, self).__init__()
        layers = []
        current_channels = in_channels

        # Placeholder for deconvolutional/upsampling layers
        # This is a simplified example. A real implementation would be more complex,
        # potentially with multiple deconv layers, and careful channel management.
        for i in range(num_deconv_layers):
            # Example: A transpose convolution layer
            layers.append(nn.ConvTranspose2d(current_channels, 
                                             head_channels, 
                                             kernel_size=deconv_kernel_size, 
                                             stride=deconv_stride, 
                                             padding=deconv_padding, 
                                             bias=False))
            layers.append(nn.BatchNorm2d(head_channels)) # Common to use BatchNorm
            layers.append(nn.ReLU(inplace=True))
            current_channels = head_channels
        
        # Final convolutional layer to get to the desired out_channels
        # This layer might operate on the upsampled feature map.
        # It could be similar to the final layer in ConvolutionalHead or more complex.
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True))
        
        self.head = nn.Sequential(*layers)
        # Note: The output spatial size depends on the deconv parameters and input size.

    def forward(self, x):
        return self.head(x)

def get_head_module(head_type, **kwargs):
    """
    Factory function to get a specific head module.
    
    Args:
        head_type (str): Type of head to create ('conv', 'deconv').
        **kwargs: Arguments to pass to the head constructor.
                  Common args: in_channels, head_channels, out_channels.
                  Specific args: num_conv_layers (for conv), 
                                 num_deconv_layers, etc. (for deconv).
    """
    if head_type == 'conv':
        # Pop deconv-specific args if they exist to avoid passing them to ConvolutionalHead
        kwargs.pop('num_deconv_layers', None)
        kwargs.pop('deconv_kernel_size', None)
        kwargs.pop('deconv_stride', None)
        kwargs.pop('deconv_padding', None)
        return ConvolutionalHead(**kwargs)
    elif head_type == 'deconv':
        # Pop conv-specific args
        kwargs.pop('num_conv_layers', None)
        return DeconvolutionalHead(**kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}")

