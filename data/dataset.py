import torch
from torch.utils.data import Dataset
import numpy as np

class CenterNetDataset(Dataset):
    """
    A dummy dataset for CenterNet training.
    This dataset generates random data to simulate image and ground truth targets.
    In a real scenario, this would load actual images and annotations (e.g., from COCO).
    """
    def __init__(self, num_samples=100, img_size=(512, 512), num_classes=80, output_stride=4):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.output_size = (img_size[0] // output_stride, img_size[1] // output_stride)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy image: [3, H, W]
        img = torch.randn(3, self.img_size[0], self.img_size[1])

        # Dummy heatmap: [num_classes, H_out, W_out]
        # Simulate one positive instance per heatmap for simplicity
        hm = torch.zeros(self.num_classes, self.output_size[0], self.output_size[1])
        # Randomly place a "positive" point for one class
        c = np.random.randint(0, self.num_classes)
        x, y = np.random.randint(0, self.output_size[1]), np.random.randint(0, self.output_size[0])
        hm[c, y, x] = 1.0 # Simple positive point, in real data this would be a Gaussian blob

        # Dummy width/height (wh): [num_pos, 2]
        # For simplicity, let's assume 1 positive instance per image for now
        # In real data, this would be actual box dimensions for positive instances
        wh = torch.randn(1, 2) * 10 + 50 # Random width/height values

        # Dummy local offset (reg): [num_pos, 2]
        # In real data, this would be sub-pixel offsets for positive instances
        reg = torch.randn(1, 2) * 0.5 # Small random offsets

        # Dummy indices (ind): [num_pos]
        # Flat index into the output feature map (H_out * W_out)
        # For our single positive instance at (y, x)
        ind = torch.tensor([y * self.output_size[1] + x], dtype=torch.long)

        # Dummy regression mask (reg_mask): [num_pos]
        # Indicates if a positive instance has valid regression targets
        reg_mask = torch.tensor([True], dtype=torch.bool)

        target = {
            'hm': hm,
            'wh': wh,
            'reg': reg,
            'ind': ind,
            'reg_mask': reg_mask
        }

        return img, target


