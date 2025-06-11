import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.centernet.model import CenterNet
from data.dataset import CenterNetDataset

def collate_fn(batch):
    """
    Custom collate_fn for DataLoader to handle dictionary targets.
    """
    imgs = torch.stack([item[0] for item in batch])
    targets = {
        key: torch.cat([item[1][key] for item in batch], dim=0) if key in ['wh', 'reg', 'ind', 'reg_mask'] else torch.stack([item[1][key] for item in batch])
        for key in batch[0][1]
    }
    return imgs, targets

def train(num_epochs=10, batch_size=4, learning_rate=1e-4, num_classes=80,
          img_size=(512, 512), use_fpn=True, fpn_head_level='P3'): # Removed output_stride from args
    
    # 1. Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine the correct output stride for the dataset based on model configuration
    output_strides = {
        'C5': 32, # ResNet50's C5 output stride
        'P3': 8,
        'P4': 16,
        'P5': 32,
        'P6': 64,
        'P7': 128
    }
    
    # If FPN is used, get stride from fpn_head_level. Otherwise, use C5 stride.
    dataset_output_stride = output_strides.get(fpn_head_level if use_fpn else 'C5', 4) # Default to 4 if not found (shouldn't happen with valid levels)
    print(f"Using dataset output stride: {dataset_output_stride} (derived from fpn_head_level='{fpn_head_level}' and use_fpn={use_fpn})")

    # 2. Dataset and DataLoader
    print("Loading dataset...")
    # Pass the dynamically determined output_stride to the dataset
    dataset = CenterNetDataset(num_samples=100, img_size=img_size, num_classes=num_classes, output_stride=dataset_output_stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 3. Model instantiation
    print("Instantiating CenterNet model...")
    model = CenterNet(
        num_classes=num_classes,
        compute_loss=True, # Enable loss computation during forward pass
        use_fpn=use_fpn,
        fpn_head_level=fpn_head_level
    ).to(device)
    print("Model instantiated.")

    # 4. Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Optimizer: {optimizer.__class__.__name__}, Learning Rate: {learning_rate}")

    # 5. Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_epoch_loss = 0
        
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            # Move all target tensors to the same device as the model
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(device)

            optimizer.zero_grad() # Zero gradients

            # Forward pass: model returns outputs, total_loss, and loss_stats
            outputs, total_loss, loss_stats = model(imgs, targets)
            
            total_loss.backward() # Backward pass
            optimizer.step()      # Update weights

            total_epoch_loss += total_loss.item()

            if (batch_idx + 1) % 10 == 0: # Print loss every 10 batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Total Loss: {total_loss.item():.4f}, "
                      f"HM Loss: {loss_stats['hm_loss'].item():.4f}, "
                      f"WH Loss: {loss_stats['wh_loss'].item():.4f}, "
                      f"Reg Loss: {loss_stats['reg_loss'].item():.4f}")
        
        avg_epoch_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {avg_epoch_loss:.4f}")
    
    print("Training finished.")

if __name__ == '__main__':
    # You can configure training parameters here
    train(
        num_epochs=5, 
        batch_size=2, 
        learning_rate=1e-4, 
        num_classes=80, 
        img_size=(512, 512), 
        # output_stride is now determined dynamically within the train function
        use_fpn=True, 
        fpn_head_level='P3' # Or 'C5' if use_fpn=False, or 'P4', 'P5', etc.
    )

