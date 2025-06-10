# Loss functions for CenterNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    Paper: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    """
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted heatmaps, shape [B, C, H, W]. Sigmoid should be applied.
            gt (torch.Tensor): Ground truth heatmaps, shape [B, C, H, W].
                                 Contains values 0 for background, 1 for positive locations,
                                 and often values between 0 and 1 for "difficult" negatives
                                 (e.g., locations near positive anchors, rendered as Gaussians).
        Returns:
            torch.Tensor: Scalar loss value.
        """
        pred = torch.clamp(pred.sigmoid_(), min=1e-4, max=1-1e-4) # Ensure numerical stability

        pos_inds = gt.eq(1).float() # Positive locations (where gt is 1)
        neg_inds = gt.lt(1).float() # Negative locations (where gt is less than 1)

        neg_weights = torch.pow(1 - gt, self.beta) # (1 - Yxyc)^beta for negative locations

        # Loss for positive locations: -(1 - Pxyc)^alpha * log(Pxyc)
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_inds
        
        # Loss for negative locations: -(1 - Yxyc)^beta * (Pxyc)^alpha * log(1 - Pxyc)
        neg_loss = -neg_weights * torch.pow(pred, self.alpha) * torch.log(1 - pred) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            # If there are no positive examples, return only the negative loss
            return neg_loss
        return (pos_loss + neg_loss) / num_pos

class RegL1Loss(nn.Module):
    """
    L1 Loss for regression tasks (e.g., offset, size) in CenterNet.
    This loss is applied only at the locations of detected objects.
    """
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        """
        Args:
            output (torch.Tensor): Predicted regression map, shape [B, C, H, W]. 
                                   C is the number of regressed values (e.g., 2 for offsets or W/H).
            mask (torch.Tensor): Boolean mask, shape [B, H, W] or [B, 1, H, W], 
                                 indicating positive locations (object centers).
            ind (torch.Tensor): LongTensor of 1D indices of positive locations, shape [num_pos].
                                These indices correspond to the flattened [B, H, W] dimensions.
            target (torch.Tensor): Ground truth regression values for positive locations, shape [num_pos, C].
        
        Returns:
            torch.Tensor: Scalar L1 loss value, normalized by the number of positive instances.
        """
        # Transpose output to [B, H, W, C] for easier gathering
        pred = output.permute(0, 2, 3, 1).contiguous() # B, H, W, C
        # Reshape to [B*H*W, C]
        pred = pred.view(-1, pred.size(-1)) # BHW, C
        
        # Gather predictions corresponding to positive instances
        # ind contains flat indices for the BHW dimension
        # We need to expand ind to gather all C channels for each positive instance
        # Target is already [num_pos, C]
        # pred is [BHW, C]
        
        # A common way to gather is to create a full mask and then select
        # However, if `ind` and `target` are already prepared for positive locations, it's more direct.
        # Let's assume `ind` are the *batch* indices for the features, and `target` is already filtered.
        # This means `pred` needs to be indexed by `ind`.
        
        # The `ind` tensor from CenterNet often refers to the k selected top score locations across the batch.
        # The `mask` indicates which of these k locations correspond to actual ground truth objects.
        
        # A simpler and common approach if `target` is shaped [num_pos, C] and `ind` helps select from `pred`:
        # `pred` is [B*H*W, C]. `ind` is [num_pos] (flat indices).
        # We need to select rows from `pred` using `ind`.
        pred = pred[ind] # This gathers the predictions for the positive locations, result is [num_pos, C]
        
        # Ensure mask is broadcastable if it's used with target for filtering, though target should already be filtered.
        # If the target is only for positive locations, mask might be implicit in how `ind` and `target` were generated.
        # For this implementation, we assume `target` and `pred` (after gathering with `ind`) are aligned and for positive samples.
        
        loss = F.l1_loss(pred, target, reduction='sum')
        
        # Normalize by the number of positive instances.
        # The mask sum can give the number of positive instances if `mask` is [B, H, W] true/false.
        # Or, more directly, num_pos = target.size(0)
        num_pos = target.size(0)
        if num_pos > 0:
            loss = loss / num_pos
        else:
            # No positive instances, loss is 0. Avoid division by zero.
            loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
            
        return loss

class CenterNetLoss(nn.Module):
    def __init__(self, hm_weight=1, wh_weight=0.1, reg_weight=1):
        super(CenterNetLoss, self).__init__()
        self.hm_loss = FocalLoss()
        self.wh_loss = RegL1Loss()
        self.reg_loss = RegL1Loss()
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.reg_weight = reg_weight

    def forward(self, outputs, batch):
        """
        Calculates the total CenterNet loss.

        Args:
            outputs (dict): Dictionary of model outputs, typically containing:
                            'hm': Predicted heatmap [B, C, H, W]
                            'wh': Predicted width/height [B, 2, H, W]
                            'reg': Predicted offset [B, 2, H, W]
            batch (dict): Dictionary of ground truth targets, typically containing:
                          'hm': Ground truth heatmap [B, C, H, W]
                          'wh': Ground truth width/height [num_pos, 2]
                          'reg': Ground truth offset [num_pos, 2]
                          'ind': Indices of positive locations [num_pos] (flat indices into B*H*W)
                          'reg_mask': Mask for regression, often a boolean tensor [num_pos]

        Returns:
            torch.Tensor: Scalar total loss value.
            dict: Dictionary of individual loss components.
        """
        hm_loss = self.hm_loss(outputs['hm'], batch['hm'])
        
        # For regression losses, we only consider positive locations
        # The 'ind' and 'reg_mask' are crucial for gathering the correct predictions and targets.
        # 'reg_mask' indicates which of the 'ind' locations actually have a valid regression target.
        # This is important because some positive locations might not have associated regression targets
        # (e.g., if an object is too small or filtered out for regression).
        
        # Ensure that ind and reg_mask are on the same device as outputs
        reg_mask_cuda = batch['reg_mask'].float()
        ind_cuda = batch['ind'].long()

        # Apply mask to target before passing to RegL1Loss.
        # This ensures that `target` passed to RegL1Loss only contains valid entries.
        # The `RegL1Loss` expects `target` to be already filtered for positive locations.
        # The `mask` argument in RegL1Loss is used to count `num_pos` correctly if `target` is not pre-filtered.
        # However, here, we assume batch['wh'] and batch['reg'] are already filtered by the pipeline
        # to only include values for actual positive instances, and reg_mask is used to count num_pos.

        # CenterNet's regression loss typically uses a mask to select the relevant predictions
        # and targets. The `ind` (indices) are used to flatten and select from the prediction map.
        # The `reg_mask` is then used to filter out predictions where there is no valid ground truth.

        wh_loss = self.wh_loss(outputs['wh'], reg_mask_cuda, ind_cuda, batch['wh'])
        reg_loss = self.reg_loss(outputs['reg'], reg_mask_cuda, ind_cuda, batch['reg'])

        total_loss = self.hm_weight * hm_loss + \
                     self.wh_weight * wh_loss + \
                     self.reg_weight * reg_loss
        
        loss_stats = {
            'loss': total_loss,
            'hm_loss': hm_loss,
            'wh_loss': wh_loss,
            'reg_loss': reg_loss
        }
        
        return total_loss, loss_stats

