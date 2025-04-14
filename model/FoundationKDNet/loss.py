import torch
import torch.nn as nn
import torch.nn.functional as F

# Pair-wise loss class to compare feature maps
class PairWiseLoss(nn.Module):
    def __init__(self):
        super(PairWiseLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_fm, teacher_fm):
        return self.mse(student_fm, teacher_fm)

class DiceLossSoftmax(nn.Module):
    def __init__(self, smooth=1e-3, reduction='mean'):
        """
        Initialize DiceLoss with softmax.

        Args:
            smooth (float): Small value to avoid division by zero (default: 1e-5).
            reduction (str): Reduction method for the loss ('mean', 'sum', or 'none').
        """
        super(DiceLossSoftmax, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute Dice Loss between pred and target using softmax.

        Args:
            pred (torch.Tensor): Model predictions, shape [B, C, H, W] (logits or probabilities).
            target (torch.Tensor): Ground truth or teacher output, shape [B, C, H, W].

        Returns:
            torch.Tensor: Dice Loss value.
        """
        # Apply softmax if pred is not already probabilities
        pred = F.softmax(pred.float(), dim=1)  # Convert logits to probabilities
        target = F.softmax(target.float(), dim=1)  # Convert target to probabilities

        # Flatten for easier computation
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()

        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        # Loss is 1 - Dice
        loss = 1 - dice

        # Apply reduction
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss
        else:  # 'none'
            # Return loss expanded to batch size
            batch_size = pred.size(0)
            return loss.expand(batch_size)

class DiceLossArgmax(nn.Module):
    def __init__(self, smooth=1e-3, reduction='mean'):
        """
        Initialize DiceLoss with argmax.

        Args:
            smooth (float): Small value to avoid division by zero (default: 1e-5).
            reduction (str): Reduction method for the loss ('mean', 'sum', or 'none').
        """
        super(DiceLossArgmax, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Compute Dice Loss between pred (student output) and target (hard labels).

        Args:
            pred (torch.Tensor): Model predictions, shape [B, C, H, W] (logits or probabilities).
            target (torch.Tensor): Hard labels, shape [B, H, W] (integer class indices).

        Returns:
            torch.Tensor: Dice Loss value.
        """
        # Convert pred to hard labels using argmax
        pred = pred.argmax(dim=1)  # Shape [B, H, W], integer class indices

        # Ensure target is integer type (hard labels)
        target = target.long()

        # Number of classes (inferred from pred)
        num_classes = pred.max().item() + 1 if pred.max() > 0 else pred.shape[1]

        # Convert to one-hot format for both pred and target
        pred_one_hot = torch.zeros_like(pred).unsqueeze(1).repeat(1, num_classes, 1, 1).float()
        target_one_hot = torch.zeros_like(target).unsqueeze(1).repeat(1, num_classes, 1, 1).float()

        pred_one_hot.scatter_(1, pred.unsqueeze(1), 1)  # Shape [B, C, H, W]
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)  # Shape [B, C, H, W]

        # Compute intersection and union
        intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))  # Sum over H, W
        union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Loss is 1 - Dice, averaged over classes
        loss = 1 - dice.mean(dim=1)  # Average over classes per batch

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class WassersteinLoss(nn.Module):
    def __init__(self, fun='sqrt', tau=2.0):
        """Initialize Gaussian Wasserstein Distance loss.

        Args:
            fun (str): The function applied to distance. Options: 'sqrt', 'log1p', 'exp'
            tau (float): Temperature parameter for loss scaling
        """
        super(WassersteinLoss, self).__init__()
        self.fun = fun
        self.tau = tau

    def compute_gaussian_params(self, x):
        """Compute mean and covariance matrix from input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W] or [B, C]

        Returns:
            tuple: (mean, covariance) tensors
        """
        if len(x.shape) == 4:  # [B, C, H, W]
            # Reshape to [B, C, H*W]
            x = x.view(x.size(0), x.size(1), -1)

        # Compute mean
        mu = torch.mean(x, dim=-1)  # [B, C]

        # Compute covariance
        x_centered = x - mu.unsqueeze(-1)
        sigma = torch.matmul(x_centered, x_centered.transpose(-2, -1))
        sigma = sigma / (x.size(-1) - 1)  # Normalize by N-1 for unbiased estimate

        return mu, sigma

    def forward(self, pred, target):
        """Compute Gaussian Wasserstein Distance loss.

        Args:
            pred (torch.Tensor): Predicted outputs [B, C, H, W] or [B, C]
            target (torch.Tensor): Target outputs [B, C, H, W] or [B, C]

        Returns:
            torch.Tensor: Computed loss value
        """
        # Compute Gaussian parameters for both prediction and target
        mu_p, sigma_p = self.compute_gaussian_params(pred)
        mu_t, sigma_t = self.compute_gaussian_params(target)

        # Compute xy_distance (mean distance)
        xy_distance = (mu_p - mu_t).square().sum(dim=-1)

        # Compute whr_distance (covariance distance)
        whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        whr_distance = whr_distance + sigma_t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        # Compute trace term
        _t_tr = (sigma_p.bmm(sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        _t_det_sqrt = (sigma_p.det() * sigma_t.det()).clamp(0).sqrt()
        whr_distance += (-2) * (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()

        # Combine distances
        dis = xy_distance + whr_distance
        gwd_dis = dis.clamp(min=1e-6)

        # Apply different loss functions
        if self.fun == 'sqrt':
            loss = 1 - 1 / (self.tau + torch.sqrt(gwd_dis))
        elif self.fun == 'log1p':
            loss = 1 - 1 / (self.tau + torch.log1p(gwd_dis))
        else:
            scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
            loss = torch.log1p(torch.sqrt(gwd_dis) / scale)

        return loss.mean()  # Average over batch
