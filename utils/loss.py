import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, weight=None):
        """
        Multiclass Dice Loss with class weights.

        Args:
            smooth (float): Smoothing value to avoid division by zero.
            weight (torch.Tensor, optional): Class weights, shape (num_classes,). Default is None.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, inputs, targets):
        """
        Compute the Dice Loss for multi-class segmentation with class weights.

        Args:
            inputs (torch.Tensor): Predicted logits, shape (B, C, H, W).
            targets (torch.Tensor): Ground truth labels, shape (B, H, W).

        Returns:
            torch.Tensor: Scalar Dice Loss.
        """
        num_classes = inputs.size(1)

        # One-hot encode targets to shape (B, C, H, W)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Apply softmax to inputs to convert logits to probabilities
        inputs_soft = torch.nn.functional.softmax(inputs, dim=1)

        # Compute Dice coefficient for each class
        intersection = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
        union = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Apply weights to Dice scores
        if self.weight is not None:
            assert self.weight.size(0) == num_classes, \
                f"Weight size {self.weight.size(0)} must match number of classes {num_classes}"
            weighted_dice_score = dice_score * self.weight  # Element-wise multiplication
            dice_loss = 1.0 - weighted_dice_score.mean()
        else:
            # Average over batch and classes
            dice_loss = 1.0 - dice_score.mean()

        return dice_loss
