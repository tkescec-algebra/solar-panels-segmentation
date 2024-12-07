import torch
from torchvision.models.segmentation import FCN_ResNet50_Weights
import segmentation_models_pytorch as smp

import torch.nn as nn

from torchvision import models
from tqdm import tqdm

from utils.dataset import SolarPanelDataset3C, SolarPanelDataset6C


# Dice loss function
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1e-5
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return dice

# Custom loss function
class ComboLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super(ComboLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)

        inputs = torch.sigmoid(inputs)

        smooth = 1e-5
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss

# Intersection over Union (IoU) metric
def calculate_iou(pred, mask, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + mask.sum(dim=(1,2)) - (pred * mask).sum(dim=(1,2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# Dice coefficient metric
def calculate_dice(pred, mask, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum(dim=(1,2))
    dice = (2 * intersection + 1e-6) / (pred.sum(dim=(1,2)) + mask.sum(dim=(1,2)) + 1e-6)
    return dice.mean().item()

# Model definition
def get_model(num_classes=1, model_name='fcn_resnet50', in_channels=3):
    if model_name == 'fcn_resnet50':
        model = models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, num_classes=21)
        # Get the original number of input channels
        original_conv = model.backbone.conv1

        if in_channels == 1:
            # Change the number of input channels
            model.backbone.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            nn.init.kaiming_normal_(model.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
            print(in_channels)

        elif in_channels == 6:
            new_conv = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )

            # Kopiranje težina iz originalnih 3 kanala u nove
            new_conv.weight.data[:, :3, :, :] = original_conv.weight.data
            # Za preostalih 3 kanala, možete inicijalizirati na nulu ili na isti način kao originalne
            # new_conv.weight.data[:, 3:, :, :] = original_conv.weight.data[:, :3, :, :]
            new_conv.weight.data[:, 3:, :, :].zero_()

            model.backbone.conv1 = new_conv

        # Change the number of output classes
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif model_name == 'resnet34':
        model = smp.Unet('resnet34', in_channels=in_channels, classes=num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported")

    return model

# Loss function definition
def get_criterion(criterion_name='ComboLoss', **kwargs):
    if criterion_name == 'ComboLoss':
        return ComboLoss(bce_weight=1.0, dice_weight=1.0)
    elif criterion_name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif criterion_name == 'DiceLoss':
        return DiceLoss()
    else:
        raise ValueError(f"Criterion {criterion_name} not supported")

# Optimizer definition
def get_optimizer(optimizer_name, model, lr, weight_decay):
    if optimizer_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")

# Scheduler definition
def get_scheduler(optimizer, scheduler_name):
    if scheduler_name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    elif scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported")

def get_dataset(channels, data_dir, transforms, grayscale=True):
    if channels == 1 or channels == 3:
        return SolarPanelDataset3C(data_dir=data_dir, transforms=transforms, grayscale=grayscale)
    elif channels == 6:
        return SolarPanelDataset6C(data_dir=data_dir, transforms=transforms)
    else:
        raise ValueError(f"Channels {channels} not supported")

# Training function for one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for images, masks, filenames in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        outputs = outputs.squeeze(1)  # [B, H, W]

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)

    return avg_loss

# Validation function for one epoch
def validate_dice_iou(model, dataloader, criterion, device):
    model.eval()
    epoch_iou = 0
    epoch_dice = 0
    epoch_loss = 0
    with torch.no_grad():
        for images, masks, filenames in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            outputs = outputs.squeeze(1)  # [B, H, W]

            loss = criterion(outputs, masks)

            epoch_loss += loss.item()
            epoch_iou += calculate_iou(outputs, masks)
            epoch_dice += calculate_dice(outputs, masks)

    avg_loss = epoch_loss / len(dataloader)
    avg_iou = epoch_iou / len(dataloader)
    avg_dice = epoch_dice / len(dataloader)

    return avg_iou, avg_dice, avg_loss