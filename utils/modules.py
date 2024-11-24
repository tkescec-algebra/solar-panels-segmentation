import os
import numpy as np
import torch
from torchvision.models.segmentation import FCN_ResNet50_Weights

import albumentations as A
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torchvision import models
from tqdm import tqdm


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

# Dataset class for loading images and masks from the same folder
class SolarPanelDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load mask
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_mask.png"
        mask_path = os.path.join(self.data_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert("L"))

        # Convert mask to binary format
        mask = np.where(mask > 0, 1, 0).astype(np.float32)  # Binary mask

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = torch.squeeze(mask.float())

        return image, mask, img_name

# Transforms for images and masks
def get_transforms(train=True):
    if train:
        return A.Compose([
            #A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.1, shift_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            #A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

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
def get_model(num_classes=1):
    model = models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, num_classes=21)
    # model = models.segmentation.fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT, num_classes=21)

    # Change the number of output classes
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    return model

# Training function for one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
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
def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_iou = 0
    epoch_dice = 0
    epoch_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
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