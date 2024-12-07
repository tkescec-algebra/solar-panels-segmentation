import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from tqdm import tqdm

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

        return image, mask

# Transforms for images and masks
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.1, shift_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# Intersection over Union (IoU) metric
def calculate_iou(pred, mask, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + mask.sum(dim=(1,2)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# Dice coefficient metric
def calculate_dice(pred, mask, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum(dim=(1,2))
    dice = (2 * intersection + 1e-6) / (pred.sum(dim=(1,2)) + mask.sum(dim=(1,2)) + 1e-6)
    return dice.mean().item()

#### Custom UNet model ####
class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Ako koristimo bilinearno skaliranje
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Ako koristimo transponiranu konvoluciju
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Podešavanje dimenzija ako je potrebno
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //2,
                        diffY //2, diffY - diffY //2])

        # Spajanje skip veze
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Kodirajući put (kontrakcija)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)   # Dimenzije: 128x128
        self.down2 = Down(128, 256)  # Dimenzije: 64x64
        self.down3 = Down(256, 512)  # Dimenzije: 32x32
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # Dimenzije: 16x16

        # Dekodirajući put (ekspanzija)
        self.up1 = Up(1024, 512 // factor, bilinear)  # Dimenzije: 32x32
        self.up2 = Up(512, 256 // factor, bilinear)   # Dimenzije: 64x64
        self.up3 = Up(256, 128 // factor, bilinear)   # Dimenzije: 128x128
        self.up4 = Up(128, 64, bilinear)              # Dimenzije: 256x256
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # Ulazni sloj
        x2 = self.down1(x1)   # Sloj 1
        x3 = self.down2(x2)   # Sloj 2
        x4 = self.down3(x3)   # Sloj 3
        x5 = self.down4(x4)   # Bottleneck

        x = self.up1(x5, x4)  # UpSample 1
        x = self.up2(x, x3)   # UpSample 2
        x = self.up3(x, x2)   # UpSample 3
        x = self.up4(x, x1)   # UpSample 4
        logits = self.outc(x) # Izlazni sloj
        return logits
#### Custom UNet model ####

def get_model(num_classes=1):
    model = UNet(n_channels=3, n_classes=num_classes, bilinear=True)
    return model

# Training function for one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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

            outputs = model(images)
            outputs = outputs.squeeze(1)  # [B, H, W]

            loss = criterion(outputs, masks)

            epoch_loss += loss.item()
            epoch_iou += calculate_iou(outputs, masks)
            epoch_dice += calculate_dice(outputs, masks)

    avg_iou = epoch_iou / len(dataloader)
    avg_dice = epoch_dice / len(dataloader)
    avg_loss = epoch_loss / len(dataloader)

    return avg_iou, avg_dice, avg_loss

def main():
    # Initialize Weights & Biases
    wandb.init(
        project="Solar-Panel-Segmentation",
        entity="tomislav-kescec-algebra",
        config={
            "name": "Custom-UNet",
            "learning_rate": 0.00001,  # Adjusted learning rate for Adam
            "epochs": 50,
            "batch_size": 32,  # Adjusted batch size
            "optimizer": "Adam",
            "weight_decay": 0.0001,  # Adjusted weight decay
            "model": "Custom-UNet",
            "num_classes": 1,  # Binary segmentation
            "loss": "BCEWithLogitsLoss"
        }
    )

    # Paths to training and validation datasets
    train_data_dir = "../datasets/dataset/train"
    val_data_dir = "../datasets/dataset/valid"

    # Dataset i DataLoader
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # Load datasets
    train_dataset = SolarPanelDataset(data_dir=train_data_dir, transforms=train_transform)
    val_dataset = SolarPanelDataset(data_dir=val_data_dir, transforms=val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

    # Initialize model, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=wandb.config.num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate, weight_decay=wandb.config.weight_decay)

    # Optionally, initialize scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # Watch the model with W&B
    wandb.watch(model, log="all", log_freq=10)

    best_val_iou = 0

    # Training and validation loop
    for epoch in range(wandb.config.epochs):
        print(f"Epoch {epoch + 1}/{wandb.config.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_iou, val_dice, val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

        # Save the best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            if not os.path.exists("../models"):
                os.makedirs("../models")
            torch.save(model.state_dict(), f"models/{wandb.config.name}_best_model.pth")
            print("Saved Best Model")

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice
        })

        # Optionally, step the scheduler
        # scheduler.step(train_loss)

    # Finish the W&B run
    wandb.finish()

if __name__ == '__main__':
    import multiprocessing
    import sys

    # For Windows compatibility
    multiprocessing.freeze_support()
    main()
