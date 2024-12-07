import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.edge_detection import sobel_edge_detection


# Dataset class for loading images in 3 channels or 1 channel and masks from the same folder
class SolarPanelDataset3C(Dataset):
    def __init__(self, data_dir, transforms=None, grayscale=False, channels=3):
        self.data_dir = data_dir
        self.transforms = transforms
        self.grayscale = grayscale
        self.channels = channels
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

        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.channels == 3:
                image = np.stack([image] * 3, axis=-1) # Pretvorba u 3-kanalni format
            # grayscale_transform = Compose([ToGray(p=1)])
            # image = grayscale_transform(image=image)['image']

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = torch.squeeze(mask.float())

        return image, mask, img_name

# Dataset class for loading images in 6 channels and masks from the same folder
class SolarPanelDataset6C(Dataset):
    def __init__(self, data_dir, transforms=None, grayscale=False, channels=6):
        self.data_dir = data_dir
        self.transforms = transforms
        self.images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate Sobel edges
        sobel_edges = sobel_edge_detection(image)  # Lista od 3 (H, W, 1)

        # Stack Sobel edges into a single array
        sobel_edges = np.concatenate(sobel_edges, axis=2)  # (H, W, 3)

        # Combine original image and Sobel edges
        combined = np.concatenate([image, sobel_edges], axis=2)  # (H, W, 6)

        # Load mask
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_mask.png"
        mask_path = os.path.join(self.data_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert("L"))

        # Convert mask to binary format
        mask = np.where(mask > 0, 1, 0).astype(np.float32)  # Binary mask

        if self.transforms:
            augmented = self.transforms(image=combined, mask=mask)
            combined = augmented['image']  # Još uvek u HWC formatu
            mask = augmented['mask']

        # Normalizacija na [0,1] ako nije već urađena u transformacijama
        if not self.transforms:
            combined = combined.astype(np.float32) / 255.0

        # Transpose to (C, H, W) ako koristite ToTensorV2, ovo je već urađeno
        # Ako koristite ToTensorV2, image je već tensor
        if self.transforms:
            image_tensor = combined  # Već je tensor zbog ToTensorV2
        else:
            combined = combined.transpose((2, 0, 1))  # (C, H, W)
            # image_tensor = torch.tensor(combined).float()
            image_tensor = torch.squeeze(combined.float())

        # Convert mask to tensor
        mask_tensor = torch.squeeze(mask.float())  # Shape: (H, W)

        return image_tensor, mask_tensor, img_name