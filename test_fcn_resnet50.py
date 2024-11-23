import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torchvision.models.segmentation import FCN_ResNet50_Weights
from tqdm import tqdm

# Dataset class for loading images and masks from the same folder
class SolarPanelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Load mask
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_mask.png"
        mask_path = os.path.join(self.data_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert('L'))  # Grayscale mask

        # Convert mask to binary format
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = torch.squeeze(mask.float())

        return image, mask

# Transforms for images and masks
def get_transforms(train=False):
    if train:
        return A.Compose([
            # A.Resize(512, 512),
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
            # A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

# Model definition
def get_model(num_classes=1):
    model = models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, num_classes=21)
    # Change the number of output classes
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    return model

# Function for visualizing predictions
def visualize_predictions(image, truth_mask, pred_mask, idx, save_dir=None, set_type='Validation'):
    """
    Vizualizira originalnu sliku, stvarnu masku i predviđenu masku.

    Args:
        image (Tensor): Slika [3, H, W].
        truth_mask (Tensor): Stvarna maska [H, W].
        pred_mask (Tensor): Predviđena maska [1, H, W] ili [H, W].
        idx (int): Indeks slike.
        save_dir (str, optional): Direktorij za spremanje vizualizacija. Defaults to None.
        set_type (str, optional): Tip seta ('Validation' ili 'Test'). Defaults to 'Validation'.
    """
    # Pretvaranje tenzora u NumPy nizove
    image_np = image.permute(1, 2, 0).cpu().numpy()
    # Undo normalizacije
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)

    truth_mask_np = truth_mask.cpu().numpy()
    pred_mask_np = pred_mask.cpu().numpy()

    # Provjera i uklanjanje dodatne dimenzije ako postoji
    if pred_mask_np.ndim == 3 and pred_mask_np.shape[0] == 1:
        pred_mask_np = np.squeeze(pred_mask_np, axis=0)
    elif pred_mask_np.ndim == 3:
        # Ako ima više kanala, možeš odabrati najjači kanal ili drugačije obraditi
        pred_mask_np = np.argmax(pred_mask_np, axis=0)
    elif pred_mask_np.ndim == 2:
        pass  # Već je u pravilnom obliku
    else:
        raise ValueError(f"Unexpected pred_mask shape: {pred_mask_np.shape}")

    # Kreiranje figure
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.imshow(image_np)
    plt.title(f"{set_type} Image #{idx}")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(truth_mask_np, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(pred_mask_np, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{set_type}_image_{idx}.png"))
        plt.close()
    else:
        plt.show()

# Function for creating DataLoaders
def create_dataloaders(train_dir, val_dir, test_dir, batch_size=8, num_workers=4):
    # Transformation definition
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    test_transform = get_transforms(train=False)

    # Dataset creation
    train_dataset = SolarPanelDataset(data_dir=train_dir, transform=train_transform)
    val_dataset = SolarPanelDataset(data_dir=val_dir, transform=val_transform)
    test_dataset = SolarPanelDataset(data_dir=test_dir, transform=test_transform)

    # DataLoaders creation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


# Function for testing the model
def test_model(model, dataloaders, device, save_dir_val=None, save_dir_test=None, num_visualize=5):
    model.eval()
    with torch.no_grad():
        for set_type, loader, save_dir in [('Validation', dataloaders['val'], save_dir_val),
                                          ('Test', dataloaders['test'], save_dir_test)]:
            print(f"Processing {set_type} Set")
            visualize_count = 0
            for batch_idx, (images, masks) in enumerate(loader):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)['out']
                outputs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).float()

                for i in range(images.size(0)):
                    if visualize_count >= num_visualize:
                        break
                    visualize_predictions(images[i], masks[i], preds[i], visualize_count,
                                         save_dir=save_dir, set_type=set_type)
                    visualize_count += 1
                if visualize_count >= num_visualize:
                    break
    print("Testiranje i vizualizacija završeni.")


# Main function for testing the model
def main_test():
    # Dataset paths
    train_images_dir = "dataset/train"  # Možeš ga ignorirati ako testiraš samo na val i test setovima
    val_images_dir = "dataset/valid"
    test_images_dir = "dataset/test"

    # Hyperparameters
    batch_size = 1
    num_workers = 4
    num_visualize = 200  # Broj uzoraka za vizualizaciju

    # DataLoaders
    dataloaders = create_dataloaders(train_images_dir, val_images_dir, test_images_dir,
                                     batch_size=batch_size, num_workers=num_workers)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Koristi uređaj: {device}")

    # Model loading
    num_classes = 1
    model = get_model(num_classes=num_classes)
    model_path = "models/FCN-ResNet50_111.pth"  # Putanja do treniranog modela
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model uspješno učitan.")

    # Create directories for saving visualizations
    save_dir_val = "visualizations/validation"
    save_dir_test = "visualizations/test"

    # Test the model and visualize predictions
    test_model(model, dataloaders, device, save_dir_val=save_dir_val, save_dir_test=save_dir_test, num_visualize=num_visualize)

if __name__ == "__main__":
    main_test()
