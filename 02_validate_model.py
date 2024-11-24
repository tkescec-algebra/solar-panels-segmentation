import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils.modules import SolarPanelDataset, get_transforms, get_model

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
def create_dataloaders(val_dir, test_dir, batch_size=8, num_workers=4):
    # Transformation definition
    val_transform = get_transforms(train=False)
    test_transform = get_transforms(train=False)

    # Dataset creation
    val_dataset = SolarPanelDataset(data_dir=val_dir, transforms=val_transform)
    test_dataset = SolarPanelDataset(data_dir=test_dir, transforms=test_transform)

    # DataLoaders creation
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'val': val_loader, 'test': test_loader}


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
    val_images_dir = "dataset/valid"
    test_images_dir = "dataset/test"

    # Hyperparameters
    batch_size = 1
    num_workers = 4
    num_visualize = 200  # Broj uzoraka za vizualizaciju

    # DataLoaders
    dataloaders = create_dataloaders(val_images_dir, test_images_dir,
                                     batch_size=batch_size, num_workers=num_workers)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Koristi uređaj: {device}")

    # Model loading
    num_classes = 1
    model = get_model(num_classes=num_classes)
    model_path = "models/FCN-ResNet50_112.pth"  # Putanja do treniranog modela
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
