import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import SolarPanelDataset3C
from utils.transforms import get_transforms
from utils.helpers import get_model

# Function for visualizing predictions
def visualize_predictions(image, truth_mask, pred_mask, image_id, save_dir=None, set_type='Validation'):
    """
    Vizualizira originalnu sliku, stvarnu masku i predviđenu masku.

    Args:
        image (Tensor): Slika [3, H, W].
        truth_mask (Tensor): Stvarna maska [H, W].
        pred_mask (Tensor): Predviđena maska [1, H, W] ili [H, W].
        image_id (str): Identifikator slike (prvi dio originalnog imena).
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
    plt.title(f"{set_type} Image {image_id}")
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
        plt.savefig(os.path.join(save_dir, f"{set_type}_image_{image_id}.png"))
        plt.close()
    else:
        plt.show()

# Function for creating DataLoaders
def create_dataloaders(test_dir, batch_size=8, num_workers=4):
    # Transformation definition
    test_transform = get_transforms(train=False)

    # Dataset creation
    test_dataset = SolarPanelDataset3C(data_dir=test_dir, transforms=test_transform, grayscale=True)

    # DataLoaders creation
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'test': test_loader}

# Function for calculating Intersection over Union (IoU)
def calculate_iou(truth_mask, pred_mask):
    intersection = np.logical_and(truth_mask, pred_mask).sum()
    union = np.logical_or(truth_mask, pred_mask).sum()
    if union == 0:
        return np.nan
    return intersection / union

# Function for 00_explore_random_ideas the model
def test_model(model, dataloaders, device, save_dir_test=None, num_visualize=5, camera_heights=None):
    model.eval()
    iou_per_height = defaultdict(list)

    with torch.no_grad():
        for set_type, loader, save_dir in [('Test', dataloaders['test'], save_dir_test)]:
            print(f"Processing {set_type} Set")
            visualize_count = 0
            for batch_idx, (images, masks, filenames) in enumerate(loader):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)['out']
                outputs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).float()

                for i in range(images.size(0)):
                    if visualize_count < num_visualize:
                        filename = filenames[i]
                        image_id = filename.split('_')[0]
                        visualize_predictions(images[i], masks[i], preds[i], image_id,
                                             save_dir=save_dir, set_type=set_type)
                        visualize_count += 1

                    # Calculate IoU
                    truth_mask_np = masks[i].cpu().numpy().astype(bool)
                    pred_mask_np = preds[i].cpu().numpy().astype(bool)
                    iou = calculate_iou(truth_mask_np, pred_mask_np)

                    # Extract height from filename
                    if camera_heights:
                        try:
                            height_str = filename.split('_')[0]
                            height = int(height_str)
                            if height in camera_heights:
                                iou_per_height[height].append(iou)
                            else:
                                print(f"Unknown height {height} for file {filename}")
                        except (IndexError, ValueError) as e:
                            print(f"Error extracting height from {filename}: {e}")


            if visualize_count >= num_visualize:
                break
    print("Testiranje i vizualizacija završeni.")
    return iou_per_height

# Function for plotting IoU per height for multiple datasets
def plot_multiple_iou_per_height(datasets_iou, datasets_heights, labels, colors):
    """
    Kreira graf IoU vrijednosti po visini kamere za više skupina podataka.

    Args:
        datasets_iou (list of dict): Lista rječnika s IoU vrijednostima po visini za svaku skupinu.
        datasets_heights (list of list): Lista lista visina za svaku skupinu.
        labels (list of str): Lista oznaka za svaku skupinu.
        colors (list of str): Lista boja za svaku skupinu.
    """
    plt.figure(figsize=(12, 8))

    for i, (iou_per_height, heights_ordered, label, color) in enumerate(zip(datasets_iou, datasets_heights, labels, colors)):
        average_iou = []
        for height in heights_ordered:
            ious = [iou for iou in iou_per_height.get(height, []) if not np.isnan(iou)]
            if ious:
                avg = np.mean(ious)
            else:
                avg = np.nan
            average_iou.append(avg)

        plt.plot(heights_ordered, average_iou, marker='o', linestyle='-', label=label, color=color)

        # Dodavanje anotacija za svaku točku
        for height, iou in zip(heights_ordered, average_iou):
            if not np.isnan(iou):
                plt.text(height, iou + 0.01, str(height), fontsize=9, ha='center', va='bottom', color=color)

    plt.title("Average IoU per Camera Height - FCN-ResNet50_02")
    plt.xlabel("Camera Height (m)")
    plt.ylabel("Average IoU")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xlim(0, 3000)
    plt.ylim(0, 1)

    # Kombinirajte sve visine za X-tickse i sortirajte ih
    # combined_heights = sorted(set([h for heights in datasets_heights for h in heights]))
    # plt.xticks(combined_heights, labels=[str(h) for h in combined_heights], rotation=45)

    plt.legend()
    plt.tight_layout()
    # Save plot
    plt.savefig("visualizations/iou_per_height_01.png")

    plt.show()

# Main function for 00_explore_random_ideas the model
def main_test(images_dir, camera_heights = None):
    # Dataset paths
    test_images_dir = f"dataset_000/{images_dir}"

    # Hyperparameters
    batch_size = 1
    num_workers = 4
    num_visualize = 200  # Broj uzoraka za vizualizaciju

    # DataLoaders
    dataloaders = create_dataloaders(test_images_dir,
                                     batch_size=batch_size, num_workers=num_workers)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Koristi uređaj: {device}")

    # Model loading
    num_classes = 1
    model = get_model(num_classes=num_classes)
    model_path = "models/FCN-ResNet50_01_rgb.pth"  # Putanja do treniranog modela
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model uspješno učitan.")

    # Create directories for saving visualizations
    save_dir_test = f"visualizations/{images_dir}/02"

    # Test the model and visualize predictions
    iou_per_height  = test_model(model, dataloaders, device, save_dir_test=save_dir_test, num_visualize=num_visualize, camera_heights=camera_heights)

    return iou_per_height

if __name__ == "__main__":
    roofs_height = [67, 93, 132, 186, 258, 357, 490, 632, 870, 1216]
    lands_height = [66, 91, 152, 211, 290, 404, 555, 764, 1053, 1453, 2000, 2740]

    # Testiranje modela na 'roofs' skupini
    print("Testing on 'roofs' dataset...")
    iou_roofs = main_test('test_custom/roofs', camera_heights=roofs_height)

    # Testiranje modela na 'lands' skupini
    print("Testing on 'lands' dataset...")
    iou_lands = main_test('test_custom/lands', camera_heights=lands_height)

    # Priprema podataka za plotanje
    datasets_iou = [iou_roofs, iou_lands]
    datasets_heights = [roofs_height, lands_height]
    labels = ['Roofs', 'Lands']
    colors = ['blue', 'green']

    # Plotanje rezultata na istom grafu
    plot_multiple_iou_per_height(datasets_iou, datasets_heights, labels, colors)
