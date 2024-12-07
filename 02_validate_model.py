import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import SolarPanelDataset3C, SolarPanelDataset6C
from utils.transforms import get_transforms
from utils.helpers import get_model


# Konfiguracija 6 modela s display_name atributima
models_config = [
    {
        "model_path": "models/FCN-ResNet50_1channel.pth",
        "in_channels": 1,
        "dataset_cls": SolarPanelDataset3C,
        "grayscale": True,
        "model_name": "fcn_resnet50",
        "display_name": "FCN-ResNet50_1channel"
    },
    {
        "model_path": "models/Unet-ResNet34_1channel.pth",
        "in_channels": 1,
        "dataset_cls": SolarPanelDataset3C,
        "grayscale": True,
        "model_name": "resnet34",
        "display_name": "Unet-ResNet34_1channel"
    },
    {
        "model_path": "models/FCN-ResNet50_3channels.pth",
        "in_channels": 3,
        "dataset_cls": SolarPanelDataset3C,
        "grayscale": False,
        "model_name": "fcn_resnet50",
        "display_name": "FCN-ResNet50_3channels"
    },
    {
        "model_path": "models/Unet-ResNet34_3channels.pth",
        "in_channels": 3,
        "dataset_cls": SolarPanelDataset3C,
        "grayscale": False,
        "model_name": "resnet34",
        "display_name": "Unet-ResNet34_3channels"
    },
    {
        "model_path": "models/FCN-ResNet50_6channels.pth",
        "in_channels": 6,
        "dataset_cls": SolarPanelDataset6C,
        "grayscale": False,
        "model_name": "fcn_resnet50",
        "display_name": "FCN-ResNet50_6channels"
    },
    {
        "model_path": "models/Unet-ResNet34_6channels.pth",
        "in_channels": 6,
        "dataset_cls": SolarPanelDataset6C,
        "grayscale": False,
        "model_name": "resnet34",
        "display_name": "Unet-ResNet34_6channels"
    }
]

def visualize_multiple_predictions(image, truth_mask, pred_masks, model_names, idx, save_dir=None):
    """
    Vizualizira originalnu sliku, stvarnu masku i 6 predviđenih maski.
    pred_masks je lista od 6 tenzora [H, W].
    model_names je lista stringova s imenima modela.
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    # Undo normalizacije
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)

    truth_mask_np = truth_mask.cpu().numpy()
    pred_masks_np = []
    for pm in pred_masks:
        pm_np = pm.cpu().numpy()
        if pm_np.ndim == 3 and pm_np.shape[0] == 1:
            pm_np = np.squeeze(pm_np, axis=0)
        elif pm_np.ndim == 3:
            pm_np = np.argmax(pm_np, axis=0)
        pred_masks_np.append(pm_np)

    fig, axes = plt.subplots(1, 2 + len(pred_masks_np), figsize=(4*(2+len(pred_masks_np)), 5))

    axes[0].imshow(image_np)
    axes[0].set_title(f"Original #{idx}")
    axes[0].axis('off')

    axes[1].imshow(truth_mask_np, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    for i, pm_np in enumerate(pred_masks_np):
        axes[i+2].imshow(pm_np, cmap='gray')
        axes[i+2].set_title(model_names[i])
        axes[i+2].axis('off')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"Train_image_{idx}.png"))
        plt.close()
    else:
        plt.show()


def create_loader(data_dir, dataset_cls, grayscale, in_channels, batch_size=1, num_workers=4, shuffle=False):
    transform = get_transforms(train=False)
    dataset = dataset_cls(data_dir=data_dir, transforms=transform, grayscale=grayscale, channels=in_channels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def load_model(model_path, model_name, in_channels, device):
    num_classes = 1
    model = get_model(num_classes=num_classes, model_name=model_name, in_channels=in_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main_visualization():
    # Uzet ćemo test slike
    train_images_dir = "datasets/dataset/test"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Koristi uređaj: {device}")

    # Učitavamo 6 modela i njihovih 6 loadera
    models = []
    loaders = []
    for cfg in models_config:
        mdl = load_model(cfg['model_path'], cfg['model_name'], cfg['in_channels'], device)
        models.append((mdl, cfg['display_name']))
        loader = create_loader(train_images_dir, cfg['dataset_cls'], cfg['grayscale'], cfg['in_channels'])
        loaders.append(loader)

    # Pretvaramo loadere u liste kako bi mogli indeksirati svaki 10-ti primjerak
    all_data = [list(l) for l in loaders]

    # Uzimamo svaki 10-ti indeks
    indices = range(0, len(all_data[0]), 10)

    save_dir = "visualizations/test"

    for idx in indices:
        batches = [data[idx] for data in all_data]
        # Uzimamo referentnu sliku i masku iz prvog batcha
        ref_images, ref_masks, ref_filenames = batches[0]

        ref_images = ref_images.to(device)
        ref_masks = ref_masks.to(device)

        pred_masks = []
        model_names = [m[1] for m in models]

        for i, (images, masks, filenames) in enumerate(batches):
            images = images.to(device)
            with torch.no_grad():
                if models_config[i]['model_name'] == 'resnet34':
                    # Unet - pretpostavka da je implementacija takva da izlaz dobijemo direktno
                    outputs = models[i][0](images)
                else:
                    # FCN - koristimo ['out']
                    outputs = models[i][0](images)['out']
                outputs = torch.sigmoid(outputs)
                preds = (outputs > 0.5).float()
            pred_masks.append(preds[0])

        visualize_multiple_predictions(ref_images[0], ref_masks[0], pred_masks, model_names, idx, save_dir=save_dir)

    print("Vizualizacija završena.")


if __name__ == "__main__":
    main_visualization()
