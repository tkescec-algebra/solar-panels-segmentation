import os
import wandb
import torch

from torch.utils.data import DataLoader

from utils.dataset import SolarPanelDataset6C, SolarPanelDataset3C
from utils.transforms import get_transforms
from utils.helpers import get_model, train_one_epoch, validate_dice_iou, get_criterion, get_optimizer, get_scheduler


GRAYSCALE = True
IN_CHANNELS = 3
DATASET = SolarPanelDataset3C
WANDB_MODE = "online"  # Change to "online" to log to W&B

# Main function
def train(retrain=False):
    os.environ["WANDB_MODE"] = WANDB_MODE

    # Initialize Weights & Biases
    wandb.init(
        project="Solar-Panel-Segmentation",
        entity="tomislav-kescec-algebra",
        config={
            "name": "FCN-ResNet50",
            "learning_rate": 0.0001,  # Adjusted learning rate for Adam
            "epochs": 50,
            "batch_size": 32,  # Adjusted batch size
            "optimizer": "Adam",
            "weight_decay": 0.0001,  # Adjusted weight decay
            "model": "fcn_resnet50",
            "num_classes": 1,  # Binary segmentation
            "loss": "ComboLoss(BCE + Dice)",
            "width": 256,
            "height": 256,
            "scheduler": "StepLR",
            "in_channels": IN_CHANNELS
        }
    )

    # Paths to training and validation datasets
    train_data_dir = "datasets/dataset/train"
    val_data_dir = "datasets/dataset/valid"

    # Dataset i DataLoader
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # Load datasets
    train_dataset = DATASET(data_dir=train_data_dir, transforms=train_transform, grayscale=GRAYSCALE, channels=IN_CHANNELS)
    val_dataset = DATASET(data_dir=val_data_dir, transforms=val_transform, grayscale=GRAYSCALE, channels=IN_CHANNELS)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

    # Initialize model, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        num_classes=wandb.config.num_classes,
        model_name=wandb.config.model,
        in_channels=wandb.config.in_channels
    ).to(device)

    # Load model weights if available
    if retrain:
        model.load_state_dict(torch.load('models/FCN-ResNet50_112.pth'))

    # Loss function
    criterion = get_criterion(criterion_name="ComboLoss")

    # Initialize optimizer
    optimizer = get_optimizer(
        optimizer_name=wandb.config.optimizer,
        model=model,
        lr=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay
    )

    # Optionally, initialize scheduler
    scheduler = get_scheduler(optimizer, scheduler_name=wandb.config.scheduler)

    # Watch the model with W&B
    wandb.watch(model, log="all", log_freq=10)

    best_val_iou = 0

    # Training and validation loop
    for epoch in range(wandb.config.epochs):
        print(f"Epoch {epoch + 1}/{wandb.config.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_iou, val_dice, val_loss = validate_dice_iou(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

        # Save the best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), f"models/{wandb.config.name}_best_model.pth")
            print("Saved Best Model")

        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice
        })

        if epoch > 10:
            scheduler.step()

    # Finish the W&B run
    wandb.finish()

if __name__ == '__main__':
    import multiprocessing

    # For Windows compatibility
    multiprocessing.freeze_support()
    train(retrain=False)