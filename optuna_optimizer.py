import os
import wandb
import torch
import optuna
import optuna.visualization as vis

from torch.utils.data import DataLoader

from utils.dataset import SolarPanelDataset6C, SolarPanelDataset3C
from utils.transforms import get_transforms
from utils.helpers import (
    get_model,
    train_one_epoch,
    validate_dice_iou,
    get_criterion,
    get_optimizer,
    get_scheduler,
)

def prepare_data(n_channels, batch_size):
    # Odabir klase dataset-a na temelju broja kanala
    if n_channels == 1:
        DatasetClass = SolarPanelDataset3C
        grayscale = True
    elif n_channels == 3:
        DatasetClass = SolarPanelDataset3C
        grayscale = False
    elif n_channels == 6:
        DatasetClass = SolarPanelDataset6C
        grayscale = False
    else:
        raise ValueError("Ne podržani broj kanala")

    # Putanje do trening i validacijskih podataka
    train_data_dir = "datasets/reduced_dataset/train"
    val_data_dir = "datasets/reduced_dataset/valid"

    # Transformacije
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    # Učitavanje dataset-a
    train_dataset = DatasetClass(data_dir=train_data_dir, transforms=train_transform, grayscale=grayscale, channels=n_channels)
    val_dataset = DatasetClass(data_dir=val_data_dir, transforms=val_transform, grayscale=grayscale, channels=n_channels)

    # Kreiranje DataLoader-a
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def objective(trial):
    # Sugeriranje hiperparametara
    n_channels = trial.suggest_categorical('in_channels', [1, 3, 6])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    scheduler_name = trial.suggest_categorical('scheduler', ['CosineAnnealingLR', 'StepLR', 'None'])
    epochs = trial.suggest_int('epochs', 30, 50)

    # Generiranje dinamičkog imena runa
    run_name = f"Trial-{trial.number}_Channels-{n_channels}_LR-{learning_rate:.1e}_WD-{weight_decay:.1e}"

    # Inicijalizacija wandb za ovaj trial
    wandb.init(
        project="Solar-Panel-Segmentation-V1",
        entity="tomislav-kescec-algebra",
        config={
            "name": "FCN-ResNet50",
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam",
            "weight_decay": weight_decay,
            "model": "fcn_resnet50",
            "num_classes": 1,  # Binary segmentation
            "loss": "ComboLoss(BCE + Dice)",
            "width": 256,
            "height": 256,
            "scheduler": scheduler_name,
            "in_channels": n_channels
        },
        group='optuna_trials',
        name=run_name,  # Postavljanje imena runa
        reinit=True
    )

    # Priprema podataka
    train_loader, val_loader = prepare_data(n_channels, batch_size)

    # Inicijalizacija modela, loss funkcije, optimizatora i schedulera
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        num_classes=1,
        model_name="fcn_resnet50",
        in_channels=n_channels
    ).to(device)

    # Inicijalizacija optimizatora
    optimizer = get_optimizer(
        optimizer_name=wandb.config.optimizer,
        model=model,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Inicijalizacija schedulera
    scheduler = get_scheduler(optimizer, scheduler_name=scheduler_name) if scheduler_name != 'None' else None

    # Definicija loss funkcije
    criterion = get_criterion(criterion_name="ComboLoss")

    # Praćenje modela s wandb
    wandb.watch(model, log="all", log_freq=10)

    best_val_iou = 0

    # Trening i validacija
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_iou, val_dice, val_loss = validate_dice_iou(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

        # Spremanje najboljeg modela
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            #torch.save(model.state_dict(), f"models/fcn_resnet50_best_model_trial_{trial.number}.pth")
            # print("Saved Best Model")

        # Logiranje metrika
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice
        })

        # Reportiranje metrika Optuni za pruning
        trial.report(val_iou, epoch)

        # Provjera treba li prekinuti trial
        if trial.should_prune():
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

        # Korak schedulera
        if scheduler:
            scheduler.step()

    # Završetak wandb runa
    wandb.finish()

    return best_val_iou  # Optuna će maksimizirati ovu vrijednost


def main():
    # Kreiranje Optuna studije s prunerom
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', study_name='SolarPanelSegmentationOptimization', pruner=pruner)

    # Pokretanje optimizacije
    study.optimize(objective, n_trials=50, timeout=18000)  # prilagoditi n_trials i timeout prema potrebama, moguće je i koristiti n_jobs=4 za paralelizaciju

    # Ispis najboljeg triala
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    plot_visualization_folder = "visualizations/optuna/plots"
    html_visualization_folder = "visualizations/optuna/html"

    # Generiranje vizualizacija
    optimization_history = vis.plot_optimization_history(study)
    param_importances = vis.plot_param_importances(study)
    slice_plot = vis.plot_slice(study)
    contour_plot = vis.plot_contour(study)

    # Spremanje vizualizacija kao slike (PNG)
    optimization_history.write_image(f"{plot_visualization_folder}/optimization_history.png")
    param_importances.write_image(f"{plot_visualization_folder}/param_importances.png")
    slice_plot.write_image(f"{plot_visualization_folder}/slice_plot.png")
    contour_plot.write_image(f"{plot_visualization_folder}/contour_plot.png")

    # Opcionalno: Spremanje vizualizacija kao HTML datoteke
    optimization_history.write_html(f"{html_visualization_folder}/optimization_history.html")
    param_importances.write_html(f"{html_visualization_folder}/param_importances.html")
    slice_plot.write_html(f"{html_visualization_folder}/slice_plot.html")
    contour_plot.write_html(f"{html_visualization_folder}/contour_plot.html")

    # Opcionalno: Logiranje vizualizacija u W&B kao artefakti
    wandb.init(project="Solar-Panel-Segmentation-V1", entity="tomislav-kescec-algebra")
    wandb.log({
        "optimization_history": wandb.Image("optimization_history.png"),
        "param_importances": wandb.Image("param_importances.png"),
        "slice_plot": wandb.Image("slice_plot.png"),
        "contour_plot": wandb.Image("contour_plot.png"),
    })


if __name__ == '__main__':
    import multiprocessing

    # Za kompatibilnost s Windowsom
    multiprocessing.freeze_support()

    main()
