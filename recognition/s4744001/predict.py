import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from modules import UNet
from dataset import get_oasis_dataloaders
from train import evaluate 

class Config:
    BASE_DATA_PATH = r"C:\Users\andre\OneDrive\Desktop\Uni\COMP3710\Oasis Dataset"
    NUM_CLASSES = 4
    IN_CHANNELS = 1
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "trained_model.pt"
    LOG_PATH = "training_logs.pth"
    NUM_VISUALS = 5

@torch.no_grad()
def visualize_predictions(model, loader, device, num_visuals=5, out_prefix="output"):
    model.eval()
    shown = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        imgs = imgs.cpu().numpy()
        masks = masks.cpu().numpy()

        for i in range(len(imgs)):
            if shown >= num_visuals:
                return
            img = imgs[i, 0]
            mask = masks[i]
            pred = preds[i]

            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

            fig, axes = plt.subplots(1, 3, figsize=(10, 4))
            axes[0].imshow(img_norm, cmap='gray')
            axes[0].set_title("Input Image")
            axes[1].imshow(mask, cmap='viridis', vmin=0, vmax=model.num_classes - 1)
            axes[1].set_title("Ground Truth")
            axes[2].imshow(pred, cmap='viridis', vmin=0, vmax=model.num_classes - 1)
            axes[2].set_title("Prediction")

            for ax in axes:
                ax.axis('off')

            plt.tight_layout()
            out_path = f"{out_prefix}_sample{shown+1}.png"
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"[Saved] {out_path}")
            shown += 1


def plot_curves(train_losses, val_losses, train_dices, val_dices,
                loss_path="training_curves_loss.png",
                dice_path="training_curves_dice.png"):
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(train_dices, label="train_meanDice")
    plt.plot(val_dices, label="val_meanDice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Mean Dice vs Epoch")
    plt.legend()
    plt.savefig(dice_path, dpi=200)
    plt.close()

    print(f"[Saved] {loss_path} and {dice_path}")


def safe_plot_from_logs(log_path):
    if not os.path.exists(log_path):
        print(f"[Warn] No log file found at {log_path}, skipping curve plots.")
        return

    logs = torch.load(log_path, map_location="cpu")
    required = ["train_losses", "val_losses", "train_dices", "val_dices"]

    if not all(k in logs for k in required):
        print("[Warn] Log file missing keys. Found keys:", list(logs.keys()))
        return

    print("[Info] Plotting training curves from saved logs...")
    plot_curves(
        train_losses=logs["train_losses"],
        val_losses=logs["val_losses"],
        train_dices=logs["train_dices"],
        val_dices=logs["val_dices"],
    )

def main():
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    print(f"[Device] Using {device}")

    _, val_loader, test_loader = get_oasis_dataloaders(
        base_path=cfg.BASE_DATA_PATH,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        resize_hw=cfg.IMG_SIZE,
    )


    model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES, 64).to(device)
    model.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    model.num_classes = cfg.NUM_CLASSES
    print(f"[Loaded] Model weights from {cfg.MODEL_PATH}")

    criterion = nn.CrossEntropyLoss()

    val_loss, val_dice, val_per_class = evaluate(model, val_loader, criterion, device, cfg.NUM_CLASSES)
    test_loss, test_dice, test_per_class = evaluate(model, test_loader, criterion, device, cfg.NUM_CLASSES)

    print(f"[Val]  Loss={val_loss:.4f}  MeanDice={val_dice:.4f}  PerClass={val_per_class.numpy()}")
    print(f"[Test] Loss={test_loss:.4f}  MeanDice={test_dice:.4f}  PerClass={test_per_class.numpy()}")

    print("\n[Visualizing outputs on validation set...]")
    visualize_predictions(model, val_loader, device, num_visuals=cfg.NUM_VISUALS)

    print("\n[Loading and plotting training curves from training_logs.pth]")
    safe_plot_from_logs(cfg.LOG_PATH)


if __name__ == "__main__":
    main()
