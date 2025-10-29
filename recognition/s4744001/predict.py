import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from modules import UNet, dice_per_class
from dataset import get_oasis_dataloaders

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_dice = 0, 0
    dice_sums = None
    n_batches = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        per_class = dice_per_class(logits, masks)
        total_dice += per_class.mean().item()

        if dice_sums is None:
            dice_sums = per_class.clone().detach()
        else:
            dice_sums += per_class.detach()

        n_batches += 1

    return (
        total_loss / n_batches,
        total_dice / n_batches,
        (dice_sums / n_batches).cpu()
    )


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