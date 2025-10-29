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