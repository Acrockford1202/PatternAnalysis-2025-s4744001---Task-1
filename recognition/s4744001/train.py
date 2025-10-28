import torch
import torch.nn as nn
from modules import UNet
from dataset import get_oasis_dataloaders

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes, log_interval=10):
    model.train()
    total_loss = 0
    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            print(f"[Train] Batch {batch_idx+1}/{len(loader)} Loss={loss.item():.4f}")
    return total_loss / len(loader)
