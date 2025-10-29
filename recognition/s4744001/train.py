import torch
import torch.nn as nn
from modules import UNet, dice_per_class
from dataset import get_oasis_dataloaders

class Config:
    BASE_DATA_PATH = r"C:\Users\andre\OneDrive\Desktop\Uni\COMP3710\Oasis Dataset"
    NUM_CLASSES = 4
    IN_CHANNELS = 1
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 20
    PRINT_EVERY = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "trained_model.pt"
    TRAIN_LOG_PATH = "training_logs.pth"


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

def main():
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    print(f"[Device] Using {device}")

    train_loader, val_loader, test_loader = get_oasis_dataloaders(
        base_path=cfg.BASE_DATA_PATH,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        resize_hw=cfg.IMG_SIZE,
    )

    model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES, 64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{cfg.NUM_EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.NUM_CLASSES, cfg.PRINT_EVERY)
        val_loss, val_dice, val_per_class = evaluate(model, val_loader, criterion, device, cfg.NUM_CLASSES)
        train_eval_loss, train_dice, _ = evaluate(model, train_loader, criterion, device, cfg.NUM_CLASSES)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        print(f"[Epoch {epoch+1}] TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | TrainDice={train_dice:.4f} | ValDice={val_dice:.4f}")

    torch.save(model.state_dict(), cfg.MODEL_PATH)
    print(f"[Saved] Model weights to {cfg.MODEL_PATH}")

    logs = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_dices": train_dices,
        "val_dices": val_dices,
    }
    torch.save(logs, cfg.TRAIN_LOG_PATH)
    print(f"[Saved] Training logs to {cfg.TRAIN_LOG_PATH}")


if __name__ == "__main__":
    main()