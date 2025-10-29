import torch
import torch.nn as nn
from modules import UNet
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


def main():
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    print(f"[Device] Using {device}")

    train_loader, _, _ = get_oasis_dataloaders(
        base_path=cfg.BASE_DATA_PATH,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        resize_hw=cfg.IMG_SIZE,
    )

    model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES, 64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    train_losses = []

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{cfg.NUM_EPOCHS} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg.NUM_CLASSES, cfg.PRINT_EVERY)
        train_losses.append(train_loss)
        print(f"[Epoch {epoch+1}] Avg Train Loss={train_loss:.4f}")

    torch.save(model.state_dict(), cfg.MODEL_PATH)
    print(f"[Saved] Model weights to {cfg.MODEL_PATH}")

    torch.save({"train_losses": train_losses}, cfg.TRAIN_LOG_PATH)
    print(f"[Saved] Training losses to {cfg.TRAIN_LOG_PATH}")

if __name__ == "__main__":
    main()
