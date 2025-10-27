import os
import cv2
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader


def find_all_slices(root_dir: str):
    files = glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f"No .png slice files found under {root_dir}")
    return files


def load_grayscale(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"Failed to read {path}")
    return arr.astype(np.float32)


def normalize_zscore(img: np.ndarray) -> np.ndarray:
    return (img - img.mean()) / (img.std() + 1e-8)


class OASISSliceDataset(Dataset):
    def __init__(self, image_root: str, mask_root: str, resize_hw=(256, 256)):
        self.resize_hw = resize_hw

        self.image_paths = find_all_slices(image_root)
        self.mask_paths = find_all_slices(mask_root)

        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError(
                f"Image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}"
            )

        print(f"[OASISSliceDataset] {len(self.image_paths)} pairs from:")
        print("  Images:", image_root)
        print("  Masks :", mask_root)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_grayscale(self.image_paths[idx])
        msk = load_grayscale(self.mask_paths[idx])

        img = normalize_zscore(img)

        h, w = self.resize_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img).unsqueeze(0).float()  
        msk = torch.from_numpy(msk).long()                

        return img, msk
    
def get_oasis_dataloaders(base_path: str, batch_size=4, num_workers=0, resize_hw=(256, 256)):
    root = os.path.join(base_path, "keras_png_slices_data")

    train_images = os.path.join(root, "keras_png_slices_train")
    train_masks  = os.path.join(root, "keras_png_slices_seg_train")

    val_images   = os.path.join(root, "keras_png_slices_validate")
    val_masks    = os.path.join(root, "keras_png_slices_seg_validate")

    test_images  = os.path.join(root, "keras_png_slices_test")
    test_masks   = os.path.join(root, "keras_png_slices_seg_test")

    train_dataset = OASISSliceDataset(train_images, train_masks, resize_hw)
    val_dataset   = OASISSliceDataset(val_images, val_masks, resize_hw)
    test_dataset  = OASISSliceDataset(test_images, test_masks, resize_hw)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    base_path = r"C:\Users\andre\OneDrive\Desktop\Uni\COMP3710\Oasis Dataset"
    train_loader, val_loader, test_loader = get_oasis_dataloaders(base_path)