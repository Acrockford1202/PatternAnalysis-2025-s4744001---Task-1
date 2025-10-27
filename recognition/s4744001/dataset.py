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

        img = torch.from_numpy(img).unsqueeze(0).float()  # [1,H,W]
        msk = torch.from_numpy(msk).long()                # [H,W]

        return img, msk