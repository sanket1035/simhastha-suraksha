"""PyTorch datasets over the index CSVs written by src.data_prep.

mode="count"   : image resized to count_size x count_size, target = people count (MLP, CNN)
mode="density" : full image (optionally cropped/padded), target = density map
                 at 1/8 resolution using sum pooling (CSRNet)
"""
import csv
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .density import downsample_density_sum

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FACTOR = 8


def read_index(index_csv):
    with open(index_csv, newline="") as f:
        return list(csv.DictReader(f))


def split_train_val(rows, val_fraction=0.1, seed=42):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction)) if len(rows) > 1 else 0
    return rows[n_val:], rows[:n_val]


def _normalise(img_float):
    return ((img_float - MEAN) / STD).transpose(2, 0, 1)


def _pad_to(img, dens, size):
    h, w = dens.shape
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
        dens = np.pad(dens, ((0, ph), (0, pw)))
    return img, dens


class CrowdDataset(Dataset):
    def __init__(self, rows, mode="density", train=False, crop_size=0, count_size=128):
        if mode not in ("density", "count"):
            raise ValueError("mode must be 'density' or 'count'")
        self.rows, self.mode, self.train = rows, mode, train
        self.crop_size, self.count_size = crop_size, count_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        dens = np.load(row["density_path"]).astype(np.float32)
        with Image.open(row["image_path"]) as im:
            im = im.convert("RGB")
            if self.mode == "count":
                small = np.asarray(im.resize((self.count_size, self.count_size)), dtype=np.float32) / 255.0
                if self.train and random.random() < 0.5:
                    small = small[:, ::-1]
                x = torch.from_numpy(np.ascontiguousarray(_normalise(small)))
                return x, torch.tensor(float(dens.sum()), dtype=torch.float32)
            img = np.asarray(im, dtype=np.float32) / 255.0

        if self.crop_size:
            img, dens = _pad_to(img, dens, self.crop_size)
        if self.train:
            if self.crop_size:
                h, w = dens.shape
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)
                img = img[top:top + self.crop_size, left:left + self.crop_size]
                dens = dens[top:top + self.crop_size, left:left + self.crop_size]
            if random.random() < 0.5:
                img, dens = img[:, ::-1], dens[:, ::-1]
        h, w = dens.shape
        h8, w8 = h // FACTOR * FACTOR, w // FACTOR * FACTOR
        img, dens = img[:h8, :w8], dens[:h8, :w8]
        target = downsample_density_sum(np.ascontiguousarray(dens), FACTOR)
        x = torch.from_numpy(np.ascontiguousarray(_normalise(np.ascontiguousarray(img))))
        return x, torch.from_numpy(np.ascontiguousarray(target))[None]
