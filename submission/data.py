"""EuroSAT loading, deterministic subset selection, and sensor-like corruption."""
from pathlib import Path
import json
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision.datasets import EuroSAT, ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor


class Corruption:
    @staticmethod
    def apply(x, kind, severity, rng):
        if kind == "clean" or severity <= 0:
            return x
        x = x.clone()
        if kind == "gaussian":
            x.add_(torch.randn(x.shape, generator=rng, dtype=x.dtype) * severity).clamp_(0, 1)
        elif kind == "salt_pepper":
            mask = torch.rand(x.shape, generator=rng) < severity
            salt = torch.rand(x.shape, generator=rng) < 0.5
            x[mask & salt] = 1.0
            x[mask & ~salt] = 0.0
        elif kind == "channel_dropout":
            # RGB channels stand in for input bands; drop an entire channel.
            if torch.rand((), generator=rng).item() < severity:
                channel = int(torch.randint(0, x.shape[0], (), generator=rng).item())
                x[channel] = 0.0
        else:
            raise ValueError(f"Unknown corruption: {kind}")
        return x


class EuroSATSplit(Dataset):
    def __init__(self, samples, indices, train=False, augment_noise=False, seed=0):
        self.samples, self.indices = samples, np.asarray(indices)
        self.train, self.augment_noise, self.seed = train, augment_noise, seed
        self.transform = Compose([Resize((64, 64)), ToTensor()])
        self.epoch = 0

    def __len__(self): return len(self.indices)

    def __getitem__(self, item):
        global_index = int(self.indices[item])
        path, label = self.samples[global_index]
        from PIL import Image
        image = Image.open(path).convert("RGB")
        if self.train:
            # Land-cover labels are invariant to right-angle rotations and flips.
            if torch.rand(()) < 0.5: image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if torch.rand(()) < 0.5: image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            image = image.rotate(90 * int(torch.randint(0, 4, ()).item()))
        x = self.transform(image)
        quality = 0
        if self.train and self.augment_noise and torch.rand(()) < 0.5:
            # Per-item deterministic RNG keeps results reproducible across workers.
            rng = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + global_index)
            # Sample channel dropout twice as often as either pixel-noise type;
            # when selected, remove one channel to expose the classifier to this failure mode.
            draw = int(torch.randint(0, 4, (), generator=rng).item())
            kind_index = min(draw, 2)
            kind = ("gaussian", "salt_pepper", "channel_dropout")[kind_index]
            severity = (0.15, 0.15, 1.0)[kind_index]
            x = Corruption.apply(x, kind, severity, rng)
            quality = 1
        return x, int(label), quality


def load_splits(root, download, count=2000, seed=42):
    root = Path(root)
    # Accept either a direct class-folder dataset or torchvision's EuroSAT location.
    candidate = root / "2750"
    if not candidate.exists(): candidate = root
    try:
        ds = ImageFolder(candidate)
        if len(ds.classes) != 10: raise ValueError("Expected 10 class folders")
    except (FileNotFoundError, ValueError):
        ds = EuroSAT(root=str(root), download=download)
    samples = ds.samples
    # Match the handout's shared subset exactly:
    # torch.manual_seed(42); torch.randperm(len(EuroSAT))[:2000].
    subset_generator = torch.Generator().manual_seed(42)
    chosen = torch.randperm(len(samples), generator=subset_generator)[:count].numpy()
    # Reproduce the guide's exact 80/20 random_split on the selected Subset.
    train_pool_size = int(count * 0.8)
    test_size = count - train_pool_size
    train_pool_local, test_local = random_split(
        range(count), [train_pool_size, test_size],
        generator=torch.Generator().manual_seed(42))
    train_pool = chosen[np.asarray(train_pool_local.indices)]
    test = chosen[np.asarray(test_local.indices)]

    # Keep the official 400-item test set untouched. Carve an internal
    # validation slice from the official 1,600-item training pool for checkpoint selection.
    val_size = max(1, int(round(train_pool_size * 0.10)))
    fit_size = train_pool_size - val_size
    fit_local, val_local = random_split(
        range(train_pool_size), [fit_size, val_size],
        generator=torch.Generator().manual_seed(43))
    train = train_pool[np.asarray(fit_local.indices)]
    val = train_pool[np.asarray(val_local.indices)]
    manifest = {"subset_protocol": "torch.randperm with seed 42",
                "split_protocol": "official 80/20 random_split with seed 42; internal 90/10 validation from training pool with seed 43",
                "subset_seed": 42, "official_split_seed": 42, "validation_seed": 43,
                "selected_indices": chosen.tolist(), "class_names": ds.classes,
                "train_pool_indices": train_pool.tolist(),
                "train_indices": train.tolist(), "val_indices": val.tolist(),
                "test_indices": test.tolist()}
    (root / "edgesat_split.json").parent.mkdir(parents=True, exist_ok=True)
    with (root / "edgesat_split.json").open("w") as f: json.dump(manifest, f, indent=2)
    return samples, manifest
