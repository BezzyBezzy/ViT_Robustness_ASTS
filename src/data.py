from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms


def build_transform(image_size: int):
    # IMPORTANT: return pixel-space tensors in [0,1], no normalization here.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def get_dataset(name: str, root: str, image_size: int, train: bool, imagenet_dir: Optional[str] = None) -> Dataset:
    tfm = build_transform(image_size)
    n = name.lower()

    if n == "cifar10":
        return datasets.CIFAR10(root=root, train=train, transform=tfm, download=True)
    if n == "cifar100":
        return datasets.CIFAR100(root=root, train=train, transform=tfm, download=True)
    if n == "imagenet1k":
        if imagenet_dir is None:
            raise ValueError("For ImageNet-1K, set data.imagenet_dir in YAML.")
        split = "train" if train else "val"
        return datasets.ImageFolder(os.path.join(imagenet_dir, split), transform=tfm)

    raise ValueError(f"Unknown dataset: {name}")


def stratified_subset(dataset: Dataset, ratio: float, seed: int = 42) -> Subset:
    """
    Create a stratified subset of the dataset maintaining class distribution.
    
    Args:
        dataset: Source dataset (must have .targets attribute or be indexable with labels)
        ratio: Fraction of data to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Subset with stratified sampling
    """
    if ratio >= 1.0:
        return dataset
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Try to extract labels by iterating (slower fallback)
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    # Sample from each class proportionally
    rng = np.random.RandomState(seed)
    selected_indices = []
    
    for class_label, indices in class_indices.items():
        n_samples = max(1, int(len(indices) * ratio))  # At least 1 sample per class
        selected = rng.choice(indices, size=n_samples, replace=False)
        selected_indices.extend(selected)
    
    # Shuffle the selected indices
    rng.shuffle(selected_indices)
    
    return Subset(dataset, selected_indices)


@dataclass
class DataModule:
    dataset_name: str
    data_root: str
    image_size: int
    batch_size: int
    num_workers: int
    val_split: float
    imagenet_dir: Optional[str] = None
    train_ratio: float = 1.0  # Ratio of training data to use (stratified)
    eval_ratio: float = 1.0   # Ratio of test/val data to use (stratified)
    seed: int = 42

    def setup(self) -> None:
        train_full = get_dataset(self.dataset_name, self.data_root, self.image_size, train=True, imagenet_dir=self.imagenet_dir)
        test_full = get_dataset(self.dataset_name, self.data_root, self.image_size, train=False, imagenet_dir=self.imagenet_dir)
        
        # Apply stratified sampling to test set if ratio < 1.0
        if self.eval_ratio < 1.0:
            self.test_ds = stratified_subset(test_full, self.eval_ratio, seed=self.seed)
        else:
            self.test_ds = test_full

        if self.val_split and 0.0 < self.val_split < 1.0:
            n_total = len(train_full)
            n_val = int(n_total * self.val_split)
            n_train = n_total - n_val
            train_ds, val_ds = random_split(train_full, [n_train, n_val])
            
            # Apply stratified sampling if ratio < 1.0
            if self.train_ratio < 1.0:
                # For random_split subsets, we need to handle differently
                self.train_ds = self._stratified_subset_from_split(train_ds, train_full, self.train_ratio)
            else:
                self.train_ds = train_ds
                
            if self.eval_ratio < 1.0:
                self.val_ds = self._stratified_subset_from_split(val_ds, train_full, self.eval_ratio)
            else:
                self.val_ds = val_ds
        else:
            if self.train_ratio < 1.0:
                self.train_ds = stratified_subset(train_full, self.train_ratio, seed=self.seed)
            else:
                self.train_ds = train_full
            self.val_ds = None
    
    def _stratified_subset_from_split(self, split_ds: Subset, original_ds: Dataset, ratio: float) -> Subset:
        """Apply stratified sampling to a random_split result."""
        # Get the indices from the split
        split_indices = split_ds.indices
        
        # Get labels for these indices
        if hasattr(original_ds, 'targets'):
            labels = np.array([original_ds.targets[i] for i in split_indices])
        elif hasattr(original_ds, 'labels'):
            labels = np.array([original_ds.labels[i] for i in split_indices])
        else:
            labels = np.array([original_ds[i][1] for i in split_indices])
        
        # Group by class
        class_indices = defaultdict(list)
        for local_idx, label in enumerate(labels):
            class_indices[label].append(local_idx)
        
        # Sample from each class
        rng = np.random.RandomState(self.seed + 1)  # Different seed for val
        selected_local = []
        
        for class_label, indices in class_indices.items():
            n_samples = max(1, int(len(indices) * ratio))
            selected = rng.choice(indices, size=n_samples, replace=False)
            selected_local.extend(selected)
        
        rng.shuffle(selected_local)
        
        # Map back to original indices
        selected_original = [split_indices[i] for i in selected_local]
        return Subset(original_ds, selected_original)

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_loader(self) -> Optional[DataLoader]:
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
