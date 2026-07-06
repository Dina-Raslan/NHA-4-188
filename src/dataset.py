"""
Handles loading the dataset from disk and building DataLoaders
for train, validation, and test splits.
"""

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

import config
from transforms import train_transform, val_transform, test_transform


def compute_class_counts(dataset):
    """
    Returns a list of sample counts, one per class, in the same order
    as dataset.classes. Used for both the sampler and the loss weights.
    """
    counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts


def get_train_loader(
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    use_weighted_sampler=True,
):
    train_dataset = ImageFolder(root=config.TRAIN_DIR, transform=train_transform)

    if use_weighted_sampler:
        class_counts = compute_class_counts(train_dataset)
        # inverse frequency: rarer classes get a higher sampling weight
        class_weights = [1.0 / c if c > 0 else 0.0 for c in class_counts]
        sample_weights = [class_weights[label] for _, label in train_dataset.samples]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,          # sampler replaces shuffle=True
            num_workers=num_workers,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=config.PERSISTENT_WORKERS and num_workers > 0,
            drop_last=config.DROP_LAST_TRAIN,
        )

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS and num_workers > 0,
        drop_last=config.DROP_LAST_TRAIN,
    )


def get_loss_class_weights():
    """
    Class weights for CrossEntropyLoss, based on the train split.

    Since get_train_loader() already balances exposure via WeightedRandomSampler,
    these weights use sqrt(inverse frequency) instead of raw inverse frequency —
    a softer correction so the two techniques don't compound into an
    over-correction that hurts the majority classes too much.
    """
    import torch

    train_dataset = ImageFolder(root=config.TRAIN_DIR)
    counts = compute_class_counts(train_dataset)
    total = sum(counts)

    weights = []
    for c in counts:
        freq = c / total if total > 0 else 0
        weight = 1.0 / (freq ** 0.5) if freq > 0 else 0.0
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # normalize so weights average to 1.0 (keeps loss magnitude stable)
    weights_tensor = weights_tensor / weights_tensor.mean()

    return weights_tensor


def get_val_loader(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    val_dataset = ImageFolder(root=config.VAL_DIR, transform=val_transform)
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS and num_workers > 0,
    )


def get_test_loader(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS):
    test_dataset = ImageFolder(root=config.TEST_DIR, transform=test_transform)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.PERSISTENT_WORKERS and num_workers > 0,
    )


def get_class_names():
    train_dataset = ImageFolder(root=config.TRAIN_DIR)
    return train_dataset.classes


if __name__ == "__main__":
    train_loader = get_train_loader()
    val_loader = get_val_loader()
    test_loader = get_test_loader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {get_class_names()}")
