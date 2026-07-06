"""
Defines the image transformations applied to the dataset.

Updated to fight the overfitting gap observed in training
(Train Acc ~93-96% vs Val Acc ~80-85%). Changes vs the previous version:

  - Wider RandomResizedCrop scale range (0.7-1.0 instead of 0.8-1.0)
    -> forces the model to recognize lesions from more varied framings.
  - Added RandomRotation (skin lesions have no fixed orientation, this is safe).
  - Slightly stronger ColorJitter (+ saturation) to reduce reliance on exact
    color/lighting conditions, since the real dataset likely has inconsistent
    lighting/camera sources.
  - Added RandomErasing (applied AFTER ToTensor/Normalize) — randomly blacks
    out a small rectangular patch of the image each time it's seen. This is
    one of the most effective, well-established regularizers for closing an
    overfitting gap in image classifiers, because the model can no longer
    rely on memorizing one specific patch/region of a training image.

Note: if you are also using augmentation.py's balance_classes() to generate
extra OFFLINE copies of underrepresented classes on disk, that's fine to use
alongside this — these are online/random transforms applied fresh every
epoch, so they compound with (not duplicate) the offline augmentation.
"""

from torchvision import transforms
import config

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

val_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
])

test_transform = val_transform
