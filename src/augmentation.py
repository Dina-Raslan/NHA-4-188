"""
Standalone augmentation utility used to balance underrepresented classes
before training. This was already applied once to generate the current
train/ folder, but is kept here for reproducibility or future re-runs
on new raw data.

Note: VerticalFlip was removed on purpose. Skin lesions have a natural
orientation (relative to the body), so flipping vertically can introduce
unrealistic samples. HorizontalFlip, small rotations, and Affine are safe.
"""

import os
import cv2
import pandas as pd
import albumentations as A

AUGMENTATION_PIPELINE = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
])


def balance_classes(train_df, output_dir, target_multiplier=1.5):
    """
    Generates augmented images for underrepresented classes so that
    every class reaches roughly target_multiplier x the median class size.

    train_df must have columns: 'path', 'class'.
    Returns a DataFrame of the newly generated augmented images.
    """
    os.makedirs(output_dir, exist_ok=True)

    class_counts = train_df["class"].value_counts()
    target_count = int(class_counts.median() * target_multiplier)

    augmented_records = []

    for cls, count in class_counts.items():
        need = target_count - count
        if need <= 0:
            continue

        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        cls_images = train_df[train_df["class"] == cls]["path"].tolist()

        for i in range(need):
            source_path = cls_images[i % len(cls_images)]
            img = cv2.imread(source_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            augmented = AUGMENTATION_PIPELINE(image=img)["image"]
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

            save_name = f"aug_{i}_{os.path.basename(source_path)}"
            save_path = os.path.join(cls_dir, save_name)
            cv2.imwrite(save_path, augmented_bgr)

            augmented_records.append({"path": save_path, "class": cls})

        print(f"{cls}: generated {need} augmented images")

    return pd.DataFrame(augmented_records)
