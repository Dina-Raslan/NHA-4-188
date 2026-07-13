# evaluation/optuna_tuning.py
import os
import sys
import json
import optuna
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Path Configuration and Module Imports
SRC_PATH = r"C:\Users\jessi\Downloads\NHA-4-188-main\NHA-4-188-main\src"
sys.path.append(SRC_PATH)
import config
from model import SkinLesionClassifier

DATA_PATH = r"C:\Users\jessi\Downloads\skin_project\data"

# Custom Dataset Class Definition
class SkinDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
        for label_idx, folder in enumerate(folders):
            folder_path = os.path.join(root, folder)
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(folder_path, img_file), label_idx))
        self.num_classes = len(folders)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Optuna Objective Function for Hyperparameter Tuning
def objective(trial):
    # Suggest hyperparameters to tune
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    image_size = trial.suggest_categorical("image_size", [224, 256])

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    dataset = SkinDataset(DATA_PATH, transform=transform)

    # Use a subset of data for fast evaluation
    subset = torch.utils.data.Subset(dataset, list(range(200)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize and load model
    model = SkinLesionClassifier(
        model_name=config.MODEL_NAME,
        num_classes=dataset.num_classes,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained=False
    )
    model_path = os.path.join(SRC_PATH, "..", "models", "best_model_ResNet50.pt")
    state = torch.load(model_path, map_location=config.DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(config.DEVICE)
    model.eval()
    
    # Evaluation loop
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return correct / total

# Execution Entry Point and Result Saving
if __name__ == "__main__":
    print(" Starting Optuna Hyperparameter Tuning...")
    
    # Create and run study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, show_progress_bar=True)
    
    # Display best trial results
    best = study.best_trial
    print(f"\n Best Trial:")
    print(f"   Accuracy  : {best.value * 100:.2f}%")
    print(f"   Batch Size: {best.params['batch_size']}")
    print(f"   Image Size: {best.params['image_size']}")
    #Save best configuration to JSON
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/best_config.json", "w") as f:
        json.dump({
            "best_accuracy": round(best.value, 4),
            "batch_size": best.params["batch_size"],
            "image_size": best.params["image_size"]
        }, f, indent=4)

    print(f"\n Saved → outputs/best_config.json")