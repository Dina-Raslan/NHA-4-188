# src/predict.py
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Path Configuration and Module Imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from model import SkinLesionClassifier

# Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD
    )
])

# Model Loading and Initialization
def load_model(model_path):
    model = SkinLesionClassifier(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained=False
    )
    state = torch.load(model_path, map_location=config.DEVICE)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(config.DEVICE)
    model.eval()
    print(f" Model loaded from {model_path}")
    return model

# Prediction Logic and Output Formatting
def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits, embedding = model(tensor, return_embedding=True)
        probs = F.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    top_class = config.CLASS_NAMES[top_idx]
    top_conf = probs[top_idx].item()

    all_scores = {
        config.CLASS_NAMES[i]: round(probs[i].item(), 4)
        for i in range(len(config.CLASS_NAMES))
    }

    return {
        "class": top_class,
        "confidence": round(top_conf, 4),
        "all_scores": all_scores,
        "embedding": embedding[0].cpu().tolist()
    }

# Main Execution Entry Point
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "models", "best_model_ResNet50.pt"
    )

    model = load_model(model_path)
    test_image = input("Enter image path: ")
    result = predict_image(model, test_image)

    print(f"\n Result:")
    print(f"  Class     : {result['class']}")
    print(f"  Confidence: {result['confidence'] * 100:.1f}%")
    print(f"\n All Condition Scores:")
    for cls, score in sorted(result['all_scores'].items(),
                             key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {score * 100:.1f}%")