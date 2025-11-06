import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import ResNet50_Weights
from pathlib import Path
from PIL import Image

def load_model(weights_path: str, num_classes: int):
    """Loads fine-tuned ResNet-50 model with trained weights."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path: str):
    """Applies same preprocessing as during training/test."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension


def predict(model, image_tensor, class_names):
    """Runs forward pass and returns top-1 and top-5 predictions."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top5_prob, top5_idx = probs.topk(5, dim=1)

    results = [
        {"label": class_names[idx], "probability": float(prob)}
        for idx, prob in zip(top5_idx[0], top5_prob[0])
    ]
    return results


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "car_recognition" / "dataset"
    model_path = base_dir / "resnet50_car_recognition_best.pth"

    dataset = datasets.ImageFolder(data_dir)
    class_names = dataset.classes
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes from dataset folder")

    model = load_model(str(model_path), num_classes)
    print(f"Model loaded successfully from {model_path}")

    test_image = input("\nEnter full path to the image you want to test: ").strip()
    image_tensor = preprocess_image(test_image)

    predictions = predict(model, image_tensor, class_names)
    print("\nTop Predictions:")
    for i, p in enumerate(predictions, 1):
        print(f"{i}. {p['label']} — {p['probability']*100:.2f}%")
