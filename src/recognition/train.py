import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from pathlib import Path
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)
    # define dataset paths 
    base_dir = Path(__file__).resolve().parents[2] # moves 2 folders up
    data_dir = base_dir / "data" / "car_recognition" # points to car recognition datset directory
    train_dir = data_dir / "train"

    # device setup: checks what hardware you have 
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") #prints which one it's using

    # resizes to ResNet's default input size, randomly flips image to make the model generalize better, 
    # converts image pixels into normalised tensor and adjusts color values to match ImageNet pre-trained stats
    transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    ])

    #80:20 training + validation split
    base_dataset = datasets.ImageFolder(train_dir) # read all image paths + class ids
    class_names = base_dataset.classes             # list of class names
    targets = [y for _, y in base_dataset.samples] # one class id per image

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))

    # Build the actual datasets with their own transforms
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset   = datasets.ImageFolder(train_dir, transform=transform_val)

    # Point to the chosen indices (no files moved)
    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(val_dataset, val_idx)

    #sanity check - confirming if stratified split worked
    print(f"Split → train: {len(train_subset)} images, val: {len(val_subset)} images")
    
    num_classes = len(class_names)

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # selectively unfreeze deeper layers for fine-tuning
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:   # last block + final layer
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)

     # DataLoader performance options
    pin = torch.cuda.is_available()  # True if CUDA, False otherwise (MPS ignores)
    print(f"pin_memory: {pin}")

    # feed data in batches
    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()                # loss for multi-class classification
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )
    num_epochs = 20 #no of times we will go through entire dataset

    def topk_acc(logits: torch.Tensor, y: torch.Tensor, k: int = 5) -> float:
        # fraction of samples where true label is in top-k predictions
        k = min(k, logits.size(1))
        _, pred = logits.topk(k, dim=1)         # [B, k]
        return pred.eq(y.view(-1, 1)).any(dim=1).float().mean().item()


    for epoch in range(num_epochs):
        model.train()           # put model in training mode
        running_loss = 0.0      # tracker to see how wrong the model was for each batch
        correct = 0             # how many predictions were right
        total = 0               # how many total images we have seen

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()           # reset gradients
            outputs = model(images)         # forward pass
            loss = criterion(outputs, labels)  # calculate loss
            loss.backward()                 # backpropagation
            optimizer.step()                # update weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0) # counts batch length
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # summarize training metrics
        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)

        # ----- VALIDATE -----
        model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_correct1 = 0
        val_top5_sum = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                val_total += labels.size(0)

                # top-1 accuracy
                _, pred1 = outputs.max(1)
                val_correct1 += pred1.eq(labels).sum().item()

                # top-5 accuracy (batch fraction)
                val_top5_sum += topk_acc(outputs, labels, k=5)

        val_loss = val_running_loss / max(1, len(val_loader))
        val_acc1 = 100.0 * val_correct1 / max(1, val_total)
        val_acc5 = 100.0 * (val_top5_sum / max(1, len(val_loader)))

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Top-1: {val_acc1:.2f}%, Val Top-5: {val_acc5:.2f}%"
        )
        scheduler.step(val_loss)


    print("Training complete!")

    torch.save(model.state_dict(), "resnet50_car_recognition_finetuned.pth")
    print("Saved: resnet50_car_recognition_finetuned.pth")