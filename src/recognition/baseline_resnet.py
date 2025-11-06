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

def weighted_top5(outputs, labels, class_correct, class_total):
    """Accumulate per-class Top-5 counts for macro averaging."""
    _, pred5 = outputs.topk(5, dim=1)
    correct_mask = pred5.eq(labels.view(-1, 1)).any(dim=1)
    for lbl, correct_flag in zip(labels.cpu().numpy(), correct_mask.cpu().numpy()):
        class_correct[lbl] = class_correct.get(lbl, 0) + int(correct_flag)
        class_total[lbl] = class_total.get(lbl, 0) + 1

if __name__ == "__main__":
    set_seed(42)
    # define dataset paths 
    base_dir = Path(__file__).resolve().parents[2] # moves 2 folders up
    data_dir = base_dir / "data" / "car_recognition" # points to car recognition datset directory
    dataset_dir = data_dir / "dataset" #merged dataset folder

    # device setup: checks what hardware you have 
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") #prints which one it's using

    # resizes to ResNet's default input size, randomly flips image to make the model generalize better, 
    # converts image pixels into normalised tensor and adjusts color values to match ImageNet pre-trained stats
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Dataset + Split (80:20 train/test)
    base_dataset = datasets.ImageFolder(dataset_dir) # read all image paths + class ids
    class_names = base_dataset.classes             # list of class names
    targets = [y for _, y in base_dataset.samples] # one class id per image

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(targets)), targets))

    # Build the actual datasets with their own transforms
    train_dataset = datasets.ImageFolder(dataset_dir, transform=transform_train)
    test_dataset   = datasets.ImageFolder(dataset_dir, transform=transform_test)

    # Point to the chosen indices (no files moved)
    train_subset = Subset(train_dataset, train_idx)
    test_subset   = Subset(test_dataset, test_idx)

    #sanity check - confirming if stratified split worked
    print(f"Split → train: {len(train_subset)} images, test: {len(test_subset)} images")
    num_classes = len(class_names)

    # Model setup
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

    test_loader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    model = model.to(device)
    # loss for multi-class classification using label smoothing to reduce overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)                
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    num_epochs = 12 #no of times we will go through entire dataset

    # Best-checkpoint setup (saves model when validation improves)
    best_loss = float('inf')
    best_checkpoint_path = "resnet50_best_checkpoint.pth"

    def save_checkpoint(epoch, model, optimizer, scheduler, best_loss, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_random_state': random.getstate(),
        }, path)
        print(f"Saved best checkpoint: {path}")

    print("Starting fresh training — no checkpoint resume.")

    # ---- Early stopping setup ----
    early_stop_patience = 2      # stop after 3 epochs with no improvement
    epochs_no_improve = 0        # how many consecutive epochs with no improvement

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
        
        # summarize training metrics
        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)

        # test the model
        model.eval()
        test_running_loss = 0.0
        test_total = 0
        test_correct1 = 0
        class_correct_top5 = {}
        class_total_top5 = {}

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_running_loss += loss.item()
                test_total += labels.size(0)

                # top-1 accuracy
                _, pred1 = outputs.max(1)
                test_correct1 += pred1.eq(labels).sum().item()

                # weighted top-5 accuracy 
                weighted_top5(outputs, labels, class_correct_top5, class_total_top5)
        #final metrics
        per_class_acc = [class_correct_top5[c] / class_total_top5[c] for c in class_correct_top5]
        test_acc5 = 100.0 * np.mean(per_class_acc)
        test_loss = test_running_loss / len(test_loader)
        test_acc1 = 100.0 * test_correct1 / test_total

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Top-1: {test_acc1:.2f}%, Test Top-5: {test_acc5:.2f}%"
        )

        # Save best checkpoint
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, best_checkpoint_path)
            print(f"Saved new best checkpoint → {best_checkpoint_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # ---- Early stopping ----
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        scheduler.step(test_loss)

    print("Training complete!")

    # ---- Load best checkpoint before final save ----
    print("Loading best checkpoint before final save")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Save only model weights (clean version, for deployment/inference)
    final_model_path = "resnet50_car_recognition_best.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved best-performing model weights: {final_model_path}")