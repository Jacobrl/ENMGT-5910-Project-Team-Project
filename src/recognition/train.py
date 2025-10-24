import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from pathlib import Path

if __name__ == "__main__":
    # define dataset paths 
    base_dir = Path(__file__).resolve().parents[2] # moves 2 folders up
    data_dir = base_dir / "data" / "car_recognition" # points to car recognition datset directory

    train_dir = data_dir / "train"

    # device setup: checks what hardware you have 
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}") #prints which one it's using

    # resizes to ResNet's default input size, randomly flips image to make the model generalize better, 
    # converts image pixels into normalised tensor and adjusts color values to match ImageNet pre-trained stats
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),   
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)

    print(f"Loaded {len(train_dataset)} training images across {len(train_dataset.classes)} classes")

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    #freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    num_classes = len(train_dataset.classes)  # 196 
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    #feed data in batches
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,   # number of images processed together
        shuffle=True,    # mix data each epoch for better generalization
        num_workers=0,   # parallel loading threads 
    )


    model = model.to(device)
    criterion = nn.CrossEntropyLoss()                # loss for multi-class classification
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # optimizer (only trains last layer)
    num_epochs = 5 #no of times we will go through entire dataset

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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    print("Training complete!")

    # torch.save(model.state_dict(), "resnet50_car_recognition.pth")
    # print("Model saved as resnet50_car_recognition.pth")