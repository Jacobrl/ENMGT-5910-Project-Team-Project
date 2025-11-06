import os
import shutil
from pathlib import Path

# ---- Paths ----
base_dir = Path(__file__).resolve().parents[2]
data_dir = base_dir / "data" / "car_recognition"
train_dir = data_dir / "train"
test_dir = data_dir / "test"
merged_dir = data_dir / "dataset"

print("Merging car_recognition/train and car_recognition/test into:", merged_dir)

# ---- Create merged dataset folder ----
merged_dir.mkdir(exist_ok=True)

# ---- Helper function to clean hidden files ----
def is_valid_folder(path):
    return path.is_dir() and not path.name.startswith(".")

# ---- Loop through each class folder ----
for class_folder in sorted(os.listdir(train_dir)):
    train_class_path = train_dir / class_folder
    test_class_path = test_dir / class_folder
    merged_class_path = merged_dir / class_folder

    # Skip anything that’s not a folder (e.g. .DS_Store)
    if not is_valid_folder(train_class_path):
        continue

    merged_class_path.mkdir(exist_ok=True)

    # ---- Copy training images ----
    for file in os.listdir(train_class_path):
        if file.startswith("."):  # skip hidden files
            continue
        src = train_class_path / file
        dst = merged_class_path / f"train_{file}"
        shutil.copy2(src, dst)

    # ---- Copy test images ----
    if test_class_path.exists() and is_valid_folder(test_class_path):
        for file in os.listdir(test_class_path):
            if file.startswith("."):
                continue
            src = test_class_path / file
            dst = merged_class_path / f"test_{file}"
            shutil.copy2(src, dst)

print("\nMerged dataset created successfully at:", merged_dir)