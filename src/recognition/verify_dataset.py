import os
from pathlib import Path
from collections import defaultdict
from PIL import Image

# Define dataset paths
base_dir = Path(__file__).resolve().parents[2]  # goes up to project root
data_dir = base_dir / "data" / "car_recognition"
train_dir = data_dir / "train"
test_dir = data_dir / "test"

def count_images(folder):
    class_counts = defaultdict(int)
    total = 0
    for class_name in sorted(os.listdir(folder)):
        class_path = folder / class_name
        if not class_path.is_dir():
            continue
        num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = num_images
        total += num_images
    return class_counts, total

# Count training and test images
train_counts, train_total = count_images(train_dir)
test_counts, test_total = count_images(test_dir)

print("✅ Dataset structure verified!\n")
print(f"Train classes: {len(train_counts)} | Train images: {train_total}")
print(f"Test classes:  {len(test_counts)} | Test images:  {test_total}\n")

print("📂 Sample classes:")
for cls in list(train_counts.keys())[:5]:
    print(f"  {cls} -> {train_counts[cls]} images")

# Optional: check one sample image to confirm loading works
sample_class = next(iter(train_counts))
sample_path = next((train_dir / sample_class).glob("*.jpg"))
try:
    img = Image.open(sample_path)
    print(f"\n🖼️ Sample image loaded successfully from '{sample_class}' ({img.size}, {img.mode})")
except Exception as e:
    print(f"\n⚠️ Could not open image: {e}")
