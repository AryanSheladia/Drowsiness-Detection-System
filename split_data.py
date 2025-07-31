import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths to your dataset
dataset_dir = os.path.expanduser('~/drowsiness_data/nthuddd/train_data')  # Original dataset path
output_dir = os.path.expanduser('~/drowsiness_data/sorted_data')         # New root directory

# Define the subdirectories
classes = ['drowsy', 'notdrowsy']
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

# Create root and class subdirectories
for dir_path in [train_dir, val_dir, test_dir]:
    os.makedirs(dir_path, exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

# Load and store image paths
image_paths = {}
for cls in classes:
    class_path = os.path.join(dataset_dir, cls)
    image_paths[cls] = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.lower().endswith('.jpg')]

# Balance training set by undersampling majority class
min_class = min(image_paths, key=lambda k: len(image_paths[k]))
max_class = max(image_paths, key=lambda k: len(image_paths[k]))
min_count = len(image_paths[min_class])

# Shuffle for reproducibility
random.seed(42)
random.shuffle(image_paths[max_class])
random.shuffle(image_paths[min_class])

balanced_train_max = image_paths[max_class][:min_count]
balanced_train_min = image_paths[min_class]

# Combine balanced training set
balanced_train = {
    max_class: balanced_train_max,
    min_class: balanced_train_min
}

# Now split train, val, and test from the balanced sets
for cls in classes:
    train_imgs, temp_imgs = train_test_split(balanced_train[cls], test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    # Copy to respective folders
    for img in train_imgs:
        shutil.copy(img, os.path.join(train_dir, cls, os.path.basename(img)))
    for img in val_imgs:
        shutil.copy(img, os.path.join(val_dir, cls, os.path.basename(img)))
    for img in test_imgs:
        shutil.copy(img, os.path.join(test_dir, cls, os.path.basename(img)))

print("✅ Balanced dataset split into 'train', 'val', and 'test' at:")
print(f"📁 {output_dir}")
