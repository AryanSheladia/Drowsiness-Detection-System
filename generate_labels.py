import os
from pathlib import Path
from PIL import Image

# Set root paths
base_path = Path("/home/cl502_23/drowsiness_data/yolov8_train/images")
sets = ['train', 'val', 'test']
class_map = {'notdrowsy': 0, 'drowsy': 1}  # Adjust if needed

for split in sets:
    image_dir = base_path / split
    label_dir = base_path / f"labels/{split}"
    label_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_dir.glob("*.*"):
        img_name = img_path.name.lower()

        # Skip non-image files
        if not img_name.endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Determine class from filename
        cls = None
        if "notdrowsy" in img_name:
            cls = class_map['notdrowsy']
        elif "drowsy" in img_name:
            cls = class_map['drowsy']
        else:
            print(f"⚠️  Skipped: {img_name} (cannot determine class)")
            continue

        # Get image size
        with Image.open(img_path) as im:
            w, h = im.size

        # Full image bounding box (centered, normalized)
        x_center, y_center, box_w, box_h = 0.5, 0.5, 1.0, 1.0
        label_line = f"{cls} {x_center} {y_center} {box_w} {box_h}\n"

        # Save label file
        label_path = label_dir / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write(label_line)

print("✅ Label generation complete.")
