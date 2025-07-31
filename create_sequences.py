import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# === GPU CONFIG ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("❌ No GPU detected. Using CPU.")

# === CONFIG ===
SRC_DIR = "F:/drowsiness_datasets/nthuddd"
DEST_DIR = "F:/drowsiness_datasets/nthuddd_sequences"
SEQ_LENGTH = 10
IMG_SIZE = (160, 160)

os.makedirs(DEST_DIR, exist_ok=True)

def create_sequences(src_class_folder, dest_class_folder):
    images = sorted([f for f in os.listdir(src_class_folder) if f.endswith(('.jpg', '.png'))])
    os.makedirs(dest_class_folder, exist_ok=True)
    sequence_count = 0

    for i in range(0, len(images) - SEQ_LENGTH + 1, SEQ_LENGTH):
        sequence_folder = os.path.join(dest_class_folder, f"seq_{sequence_count:05d}")
        os.makedirs(sequence_folder, exist_ok=True)

        for j in range(SEQ_LENGTH):
            img_path = os.path.join(src_class_folder, images[i + j])
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            img_filename = f"frame_{j:02d}.jpg"
            img_save_path = os.path.join(sequence_folder, img_filename)
            array_to_img(img_array).save(img_save_path)

        sequence_count += 1

# Process both 'train' and 'val' folders
for split in ['train', 'val']:
    for class_name in ['drowsy', 'notdrowsy']:
        src_folder = os.path.join(SRC_DIR, split, class_name)
        dest_folder = os.path.join(DEST_DIR, split, class_name)
        create_sequences(src_folder, dest_folder)

print("✅ Sequences created successfully and saved to:", DEST_DIR)
