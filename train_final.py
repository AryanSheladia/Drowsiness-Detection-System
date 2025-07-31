import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Paths
BASE_DIR = "F:/drowsiness_datasets/nthuddd_sequences"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# Parameters
BATCH_SIZE = 6
SEQUENCE_LENGTH = 10
IMAGE_SIZE = (224, 224)
EPOCHS = 50

# Data generator
class FrameSequence(Sequence):
    def __init__(self, directory, batch_size, sequence_length, img_size):
        self.directory = directory
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.sequence_paths = []
        self.image_labels = []
        self._prepare_sequences()

    def _prepare_sequences(self):
        for label_name in os.listdir(self.directory):
            label_dir = os.path.join(self.directory, label_name)
            if not os.path.isdir(label_dir):
                continue
            for seq_folder in os.listdir(label_dir):
                seq_path = os.path.join(label_dir, seq_folder)
                if os.path.isdir(seq_path):
                    self.sequence_paths.append(seq_path)
                    self.image_labels.append(1 if label_name == 'drowsy' else 0)

    def __len__(self):
        return len(self.sequence_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_paths = self.sequence_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.image_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sequences = []

        for seq_path in batch_paths:
            frames = sorted(os.listdir(seq_path))
            frames = frames[:self.sequence_length]
            images = []

            for frame in frames:
                img_path = os.path.join(seq_path, frame)
                img = load_img(img_path, target_size=self.img_size)
                img = img_to_array(img)
                img = preprocess_input(img)
                images.append(img)

            while len(images) < self.sequence_length:
                images.append(np.zeros((*self.img_size, 3)))  # padding

            batch_sequences.append(images)

        return np.array(batch_sequences), np.array(batch_labels)

# Model
def create_model(input_shape=(SEQUENCE_LENGTH, 224, 224, 3)):
    cnn_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    cnn_base.trainable = False  # freeze base

    model_input = Input(shape=input_shape)
    x = TimeDistributed(cnn_base)(model_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = LSTM(64)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(model_input, output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load data
train_gen = FrameSequence(TRAIN_DIR, BATCH_SIZE, SEQUENCE_LENGTH, IMAGE_SIZE)
val_gen = FrameSequence(VAL_DIR, BATCH_SIZE, SEQUENCE_LENGTH, IMAGE_SIZE)

# Create and train model
model = create_model()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Save model
model.save("mobilenetv2_lstm_drowsiness.h5")

# Plot training curves
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
