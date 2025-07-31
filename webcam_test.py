import cv2
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Flatten
from playsound import playsound
import threading
import pathlib
import os
from pathlib import Path

# Ensure Windows path compatibility
pathlib.PosixPath = pathlib.WindowsPath

# Function to get Windows-compatible path
def get_path(path):
    return pathlib.Path(path).as_posix()

# Define paths
ALARM_SOUND_PATH = get_path("F:/drowsiness_datasets/alarm/mixkit-morning-clock-alarm-1003.wav")
YOLO_MODEL_PATH = get_path("F:/drowsiness_datasets/yolov5/runs/train/nthuddd/weights/best.pt")
YOLO_REPO_PATH = get_path("F:/drowsiness_datasets/yolov5")
LSTM_MODEL_PATH = get_path("F:/drowsiness_datasets/mobilenetv2_lstm_drowsiness_model.h5")

# Load YOLOv5 model with explicit string paths
yolo_model = torch.hub.load(str(YOLO_REPO_PATH), 'custom', path=str(YOLO_MODEL_PATH), source='local', force_reload=True)

# === Rebuild LSTM model and load weights === #
SEQUENCE_LENGTH = 30

sequence_input = Input(shape=(SEQUENCE_LENGTH, 224, 224, 3))
cnn_base = MobileNetV2(weights='imagenet', include_top=False)  # Remove pooling='avg'
cnn_base.trainable = False

x = TimeDistributed(cnn_base)(sequence_input)  # Shape: (batch, SEQUENCE_LENGTH, 7, 7, 1280)
x = TimeDistributed(Flatten())(x)  # Shape: (batch, SEQUENCE_LENGTH, 7 * 7 * 1280) = (batch, 30, 62720)
x = LSTM(128, return_sequences=False)(x)  # Units = 128 to match saved weights (62720, 512)
x = Dense(64, activation='relu')(x)  # Updated to 64 units to match saved weights (128, 64)
output = Dense(1, activation='sigmoid')(x)

lstm_model = Model(inputs=sequence_input, outputs=output)
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load pre-trained weights
lstm_model.load_weights(str(LSTM_MODEL_PATH))

def sound_alarm():
    playsound(str(ALARM_SOUND_PATH))

# Start webcam
cap = cv2.VideoCapture(0)

sequence = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5 detection
        results = yolo_model(frame)

        pred = [0]  # Default value for pred, to avoid NameError

        for detection in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = detection[:6]
            if conf < 0.5:
                continue

            # Assuming YOLOv5 class 0 is for faces (adjust based on your training data)
            if int(cls) != 0:  # Skip non-face detections
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Tighten the bounding box around the face (reduce size by 10% on each side)
            width = x2 - x1
            height = y2 - y1
            shrink_factor = 0.1  # Adjust this to control how much to shrink the box
            x1 = int(x1 + width * shrink_factor)
            y1 = int(y1 + height * shrink_factor)
            x2 = int(x2 - width * shrink_factor)
            y2 = int(y2 - height * shrink_factor)

            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (224, 224))
            face_array = preprocess_input(face_resized.astype(np.float32))
            face_array = np.expand_dims(face_array, axis=0)

            sequence.append(face_array)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            status = "Unknown"
            color = (255, 255, 255)

            if len(sequence) == SEQUENCE_LENGTH:
                seq_input = np.vstack(sequence).reshape(1, SEQUENCE_LENGTH, 224, 224, 3)
                pred = lstm_model.predict(seq_input, verbose=0)[0]

                # Debugging output for prediction
                print(f"Prediction value: {pred[0]:.4f}")

                # Lower the threshold to make it more sensitive to eye closure
                status = "Drowsy" if pred[0] > 0.5 else "Alert"  # Lowered threshold from 0.7 to 0.5
                color = (0, 0, 255) if status == "Drowsy" else (0, 255, 0)

                if status == "Drowsy":
                    threading.Thread(target=sound_alarm, daemon=True).start()

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Status: {status} ({pred[0]:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting... 'q' pressed.")
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Window closed.")


