from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Use yolov8s.pt or yolov8m.pt if you want higher accuracy

model.train(
    data="/home/cl502_23/drowsiness_data/yolov8_train/drowsiness.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    name="drowsiness_yolov8",
    workers=2
)
