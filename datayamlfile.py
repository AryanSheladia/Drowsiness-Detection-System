yaml_content = """
path: /home/cl502_23/drowsiness_data/yolov8_train
train: images/train
val: images/val
test: images/test

names:
  0: notdrowsy
  1: drowsy
"""

with open("/home/cl502_23/drowsiness_data/yolov8_train/drowsiness.yaml", "w") as f:
    f.write(yaml_content.strip())

print("✅ YAML created at /home/cl502_23/drowsiness_data/yolov8_train/drowsiness.yaml")

