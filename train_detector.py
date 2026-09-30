"""
OPTIONAL: fine-tune YOLOv8 on your own traffic dataset (needed only if the
pretrained COCO model misses vehicles in your images).

Create traffic.yaml:
    path: datasets/traffic
    train: images/train
    val: images/val
    names:
      0: car
      1: bus
      2: truck
      3: motorcycle

Labels must be in YOLO format (one .txt per image: class cx cy w h, normalized).
Free datasets: UA-DETRAC, Roboflow Universe "vehicle detection", Open Images (vehicle classes).

Run:  python train_detector.py

requirements:
    pip install torch torchvision ultralytics opencv-python scikit-learn matplotlib pillow
"""
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data="traffic.yaml",
    epochs=60,
    imgsz=640,
    batch=16,
    patience=15,
    augment=True,
    mosaic=1.0,
    fliplr=0.5,
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
)
metrics = model.val()
print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
# Best weights: runs/detect/train/weights/best.pt
