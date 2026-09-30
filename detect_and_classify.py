"""
Detect vehicles with YOLOv8, then classify each crop with the trained CNN.

Run on an image:   python detect_and_classify.py --source road.jpg
Run on a video:    python detect_and_classify.py --source traffic.mp4
Run on webcam:     python detect_and_classify.py --source 0

Use --det_weights runs/detect/train/weights/best.pt if you fine-tuned YOLO.
"""
import argparse
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

# COCO ids: 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
COCO_VEHICLES = [1, 2, 3, 5, 7]


def load_classifier(path, device):
    ckpt = torch.load(path, map_location=device)
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, len(ckpt["classes"]))
    )
    m.load_state_dict(ckpt["state_dict"])
    tf = transforms.Compose([
        transforms.Resize((ckpt["img"], ckpt["img"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return m.to(device).eval(), ckpt["classes"], tf


@torch.no_grad()
def classify_crops(crops, model, classes, tf, device):
    if not crops:
        return []
    batch = torch.stack([tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops]).to(device)
    probs = torch.softmax(model(batch), dim=1)
    conf, idx = probs.max(1)
    return [(classes[i], c) for i, c in zip(idx.tolist(), conf.tolist())]


def process_frame(frame, detector, clf, classes, tf, device, det_conf, det_classes):
    res = detector(frame, conf=det_conf, classes=det_classes, verbose=False)[0]
    boxes, crops = [], []
    h, w = frame.shape[:2]
    for b in res.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        boxes.append((x1, y1, x2, y2))
        crops.append(frame[y1:y2, x1:x2])
    labels = classify_crops(crops, clf, classes, tf, device)
    counts = {}
    for (x1, y1, x2, y2), (name, conf) in zip(boxes, labels):
        counts[name] = counts.get(name, 0) + 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 200, 0), -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    y = 25
    for k, v in counts.items():
        cv2.putText(frame, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 25
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--clf_weights", default="vehicle_classifier.pt")
    ap.add_argument("--det_weights", default="yolov8m.pt")
    ap.add_argument("--det_conf", type=float, default=0.35)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = YOLO(args.det_weights)
    clf, classes, tf = load_classifier(args.clf_weights, device)
    # Pretrained COCO model -> restrict to vehicle ids; custom-trained model -> use all classes
    det_classes = COCO_VEHICLES if args.det_weights.startswith("yolov8") and "runs" not in args.det_weights else None

    if args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img = cv2.imread(args.source)
        img = process_frame(img, detector, clf, classes, tf, device, args.det_conf, det_classes)
        cv2.imwrite(f"{args.out}.jpg", img)
        print(f"Saved {args.out}.jpg")
        return

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    size = (int(cap.get(3)), int(cap.get(4)))
    writer = cv2.VideoWriter(f"{args.out}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = process_frame(frame, detector, clf, classes, tf, device, args.det_conf, det_classes)
        writer.write(frame)
        cv2.imshow("Traffic", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"Saved {args.out}.mp4")


if __name__ == "__main__":
    main()
