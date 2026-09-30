"""
Train a vehicle classifier (EfficientNet-B0, transfer learning).

Dataset layout (ImageFolder):
    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg
    data/test/<class_name>/*.jpg

Run:
    python train_classifier.py --data data --epochs 25
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_loaders(root, img, bs, workers):
    norm = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img, img)), transforms.ToTensor(), norm,
    ])
    train_ds = datasets.ImageFolder(f"{root}/train", train_tf)
    val_ds = datasets.ImageFolder(f"{root}/val", eval_tf)
    test_ds = datasets.ImageFolder(f"{root}/test", eval_tf)
    mk = lambda ds, sh: DataLoader(ds, bs, shuffle=sh, num_workers=workers, pin_memory=True)
    return train_ds, mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)


def build_model(num_classes):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, num_classes)
    )
    return m


def set_backbone_trainable(model, flag):
    for p in model.features.parameters():
        p.requires_grad = flag


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="vehicle_classifier.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, train_dl, val_dl, test_dl = get_loaders(args.data, args.img, args.bs, args.workers)
    classes = train_ds.classes
    print("Classes:", classes)

    # class weights to handle imbalance
    counts = np.bincount(train_ds.targets)
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    model = build_model(len(classes)).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_acc = 0.0

    # Phase 1: train head only
    set_backbone_trainable(model, False)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    for ep in range(args.warmup_epochs):
        tl, ta = run_epoch(model, train_dl, criterion, device, opt, scaler)
        vl, va = run_epoch(model, val_dl, criterion, device)
        print(f"[warmup {ep+1}] train {ta:.3f} | val {va:.3f}")

    # Phase 2: fine-tune everything
    set_backbone_trainable(model, True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    fine_epochs = args.epochs - args.warmup_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fine_epochs)
    for ep in range(fine_epochs):
        tl, ta = run_epoch(model, train_dl, criterion, device, opt, scaler)
        vl, va = run_epoch(model, val_dl, criterion, device)
        sched.step()
        print(f"[ft {ep+1}/{fine_epochs}] loss {tl:.4f} train {ta:.3f} | val loss {vl:.4f} val {va:.3f}")
        if va > best_acc:
            best_acc = va
            torch.save({"state_dict": model.state_dict(), "classes": classes, "img": args.img}, args.out)
            print(f"  saved best ({best_acc:.3f})")

    # Final test evaluation
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in test_dl:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            gts += y.tolist()
    print(classification_report(gts, preds, target_names=classes, digits=4))

    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)
    json.dump(classes, open("classes.json", "w"))


if __name__ == "__main__":
    main()
