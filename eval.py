import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import KneeSegDataset
from src.models.model import build_deeplabv3
import pandas as pd
from rich import print
from rich.progress import track

CLASS_NAMES = {
    0: "background",
    1: "fat",
    2: "tendon",
    3: "muscle",
    4: "femur",
    5: "artery",
    6: "bakers_cyst",
}

EVAL_CLASSES = [1, 2, 3, 4, 5, 6]  # exclude background


def init_stats():
    return {c: {"TP": 0, "FP": 0, "FN": 0} for c in EVAL_CLASSES}

def update_stats(stats, pred, gt):
    for c in EVAL_CLASSES:
        pred_c = (pred == c)
        gt_c   = (gt == c)

        stats[c]["TP"] += np.logical_and(pred_c, gt_c).sum()
        stats[c]["FP"] += np.logical_and(pred_c, ~gt_c).sum()
        stats[c]["FN"] += np.logical_and(~pred_c, gt_c).sum()

def load_model(weights_path, model_name, num_classes, device):
    model = build_deeplabv3(num_classes, model_name=model_name).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def load_test_data(dataset_dir, seed):
    image_dir = f"{dataset_dir}/images"
    test_file = f"{dataset_dir}/split/seed_{seed}/test.txt"

    with open(test_file, "r") as f:
        test_images = [f"{image_dir}/{line.strip()}" for line in f.readlines()]
    test_masks = [p.replace("images", "masks") for p in test_images]

    return test_images, test_masks


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = [
        "data/processed/training/post_trans-27-random-flipped-batch_000",
        "data/processed/training/post_trans-baker_cyst-flipped-batch_000"
    ]

    seed = 16
    model_name = "deeplabv3_resnet50"
    # model_name = "deeplabv3_resnet101"

    weights_path = f"experiments/20260105-164514/deeplabv3_resnet50_seed_16/best_model.pth"

    model = load_model(weights_path, model_name, num_classes=7, device=device)

    test_imgs, test_masks = [], []
    for d in dataset_dir:
        t_imgs, t_masks = load_test_data(d, seed)
        test_imgs.extend(t_imgs)
        test_masks.extend(t_masks)


    test_loader  = DataLoader(KneeSegDataset(test_imgs, test_masks), batch_size=8, shuffle=False)

    stats = init_stats()

    with torch.no_grad():
        for images, gt_masks in track(test_loader, description="Evaluating"):
            images = images.to(device)
            gt_masks = gt_masks.cpu().numpy()      # (B, H, W)

            outputs = model(images)["out"]         # (B, 7, H, W)
            preds = torch.argmax(outputs, dim=1)   # (B, H, W)
            preds = preds.cpu().numpy()

            for i in range(preds.shape[0]):
                update_stats(stats, preds[i], gt_masks[i])

    print("\n=== Segmentation Results (6 classes) ===")

    ious, dices = [], []

    for c in EVAL_CLASSES:
        TP = stats[c]["TP"]
        FP = stats[c]["FP"]
        FN = stats[c]["FN"]

        iou  = TP / (TP + FP + FN + 1e-6)
        dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
        prec = TP / (TP + FP + 1e-6)
        rec  = TP / (TP + FN + 1e-6)

        ious.append(iou)
        dices.append(dice)

        print(f"{CLASS_NAMES[c]:15s} | Dice: {dice:.4f} | IoU: {iou:.4f} | P: {prec:.4f} | R: {rec:.4f}")

    print("\n--- Mean (macro, no background) ---")
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean IoU : {np.mean(ious):.4f}")
