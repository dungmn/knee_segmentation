import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.dataset import KneeSegDataset
from src.data.augmentations import get_val_transforms
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
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--weights", type=str, default=f"experiments/20260129-162624/deeplabv3_resnet50_seed_16/best_model.pth")
    parser.add_argument("--dataset-dir", nargs="+", default=[
        "data/processed/training/post_trans-27-random-flipped-batch_000",
        "data/processed/training/post_trans-baker_cyst-flipped-batch_000"
    ])
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get model name from weights path
    model_name = args.weights.split("/")[-2].split("-seed")[0]
    model = load_model(args.weights, model_name, num_classes=args.num_classes, device=device)

    test_imgs, test_masks = [], []
    for d in args.dataset_dir:
        t_imgs, t_masks = load_test_data(d, args.seed)
        test_imgs.extend(t_imgs)
        test_masks.extend(t_masks)

    print(f"Found {len(test_imgs)} test images from {len(args.dataset_dir)} dataset(s)")

    val_transform = get_val_transforms(img_size=512)
    test_loader  = DataLoader(KneeSegDataset(test_imgs, test_masks, transform=val_transform), batch_size=args.batch_size, shuffle=False)
    # test_loader  = DataLoader(KneeSegDataset(test_imgs, test_masks, transform=None), batch_size=args.batch_size, shuffle=False)


    stats = init_stats()

    with torch.no_grad():
        for images, gt_masks in track(test_loader, description="Evaluating"):
            images = images.to(device)

            # gt_masks may be a torch tensor (B, H, W) or (B, 1, H, W) depending on transforms
            if isinstance(gt_masks, torch.Tensor):
                gt_np = gt_masks.cpu().numpy()
            else:
                gt_np = np.array(gt_masks)

            # If mask has channel dim (B,1,H,W) squeeze to (B,H,W)
            if gt_np.ndim == 4 and gt_np.shape[1] == 1:
                gt_np = gt_np.squeeze(1)

            outputs = model(images)["out"]         # (B, C, H, W)
            preds = torch.argmax(outputs, dim=1)   # (B, H, W)
            preds = preds.cpu().numpy()

            for i in range(preds.shape[0]):
                update_stats(stats, preds[i], gt_np[i])

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
