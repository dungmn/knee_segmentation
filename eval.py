import argparse
import os
from typing import Dict, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
from src.data.dataset import KneeSegDataset
from src.data.augmentations import get_val_transforms
from src.models.model import build_deeplabv3
from src.models.unet import build_unet
from cv import segment_baker_cyst
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


def init_stats() -> Dict[int, Dict[str, int]]:
    """Initialize per-class segmentation statistics."""
    return {c: {"TP": 0, "FP": 0, "FN": 0} for c in EVAL_CLASSES}


def init_binary_stats() -> Dict[str, int]:
    """Initialize binary classification statistics (detection)."""
    return {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

def update_stats(stats: Dict[int, Dict[str, int]], pred: np.ndarray, gt: np.ndarray) -> None:
    """Update per-class segmentation statistics.
    
    Args:
        stats: Dictionary to accumulate statistics
        pred: Predicted segmentation mask
        gt: Ground truth segmentation mask
    """
    for c in EVAL_CLASSES:
        pred_c = (pred == c)
        gt_c   = (gt == c)

        stats[c]["TP"] += np.logical_and(pred_c, gt_c).sum()
        stats[c]["FP"] += np.logical_and(pred_c, ~gt_c).sum()
        stats[c]["FN"] += np.logical_and(~pred_c, gt_c).sum()

def update_binary_stats(stats, pred_positive, gt_positive):
    if pred_positive and gt_positive:
        stats["TP"] += 1
    elif pred_positive and not gt_positive:
        stats["FP"] += 1
    elif not pred_positive and gt_positive:
        stats["FN"] += 1
    else:
        stats["TN"] += 1

def load_model(weights_path, model_name, num_classes, device):
    if model_name.startswith("deeplabv3"):
        model = build_deeplabv3(num_classes, model_name=model_name).to(device)
    else:
        model = build_unet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def load_test_data(dataset_dir, seed, file_name="test.txt"):
    image_dir = f"{dataset_dir}/images"
    test_file = f"{dataset_dir}/split/seed_{seed}/{file_name}"

    with open(test_file, "r") as f:
        test_images = [f"{image_dir}/{line.strip()}" for line in f.readlines()]
    test_masks = [p.replace("images", "masks") for p in test_images]

    return test_images, test_masks


def compute_metrics(tp, fp, fn):   
    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return iou, dice, precision, recall


def evaluate_model(args, test_imgs, test_masks, device):
    model_name = args.weights.split("/")[-2].split("-seed")[0]
    print(f"Loading model '{model_name}' from {args.weights}...")
    model = load_model(args.weights, model_name, num_classes=args.num_classes, device=device)
    cyst_class_id = 6

    val_transform = get_val_transforms(img_size=512)
    test_loader = DataLoader(
        KneeSegDataset(test_imgs, test_masks, transform=val_transform),
        batch_size=args.batch_size,
        shuffle=False,
    )

    stats = init_stats()
    detection_stats = init_binary_stats()

    with torch.no_grad():
        for images, gt_masks in track(test_loader, description="Evaluating model"):
            images = images.to(device)

            if isinstance(gt_masks, torch.Tensor):
                gt_np = gt_masks.cpu().numpy()
            else:
                gt_np = np.array(gt_masks)

            if gt_np.ndim == 4 and gt_np.shape[1] == 1:
                gt_np = gt_np.squeeze(1)
            if model_name.startswith("deeplabv3"):  
                outputs = model(images)["out"]
            else:
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            preds = preds.cpu().numpy()

            for i in range(preds.shape[0]):
                update_stats(stats, preds[i], gt_np[i])
                pred_positive = bool((preds[i] == cyst_class_id).any())
                gt_positive = bool((gt_np[i] == cyst_class_id).any())
                update_binary_stats(detection_stats, pred_positive, gt_positive)

    print("\n=== Segmentation Results (6 classes) ===")

    ious, dices = [], []
    for c in EVAL_CLASSES:
        tp = stats[c]["TP"]
        fp = stats[c]["FP"]
        fn = stats[c]["FN"]

        iou, dice, prec, rec = compute_metrics(tp, fp, fn)
        ious.append(iou)
        dices.append(dice)
        print(f"{CLASS_NAMES[c]:15s} | Dice: {dice:.4f} | IoU: {iou:.4f} | P: {prec:.4f} | R: {rec:.4f}")

    print("\n--- Mean (macro, no background) ---")
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean IoU : {np.mean(ious):.4f}")

    det_tp = detection_stats["TP"]
    det_fp = detection_stats["FP"]
    det_fn = detection_stats["FN"]
    det_tn = detection_stats["TN"]
    total = det_tp + det_fp + det_fn + det_tn

    det_acc = (det_tp + det_tn) / (total + 1e-6)
    det_prec = det_tp / (det_tp + det_fp + 1e-6)
    det_rec = det_tp / (det_tp + det_fn + 1e-6)
    det_spec = det_tn / (det_tn + det_fp + 1e-6)
    det_f1 = (2 * det_prec * det_rec) / (det_prec + det_rec + 1e-6)

    print("\n--- Cyst Presence Detection (image-level) ---")
    print(f"Accuracy   : {det_acc:.4f}")
    print(f"Precision  : {det_prec:.4f}")
    print(f"Recall     : {det_rec:.4f}")
    print(f"Specificity: {det_spec:.4f}")
    print(f"F1-score   : {det_f1:.4f}")
    print(f"Confusion  : TP={det_tp}, FP={det_fp}, FN={det_fn}, TN={det_tn}")


def evaluate_cv(args, test_imgs, test_masks):
    cyst_class_id = 6
    cyst_stats = {cyst_class_id: {"TP": 0, "FP": 0, "FN": 0}}
    detection_stats = init_binary_stats()

    for img_path, mask_path in track(zip(test_imgs, test_masks), total=len(test_imgs), description="Evaluating CV method"):
        # if img_path != "data/processed/training/post_trans-27-random-flipped-batch_000/images/72bb4eec-f020-11ed-b527-0a580a5f736a_27.png":
        #     continue
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if gt_mask is None:
            raise ValueError(f"Failed to load ground-truth mask: {mask_path}")

        pred_mask, _, _, _ = segment_baker_cyst(img_path, debug=args.cv_debug)

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        gt_cyst = (gt_mask == cyst_class_id).astype(np.uint8)
        print("===")
        print(f"Evaluating image: {img_path}")
        print("count value in gt_cyst", np.unique(gt_cyst, return_counts=True))
        

        pred_cyst = (pred_mask > 0).astype(np.uint8)
        print("count value in pred_cyst", np.unique(pred_cyst, return_counts=True))


        tp = np.logical_and(pred_cyst == 1, gt_cyst == 1).sum()
        fp = np.logical_and(pred_cyst == 1, gt_cyst == 0).sum()
        fn = np.logical_and(pred_cyst == 0, gt_cyst == 1).sum()

        print(f"Image: {os.path.basename(img_path)} | TP: {tp}, FP: {fp}, FN: {fn}")

        cyst_stats[cyst_class_id]["TP"] += tp
        cyst_stats[cyst_class_id]["FP"] += fp
        cyst_stats[cyst_class_id]["FN"] += fn

        pred_positive = bool(pred_cyst.any())
        gt_positive = bool(gt_cyst.any())
        update_binary_stats(detection_stats, pred_positive, gt_positive)

    print("\n=== CV Method Results (bakers_cyst only) ===")

    tp = cyst_stats[cyst_class_id]["TP"]
    fp = cyst_stats[cyst_class_id]["FP"]
    fn = cyst_stats[cyst_class_id]["FN"]
    iou, dice, prec, rec = compute_metrics(tp, fp, fn)
    print(f"{CLASS_NAMES[cyst_class_id]:15s} | Dice: {dice:.4f} | IoU: {iou:.4f} | P: {prec:.4f} | R: {rec:.4f}")

    det_tp = detection_stats["TP"]
    det_fp = detection_stats["FP"]
    det_fn = detection_stats["FN"]
    det_tn = detection_stats["TN"]
    total = det_tp + det_fp + det_fn + det_tn

    det_acc = (det_tp + det_tn) / (total + 1e-6)
    det_prec = det_tp / (det_tp + det_fp + 1e-6)
    det_rec = det_tp / (det_tp + det_fn + 1e-6)
    det_spec = det_tn / (det_tn + det_fp + 1e-6)
    det_f1 = (2 * det_prec * det_rec) / (det_prec + det_rec + 1e-6)

    print("\n--- Cyst Presence Detection (image-level) ---")
    print(f"Accuracy   : {det_acc:.4f}")
    print(f"Precision  : {det_prec:.4f}")
    print(f"Recall     : {det_rec:.4f}")
    print(f"Specificity: {det_spec:.4f}")
    print(f"F1-score   : {det_f1:.4f}")
    print(f"Confusion  : TP={det_tp}, FP={det_fp}, FN={det_fn}, TN={det_tn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--method", type=str, default="model", choices=["model", "cv"], help="Evaluation method: deep model or classical CV")
    parser.add_argument("--weights", type=str, default=f"experiments/20260129-162624/deeplabv3_resnet50_seed_16/best_model.pth")
    parser.add_argument("--dataset-dir", nargs="+", default=[
        "data/processed/training/post_trans-27-random-flipped-batch_000",
        "data/processed/training/post_trans-baker_cyst-flipped-batch_000"
    ])
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--cv-debug", action="store_true", help="Enable debug mode for cv.segment_baker_cyst")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_imgs, test_masks = [], []
    for d in args.dataset_dir:
        t_imgs, t_masks = load_test_data(d, args.seed, file_name="all.txt")
        test_imgs.extend(t_imgs)
        test_masks.extend(t_masks)

    print(f"Found {len(test_imgs)} test images from {len(args.dataset_dir)} dataset(s)")

    if args.method == "model":
        evaluate_model(args, test_imgs, test_masks, device)
    else:
        evaluate_cv(args, test_imgs, test_masks)

    # visualize all results
    output_dir = "logs/eval_results"
    os.makedirs(output_dir, exist_ok=True)

    for img_path, mask_path in zip(test_imgs, test_masks):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if gt_mask is None:
            raise ValueError(f"Failed to load ground-truth mask: {mask_path}")

        pred_mask, _, _, _ = segment_baker_cyst(img_path, debug=args.cv_debug)

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = cv2.addWeighted(pred_mask.astype(np.uint8)*255, 0.5, gt_mask.astype(np.uint8)*255, 0.5, 0)
        cv2.imwrite(f"{output_dir}/{os.path.basename(img_path)}", overlay)