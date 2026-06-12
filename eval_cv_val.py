import argparse
import csv
import glob
import os
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
from rich import print
from rich.progress import track

from cv import segment_baker_cyst


CLASS_NAMES = {
    0: "background",
    1: "fat",
    2: "tendon",
    3: "muscle",
    4: "femur",
    5: "artery",
    6: "bakers_cyst",
}


def init_binary_stats() -> Dict[str, int]:
    return {"TP": 0, "FP": 0, "FN": 0, "TN": 0}


def update_binary_stats(stats: Dict[str, int], pred_positive: bool, gt_positive: bool) -> None:
    if pred_positive and gt_positive:
        stats["TP"] += 1
    elif pred_positive and not gt_positive:
        stats["FP"] += 1
    elif not pred_positive and gt_positive:
        stats["FN"] += 1
    else:
        stats["TN"] += 1


def compute_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float, float]:
    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return iou, dice, precision, recall


def compute_detection_metrics(det: Dict[str, int]) -> Dict[str, float]:
    det_tp = det["TP"]
    det_fp = det["FP"]
    det_fn = det["FN"]
    det_tn = det["TN"]
    total = det_tp + det_fp + det_fn + det_tn

    det_acc = (det_tp + det_tn) / (total + 1e-6)
    det_prec = det_tp / (det_tp + det_fp + 1e-6)
    det_rec = det_tp / (det_tp + det_fn + 1e-6)
    det_spec = det_tn / (det_tn + det_fp + 1e-6)
    det_f1 = (2 * det_prec * det_rec) / (det_prec + det_rec + 1e-6)

    return {
        "acc": det_acc,
        "precision": det_prec,
        "recall": det_rec,
        "specificity": det_spec,
        "f1": det_f1,
    }


def find_fold_val_files(experiment_dir: str) -> List[str]:
    pattern = os.path.join(experiment_dir, "fold_*_val_imgs.txt")
    return sorted(glob.glob(pattern))


def load_paths_from_list(list_file: str) -> List[str]:
    with open(list_file, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def image_to_mask_path(image_path: str) -> str:
    return image_path.replace("/images/", "/masks/")


def evaluate_fold(list_file: str, cyst_class_id: int, cv_debug: bool) -> Dict[str, object]:
    image_paths = load_paths_from_list(list_file)
    mask_paths = [image_to_mask_path(p) for p in image_paths]

    cyst_stats = {"TP": 0, "FP": 0, "FN": 0}
    detection_stats = init_binary_stats()

    for img_path, mask_path in track(
        zip(image_paths, mask_paths),
        total=len(image_paths),
        description=f"Evaluating {os.path.basename(list_file)}",
    ):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if gt_mask is None:
            raise ValueError(f"Failed to load ground-truth mask: {mask_path}")

        pred_mask, _, _, _, _ = segment_baker_cyst(img_path, debug=cv_debug)

        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(
                pred_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        gt_cyst = (gt_mask == cyst_class_id).astype(np.uint8)
        pred_cyst = (pred_mask > 0).astype(np.uint8)

        tp = int(np.logical_and(pred_cyst == 1, gt_cyst == 1).sum())
        fp = int(np.logical_and(pred_cyst == 1, gt_cyst == 0).sum())
        fn = int(np.logical_and(pred_cyst == 0, gt_cyst == 1).sum())

        cyst_stats["TP"] += tp
        cyst_stats["FP"] += fp
        cyst_stats["FN"] += fn

        pred_positive = bool(pred_cyst.any())
        gt_positive = bool(gt_cyst.any())
        update_binary_stats(detection_stats, pred_positive, gt_positive)

    iou, dice, prec, rec = compute_metrics(cyst_stats["TP"], cyst_stats["FP"], cyst_stats["FN"])

    return {
        "list_file": list_file,
        "n_images": len(image_paths),
        "seg_tp": cyst_stats["TP"],
        "seg_fp": cyst_stats["FP"],
        "seg_fn": cyst_stats["FN"],
        "seg_iou": iou,
        "seg_dice": dice,
        "seg_precision": prec,
        "seg_recall": rec,
        "det": detection_stats,
    }


def print_fold_result(result: Dict[str, object], cyst_class_id: int) -> None:
    det = result["det"]
    det_metrics = compute_detection_metrics(det)

    print("-" * 90)
    print(f"Fold list: {result['list_file']}")
    print(f"Images   : {result['n_images']}")
    print(
        f"{CLASS_NAMES[cyst_class_id]:15s} | Dice: {result['seg_dice']:.4f} | "
        f"IoU: {result['seg_iou']:.4f} | P: {result['seg_precision']:.4f} | R: {result['seg_recall']:.4f}"
    )
    print("Cyst Presence Detection (image-level):")
    print(f"  Accuracy: {det_metrics['acc']:.4f} | Precision: {det_metrics['precision']:.4f} | Recall: {det_metrics['recall']:.4f} | Specificity: {det_metrics['specificity']:.4f} | F1: {det_metrics['f1']:.4f}")
    det_tp = det["TP"]
    det_fp = det["FP"]
    det_fn = det["FN"]
    det_tn = det["TN"]
    print(f"Confusion: TP={det_tp}, FP={det_fp}, FN={det_fn}, TN={det_tn}")


def summarize_all(results: List[Dict[str, object]], cyst_class_id: int) -> Dict[str, object]:
    total_tp = sum(r["seg_tp"] for r in results)
    total_fp = sum(r["seg_fp"] for r in results)
    total_fn = sum(r["seg_fn"] for r in results)

    iou, dice, prec, rec = compute_metrics(total_tp, total_fp, total_fn)

    det_tp = sum(r["det"]["TP"] for r in results)
    det_fp = sum(r["det"]["FP"] for r in results)
    det_fn = sum(r["det"]["FN"] for r in results)
    det_tn = sum(r["det"]["TN"] for r in results)
    det_metrics = compute_detection_metrics({"TP": det_tp, "FP": det_fp, "FN": det_fn, "TN": det_tn})

    total_images = sum(r["n_images"] for r in results)

    print("=" * 90)
    print("Overall across all fold val lists")
    print(
        f"{CLASS_NAMES[cyst_class_id]:15s} | Dice: {dice:.4f} | "
        f"IoU: {iou:.4f} | P: {prec:.4f} | R: {rec:.4f}"
    )
    print("Cyst Presence Detection (image-level):")
    print(f"  Accuracy: {det_metrics['acc']:.4f} | Precision: {det_metrics['precision']:.4f} | Recall: {det_metrics['recall']:.4f} | Specificity: {det_metrics['specificity']:.4f} | F1: {det_metrics['f1']:.4f}")
    print(f"Confusion: TP={det_tp}, FP={det_fp}, FN={det_fn}, TN={det_tn}")

    return {
        "list_file": "overall",
        "n_images": total_images,
        "seg_tp": total_tp,
        "seg_fp": total_fp,
        "seg_fn": total_fn,
        "seg_iou": iou,
        "seg_dice": dice,
        "seg_precision": prec,
        "seg_recall": rec,
        "det": {"TP": det_tp, "FP": det_fp, "FN": det_fn, "TN": det_tn},
    }


def compute_fold_statistics(results: List[Dict[str, object]]) -> Dict[str, object]:
    """Compute mean and std statistics across all folds."""
    fold_dices = [r["seg_dice"] for r in results]
    fold_ious = [r["seg_iou"] for r in results]
    fold_precisions = [r["seg_precision"] for r in results]
    fold_recalls = [r["seg_recall"] for r in results]
    
    fold_det_accs = []
    fold_det_precs = []
    fold_det_recs = []
    fold_det_specs = []
    fold_det_f1s = []
    
    for r in results:
        det_metrics = compute_detection_metrics(r["det"])
        fold_det_accs.append(det_metrics["acc"])
        fold_det_precs.append(det_metrics["precision"])
        fold_det_recs.append(det_metrics["recall"])
        fold_det_specs.append(det_metrics["specificity"])
        fold_det_f1s.append(det_metrics["f1"])
    
    return {
        "segmentation": {
            "dice": {
                "values": fold_dices,
                "mean": float(np.mean(fold_dices)),
                "std": float(np.std(fold_dices)),
                "min": float(np.min(fold_dices)),
                "max": float(np.max(fold_dices)),
            },
            "iou": {
                "values": fold_ious,
                "mean": float(np.mean(fold_ious)),
                "std": float(np.std(fold_ious)),
                "min": float(np.min(fold_ious)),
                "max": float(np.max(fold_ious)),
            },
            "precision": {
                "values": fold_precisions,
                "mean": float(np.mean(fold_precisions)),
                "std": float(np.std(fold_precisions)),
                "min": float(np.min(fold_precisions)),
                "max": float(np.max(fold_precisions)),
            },
            "recall": {
                "values": fold_recalls,
                "mean": float(np.mean(fold_recalls)),
                "std": float(np.std(fold_recalls)),
                "min": float(np.min(fold_recalls)),
                "max": float(np.max(fold_recalls)),
            },
        },
        "detection": {
            "accuracy": {
                "values": fold_det_accs,
                "mean": float(np.mean(fold_det_accs)),
                "std": float(np.std(fold_det_accs)),
                "min": float(np.min(fold_det_accs)),
                "max": float(np.max(fold_det_accs)),
            },
            "precision": {
                "values": fold_det_precs,
                "mean": float(np.mean(fold_det_precs)),
                "std": float(np.std(fold_det_precs)),
                "min": float(np.min(fold_det_precs)),
                "max": float(np.max(fold_det_precs)),
            },
            "recall": {
                "values": fold_det_recs,
                "mean": float(np.mean(fold_det_recs)),
                "std": float(np.std(fold_det_recs)),
                "min": float(np.min(fold_det_recs)),
                "max": float(np.max(fold_det_recs)),
            },
            "specificity": {
                "values": fold_det_specs,
                "mean": float(np.mean(fold_det_specs)),
                "std": float(np.std(fold_det_specs)),
                "min": float(np.min(fold_det_specs)),
                "max": float(np.max(fold_det_specs)),
            },
            "f1": {
                "values": fold_det_f1s,
                "mean": float(np.mean(fold_det_f1s)),
                "std": float(np.std(fold_det_f1s)),
                "min": float(np.min(fold_det_f1s)),
                "max": float(np.max(fold_det_f1s)),
            },
        },
    }


def print_cv_statistics(stats: Dict[str, object], cyst_class_id: int) -> None:
    """Print cross-validation statistics."""
    print("\n" + "=" * 90)
    print("CROSS-VALIDATION STATISTICS (across all folds)")
    print("=" * 90)
    
    print("\n[Segmentation Metrics (Baker Cyst)]")
    seg_stats = stats["segmentation"]
    print(f"Dice      : {seg_stats['dice']['mean']:.4f} ± {seg_stats['dice']['std']:.4f} (min: {seg_stats['dice']['min']:.4f}, max: {seg_stats['dice']['max']:.4f})")
    print(f"IoU       : {seg_stats['iou']['mean']:.4f} ± {seg_stats['iou']['std']:.4f} (min: {seg_stats['iou']['min']:.4f}, max: {seg_stats['iou']['max']:.4f})")
    print(f"Precision : {seg_stats['precision']['mean']:.4f} ± {seg_stats['precision']['std']:.4f} (min: {seg_stats['precision']['min']:.4f}, max: {seg_stats['precision']['max']:.4f})")
    print(f"Recall    : {seg_stats['recall']['mean']:.4f} ± {seg_stats['recall']['std']:.4f} (min: {seg_stats['recall']['min']:.4f}, max: {seg_stats['recall']['max']:.4f})")
    
    print("\n[Detection Metrics (Cyst Presence - image level)]")
    det_stats = stats["detection"]
    print(f"Accuracy    : {det_stats['accuracy']['mean']:.4f} ± {det_stats['accuracy']['std']:.4f} (min: {det_stats['accuracy']['min']:.4f}, max: {det_stats['accuracy']['max']:.4f})")
    print(f"Precision   : {det_stats['precision']['mean']:.4f} ± {det_stats['precision']['std']:.4f} (min: {det_stats['precision']['min']:.4f}, max: {det_stats['precision']['max']:.4f})")
    print(f"Recall      : {det_stats['recall']['mean']:.4f} ± {det_stats['recall']['std']:.4f} (min: {det_stats['recall']['min']:.4f}, max: {det_stats['recall']['max']:.4f})")
    print(f"Specificity : {det_stats['specificity']['mean']:.4f} ± {det_stats['specificity']['std']:.4f} (min: {det_stats['specificity']['min']:.4f}, max: {det_stats['specificity']['max']:.4f})")
    print(f"F1-score    : {det_stats['f1']['mean']:.4f} ± {det_stats['f1']['std']:.4f} (min: {det_stats['f1']['min']:.4f}, max: {det_stats['f1']['max']:.4f})")
    print("=" * 90)


def save_results_csv(csv_path: str, results: List[Dict[str, object]], overall_result: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    rows = results + [overall_result]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "list_file",
                "n_images",
                "seg_tp",
                "seg_fp",
                "seg_fn",
                "seg_dice",
                "seg_iou",
                "seg_precision",
                "seg_recall",
                "det_tp",
                "det_fp",
                "det_fn",
                "det_tn",
                "det_accuracy",
                "det_precision",
                "det_recall",
                "det_specificity",
                "det_f1",
            ]
        )

        for r in rows:
            det = r["det"]
            det_metrics = compute_detection_metrics(det)
            split = "overall" if r["list_file"] == "overall" else os.path.basename(r["list_file"]).replace("_val_imgs.txt", "")
            writer.writerow(
                [
                    split,
                    r["list_file"],
                    r["n_images"],
                    r["seg_tp"],
                    r["seg_fp"],
                    r["seg_fn"],
                    f"{r['seg_dice']:.6f}",
                    f"{r['seg_iou']:.6f}",
                    f"{r['seg_precision']:.6f}",
                    f"{r['seg_recall']:.6f}",
                    det["TP"],
                    det["FP"],
                    det["FN"],
                    det["TN"],
                    f"{det_metrics['acc']:.6f}",
                    f"{det_metrics['precision']:.6f}",
                    f"{det_metrics['recall']:.6f}",
                    f"{det_metrics['specificity']:.6f}",
                    f"{det_metrics['f1']:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate classical CV Baker cyst segmentation on saved fold val image lists"
    )
    parser.add_argument(
        "-e",
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment folder containing fold_XX_val_imgs.txt files",
    )
    parser.add_argument(
        "--cyst-class-id",
        type=int,
        default=6,
        help="Class id for baker cyst in mask annotations",
    )
    parser.add_argument(
        "--cv-debug",
        action="store_true",
        help="Enable debug mode for cv.segment_baker_cyst",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to save CSV results (default: <experiment-dir>/cv_val_results.csv)",
    )
    args = parser.parse_args()

    fold_files = find_fold_val_files(args.experiment_dir)
    if not fold_files:
        raise ValueError(
            f"No fold val list files found in {args.experiment_dir}. "
            "Expected files like fold_01_val_imgs.txt"
        )

    print(f"Found {len(fold_files)} fold validation list file(s)")

    results = []
    for fold_file in fold_files:
        result = evaluate_fold(fold_file, cyst_class_id=args.cyst_class_id, cv_debug=args.cv_debug)
        print_fold_result(result, cyst_class_id=args.cyst_class_id)
        results.append(result)

    overall_result = summarize_all(results, cyst_class_id=args.cyst_class_id)

    csv_path = args.csv_path or os.path.join(args.experiment_dir, "cv_val_results.csv")
    save_results_csv(csv_path, results, overall_result)
    print(f"Saved CSV results to: {csv_path}")
    
    # Compute and print cross-validation statistics
    stats = compute_fold_statistics(results)
    print_cv_statistics(stats, cyst_class_id=args.cyst_class_id)
    
    # Save statistics to JSON
    stats_json_path = os.path.join(args.experiment_dir, "cv_statistics.json")
    with open(stats_json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to: {stats_json_path}")


if __name__ == "__main__":
    main()
