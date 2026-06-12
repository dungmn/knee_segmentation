import argparse
import subprocess
import os
import glob
import json
from pathlib import Path
import numpy as np


def find_fold_files(experiment_dir):
    """Find all fold checkpoint and validation list files."""
    fold_pattern = os.path.join(experiment_dir, "fold_*_last_model.pth")
    fold_files = sorted(glob.glob(fold_pattern))
    
    if not fold_files:
        raise ValueError(f"No fold checkpoint files found in {experiment_dir}")
    
    folds = []
    for weights_file in fold_files:
        fold_name = os.path.basename(weights_file).replace("_last_model.pth", "")
        val_list = os.path.join(experiment_dir, f"{fold_name}_val_imgs.txt")
        
        if not os.path.exists(val_list):
            raise ValueError(f"Missing validation list: {val_list}")
        
        folds.append({
            "fold_name": fold_name,
            "weights": weights_file,
            "val_list": val_list,
        })
    
    return folds


def run_eval_fold(weights_path, val_list_path, method="model", batch_size=8, num_classes=7):
    """Run evaluation for a single fold."""
    cmd = [
        "python", "eval.py",
        "--method", method,
        "--weights", weights_path,
        "--val-list", val_list_path,
        "--batch-size", str(batch_size),
        "--num-classes", str(num_classes),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Warning: Fold evaluation returned non-zero exit code: {result.returncode}")
    
    return result.returncode


def aggregate_results(experiment_dir, results):
    """Aggregate metrics from all fold results."""
    all_mean_dices = []
    all_mean_ious = []
    all_class_dices = {}  # class_name -> [dice_fold1, dice_fold2, ...]
    all_class_ious = {}
    all_class_precisions = {}
    all_class_recalls = {}
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_specificities = []
    all_f1s = []
    
    for result in results:
        results_file = result["results_file"]
        if not os.path.exists(results_file):
            print(f"Warning: Results file not found: {results_file}")
            continue
        
        with open(results_file, "r") as f:
            fold_results = json.load(f)
        
        all_mean_dices.append(fold_results["segmentation"]["mean_dice"])
        all_mean_ious.append(fold_results["segmentation"]["mean_iou"])
        
        # Aggregate per-class metrics
        for class_name, metrics in fold_results["segmentation"]["per_class"].items():
            if class_name not in all_class_dices:
                all_class_dices[class_name] = []
                all_class_ious[class_name] = []
                all_class_precisions[class_name] = []
                all_class_recalls[class_name] = []
            all_class_dices[class_name].append(metrics["dice"])
            all_class_ious[class_name].append(metrics["iou"])
            all_class_precisions[class_name].append(metrics["precision"])
            all_class_recalls[class_name].append(metrics["recall"])
        
        all_accuracies.append(fold_results["detection"]["accuracy"])
        all_precisions.append(fold_results["detection"]["precision"])
        all_recalls.append(fold_results["detection"]["recall"])
        all_specificities.append(fold_results["detection"]["specificity"])
        all_f1s.append(fold_results["detection"]["f1"])
    
    if not all_mean_dices:
        print("No results to aggregate!")
        return
    
    # Compute statistics
    summary = {
        "segmentation": {
            "per_class": {
                class_name: {
                    "dice": {
                        "folds": all_class_dices[class_name],
                        "mean": float(np.mean(all_class_dices[class_name])),
                        "std": float(np.std(all_class_dices[class_name])),
                        "min": float(np.min(all_class_dices[class_name])),
                        "max": float(np.max(all_class_dices[class_name])),
                    },
                    "iou": {
                        "folds": all_class_ious[class_name],
                        "mean": float(np.mean(all_class_ious[class_name])),
                        "std": float(np.std(all_class_ious[class_name])),
                        "min": float(np.min(all_class_ious[class_name])),
                        "max": float(np.max(all_class_ious[class_name])),
                    },
                    "precision": {
                        "folds": all_class_precisions[class_name],
                        "mean": float(np.mean(all_class_precisions[class_name])),
                        "std": float(np.std(all_class_precisions[class_name])),
                        "min": float(np.min(all_class_precisions[class_name])),
                        "max": float(np.max(all_class_precisions[class_name])),
                    },
                    "recall": {
                        "folds": all_class_recalls[class_name],
                        "mean": float(np.mean(all_class_recalls[class_name])),
                        "std": float(np.std(all_class_recalls[class_name])),
                        "min": float(np.min(all_class_recalls[class_name])),
                        "max": float(np.max(all_class_recalls[class_name])),
                    },
                }
                for class_name in sorted(all_class_dices.keys())
            },
            "mean_dice": {
                "folds": all_mean_dices,
                "mean": float(np.mean(all_mean_dices)),
                "std": float(np.std(all_mean_dices)),
                "min": float(np.min(all_mean_dices)),
                "max": float(np.max(all_mean_dices)),
            },
            "mean_iou": {
                "folds": all_mean_ious,
                "mean": float(np.mean(all_mean_ious)),
                "std": float(np.std(all_mean_ious)),
                "min": float(np.min(all_mean_ious)),
                "max": float(np.max(all_mean_ious)),
            },
        },
        "detection": {
            "accuracy": {
                "folds": all_accuracies,
                "mean": float(np.mean(all_accuracies)),
                "std": float(np.std(all_accuracies)),
                "min": float(np.min(all_accuracies)),
                "max": float(np.max(all_accuracies)),
            },
            "precision": {
                "folds": all_precisions,
                "mean": float(np.mean(all_precisions)),
                "std": float(np.std(all_precisions)),
                "min": float(np.min(all_precisions)),
                "max": float(np.max(all_precisions)),
            },
            "recall": {
                "folds": all_recalls,
                "mean": float(np.mean(all_recalls)),
                "std": float(np.std(all_recalls)),
                "min": float(np.min(all_recalls)),
                "max": float(np.max(all_recalls)),
            },
            "specificity": {
                "folds": all_specificities,
                "mean": float(np.mean(all_specificities)),
                "std": float(np.std(all_specificities)),
                "min": float(np.min(all_specificities)),
                "max": float(np.max(all_specificities)),
            },
            "f1": {
                "folds": all_f1s,
                "mean": float(np.mean(all_f1s)),
                "std": float(np.std(all_f1s)),
                "min": float(np.min(all_f1s)),
                "max": float(np.max(all_f1s)),
            },
        },
    }
    
    # Save summary
    summary_file = os.path.join(experiment_dir, "cv_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved CV summary to: {summary_file}")
    
    # Print summary
    print("\n" + "=" * 90)
    print("CROSS-VALIDATION SUMMARY (across all folds)")
    print("=" * 90)
    print("\n[Per-Class Segmentation Metrics]")
    for class_name in sorted(summary["segmentation"]["per_class"].keys()):
        class_stats = summary["segmentation"]["per_class"][class_name]
        print(
            f"{class_name:15s} | "
            f"Dice: {class_stats['dice']['mean']:.4f} ± {class_stats['dice']['std']:.4f} | "
            f"IoU: {class_stats['iou']['mean']:.4f} ± {class_stats['iou']['std']:.4f} | "
            f"P: {class_stats['precision']['mean']:.4f} ± {class_stats['precision']['std']:.4f} | "
            f"R: {class_stats['recall']['mean']:.4f} ± {class_stats['recall']['std']:.4f}"
        )
    
    print("\n[Segmentation Metrics]")
    print(f"Mean Dice : {summary['segmentation']['mean_dice']['mean']:.4f} ± {summary['segmentation']['mean_dice']['std']:.4f} (min: {summary['segmentation']['mean_dice']['min']:.4f}, max: {summary['segmentation']['mean_dice']['max']:.4f})")
    print(f"Mean IoU  : {summary['segmentation']['mean_iou']['mean']:.4f} ± {summary['segmentation']['mean_iou']['std']:.4f} (min: {summary['segmentation']['mean_iou']['min']:.4f}, max: {summary['segmentation']['mean_iou']['max']:.4f})")
    
    print("\n[Detection Metrics (Cyst Presence)]")
    print(f"Accuracy    : {summary['detection']['accuracy']['mean']:.4f} ± {summary['detection']['accuracy']['std']:.4f} (min: {summary['detection']['accuracy']['min']:.4f}, max: {summary['detection']['accuracy']['max']:.4f})")
    print(f"Precision   : {summary['detection']['precision']['mean']:.4f} ± {summary['detection']['precision']['std']:.4f} (min: {summary['detection']['precision']['min']:.4f}, max: {summary['detection']['precision']['max']:.4f})")
    print(f"Recall      : {summary['detection']['recall']['mean']:.4f} ± {summary['detection']['recall']['std']:.4f} (min: {summary['detection']['recall']['min']:.4f}, max: {summary['detection']['recall']['max']:.4f})")
    print(f"Specificity : {summary['detection']['specificity']['mean']:.4f} ± {summary['detection']['specificity']['std']:.4f} (min: {summary['detection']['specificity']['min']:.4f}, max: {summary['detection']['specificity']['max']:.4f})")
    print(f"F1-score    : {summary['detection']['f1']['mean']:.4f} ± {summary['detection']['f1']['std']:.4f} (min: {summary['detection']['f1']['min']:.4f}, max: {summary['detection']['f1']['max']:.4f})")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all folds from a cross-validation experiment"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment directory containing fold_XX_last_model.pth and fold_XX_val_imgs.txt files",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="model",
        choices=["model", "cv"],
        help="Evaluation method: deep model or classical CV",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Number of classes in segmentation",
    )
    args = parser.parse_args()

    experiment_dir = os.path.abspath(args.experiment_dir)
    
    if not os.path.isdir(experiment_dir):
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    print(f"Evaluating experiment: {experiment_dir}")
    print()
    
    folds = find_fold_files(experiment_dir)
    print(f"Found {len(folds)} folds to evaluate\n")
    
    results = []
    for fold in folds:
        print("=" * 90)
        print(f"Evaluating {fold['fold_name']}")
        print(f"  Weights : {fold['weights']}")
        print(f"  Val list: {fold['val_list']}")
        print("=" * 90)
        
        exit_code = run_eval_fold(
            fold["weights"],
            fold["val_list"],
            method=args.method,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
        )
        
        results.append({
            "fold": fold["fold_name"],
            "status": "success" if exit_code == 0 else "failed",
            "results_file": os.path.join(experiment_dir, f"{fold['fold_name']}_results.json"),
        })
        print()
    
    print("=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)
    for result in results:
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(f"{status_icon} {result['fold']}: {result['status']}")
    print("=" * 90)
    
    # Aggregate results across folds
    if all(r["status"] == "success" for r in results):
        print("\nAGGREGATING RESULTS...")
        aggregate_results(experiment_dir, results)


if __name__ == "__main__":
    main()
