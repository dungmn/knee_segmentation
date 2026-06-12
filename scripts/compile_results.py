#!/usr/bin/env python3
"""Compile results from all four methods (Classical CV, U-Net, DeepLabV3, DeepLabV3+)
and display a formatted Markdown + LaTeX table.
"""

import argparse
import os
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Compile thesis evaluation results table")
    parser.add_argument("--unet-dir", type=str, required=True, help="U-Net experiment dir")
    parser.add_argument("--dlv3-dir", type=str, required=True, help="DeepLabV3 experiment dir")
    parser.add_argument("--dlv3p-dir", type=str, required=True, help="DeepLabV3+ experiment dir")
    parser.add_argument("--cv-dir", type=str, required=True, help="Dir containing cv_statistics.json (usually same as unet-dir)")
    return parser.parse_args()


def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Required file not found: {filepath}", file=sys.stderr)
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None


def main():
    args = parse_args()

    # Load results
    unet_summary = load_json(os.path.join(args.unet_dir, "cv_summary.json"))
    dlv3_summary = load_json(os.path.join(args.dlv3_dir, "cv_summary.json"))
    dlv3p_summary = load_json(os.path.join(args.dlv3p_dir, "cv_summary.json"))
    cv_summary = load_json(os.path.join(args.cv_dir, "cv_statistics.json"))

    missing = False
    if unet_summary is None:
        print(f"Missing U-Net results in {args.unet_dir}")
        missing = True
    if dlv3_summary is None:
        print(f"Missing DeepLabV3 results in {args.dlv3_dir}")
        missing = True
    if dlv3p_summary is None:
        print(f"Missing DeepLabV3+ results in {args.dlv3p_dir}")
        missing = True
    if cv_summary is None:
        print(f"Missing Classical CV results in {args.cv_dir}")
        missing = True

    if missing:
        print("\nCould not compile table because one or more results files are missing.", file=sys.stderr)
        sys.exit(1)

    # Extract metrics
    # Format: (Dice Mean, Dice Std, IoU Mean, IoU Std, Det Acc, Det F1)
    
    # Classical CV
    cv_dice_mean = cv_summary["segmentation"]["dice"]["mean"]
    cv_dice_std = cv_summary["segmentation"]["dice"]["std"]
    cv_iou_mean = cv_summary["segmentation"]["iou"]["mean"]
    cv_iou_std = cv_summary["segmentation"]["iou"]["std"]
    cv_det_acc = cv_summary["detection"]["accuracy"]["mean"]
    cv_det_f1 = cv_summary["detection"]["f1"]["mean"]
    
    # U-Net (smp)
    unet_dice_mean = unet_summary["segmentation"]["per_class"]["bakers_cyst"]["dice"]["mean"]
    unet_dice_std = unet_summary["segmentation"]["per_class"]["bakers_cyst"]["dice"]["std"]
    unet_iou_mean = unet_summary["segmentation"]["per_class"]["bakers_cyst"]["iou"]["mean"]
    unet_iou_std = unet_summary["segmentation"]["per_class"]["bakers_cyst"]["iou"]["std"]
    unet_det_acc = unet_summary["detection"]["accuracy"]["mean"]
    unet_det_f1 = unet_summary["detection"]["f1"]["mean"]

    # DeepLabV3
    dlv3_dice_mean = dlv3_summary["segmentation"]["per_class"]["bakers_cyst"]["dice"]["mean"]
    dlv3_dice_std = dlv3_summary["segmentation"]["per_class"]["bakers_cyst"]["dice"]["std"]
    dlv3_iou_mean = dlv3_summary["segmentation"]["per_class"]["bakers_cyst"]["iou"]["mean"]
    dlv3_iou_std = dlv3_summary["segmentation"]["per_class"]["bakers_cyst"]["iou"]["std"]
    dlv3_det_acc = dlv3_summary["detection"]["accuracy"]["mean"]
    dlv3_det_f1 = dlv3_summary["detection"]["f1"]["mean"]

    # DeepLabV3+
    dlv3p_dice_mean = dlv3p_summary["segmentation"]["per_class"]["bakers_cyst"]["dice"]["mean"]
    dlv3p_dice_std = dlv3p_summary["segmentation"]["per_class"]["bakers_cyst"]["dice"]["std"]
    dlv3p_iou_mean = dlv3p_summary["segmentation"]["per_class"]["bakers_cyst"]["iou"]["mean"]
    dlv3p_iou_std = dlv3p_summary["segmentation"]["per_class"]["bakers_cyst"]["iou"]["std"]
    dlv3p_det_acc = dlv3p_summary["detection"]["accuracy"]["mean"]
    dlv3p_det_f1 = dlv3p_summary["detection"]["f1"]["mean"]

    methods = [
        {
            "name": "Classical CV",
            "dice_mean": cv_dice_mean, "dice_std": cv_dice_std,
            "iou_mean": cv_iou_mean, "iou_std": cv_iou_std,
            "det_acc": cv_det_acc, "det_f1": cv_det_f1
        },
        {
            "name": "U-Net (ResNet-34)",
            "dice_mean": unet_dice_mean, "dice_std": unet_dice_std,
            "iou_mean": unet_iou_mean, "iou_std": unet_iou_std,
            "det_acc": unet_det_acc, "det_f1": unet_det_f1
        },
        {
            "name": "DeepLabV3 (ResNet-101)",
            "dice_mean": dlv3_dice_mean, "dice_std": dlv3_dice_std,
            "iou_mean": dlv3_iou_mean, "iou_std": dlv3_iou_std,
            "det_acc": dlv3_det_acc, "det_f1": dlv3_det_f1
        },
        {
            "name": "DeepLabV3+ (ResNet-101)",
            "dice_mean": dlv3p_dice_mean, "dice_std": dlv3p_dice_std,
            "iou_mean": dlv3p_iou_mean, "iou_std": dlv3p_iou_std,
            "det_acc": dlv3p_det_acc, "det_f1": dlv3p_det_f1
        }
    ]

    print("\n" + "=" * 90)
    print("COMPARISON RESULTS TABLE (Markdown)")
    print("=" * 90)
    print("| Method | Dice (mean±std) | IoU (mean±std) | Det-Acc | Det-F1 |")
    print("|---|---|---|---|---|")
    for m in methods:
        print(f"| {m['name']} | {m['dice_mean']:.4f} ± {m['dice_std']:.4f} | {m['iou_mean']:.4f} ± {m['iou_std']:.4f} | {m['det_acc']:.4f} | {m['det_f1']:.4f} |")

    print("\n" + "=" * 90)
    print("COMPARISON RESULTS TABLE (LaTeX)")
    print("=" * 90)
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Comparison of Segmentation and Detection Performance on Baker's Cyst}")
    print(r"\label{tab:bakers_cyst_comparison}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Method & Dice Score & IoU Score & Detection Acc & Detection F1 \\")
    print(r"\hline")
    for m in methods:
        print(f"{m['name']} & {m['dice_mean']:.4f} $\\pm$ {m['dice_std']:.4f} & {m['iou_mean']:.4f} $\\pm$ {m['iou_std']:.4f} & {m['det_acc']:.4f} & {m['det_f1']:.4f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
