"""Generate comparison report for all models including per-class and Baker's cyst metrics."""
import json
import os

EVAL_DIR = "experiments/20260612-174629"
MODELS = [
    ("U-Net (resnet34)",       "unet_resnet34-seed_16-cv5"),
    ("U-Net (resnet101)",      "Unet_resnet101-seed_16-cv5"),
    ("DeepLabV3 (resnet101)",  "deeplabv3_resnet101-seed_16-cv5"),
    ("DeepLabV3+ (resnet101)", "deeplabv3plus_resnet101-seed_16-cv5"),
]
CV_STATS_PATH = "experiments/20260403-104345/deeplabv3_resnet101-seed_16-cv5/cv_statistics.json"
CLASS_NAMES = ["fat", "tendon", "muscle", "femur", "artery", "bakers_cyst"]


def pm(mean, std):
    return f"{mean:.4f} +/- {std:.4f}"


def main():
    results = {}

    for label, model_dir in MODELS:
        path = os.path.join(EVAL_DIR, model_dir, "cv_summary.json")
        with open(path) as f:
            data = json.load(f)

        model_results = {"per_class": {}, "mean_all_classes": {}, "detection": {}}

        for cls in CLASS_NAMES:
            c = data["segmentation"]["per_class"][cls]
            model_results["per_class"][cls] = {
                "dice":      {"mean": c["dice"]["mean"],      "std": c["dice"]["std"]},
                "iou":       {"mean": c["iou"]["mean"],       "std": c["iou"]["std"]},
                "precision": {"mean": c["precision"]["mean"], "std": c["precision"]["std"]},
                "recall":    {"mean": c["recall"]["mean"],    "std": c["recall"]["std"]},
            }

        model_results["mean_all_classes"] = {
            "dice": {"mean": data["segmentation"]["mean_dice"]["mean"], "std": data["segmentation"]["mean_dice"]["std"]},
            "iou":  {"mean": data["segmentation"]["mean_iou"]["mean"],  "std": data["segmentation"]["mean_iou"]["std"]},
        }

        det = data["detection"]
        for k in ["accuracy", "precision", "recall", "specificity", "f1"]:
            model_results["detection"][k] = {"mean": det[k]["mean"], "std": det[k]["std"]}

        results[label] = model_results

    with open(CV_STATS_PATH) as f:
        cv = json.load(f)
    results["Classical CV"] = {
        "per_class": {
            "bakers_cyst": {
                "dice":      {"mean": cv["segmentation"]["dice"]["mean"],      "std": cv["segmentation"]["dice"]["std"]},
                "iou":       {"mean": cv["segmentation"]["iou"]["mean"],       "std": cv["segmentation"]["iou"]["std"]},
                "precision": {"mean": cv["segmentation"]["precision"]["mean"], "std": cv["segmentation"]["precision"]["std"]},
                "recall":    {"mean": cv["segmentation"]["recall"]["mean"],    "std": cv["segmentation"]["recall"]["std"]},
            }
        },
        "mean_all_classes": None,
        "detection": {
            "accuracy":    {"mean": cv["detection"]["accuracy"]["mean"],    "std": cv["detection"]["accuracy"]["std"]},
            "precision":   {"mean": cv["detection"]["precision"]["mean"],   "std": cv["detection"]["precision"]["std"]},
            "recall":      {"mean": cv["detection"]["recall"]["mean"],      "std": cv["detection"]["recall"]["std"]},
            "specificity": {"mean": cv["detection"]["specificity"]["mean"], "std": cv["detection"]["specificity"]["std"]},
            "f1":          {"mean": cv["detection"]["f1"]["mean"],          "std": cv["detection"]["f1"]["std"]},
        },
    }

    # Save JSON
    json_path = os.path.join(EVAL_DIR, "bakers_cyst_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Write text report
    txt_path = os.path.join(EVAL_DIR, "bakers_cyst_comparison.txt")
    with open(txt_path, "w") as fp:
        fp.write("=" * 120 + "\n")
        fp.write("5-Fold Cross-Validation Results — All Methods\n")
        fp.write("=" * 120 + "\n")

        # Section 1
        fp.write("\n")
        fp.write("1. SEGMENTATION — MEAN ACROSS ALL CLASSES (6 classes, excl. background)\n")
        fp.write("-" * 100 + "\n")
        fp.write(f"{'Model':<25} | {'Mean Dice':>16} | {'Mean IoU':>16}\n")
        fp.write("-" * 100 + "\n")
        for label in results:
            mac = results[label]["mean_all_classes"]
            if mac:
                fp.write(f"{label:<25} | {pm(mac['dice']['mean'], mac['dice']['std']):>16} | {pm(mac['iou']['mean'], mac['iou']['std']):>16}\n")
            else:
                fp.write(f"{label:<25} | {'N/A':>16} | {'N/A':>16}\n")

        # Section 2
        fp.write("\n\n")
        fp.write("2. SEGMENTATION — PER-CLASS DICE SCORES\n")
        fp.write("-" * 145 + "\n")
        header = f"{'Model':<25}"
        for cls in CLASS_NAMES:
            header += f" | {cls:>16}"
        fp.write(header + "\n")
        fp.write("-" * 145 + "\n")
        for label in results:
            row = f"{label:<25}"
            for cls in CLASS_NAMES:
                pc = results[label]["per_class"].get(cls)
                if pc:
                    row += f" | {pm(pc['dice']['mean'], pc['dice']['std']):>16}"
                else:
                    row += f" | {'N/A':>16}"
            fp.write(row + "\n")

        # Section 3
        fp.write("\n\n")
        fp.write("3. SEGMENTATION — BAKER'S CYST (class 6) DETAILED\n")
        fp.write("-" * 110 + "\n")
        fp.write(f"{'Model':<25} | {'Dice':>16} | {'IoU':>16} | {'Precision':>16} | {'Recall':>16}\n")
        fp.write("-" * 110 + "\n")
        for label in results:
            bc = results[label]["per_class"].get("bakers_cyst")
            if bc:
                fp.write(
                    f"{label:<25} | {pm(bc['dice']['mean'], bc['dice']['std']):>16}"
                    f" | {pm(bc['iou']['mean'], bc['iou']['std']):>16}"
                    f" | {pm(bc['precision']['mean'], bc['precision']['std']):>16}"
                    f" | {pm(bc['recall']['mean'], bc['recall']['std']):>16}\n"
                )

        # Section 4
        fp.write("\n\n")
        fp.write("4. DETECTION — IMAGE-LEVEL CYST PRESENCE\n")
        fp.write("-" * 130 + "\n")
        fp.write(f"{'Model':<25} | {'Accuracy':>16} | {'Precision':>16} | {'Recall':>16} | {'Specificity':>16} | {'F1':>16}\n")
        fp.write("-" * 130 + "\n")
        for label in results:
            d = results[label]["detection"]
            fp.write(
                f"{label:<25} | {pm(d['accuracy']['mean'], d['accuracy']['std']):>16}"
                f" | {pm(d['precision']['mean'], d['precision']['std']):>16}"
                f" | {pm(d['recall']['mean'], d['recall']['std']):>16}"
                f" | {pm(d['specificity']['mean'], d['specificity']['std']):>16}"
                f" | {pm(d['f1']['mean'], d['f1']['std']):>16}\n"
            )

        fp.write("\n" + "=" * 130 + "\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
