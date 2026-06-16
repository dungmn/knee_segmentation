import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str, required=True)
    args = parser.parse_args()
    
    models = {
        "U-Net (smp, resnet34)": "unet_resnet34-seed_16-cv5",
        "U-Net (smp, resnet101)": "Unet_resnet101-seed_16-cv5",
        "DeepLabV3 (torchvision, resnet101)": "deeplabv3_resnet101-seed_16-cv5",
        "DeepLabV3+ (smp, resnet101)": "deeplabv3plus_resnet101-seed_16-cv5"
    }

    for name, folder in models.items():
        p = os.path.join(args.eval_dir, folder, "cv_summary.json")
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            seg = data["segmentation"]
            det = data["detection"]
            cyst = seg["per_class"]["bakers_cyst"]["dice"]
            print(f"{name}:")
            print(f"  Mean Dice: {seg['mean_dice']['mean']:.4f} \u00b1 {seg['mean_dice']['std']:.4f}")
            print(f"  Cyst Dice: {cyst['mean']:.4f} \u00b1 {cyst['std']:.4f}")
            print(f"  Accuracy:  {det['accuracy']['mean']:.4f} \u00b1 {det['accuracy']['std']:.4f}")
            print(f"  Recall:    {det['recall']['mean']:.4f} \u00b1 {det['recall']['std']:.4f}")
            print(f"  Precision: {det['precision']['mean']:.4f} \u00b1 {det['precision']['std']:.4f}")
            print(f"  F1-score:  {det['f1']['mean']:.4f} \u00b1 {det['f1']['std']:.4f}")
        else:
            print(f"{name}: No results")
        print()

if __name__ == "__main__":
    main()
