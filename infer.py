import os
import torch
import cv2
import matplotlib.pyplot as plt
from src.models.model import build_deeplabv3
from src.visualization.draw_mask import draw_mask
import numpy as np


def predict_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_normed = (img_resized.astype("float32") / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img_normed.astype("float32")).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)["out"].argmax(1).squeeze().cpu().numpy()
    return img_resized, pred

if __name__ == "__main__":
    seed = 16
    image_dir = "data/processed/training/post_trans-baker_cyst-flipped-batch_000/images"
    test_file = f"data/processed/training/post_trans-baker_cyst-flipped-batch_000/split/seed_{seed}/test.txt"

    #MODEL CONFIG
    model_name = "deeplabv3_resnet101"
    weights_path = f"experiments/20260227-172930/deeplabv3_resnet101-seed_16/best_model.pth"

    output_dir = f"tests/{model_name}_{seed}_results"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_deeplabv3(num_classes=7, model_name=model_name).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    palette = np.array([
        [0,   0,   0],     # 0 background
        [255,255,0],     # 1 - fat
        [250, 125, 187],     # 2 - tendon
        [255, 0, 0],   # 3 - muscle
        [204, 207, 210],     # 4 - femur
        [42, 125, 209],   # 5 - artery
        [170, 240, 209],   # 6 - baker's cyst
    ], dtype=np.uint8)

    with open(test_file, "r") as f:
        test_images = f.read().splitlines()

    for test_image in test_images:
        img, mask_pred = predict_image(f"{image_dir}/{test_image}")
        gt_mask = cv2.imread(f"{image_dir.replace('images','masks')}/{test_image}", cv2.IMREAD_UNCHANGED)
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        draw_mask(img, mask_pred, gt_mask, output_path=f"{output_dir}/{test_image}")