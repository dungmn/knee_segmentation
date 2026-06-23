import os
import torch
import cv2
import matplotlib.pyplot as plt
from src.models.model import build_deeplabv3, build_deeplabv3plus
from src.visualization.draw_mask import draw_mask
import numpy as np


def predict_image(img_path, model_name):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_normed = (img_resized.astype("float32") / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(img_normed.astype("float32")).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        if model_name.startswith("deeplabv3"):  
            pred = model(tensor)["out"].argmax(1).squeeze().cpu().numpy()
        else:
            pred = model(tensor).argmax(1).squeeze().cpu().numpy()
    return img_resized, pred

if __name__ == "__main__":
    seed = 16
    image_dir = "data/processed/training/post_trans-baker_cyst-flipped-batch_000/images"
    test_file = f"data/processed/training/post_trans-baker_cyst-flipped-batch_000/split/seed_{seed}/test.txt"

    #MODEL CONFIG

    weights_path = f"experiments/20260330-141417/Unet-seed_16/best_model.pth"
    model_name = weights_path.split("/")[-2].split("-")[0]  # Extract model name from path

    output_dir = f"tests/{model_name}_{seed}_results"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: check deeplabv3plus BEFORE deeplabv3
    if model_name.startswith("deeplabv3plus"):
        model = build_deeplabv3plus(num_classes=7, model_name=model_name).to(device)
    elif model_name.startswith("deeplabv3"):
        model = build_deeplabv3(num_classes=7, model_name=model_name).to(device)
    elif model_name.lower().startswith("unet"):
        from src.models.unet import build_unet
        encoder = model_name.split("_", 1)[1] if "_" in model_name else "resnet34"
        model = build_unet(num_classes=7, encoder_name=encoder).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

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
        img, mask_pred = predict_image(f"{image_dir}/{test_image}", model_name=model_name)
        gt_mask = cv2.imread(f"{image_dir.replace('images','masks')}/{test_image}", cv2.IMREAD_UNCHANGED)
        gt_mask = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        draw_mask(img, mask_pred, gt_mask, output_path=f"{output_dir}/{test_image}")