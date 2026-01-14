import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from rich.progress import track
from rich import print

from src.data.dataset import KneeSegDataset
from src.models.model import build_deeplabv3


def load_dataset(dataset_dir, seed):
    with open(f"{dataset_dir}/split/seed_{seed}/train.txt", "r") as f:
        train_imgs = [f"{dataset_dir}/images/{line.strip()}" for line in f.readlines()]
    with open(f"{dataset_dir}/split/seed_{seed}/val.txt", "r") as f:
        val_imgs = [f"{dataset_dir}/images/{line.strip()}" for line in f.readlines()]

    train_masks = [p.replace("images", "masks") for p in train_imgs]
    val_masks = [p.replace("images", "masks") for p in val_imgs]

    return train_imgs, train_masks, val_imgs, val_masks


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 16
    model_name = "deeplabv3_resnet50"
    # model_name = "deeplabv3_resnet101"
    experiment_dir = f"experiments/{time.strftime('%Y%m%d-%H%M%S')}/{model_name}_seed_{seed}"
    saved_weights_path = f"{experiment_dir}/best_model.pth"

    os.makedirs(experiment_dir, exist_ok=True)



    dataset_dir = [
        "data/processed/training/post_trans-27-random-flipped-batch_000",
        "data/processed/training/post_trans-baker_cyst-flipped-batch_000"
    ]


    train_imgs, train_masks, val_imgs, val_masks = [], [], [], []
    for d in dataset_dir:
        t_imgs, t_masks, v_imgs, v_masks = load_dataset(d, seed)
        train_imgs.extend(t_imgs)
        train_masks.extend(t_masks)
        val_imgs.extend(v_imgs)
        val_masks.extend(v_masks)


    # # Model
    num_classes = 7
    batch_size = 8
    epochs = 50
    model = build_deeplabv3(num_classes, model_name=model_name).to(device)
    model.train()
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    train_loader = DataLoader(KneeSegDataset(train_imgs, train_masks), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(KneeSegDataset(val_imgs, val_masks), batch_size=batch_size, shuffle=False)

    # # Train Loop
    best_dice = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, masks in track(train_loader, description=f"Epoch {epoch+1} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)["out"]
            loss = criterion(out, F.interpolate(masks.unsqueeze(1).float(), out.shape[2:], mode="nearest").long())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval(); val_loss = 0; dice_metric.reset()
        with torch.no_grad():
            for imgs, masks in track(val_loader, description=f"Epoch {epoch+1} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)["out"]
                loss = criterion(out, F.interpolate(masks.unsqueeze(1).float(), out.shape[2:], mode="nearest").long())
                val_loss += loss.item()
                preds = out.argmax(1, keepdim=True)
                dice_metric(y_pred=preds, y=masks.unsqueeze(1))
        dice = dice_metric.aggregate().item()
        print(f"Epoch {epoch+1}: TrainLoss={train_loss/len(train_loader):.4f} | ValLoss={val_loss/len(val_loader):.4f} | Dice={dice:.4f}")
        if dice > best_dice:
            best_dice = dice
            print(f"New best model found at epoch {epoch+1}, saving model...")
            torch.save(model.state_dict(), saved_weights_path)

    print(f"Training completed. Best Val Dice: {best_dice:.4f}")

