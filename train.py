import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from rich.progress import track
from rich import print
from sklearn.model_selection import KFold

from src.data.dataset import KneeSegDataset
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.models.model import build_deeplabv3
from src.models.unet import build_unet


def build_model(model_name, num_classes, device):
    if model_name.startswith("deeplabv3"):
        return build_deeplabv3(num_classes, model_name=model_name).to(device)
    if model_name == "Unet":
        return build_unet(num_classes=num_classes).to(device)
    raise ValueError(f"Unsupported model name: {model_name}")


def image_to_mask_path(image_path):
    return image_path.replace("/images/", "/masks/")


def load_image_list(dataset_dir, seed):
    split_root = f"{dataset_dir}/split/seed_{seed}"
    all_file = f"{split_root}/all.txt"

    if os.path.exists(all_file):
        with open(all_file, "r") as f:
            image_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        train_file = f"{split_root}/train.txt"
        val_file = f"{split_root}/val.txt"
        with open(train_file, "r") as f:
            train_names = [line.strip() for line in f.readlines() if line.strip()]
        with open(val_file, "r") as f:
            val_names = [line.strip() for line in f.readlines() if line.strip()]
        image_names = sorted(set(train_names + val_names))

    return [f"{dataset_dir}/images/{name}" for name in image_names]


def dice_from_counts(tp, fp, fn):
    return (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)


def save_image_list(file_path, image_paths):
    with open(file_path, "w") as f:
        for path in image_paths:
            f.write(f"{path}\n")


def train_one_fold(
    fold_idx,
    model_name,
    num_classes,
    device,
    train_imgs,
    val_imgs,
    batch_size,
    epochs,
    experiment_dir,
    best_class_id=6,
    best_class_name="bakers_cyst",
):
    saved_weights_path = f"{experiment_dir}/fold_{fold_idx:02d}_best_model.pth"
    os.makedirs(experiment_dir, exist_ok=True)

    train_masks = [image_to_mask_path(p) for p in train_imgs]
    val_masks = [image_to_mask_path(p) for p in val_imgs]

    model = build_model(model_name, num_classes, device)
    model.train()

    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_transform = get_train_transforms(img_size=512)
    val_transform = get_val_transforms(img_size=512)

    train_loader = DataLoader(
        KneeSegDataset(train_imgs, train_masks, transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        KneeSegDataset(val_imgs, val_masks, transform=val_transform),
        batch_size=1,
        shuffle=False,
    )

    best_dice = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in track(train_loader, description=f"Fold {fold_idx} Epoch {epoch + 1} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)["out"] if model_name.startswith("deeplabv3") else model(imgs)
            
            # masks_resized = F.interpolate(masks.unsqueeze(1).float(), out.shape[2:], mode="nearest").long().squeeze(1)
            # loss = criterion(out, masks_resized)

            loss = criterion(out, F.interpolate(masks.unsqueeze(1).float(), out.shape[2:], mode="nearest").long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        class_tp, class_fp, class_fn = 0, 0, 0
        with torch.no_grad():
            for imgs, masks in track(val_loader, description=f"Fold {fold_idx} Epoch {epoch + 1} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)["out"] if model_name.startswith("deeplabv3") else model(imgs)
                masks_resized = F.interpolate(masks.unsqueeze(1).float(), out.shape[2:], mode="nearest").long().squeeze(1)
                loss = criterion(out, masks_resized.unsqueeze(1))
                val_loss += loss.item()
                preds = out.argmax(1)

                pred_pos = preds == best_class_id
                gt_pos = masks_resized == best_class_id
                class_tp += torch.logical_and(pred_pos, gt_pos).sum().item()
                class_fp += torch.logical_and(pred_pos, ~gt_pos).sum().item()
                class_fn += torch.logical_and(~pred_pos, gt_pos).sum().item()

        dice = dice_from_counts(class_tp, class_fp, class_fn)
        print(
            f"Fold {fold_idx} Epoch {epoch + 1}: "
            f"TrainLoss={train_loss / len(train_loader):.4f} | "
            f"ValLoss={val_loss / len(val_loader):.4f} | "
            f"{best_class_name}_Dice={dice:.4f}"
        )

        if dice > best_dice:
            best_dice = dice
            print(f"Fold {fold_idx}: new best model at epoch {epoch + 1}, saving to {saved_weights_path}")
            torch.save(model.state_dict(), saved_weights_path)

    return best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model with 5-fold cross-validation")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--model-name", type=str, default="deeplabv3_resnet101")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--dataset-dir",
        nargs="+",
        default=[
            "data/processed/training/post_trans-27-random-flipped-batch_000",
            "data/processed/training/post_trans-baker_cyst-flipped-batch_000",
        ],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    model_name = args.model_name

    experiment_dir = f"experiments/{time.strftime('%Y%m%d-%H%M%S')}/{model_name}-seed_{seed}-cv{args.n_folds}"
    os.makedirs(experiment_dir, exist_ok=True)

    all_imgs = []
    for d in args.dataset_dir:
        all_imgs.extend(load_image_list(d, seed))

    if len(all_imgs) < args.n_folds:
        raise ValueError(f"Not enough samples ({len(all_imgs)}) for {args.n_folds}-fold CV")

    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=seed)
    fold_best_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_imgs), start=1):
        train_imgs = [all_imgs[i] for i in train_idx]
        val_imgs = [all_imgs[i] for i in val_idx]

        # val_split_file = f"{experiment_dir}/fold_{fold_idx:02d}_val_imgs.txt"
        # save_image_list(val_split_file, val_imgs)
        # continue

        print("-" * 90)
        print(f"Fold {fold_idx}/{args.n_folds} | Train={len(train_imgs)} | Val={len(val_imgs)}")

        fold_best = train_one_fold(
            fold_idx=fold_idx,
            model_name=model_name,
            num_classes=args.num_classes,
            device=device,
            train_imgs=train_imgs,
            val_imgs=val_imgs,
            batch_size=args.batch_size,
            epochs=args.epochs,
            experiment_dir=experiment_dir,
        )
        fold_best_scores.append(fold_best)

    mean_best = sum(fold_best_scores) / len(fold_best_scores)
    print("=" * 90)
    for i, score in enumerate(fold_best_scores, start=1):
        print(f"Fold {i} best Dice: {score:.4f}")
    print(f"5-fold CV mean best Dice: {mean_best:.4f}")

