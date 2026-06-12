# AGENTS.md — Repository guidance for AI coding agents

## Purpose
Provide concise, actionable guidance for AI coding agents working on this repo.

---

## Project overview
Knee / popliteal-fossa ultrasound segmentation for Baker's cyst detection (thesis project).  
Two complementary approaches are implemented and compared:

1. **Deep-learning (DL)** — supervised semantic segmentation with DeepLabV3+ (ResNet-50/101) or U-Net, trained with 5-fold cross-validation.
2. **Classical CV** — rule-based Baker's cyst detection in [`cv.py`](cv.py) using bilateral filtering, Otsu thresholding, morphological operations, and contour scoring.

### Segmentation classes (7 total, class id → name)
| ID | Name |
|----|------|
| 0  | background |
| 1  | fat |
| 2  | tendon |
| 3  | muscle |
| 4  | femur |
| 5  | artery |
| 6  | bakers_cyst |

---

## Quick setup
```bash
conda activate idpvlm          # project conda environment
pip install -r requirements.txt
```

**Key dependencies** (from `requirements.txt`): `torch`/`torchvision` (assumed installed in env), `monai` (DiceCELoss), `albumentations` (augmentations), `opencv-python`, `pycocotools`, `scikit-learn`, `rich`, `matplotlib`, `pandas`.

---

## Key file locations

### Entrypoints
| File | Purpose |
|------|---------|
| [`train.py`](train.py) | Train DL model with K-fold CV |
| [`eval.py`](eval.py) | Evaluate a single fold — DL model **or** classical CV |
| [`eval_folds.py`](eval_folds.py) | Orchestrate per-fold `eval.py` calls and aggregate CV summary JSON |
| [`eval_cv_val.py`](eval_cv_val.py) | Classical CV evaluation across fold val lists; outputs CSV + JSON stats |
| [`infer.py`](infer.py) | Run inference on test split and save side-by-side visualisations |
| [`cv.py`](cv.py) | Classical CV pipeline (`segment_baker_cyst()`); also runnable as a script |

### Source code (`src/`)
| Path | Purpose |
|------|---------|
| [`src/models/model.py`](src/models/model.py) | `build_deeplabv3()` — DeepLabV3+ with ResNet-50/101 backbone, custom head (ASPP → Dropout(0.3) → Conv1×1) |
| [`src/models/unet.py`](src/models/unet.py) | `build_unet()` — custom U-Net with `dual_conv` blocks and `crop_tensor` skip connections |
| [`src/data/dataset.py`](src/data/dataset.py) | `KneeSegDataset` — loads PNG images + grayscale masks, applies albumentations transforms |
| [`src/data/augmentations.py`](src/data/augmentations.py) | `get_train_transforms()` / `get_val_transforms()` — albumentations pipelines (resize 512², ImageNet norm) |
| [`src/data/process_cvat_data.py`](src/data/process_cvat_data.py) | Convert CVAT COCO JSON annotations → class-id PNG masks; optional 80/10/10 train/val/test split |
| [`src/data/split_data.py`](src/data/split_data.py) | Batch-split a flat image folder into numbered `batch_NNN/` subdirs |
| [`src/utils/convert_coco_to_mask.py`](src/utils/convert_coco_to_mask.py) | COCO JSON → single-channel PNG mask converter (used by `process_cvat_data.py`) |
| [`src/visualization/draw_mask.py`](src/visualization/draw_mask.py) | `draw_mask()` — 3-panel matplotlib figure (input / pred / GT) saved as PDF |

### Scripts & configs
| Path | Purpose |
|------|---------|
| [`scripts/evaluation_dl.sh`](scripts/evaluation_dl.sh) | Example: run `eval_folds.py` for a DL experiment |
| [`scripts/evaluation_cv.sh`](scripts/evaluation_cv.sh) | Example: run `eval_cv_val.py` for classical CV evaluation |
| [`scripts/inference.sh`](scripts/inference.sh) | Example: process raw CVAT annotations → training-ready dataset |
| [`configs/cvat/`](configs/cvat/) | CVAT class configuration files |

### Data & experiment layout
```
data/processed/
  training/
    <dataset-name>/          # e.g. post_trans-27-random-flipped-batch_000
      images/                # PNG frames
      masks/                 # single-channel PNG class-id masks (mirrors images/)
      split/seed_<N>/        # train.txt / val.txt / test.txt  (or all.txt for CV mode)

experiments/
  <YYYYMMDD-HHMMSS>/
    <model>-seed_<N>-cv<K>/  # e.g. deeplabv3_resnet101-seed_16-cv5
      fold_01_last_model.pth
      fold_01_val_imgs.txt   # absolute image paths used for that fold's validation
      fold_01_results.json   # per-fold segmentation + detection metrics
      cv_summary.json        # aggregated summary across all folds (written by eval_folds.py)

logs/
  eval_results/              # per-image overlay visualisations from eval.py
  output/                    # cv.py standalone output
  0original.png … 4morphological.png  # cv.py debug intermediate images (logs/*)

notebooks/
  data_analysis.ipynb        # exploratory data analysis
```

---

## Common commands

### Data preparation
```bash
# Convert CVAT annotations → training dataset (with train/val/test split)
python src/data/process_cvat_data.py \
  --input_dir  data/processed/annotations/<batch> \
  --output_dir data/processed/training/<dataset-name> \
  --seed 16 \
  --split
```

### Training (5-fold CV)
```bash
# Full run (default: deeplabv3_resnet101, seed 16, 5 folds, 50 epochs, batch 6)
python train.py \
  --model-name deeplabv3_resnet101 \
  --num-classes 7 \
  --n-folds 5 \
  --epochs 50 \
  --batch-size 6 \
  --dataset-dir data/processed/training/post_trans-27-random-flipped-batch_000 \
               data/processed/training/post_trans-baker_cyst-flipped-batch_000

# Quick smoke-run (1 epoch, small batch)
python train.py --epochs 1 --batch-size 2 --n-folds 2

# Supported --model-name values: deeplabv3_resnet50 | deeplabv3_resnet101 | deeplabv3plus_resnet50 | deeplabv3plus_resnet101 | Unet
```

> **Note:** `train.py` currently hard-codes `if fold_idx != 5: continue` — only fold 5 is trained. Remove or adjust this guard to train all folds.

> **Dispatch rule:** All dispatch sites check `deeplabv3plus` **before** `deeplabv3` because the plus variant also starts with that prefix.

### Evaluation — DL model (single fold)
```bash
python eval.py \
  --method model \
  --weights experiments/<run>/<model>/fold_05_last_model.pth \
  --val-list experiments/<run>/<model>/fold_05_val_imgs.txt \
  --num-classes 7
# Results saved as fold_05_results.json next to the val-list file.
```

### Evaluation — DL model (all folds + aggregate)
```bash
python eval_folds.py \
  --experiment-dir experiments/<run>/<model> \
  --batch-size 16 \
  --method model
# → writes cv_summary.json in the experiment dir
# See also: scripts/evaluation_dl.sh
```

### Evaluation — classical CV (all fold val lists)
```bash
python eval_cv_val.py \
  -e experiments/<run>/<model> \
  [--cv-debug]
# → writes cv_val_results.csv and cv_statistics.json
# See also: scripts/evaluation_cv.sh
```

### Classical CV — standalone (single image or directory)
```bash
python cv.py [image_name] \
  --input-dir data/processed/annotations/post_trans-baker_cyst/batch_000 \
  --output-dir logs/output \
  --debug --verbose
```

### Inference (generate prediction visualisations)
```bash
# Edit weights_path and test_file inside infer.py, then:
python infer.py
# → saves <model>_16_results/<image>.pdf side-by-side comparisons
```

---

## Architecture notes

### DeepLabV3+ (`src/models/model.py` — via `segmentation_models_pytorch`)
- Built with `smp.DeepLabV3Plus(encoder_name, encoder_weights="imagenet", classes=num_classes, activation=None)`.
- Encoder: ImageNet-pretrained ResNet-50 or ResNet-101 (selected by model name).
- Wrapped in `_SmpDictWrapper` so forward returns `{"out": logits}` — **same dict convention as DeepLabV3**; the `model(x)["out"]` dispatch works unchanged.
- Model names: `deeplabv3plus_resnet50`, `deeplabv3plus_resnet101`.
- Requires: `segmentation-models-pytorch` (added to `requirements.txt`).

### DeepLabV3 (`src/models/model.py` — torchvision wrapper)
- Pretrained backbone: ResNet-50 or ResNet-101 with COCO weights.
- Classifier head replaced with: `ASPP block → Dropout(0.3) → Conv2d(in_ch, num_classes, 1)`.
- Forward pass returns a dict; use `model(x)["out"]` for logits.

### U-Net (`src/models/unet.py`)
- Custom implementation with `dual_conv` (BN→Conv3×3→BN→ReLU ×2) blocks.
- 4 encoder stages (64 → 128 → 256 → 512 → 1024), 4 decoder stages with `ConvTranspose2d` + skip connections via `crop_tensor`.
- Output activation: `Sigmoid` (use `logits.argmax(1)` for class prediction regardless).
- Returns a plain tensor (no dict wrapper).

### Training loop (`train.py`)
- Loss: `monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)`.
- Optimizer: `AdamW(lr=1e-4)`.
- Best checkpoint tracked by Baker's cyst Dice (class 6); **last** weights saved per fold.
- Mask is resized to match output resolution via `F.interpolate(..., mode="nearest")` before loss.

### Classical CV pipeline (`cv.py → segment_baker_cyst()`)
1. Grayscale load → ROI crop (configurable fraction of image).
2. Bilateral filter (d=9, σ=75) for edge-preserving denoising.
3. Otsu thresholding (adjusted by −30) → `THRESH_BINARY_INV` to highlight dark fluid.
4. Morphological open + close (ellipse kernel 7×7, 2 iterations).
5. Contour filtering: area, aspect ratio (1.2–4.0), solidity ≥ 0.75, extent ≥ 0.45, circularity ≥ 0.2.
6. Scoring: weighted combination of normalised area score + distance-to-ROI-centre score + upper-half bonus.

---

## Evaluation metrics
Both DL and CV paths report:
- **Segmentation** (pixel-level): Dice, IoU, Precision, Recall — per class and macro mean.
- **Detection** (image-level, Baker's cyst presence): Accuracy, Precision, Recall, Specificity, F1, confusion matrix.

---

## Conventions & tips for agents
- **Minimal changes**: Prefer localised edits; use `--epochs 1 --batch-size 2 --n-folds 2` for smoke-tests.
- **Model dispatch**: Both `build_model()` in `train.py` and loading in `eval.py`/`infer.py` infer the model type from the name string (`deeplabv3*` vs `Unet`). Keep this convention when adding new architectures.
- **DeepLabV3 output**: always index with `["out"]`; U-Net returns a plain tensor.
- **Data layout**: datasets must follow `<root>/images/` + `<root>/masks/` with `<root>/split/seed_<N>/` split files.
- **Experiment naming**: `experiments/<YYYYMMDD-HHMMSS>/<model>-seed_<N>-cv<K>/` — generated automatically by `train.py`.
- **Do not commit** large `.pth` weight files; use `experiments/` for run artifacts.
- **`logs/`**: debug images from `cv.py` and `eval.py` visualisations land here; safe to ignore in version control.

---

## Suggested next customisations
- Remove the `if fold_idx != 5: continue` guard in `train.py` to enable full K-fold training.
- Add a `scripts/train.sh` with safe default flags for reproducible full runs.
- Add `--save-best` flag to `train.py` to also checkpoint the best-Dice model per fold (currently only the last epoch is saved).
- Add `--output-json` flag to `eval.py` for standalone (non-fold) runs to export structured results.
- Create skill `run-experiment` wrapping experiment-dir creation + `train.py` with safe defaults.
