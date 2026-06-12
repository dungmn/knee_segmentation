#!/usr/bin/env bash
# Train all 3 DL models for the 4-method comparison (30 epochs, 5-fold CV, seed 16)
# Usage: bash scripts/train.sh

set -e

# Change directory to the repository root
cd "$(dirname "$0")/.."

echo "========================================================================"
echo "Starting Knee Segmentation 5-fold CV training pipeline"
echo "Models to train:"
echo "  1. U-Net (smp, resnet34)"
echo "  2. DeepLabV3 (torchvision, resnet101)"
echo "  3. DeepLabV3+ (smp, resnet101)"
echo "========================================================================"

DATASETS=(
  "data/processed/training/post_trans-27-random-flipped-batch_000"
  "data/processed/training/post_trans-baker_cyst-flipped-batch_000"
)

# Verify dataset directories exist
for ds in "${DATASETS[@]}"; do
  if [ ! -d "$ds" ]; then
    echo "Error: Dataset directory $ds does not exist!"
    exit 1
  fi
done

# Train U-Net
echo ""
echo "--------------------------------------------------------"
echo "1/3: Training U-Net (smp, resnet34)..."
echo "--------------------------------------------------------"
python train.py \
  --model-name Unet \
  --num-classes 7 \
  --n-folds 5 \
  --epochs 30 \
  --batch-size 6 \
  --seed 16 \
  --dataset-dir "${DATASETS[@]}"

# Train DeepLabV3
echo ""
echo "--------------------------------------------------------"
echo "2/3: Training DeepLabV3 (torchvision, resnet101)..."
echo "--------------------------------------------------------"
python train.py \
  --model-name deeplabv3_resnet101 \
  --num-classes 7 \
  --n-folds 5 \
  --epochs 30 \
  --batch-size 6 \
  --seed 16 \
  --dataset-dir "${DATASETS[@]}"

# Train DeepLabV3+
echo ""
echo "--------------------------------------------------------"
echo "3/3: Training DeepLabV3+ (smp, resnet101)..."
echo "--------------------------------------------------------"
python train.py \
  --model-name deeplabv3plus_resnet101 \
  --num-classes 7 \
  --n-folds 5 \
  --epochs 30 \
  --batch-size 6 \
  --seed 16 \
  --dataset-dir "${DATASETS[@]}"

echo ""
echo "========================================================================"
echo "Training pipeline completed successfully!"
echo "You can now run: bash scripts/evaluate.sh to run evaluation on all models."
echo "========================================================================"
