#!/usr/bin/env bash
# Evaluate all 3 DL models and classical CV, then compile comparison table
# Usage: bash scripts/evaluate.sh

set -e

# Change directory to the repository root
cd "$(dirname "$0")/.."

echo "========================================================================"
echo "Starting Knee Segmentation 5-fold CV evaluation pipeline"
echo "========================================================================"

# Function to get latest experiment dir for a model
get_latest_exp() {
  local model_pattern="$1"
  local match
  match=$(ls -td experiments/*/"$model_pattern" 2>/dev/null | head -n 1)
  if [ -z "$match" ]; then
    echo ""
  else
    echo "$match"
  fi
}

UNET_DIR=$(get_latest_exp "Unet-seed_16-cv5")
DLV3_DIR=$(get_latest_exp "deeplabv3_resnet101-seed_16-cv5")
DLV3P_DIR=$(get_latest_exp "deeplabv3plus_resnet101-seed_16-cv5")

# Verify all directories exist
if [ -z "$UNET_DIR" ]; then
  echo "Error: Could not find any experiment directory for U-Net (pattern: experiments/*/Unet-seed_16-cv5)"
  exit 1
fi
if [ -z "$DLV3_DIR" ]; then
  echo "Error: Could not find any experiment directory for DeepLabV3 (pattern: experiments/*/deeplabv3_resnet101-seed_16-cv5)"
  exit 1
fi
if [ -z "$DLV3P_DIR" ]; then
  echo "Error: Could not find any experiment directory for DeepLabV3+ (pattern: experiments/*/deeplabv3plus_resnet101-seed_16-cv5)"
  exit 1
fi

echo "Found latest experiment directories:"
echo "  U-Net:        $UNET_DIR"
echo "  DeepLabV3:    $DLV3_DIR"
echo "  DeepLabV3+:   $DLV3P_DIR"
echo "--------------------------------------------------------"

# 1. Evaluate DL models
echo ""
echo "--------------------------------------------------------"
echo "1/4: Evaluating U-Net (smp)..."
echo "--------------------------------------------------------"
python eval_folds.py --experiment-dir "$UNET_DIR" --batch-size 16 --method model

echo ""
echo "--------------------------------------------------------"
echo "2/4: Evaluating DeepLabV3 (torchvision)..."
echo "--------------------------------------------------------"
python eval_folds.py --experiment-dir "$DLV3_DIR" --batch-size 16 --method model

echo ""
echo "--------------------------------------------------------"
echo "3/4: Evaluating DeepLabV3+ (smp)..."
echo "--------------------------------------------------------"
python eval_folds.py --experiment-dir "$DLV3P_DIR" --batch-size 16 --method model

# 2. Evaluate Classical CV
echo ""
echo "--------------------------------------------------------"
echo "4/4: Evaluating Classical CV (using U-Net's fold splits)..."
echo "--------------------------------------------------------"
python eval_cv_val.py -e "$UNET_DIR"

# 3. Compile final comparison table
echo ""
echo "--------------------------------------------------------"
echo "Compiling comparison table..."
echo "--------------------------------------------------------"
python scripts/compile_results.py \
  --unet-dir "$UNET_DIR" \
  --dlv3-dir "$DLV3_DIR" \
  --dlv3p-dir "$DLV3P_DIR" \
  --cv-dir "$UNET_DIR"

echo "========================================================================"
echo "Evaluation and comparison compilation completed successfully!"
echo "========================================================================"
