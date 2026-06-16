#!/bin/bash

# ============================================================================
# Evaluate 5-fold CV models on public test set
# 
# Models to evaluate:
#   1. U-Net (smp, resnet34)
#   2. U-Net (smp, resnet101)
#   3. DeepLabV3 (torchvision, resnet101)
#   4. DeepLabV3+ (smp, resnet101)
# ============================================================================

# Load modules
module load anaconda/2022.05
module load cudnn/8.4.0_cuda11.6
module load cuda/11.6.0

# Activate environment
source activate knee-seg

# Evaluation directories
EVAL_DIR="experiments/20260612-174629"

echo "========================================================================"
echo "Starting 5-fold CV evaluation on public test set"
echo "Models to evaluate:"
echo "  1. U-Net (smp, resnet34)"
echo "  2. U-Net (smp, resnet101)"
echo "  3. DeepLabV3 (torchvision, resnet101)"
echo "  4. DeepLabV3+ (smp, resnet101)"
echo "========================================================================"

echo

# ============================================================================
# U-Net (smp, resnet34)
# ============================================================================
echo "------------------------------------------------------------------------"
echo "1/4: Evaluating U-Net (smp, resnet34)..."
echo "------------------------------------------------------------------------"
echo

/home/dungmn3/.conda/envs/knee-seg/bin/python eval_folds.py \
  --experiment-dir ${EVAL_DIR}/unet_resnet34-seed_16-cv5 \
  --batch-size 32 \
  --method model

echo

# ============================================================================
# U-Net (smp, resnet101)
# ============================================================================
echo "------------------------------------------------------------------------"
echo "2/4: Evaluating U-Net (smp, resnet101)..."
echo "------------------------------------------------------------------------"
echo

/home/dungmn3/.conda/envs/knee-seg/bin/python eval_folds.py \
  --experiment-dir ${EVAL_DIR}/Unet_resnet101-seed_16-cv5 \
  --batch-size 32 \
  --method model

echo

# ============================================================================
# DeepLabV3 (torchvision, resnet101)
# ============================================================================
echo "------------------------------------------------------------------------"
echo "3/4: Evaluating DeepLabV3 (torchvision, resnet101)..."
echo "------------------------------------------------------------------------"
echo

/home/dungmn3/.conda/envs/knee-seg/bin/python eval_folds.py \
  --experiment-dir ${EVAL_DIR}/deeplabv3_resnet101-seed_16-cv5 \
  --batch-size 32 \
  --method model

echo

# ============================================================================
# DeepLabV3+ (smp, resnet101)
# ============================================================================
echo "------------------------------------------------------------------------"
echo "4/4: Evaluating DeepLabV3+ (smp, resnet101)..."
echo "------------------------------------------------------------------------"
echo

/home/dungmn3/.conda/envs/knee-seg/bin/python eval_folds.py \
  --experiment-dir ${EVAL_DIR}/deeplabv3plus_resnet101-seed_16-cv5 \
  --batch-size 32 \
  --method model

echo

# ============================================================================
# Final Summary
# ============================================================================
echo "========================================================================"
echo "Evaluation completed!"
echo "Cross-validation results summary:"
echo

/home/dungmn3/.conda/envs/knee-seg/bin/python scripts/print_summary.py --eval-dir ${EVAL_DIR}

echo "========================================================================"