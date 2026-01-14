import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def build_deeplabv3(num_classes=7, model_name="deeplabv3_resnet50"):
    if model_name == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    elif model_name == "deeplabv3_resnet101":
        model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported backbone: {model_name}")

    in_ch = model.classifier[-1].in_channels
    model.classifier = nn.Sequential(
        model.classifier[0],
        nn.Dropout(0.3),
        nn.Conv2d(in_ch, num_classes, kernel_size=1)
    )
    # model.classifier = DeepLabHead(2048, num_classes)

    return model
