import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
)


# ---------------------------------------------------------------------------
# DeepLabV3  (torchvision wrapper — unchanged)
# ---------------------------------------------------------------------------

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
        nn.Conv2d(in_ch, num_classes, kernel_size=1),
    )
    return model


# ---------------------------------------------------------------------------
# DeepLabV3+  (segmentation_models_pytorch)
# ---------------------------------------------------------------------------

class _SmpDictWrapper(nn.Module):
    """Wrap an smp model to return {\"out\": logits} matching the torchvision
    DeepLabV3 convention so all dispatch sites work unchanged."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return {"out": self.model(x)}


def build_deeplabv3plus(num_classes: int = 7, model_name: str = "deeplabv3plus_resnet50") -> _SmpDictWrapper:
    """Build a DeepLabV3+ model via segmentation_models_pytorch.

    The model is wrapped so that forward() returns ``{"out": logits}``,
    matching the torchvision DeepLabV3 convention used everywhere in this repo.

    Args:
        num_classes: Number of segmentation classes.
        model_name:  ``"deeplabv3plus_resnet50"`` or ``"deeplabv3plus_resnet101"``.

    Returns:
        Wrapped smp.DeepLabV3Plus whose forward returns ``{"out": logits}``.
    """
    if "resnet50" in model_name:
        encoder = "resnet50"
    elif "resnet101" in model_name:
        encoder = "resnet101"
    else:
        raise ValueError(f"Unsupported backbone for DeepLabV3+: {model_name}")

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,      # raw logits; caller uses argmax
    )
    return _SmpDictWrapper(model)
