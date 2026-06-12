import segmentation_models_pytorch as smp


def build_unet(num_classes: int = 7, encoder_name: str = "resnet34") -> smp.Unet:
    """Build a U-Net segmentation model via segmentation_models_pytorch.

    Returns a plain tensor (no dict wrapper) — the existing dispatch in
    train.py, eval.py, and infer.py handles this correctly via::

        out = model(imgs)["out"] if model_name.startswith("deeplabv3") else model(imgs)

    Args:
        num_classes:  Number of segmentation classes.
        encoder_name: Encoder backbone (default ``"resnet34"``; any smp-
                      supported encoder works, e.g. ``"resnet50"``,
                      ``"efficientnet-b4"``).

    Returns:
        ``smp.Unet`` model with ImageNet-pretrained encoder.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,   # raw logits; caller uses argmax
    )