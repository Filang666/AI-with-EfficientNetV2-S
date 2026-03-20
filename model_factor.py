import tensorflow as tf
from tensorflow.keras import applications, layers, models

from config import IMG_SIZE


def build_efficientnet_model(
    num_classes: int, trainable_base: bool = False
) -> models.Sequential:
    """Creates an EfficientNetV2-S based model with custom top layers."""
    base_model = applications.EfficientNetV2S(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = trainable_base

    model = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model
