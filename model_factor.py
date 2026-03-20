import numpy as np
import tensorflow as tf
from tensorflow.keras import applications, layers, models

from config import CONFIDENCE_THRESHOLD, ENTROPY_THRESHOLD, IMG_SIZE


def build_efficientnet_model(
    num_classes: int, trainable_base: bool = False
) -> models.Sequential:
    """
    Creates an EfficientNetV2-S based model with custom top layers.
    Includes an explicit Input layer for better graph initialization.
    """
    # Base pretrained model
    base_model = applications.EfficientNetV2S(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = trainable_base

    # Sequential model structure
    model = models.Sequential(
        [
            layers.Input(shape=(*IMG_SIZE, 3)),
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


def predict_with_ood(model: tf.keras.Model, img_tensor: tf.Tensor):
    """
    Performs inference and detects Out-of-Distribution (OOD) data.
    Uses Temperature Scaling and Shannon Entropy for better calibration.

    Returns:
        tuple: (class_index, confidence, entropy, is_ood)
    """
    # 1. Get raw predictions from the model (Softmax output)
    # We use training=False for inference consistency
    preds = model(img_tensor, training=False).numpy()

    # 2. Temperature Scaling (T=2.0)
    # Softens probabilities to reveal true uncertainty in noise/OOD data
    temp = 2.0
    logits = np.log(preds + 1e-9) / temp
    exp_logits = np.exp(logits - np.max(logits))
    soft_probs = exp_logits / np.sum(exp_logits)

    # 3. Calculate metrics
    # Get the raw confidence (original softmax)
    confidence = float(np.max(preds))
    class_idx = int(np.argmax(preds))

    # Calculate Shannon Entropy on softened probabilities: H(p) = -sum(p * log(p))
    entropy_val = -np.sum(soft_probs * np.log2(soft_probs + 1e-9))
    entropy_val = float(entropy_val)

    # 4. Final OOD Decision logic
    # OOD is detected if confidence is too low OR uncertainty (entropy) is too high
    is_ood = (confidence < CONFIDENCE_THRESHOLD) or (entropy_val > ENTROPY_THRESHOLD)

    return class_idx, confidence, entropy_val, bool(is_ood)
