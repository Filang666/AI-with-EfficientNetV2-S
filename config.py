import os

# Dataset and model paths
DATA_DIR = "data"
MODEL_PATH = "model.h5"
CLASSES_FILE = "classes.txt"

# Image processing settings
IMG_SIZE = (300, 300)
BATCH_SIZE = 32

# Training hyperparameters
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 20
FINE_TUNE_LEARNING_RATE = 1e-5

# --- OOD Settings ---
# Correct images: 1.0, Noise: 0.97. So 0.98 or 0.99 is a good split.
CONFIDENCE_THRESHOLD = 0.985
# Entropy threshold for additional check (Shannon entropy)
ENTROPY_THRESHOLD = 0.4


def get_class_names():
    """
    Reads class names from the local text file.
    Returns a list of strings.
    """
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r") as f:
            return [line.strip() for line in f.readlines()]
    return []
