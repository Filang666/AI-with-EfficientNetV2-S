import os

# Model & Data Settings
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
DATA_DIR = 'dataset'
MODEL_PATH = 'model.h5'
CLASSES_FILE = 'classes.txt'

# Training Hyperparameters
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 20
BASE_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

def get_class_names() -> list:
    """Loads class names from file or returns default placeholders."""
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return [f"Class_{i}" for i in range(10)]
