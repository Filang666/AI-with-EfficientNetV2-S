import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# --- 1. Load Model and Classes ---
model = tf.keras.models.load_model('model.h5')

# Load class names from the file generated during training
if os.path.exists('classes.txt'):
    with open('classes.txt', 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
else:
    # Fallback if file is missing
    CLASS_NAMES = [f"Class_{i}" for i in range(10)]

# --- 2. Image Preprocessing ---
img_path = 'new.png'
img = image.load_img(img_path, target_size=(300, 300))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# --- 3. Inference ---
predictions = model.predict(img_array, verbose=0)
score = tf.nn.softmax(predictions[0]) # Use if model doesn't have softmax in last layer

idx = np.argmax(predictions)
predicted_class = CLASS_NAMES[idx]
confidence = 100 * np.max(predictions)

print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")