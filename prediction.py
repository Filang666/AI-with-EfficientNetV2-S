import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# --- 1. Load model ---
model = tf.keras.models.load_model('model.h5')

# --- 2. Load and preprocess image ---
img_path = 'new.png'
# It's important to use the same target_size as during training (300x300)
img = image.load_img(img_path, target_size=(300, 300))

# Convert to array and add batch dimension (1, 300, 300, 3)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# --- 3. Make prediction ---
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0]) # Get probabilities if not using softmax in model

# Get the class with the highest probability
predicted_class = class_names[np.argmax(predictions)]
confidence = 100 * np.max(predictions)

print(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")
