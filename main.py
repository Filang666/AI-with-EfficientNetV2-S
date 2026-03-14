import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np

# --- 1. SETTINGS ---
IMG_SIZE = (300, 300)  # High resolution for better details
BATCH_SIZE = 32        # Optimized for 16GB VRAM
DATA_DIR = 'dataset' 

# --- 2. DATA LOADING & SPLITTING (80/10/10) ---
# Train set (80%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

# Remaining 20% for Validation and Test
temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

batches = tf.data.experimental.cardinality(temp_ds)
test_ds = temp_ds.take(batches // 2)
val_ds = temp_ds.skip(batches // 2)

# Optimization for faster GPU processing
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# --- 3. MODEL ARCHITECTURE ---
# Using EfficientNetV2-S
base_model = applications.EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False # Freeze ImageNet weights

model = models.Sequential([
    # Data Augmentation (Prevents Overfitting)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')
])

# --- 4. TRAINING (STAGE 1: TOP LAYERS ONLY) ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training of top layers...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# --- 5. Learning FINE-TUNING ---
print("Unfreezing base model for fine-tuning...")
base_model.trainable = True

# Use a VERY small learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Stop training if validation loss stops improving
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop])

# --- 6. Final accuracy test ---
print("\n--- FINAL TEST ON UNSEEN DATA ---")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# --- 7. Save model ---
model.save('model.h5')
