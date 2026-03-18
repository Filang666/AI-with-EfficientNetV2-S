import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np

# --- 1. SETTINGS ---
IMG_SIZE = (300, 300)  
BATCH_SIZE = 32       
DATA_DIR = 'dataset' 

# --- 2. DATA LOADING & CLASS NAMES ---
# Loading the training set to extract class names automatically
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, 
    validation_split=0.2, 
    subset="training", 
    seed=

123,
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    label_mode='categorical'
)

# Extract CLASS_NAMES from the dataset
CLASS_NAMES = train_ds.class_names
print(f"Detected classes: {CLASS_NAMES}")

# Save class names to a file for the Inference API to use later
with open('classes.txt', 'w') as f:
    for name in CLASS_NAMES:
        f.write(f"{name}\n")

# Data splitting (80/10/10)
temp_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

batches = tf.data.experimental.cardinality(temp_ds)
test_ds = temp_ds.take(batches // 2)
val_ds = temp_ds.skip(batches // 2)

# Prefetching for performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# --- 3. MODEL ARCHITECTURE ---
base_model = applications.EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False 

model = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(CLASS_NAMES), activation='softmax') # Dynamic output size
])

# --- 4. TRAINING (STAGE 1) ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# --- 5. FINE-TUNING (STAGE 2) ---
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop])

# --- 6. SAVE MODEL ---
model.save('model.h5')