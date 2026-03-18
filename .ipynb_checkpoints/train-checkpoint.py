import tensorflow as tf
from config import *
from model_factory import build_efficientnet_model

def run_training():
    # 1. Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
    )

    class_names = train_ds.class_names
    with open(CLASSES_FILE, 'w') as f:
        f.write('\n'.join(class_names))

    # Optimization
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    # 2. Stage 1: Train top layers
    model = build_efficientnet_model(len(class_names))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS)

    # 3. Stage 2: Fine-tuning
    model.layers[3].trainable = True # Unfreeze EfficientNet base
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=[early_stop])

    model.save(MODEL_PATH)

if __name__ == "__main__":
    run_training()