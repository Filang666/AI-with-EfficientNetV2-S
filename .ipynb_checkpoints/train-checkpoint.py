import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import settings
from config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, INITIAL_EPOCHS, 
    FINE_TUNE_EPOCHS, FINE_TUNE_LEARNING_RATE, 
    MODEL_PATH, CLASSES_FILE
)
from model_factor import build_efficientnet_model

def save_plots(history, model, val_ds, class_names):
    """
    Generates and saves training history plots and a confusion matrix.
    """
    reports_dir = 'reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"📁 Created directory: {reports_dir}")

    # 1. Training History Plot (Accuracy & Loss)
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/training_history.png')
    print(f"✅ Training history saved to {reports_dir}/training_history.png")

    # 2. Confusion Matrix
    print("Generating Confusion Matrix...")
    y_true = []
    y_pred = []
    
    for x, y in val_ds:
        # Get true labels from one-hot encoding
        y_true.extend(np.argmax(y.numpy(), axis=1))
        # Get model predictions
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(f'{reports_dir}/confusion_matrix.png')
    print(f"✅ Confusion matrix saved to {reports_dir}/confusion_matrix.png")

def run_training():
    """
    Main training pipeline: Load data -> Stage 1 (Head) -> Stage 2 (Fine-tuning) -> Save results.
    """
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset directory '{DATA_DIR}' not found!")
        print("Please run 'python download_data.py' first.")
        return

    # 1. Load Datasets
    print("Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, 
        validation_split=0.2, 
        subset="training", 
        seed=123,
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        label_mode='categorical'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, 
        validation_split=0.2, 
        subset="validation", 
        seed=123,
        image_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        label_mode='categorical'
    )

    # Auto-generate classes.txt
    class_names = train_ds.class_names
    with open(CLASSES_FILE, 'w') as f:
        f.write('\n'.join(class_names))
    print(f"✅ Classes detected and saved to {CLASSES_FILE}: {class_names}")

    # Optimization for performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    # 2. Stage 1: Train top layers (Transfer Learning)
    print("\n--- Stage 1: Training Classification Head ---")
    model = build_efficientnet_model(len(class_names))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS)

    # 3. Stage 2: Fine-tuning (Unfreeze base model)
    print("\n--- Stage 2: Fine-tuning Full Model ---")
    # Finding the base model layer (EfficientNet) and unfreezing it
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=5, 
        restore_best_weights=True,
        monitor='val_loss'
    )
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=FINE_TUNE_EPOCHS, 
        callbacks=[early_stop]
    )

    # 4. Save results
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    
    save_plots(history, model, val_ds, class_names)

if __name__ == "__main__":
    run_training()