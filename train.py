import tensorflow as tf
from config import *
from model_factory import build_efficientnet_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    
def save_plots(history, model, val_ds, class_names):
    # 1. Training History Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('reports/training_history.png')

    # 2. Confusion Matrix
    y_true = []
    y_pred = []
    for x, y in val_ds:
        y_true.extend(np.argmax(y, axis=1))
        preds = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('reports/confusion_matrix.png')

if __name__ == "__main__":
    run_training()