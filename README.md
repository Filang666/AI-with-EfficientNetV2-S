# Image Classifier (EfficientNetV2-S)

High-performance image classification model optimized for 16GB VRAM.

## Features
- **EfficientNetV2-S Architecture**: Pre-trained on ImageNet for maximum accuracy.
- **High Resolution**: Processing at 300x300 for fine details.
- **Two-Stage Training**: Transfer learning followed by deep Fine-Tuning.
- **Auto-Splitting**: Automatic 80/10/10 split (Train/Val/Test).
- **Data Augmentation**: Built-in flips, rotations, and zooms to prevent overfitting.

## Setup
1. Prepare dataset: 10 folders named `0` to `9`, each with 1000 PNG images (1000x1000).
2. Install dependencies: `pip install tensorflow numpy matplotlib`
3. Set your data path in `train.py`.
4. Run: `python train.py`

## Project Structure
- `train.py` – script for loading data, training, and fine-tuning.
- `predict.py` – script for single image inference.
- `model.h5` – saved model after training.
- `dataset/` – root folder containing subfolders 0, 1, 2... 9.

## Training Process
1. **Stage 1**: Train top layers only (Adam, default LR).
2. **Stage 2 (Fine-Tuning)**: Unfreeze base model and train with `1e-5` LR.
3. **Early Stopping**: Automatically restores best weights if accuracy plateaus.
4. **Final accuracy test**: Evaluation on unseen data (10% of total).

## Usage (Inference)
To predict a single image:
1. Load `model.h5`.
2. Resize target image to **300x300**.
3. Run `np.argmax(model.predict(img))` to get the folder index (0-9).

