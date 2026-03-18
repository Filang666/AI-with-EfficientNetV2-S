# EfficientNetV2-S Image Classification Service

A production-ready deep learning microservice for high-accuracy image classification. Built with **TensorFlow 2.x**, **FastAPI**, and **Docker**.

## 🚀 Overview
This project provides a complete pipeline for training and deploying an image classifier based on the **EfficientNetV2-S** architecture. It includes automated class management, fine-tuning logic, and a high-performance REST API for inference.

### Key Features
- **Transfer Learning & Fine-tuning:** Leverages ImageNet weights with a two-stage training process.
- **Production-Ready API:** Built with FastAPI, featuring asynchronous processing and Pydantic validation.
- **Infrastructure as Code:** Fully containerized using Docker and Docker Compose.
- **Automated Testing:** Robust test suite using Pytest and FastAPI TestClient.
- **Dynamic Class Handling:** Automatically detects and maps classes from the dataset directory.

---

## 🛠 Tech Stack
- **Deep Learning:** TensorFlow, Keras
- **Web Framework:** FastAPI, Uvicorn
- **Data Processing:** NumPy, Pillow, Pandas
- **DevOps:** Docker, Docker Compose
- **Testing:** Pytest, HTTPX

---

## 📊 Model Evaluation

To ensure the model is robust and generalize well to unseen data, we track performance metrics and analyze the decision-making patterns.

<table border="0">
  <tr>
    <td>
      <p align="center"><b>Training Metrics (Accuracy & Loss)</b></p>
      <img src="reports/training_history.png" width="500">
    </td>
    <td>
      <p align="center"><b>Confusion Matrix</b></p>
      <img src="reports/confusion_matrix.png" width="400">


    </td>
  </tr>
</table>

### Performance Insights:
- **Convergence:** The model shows stable convergence with minimal gap between training and validation loss, indicating no significant overfitting.
- **Reliability:** The Confusion Matrix confirms high precision across all classes, with minor confusion only between visually similar categories.
- **Explainability:** We monitor these charts during the fine-tuning stage to ensure the learning rate is optimal for the **EfficientNetV2** backbone.


## 📦 Project Structure
- `config.py`: Centralized configuration & constants
- `model_factory.py`: Reusable model architecture logic
- `train.py`: Training & fine-tuning pipeline
- `main.py`: FastAPI production server
- `test_main.py`: Integration & unit tests
- `requirements.txt`: Fixed-version dependencies
- `Dockerfile`: Container definition
- `docker-compose.yml`: Multi-container orchestration

## ⚡ Quick Start

### 1. Requirements
 Python 3.10+ or Docker

### 2. Installation
```bash

git clone https://github.com
cd AI-with-EfficientNetV2-S
pip install -r requirements.txt
```

### 3. Training

Place your images in the dataset/ folder (organized by subfolders) and run:
bash
```bash

python train.py
```


### 4. Running with Docker (Recommended)
Launch the inference service instantly:
bash
```bash

docker-compose up --build
```

The API will be available at http://localhost:8000.
## 🧪 Testing
Run the automated test suite to ensure service stability:
```bash

pytest test_main.py
```

## 📡 API Documentation
Once the service is running, explore the interactive **Swagger UI** at:
http://localhost:8000/docs
### Example Request (Python)

```python

import requests

files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
print(response.json())
```

## 👤 Author
 GitHub: @Filang666
 LinkedIn: 
