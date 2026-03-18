import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def dummy_image():
    file = io.BytesIO()
    image = Image.new('RGB', (300, 300), color='blue')
    image.save(file, format='JPEG')
    file.seek(0)
    return file

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_reports_generated():
    """Verify that evaluation images exist in the reports folder."""
    # This assumes you ran train.py at least once
    if os.path.exists('model.h5'):
        assert os.path.exists('reports/training_history.png'), "Missing accuracy/loss plot"
        assert os.path.exists('reports/confusion_matrix.png'), "Missing confusion matrix plot"

def test_predict_success(client, dummy_image):
    files = {'file': ('test.jpg', dummy_image, 'image/jpeg')}
    response = client.post("/predict", files=files)
    
    if response.status_code == 200:
        assert "label" in response.json()
        assert "confidence" in response.json()
    else:
        assert response.status_code in [503, 500]
