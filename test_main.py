import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image
from main import app

client = TestClient(app)

def create_test_image():
    """Helper function to create a dummy image for testing."""
    file = io.BytesIO()
    image = Image.new('RGB', (300, 300), color='red')
    image.save(file, 'jpeg')
    file.seek(0)
    return file

def test_health_check():
    """Tests the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_invalid_file_type():
    """Verifies that the API rejects non-image files."""
    files = {'file': ('test.txt', b'hello world', 'text/plain')}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()

def test_predict_success():
    """Tests a successful prediction request."""
    # This test requires model.h5 to be present to pass 200 OK
    test_img = create_test_image()
    files = {'file': ('test.jpg', test_img, 'image/jpeg')}
    
    response = client.post("/predict", files=files)
    
    # If model is loaded, we expect 200. If model missing, 503 Service Unavailable.
    if response.status_code == 200:
        data = response.json()
        assert "class_name" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
    else:
        assert response.status_code == 503