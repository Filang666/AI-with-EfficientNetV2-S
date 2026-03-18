import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from main import app

# Use a fixture to initialize the test client
@pytest.fixture
def client():
    """Returns a FastAPI test client."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def dummy_image():
    """Generates a valid dummy JPEG image in memory."""
    file = io.BytesIO()
    image = Image.new('RGB', (300, 300), color='blue')
    image.save(file, format='JPEG')
    file.seek(0)
    return file

# --- Tests ---

def test_health_endpoint(client):
    """Checks if the service and model are online."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_no_file(client):
    """Ensures 422 error if no file is uploaded."""
    response = client.post("/predict")
    assert response.status_code == 422

def test_predict_wrong_extension(client):
    """Verifies rejection of unsupported file formats (e.g., .txt)."""
    files = {'file': ('test.txt', b'not-an-image', 'text/plain')}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Only JPG/PNG" in response.json()["detail"]

def test_predict_success(client, dummy_image):
    """Full integration test for a successful prediction."""
    files = {'file': ('test.jpg', dummy_image, 'image/jpeg')}
    response = client.post("/predict", files=files)
    
    # If the model is present (model.h5 exists)
    if response.status_code == 200:
        data = response.json()
        assert "class_name" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
    else:
        # If model.h5 is missing, the API should return 503 (Service Unavailable)
        assert response.status_code == 503