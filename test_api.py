from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_metrics_endpoint():
    response = client.get('/api/metrics')
    assert response.status_code == 200
    assert 'requests' in response.json()
