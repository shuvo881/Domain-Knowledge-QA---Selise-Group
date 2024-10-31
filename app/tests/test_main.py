from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_answer_endpoint():
    response = client.post("/answer", json={"question": "What is Retrieval-Augmented Generation?"})
    assert response.status_code == 200
    assert "answer" in response.json()
