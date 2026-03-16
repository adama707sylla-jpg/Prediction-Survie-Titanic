# test_app.py
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Test 1 — L'API répond ?
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    # 200 = "tout va bien" en HTTP

# Test 2 — La prédiction fonctionne ?
def test_predict():
    response = client.post("/predict", json={
        "Pclass": 1,
        "Age": 25,
        "Fare": 100.0,
        "family": 0,
        "Sex": "female",
        "Embarked": "S"
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confiance" in response.json()

# Test 3 — La confiance est entre 0 et 1 ?
def test_confiance_valide():
    response = client.post("/predict", json={
        "Pclass": 2,
        "Age": 30,
        "Fare": 50.0,
        "family": 1,
        "Sex": "male",
        "Embarked": "C"
    })
    confiance = response.json()["confiance"]
    assert 0.0 <= confiance <= 1.0
    # Si confiance = 1.5 par exemple → test échoue → pas de déploiement