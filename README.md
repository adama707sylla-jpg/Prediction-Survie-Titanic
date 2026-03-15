# 🚢 Prediction Survie Titanic — Pipeline MLOps Complet

> Modèle de classification entraîné, tracké, containerisé et déployé via une API REST.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-tracking-orange?style=flat-square&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-containerisé-blue?style=flat-square&logo=docker)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-yellow?style=flat-square&logo=scikitlearn)

---

## 📌 Objectif

Prédire la **survie d'un passager du Titanic** à partir de ses caractéristiques (classe, âge, sexe, tarif...) en suivant une approche **MLOps de bout en bout** :

```
Données brutes → Preprocessing → Entraînement → Tracking → API → Docker → GitHub
```

---

## 🗂️ Structure du projet

```
Prediction-Survie-Titanic/
│
├── app.py                  # API FastAPI — endpoint /predict
├── appint.py               # Script d'intégration / tests
├── mlflow_tracker.py       # Entraînement + logging MLflow
├── mon_outillage.py        # Pipeline de preprocessing
├── titan.ipynb             # Exploration & analyse des données
│
├── Dockerfile              # Containerisation de l'API
├── requirements.txt        # Dépendances Python
├── .gitignore
└── README.md
```

---

## 🤖 Modèle

| Paramètre | Valeur |
|---|---|
| Algorithme | Random Forest Classifier |
| `n_estimators` | 200 |
| `max_depth` | 6 |
| `random_state` | 42 |
| **Accuracy** | **~81%** |
| **Confiance max** | **92.9%** |

Plusieurs versions comparées via **MLflow** :

| Run | n_estimators | max_depth | Accuracy |
|---|---|---|---|
| RF_v1 | 100 | 6 | 0.8045 |
| RF_v2 | 200 | 6 | 0.8101 ✅ |

---

## 🔬 Tracking MLflow

Les expériences sont loggées dans MLflow avec :
- **Paramètres** : `n_estimators`, `max_depth`, `random_state`
- **Métriques** : `accuracy`, `precision`, `recall`, `f1_score`
- **Artefacts** : modèle `.pkl` sauvegardé par run

```bash
# Lancer l'interface MLflow
mlflow ui
# → http://127.0.0.1:5000
```

---

## 🚀 Lancer l'API

### Option 1 — En local

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Option 2 — Via Docker

```bash
# Build de l'image
docker build -t api-titanic .

# Lancement du container
docker run -p 8000:8000 api-titanic
```

L'API est disponible sur : `http://127.0.0.1:8000`

---

## 📡 Endpoints

### `GET /`
Vérifie que l'API est opérationnelle.

```json
{
  "message": "API Prédiction Survie Titanic operationnelle !",
  "features": ["Pclass", "Age", "Fare", "family", "Sex", "Embarked"],
  "total": 6
}
```

### `POST /predict`
Prédit la survie d'un passager.

**Request body :**
```json
{
  "Pclass": 1,
  "Age": 25,
  "Fare": 100.0,
  "family": 0,
  "Sex": "female",
  "Embarked": "S"
}
```

**Response :**
```json
{
  "prediction": "1",
  "confiance": 0.9292,
  "type": "classification"
}
```

> `prediction: "1"` = survie | `prediction: "0"` = décès

### `GET /docs`
Interface Swagger interactive pour tester l'API directement depuis le navigateur.

---

## 🧪 Features utilisées

| Feature | Description |
|---|---|
| `Pclass` | Classe du billet (1, 2, 3) |
| `Age` | Âge du passager |
| `Fare` | Prix du billet |
| `family` | Nombre de membres de la famille à bord |
| `Sex` | Sexe (`male` / `female`) |
| `Embarked` | Port d'embarquement (`S`, `C`, `Q`) |

---

## 🐋 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY modele.pkl .
COPY mon_outillage.py .
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🗺️ Pipeline MLOps

```
📊 Données         Titanic dataset (train/test split 80/20)
      ↓
🔧 Preprocessing   Encodage, gestion NaN, feature engineering
      ↓
🤖 Entraînement    Random Forest — grid sur max_depth & n_estimators
      ↓
📈 Tracking        MLflow — comparaison des runs, sélection du meilleur
      ↓
💾 Sauvegarde      modele.pkl
      ↓
🌐 API             FastAPI — endpoint /predict avec score de confiance
      ↓
🐋 Docker          Image containerisée, portable
      ↓
🐙 GitHub          Code versionné, .gitignore propre
```

---

## 📦 Installation

```bash
git clone https://github.com/adama707sylla-jpg/Prediction-Survie-Titanic.git
cd Prediction-Survie-Titanic
pip install -r requirements.txt
```

---

## 👤 Auteur

**Adama Sylla** — Projet MLOps personnel  
🔗 [GitHub](https://github.com/adama707sylla-jpg)
