# Relationship Pattern Classifier

A machine learning–based NLP system that classifies relationship communication patterns from text messages into predefined behavioral categories such as lovebombing, gaslighting, mansplaining, breadcrumbing, and normal communication.

The project demonstrates a complete end-to-end machine learning workflow including:

- Custom dataset creation
- Text preprocessing
- Feature extraction with TF-IDF
- Model training and evaluation
- REST API deployment with FastAPI
- Frontend integration
- Docker containerization
- Multi-service orchestration with Docker Compose

---

# System Architecture

```text
User Input
    ↓
Frontend UI
    ↓
FastAPI Backend
    ↓
TF-IDF Vectorizer
    ↓
Logistic Regression Model
    ↓
Prediction Response
```

---

# Features

- Multi-class text classification
- Custom manually curated NLP dataset
- TF-IDF–based feature engineering
- Logistic Regression classifier
- Real-time prediction API
- Frontend interface for live interaction
- Dockerized backend and frontend services
- Docker Compose orchestration
- Structured project organization

---

# Labels

The system currently supports the following communication pattern categories:

- lovebombing
- gaslighting
- mansplaining
- breadcrumbing
- normal

---

# Project Structure

```text
relationship-pattern-classifier/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── messages.csv
│   └── processed/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── Dockerfile
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

The dataset is manually curated and consists of labeled conversational text samples.

Each sample contains:

- `text` → input message
- `label` → communication pattern category

Example:

```csv
text,label
"Sen benim ruh eşimsin",lovebombing
"Ben öyle bir şey demedim",gaslighting
"Yanlış biliyorsun, anlatayım",mansplaining
"Belki sonra konuşuruz",breadcrumbing
"Yarın buluşalım mı?",normal
```

The dataset includes:

- Explicit examples
- Medium-strength examples
- Subtle conversational patterns
- Realistic real-world variations

---

# Machine Learning Pipeline

## 1. Data Collection

Custom conversational examples are manually created and labeled.

## 2. Text Vectorization

Text inputs are transformed into numerical representations using TF-IDF vectorization.

## 3. Model Training

The classifier is trained using Logistic Regression.

## 4. Evaluation

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

## 5. Inference

The trained model predicts communication patterns from unseen text inputs.

---

# Tech Stack

## Backend

- Python
- FastAPI
- Pydantic
- Scikit-learn

## Machine Learning

- TF-IDF Vectorization
- Logistic Regression
- NumPy
- Pandas
- Joblib

## Frontend

- HTML
- CSS
- JavaScript

## Infrastructure

- Docker
- Docker Compose

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd relationship-pattern-classifier
```

---

# Local Development Setup

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python3 src/train.py
```

## Run API

```bash
uvicorn api.main:app --reload
```

API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

# Frontend Development

Run the frontend locally:

```bash
cd frontend
python3 -m http.server 3000
```

Frontend will be available at:

```text
http://localhost:3000
```

---

# Docker Setup

## Build and Run with Docker Compose

```bash
docker compose up --build
```

Services:

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

# API Usage

## Prediction Endpoint

```http
POST /predict
```

## Example Request

```json
{
  "text": "Sen benim ruh eşimsin"
}
```

## Example Response

```json
{
  "input_text": "Sen benim ruh eşimsin",
  "predicted_label": "lovebombing",
  "confidence": 0.87
}
```

---

# Example Workflow

1. User enters a message in the frontend interface
2. Frontend sends a POST request to FastAPI
3. FastAPI preprocesses the input
4. TF-IDF vectorizer transforms the text
5. Logistic Regression model predicts the label
6. Prediction result is returned to the frontend

---

# Current Status

The project currently supports:

- End-to-end NLP classification
- REST API inference
- Frontend integration
- Dockerized deployment
- Multi-container orchestration with Docker Compose

---

# Future Improvements

Potential future enhancements include:

- Larger and more balanced datasets
- Deep learning–based NLP models
- Transformer architectures (BERT)
- Confidence calibration
- User authentication
- Database integration
- Cloud deployment
- Monitoring and logging
- CI/CD pipelines

---

# License

This project is intended for educational and portfolio purposes.