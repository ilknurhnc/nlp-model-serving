## Relationship Pattern Classifier

- A machine learning–based NLP system that classifies relationship communication patterns into predefined categories such as lovebombing, gaslighting, mansplaining, breadcrumbing, and normal communication.

- This project demonstrates an end-to-end machine learning pipeline, including dataset creation, model training, evaluation, and deployment via a REST API.

## Project Overview

The system includes:
- Custom dataset creation with labeled communication patterns

- Text preprocessing using TF-IDF

- Model training with Logistic Regression

- Evaluation using standard classification metrics

- Local inference script for testing predictions

- FastAPI-based REST API for real-time inference

## Project Structure
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
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── train.py
│   ├── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore

## Dataset
The dataset is manually curated and consists of labeled text samples.

# Each sample contains:

- text: input message

- label: communication category

Example:

text,label
"Sen benim ruh eşimsin",lovebombing
"Ben öyle bir şey demedim",gaslighting
"Yanlış biliyorsun, anlatayım",mansplaining
"Belki sonra konuşuruz",breadcrumbing
"Yarın buluşalım mı?",normal


# The dataset includes a mix of:

- Strong (explicit) examples

- Medium-level examples

- Subtle, real-world variations

- Installation

# Install dependencies:

-pip install -r requirements.txt

Training the Model

Run:

python src/train.py

This process:

Loads the dataset

Splits it into training and test sets

Converts text into numerical features using TF-IDF

Trains a Logistic Regression model

Evaluates performance

Saves the trained model and vectorizer

Output:

models/model.pkl
models/vectorizer.pkl
Local Inference

Run:

python src/predict.py

Example:

Enter a message: Sen benim kaderimsin
Predicted label: lovebombing
Confidence: 0.87
API Usage
Run the API
uvicorn api.main:app --reload

The API will be available at:

http://127.0.0.1:8000
API Documentation

Interactive documentation is available at:

http://127.0.0.1:8000/docs
Predict Endpoint

POST /predict

Request
{
  "text": "Sen benim ruh eşimsin"
}
Response
{
  "input_text": "Sen benim ruh eşimsin",
  "predicted_label": "lovebombing",
  "confidence": 0.87
}
Health Check

GET /health

Response:

{
  "status": "ok"
}
Model Details

Model: Logistic Regression

Feature extraction: TF-IDF

Task: Multi-class text classification

Labels:

lovebombing

gaslighting

mansplaining

breadcrumbing

normal

Performance

The model achieves approximately:

Accuracy: ~0.88

Performance varies by class:

Strong performance on lovebombing and breadcrumbing

Moderate performance on mansplaining and normal

Lower recall on subtle gaslighting examples

Notes

The model is trained on a custom dataset and may produce lower confidence for:

Very short inputs

Ambiguous or context-dependent messages

Confidence scores represent model certainty, not correctness.

Inputs such as "ok" or "rahat ol" may result in less reliable predictions due to lack of context.

Future Improvements

Improve gaslighting detection with more subtle examples

Add top-N prediction outputs

Implement logging and monitoring

Containerize the application using Docker

Deploy the service to a cloud environment

Add a simple user interface

Summary

This project demonstrates:

Dataset design for NLP tasks

Feature engineering using TF-IDF

Model training and evaluation

Error analysis and iterative improvement

Serving machine learning models via a REST API