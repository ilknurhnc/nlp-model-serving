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
