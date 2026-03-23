import joblib
import numpy as np

# Load saved model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def predict_message(text):
    text_vectorized = vectorizer.transform([text])

    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]

    confidence = np.max(probabilities)

    return prediction, confidence


if __name__ == "__main__":
    user_input = input("Enter a message: ")
    label, confidence = predict_message(user_input)

    print(f"Predicted label: {label}")
    print(f"Confidence: {confidence:.2f}")