import joblib


# 1. Load saved model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# 2. Prediction function
def predict_message(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return prediction


# 3. Simple test
if __name__ == "__main__":
    user_input = input("Enter a message: ")
    result = predict_message(user_input)
    print(f"Predicted label: {result}")