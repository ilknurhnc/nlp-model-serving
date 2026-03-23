import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# 1. Load dataset
df = pd.read_csv("data/raw/messages.csv")

# 2. Preview data
print("First 5 rows:")
print(df.head())

# 3. Separate features and labels
X = df["text"]
y = df["label"]

# 4. Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Convert text into numerical features
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 6. Create model
model = LogisticRegression(max_iter=1000)

# 7. Train model
model.fit(X_train_vectorized, y_train)

# 8. Predict on test set
y_pred = model.predict(X_test_vectorized)

# 9. Print evaluation results
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Save model and vectorizer
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel and vectorizer saved successfully.")