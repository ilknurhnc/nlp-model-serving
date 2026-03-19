import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# 1. CSV dosyasını oku
df = pd.read_csv("data/raw/messages.csv")

# 2. İlk birkaç satırı kontrol et
print("First 5 rows:")
print(df.head())

# 3. Text ve label kolonlarını ayır
X = df["text"]
y = df["label"]

# 4. Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Metinleri sayıya çevir
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 6. Modeli oluştur
model = LogisticRegression(max_iter=1000)

# 7. Modeli eğit
model.fit(X_train_vectorized, y_train)

# 8. Tahmin al
y_pred = model.predict(X_test_vectorized)

# 9. Sonuçları yazdır
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))