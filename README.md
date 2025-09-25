# Sentiment-Analysis-of-Product-Reviews-using-Naive-Bayes# Step 1: Imports and Setup\

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Step 2: Small Sample Dataset (Product Reviews)
data = {
    'review': [
        "This product is amazing and works great!",
        "I love the quality, highly recommend.",
        "Terrible quality, broke after one use.",
        "Waste of money, very disappointed.",
        "Good value for the price, satisfied.",
        "Not as expected, poor performance.",
        "Excellent features and durable.",
        "Awful customer service and product.",
        "Fantastic buy, exceeded expectations.",
        "Defective item, do not buy."
    ],
    'sentiment': [1, 1, -1, -1, 1, -1, 1, -1, 1, -1]  # 1: Positive, -1: Negative
}
df = pd.DataFrame(data)
print("Dataset:\n", df)

# Step 3: Text Preprocessing Function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())  # Clean: lowercase, remove non-alpha
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]  # Remove stopwords, lemmatize
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("\nCleaned Reviews:\n", df['cleaned_review'])

# Step 4: Feature Extraction and Train-Test Split
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))  # Simple TF-IDF
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\nTrain shape: {X_train_tfidf.shape}")

# Step 5: Train Naive Bayes Model
model = MultinomialNB(alpha=1.0)  # Laplace smoothing
model.fit(X_train_tfidf, y_train)

# Step 6: Predict and Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f} (e.g., 0.80 means 80% correct)")
print("Predictions:", y_pred)
print("Actual:", y_test.values)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Example Prediction on New Review
new_review = "This is a fantastic product!"
cleaned_new = preprocess_text(new_review)
new_tfidf = vectorizer.transform([cleaned_new])
pred = model.predict(new_tfidf)[0]
print(f"\nNew Review Prediction: {'Positive' if pred == 1 else 'Negative'}")
