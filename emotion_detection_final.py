# emotion_detection_final.py
# Install required packages: pandas, numpy, scikit-learn, nltk, matplotlib

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk

# Download NLTK resources with explicit paths
nltk.download('punkt', force=True)
nltk.download('wordnet', force=True)
nltk.download('omw-1.4', force=True)
nltk.download('punkt_tab', force=True)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('C:/Users/mdevr/Desktop/git/emotions.csv')


# Clean text data
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# Lemmatize text with error handling
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    try:
        tokens = word_tokenize(text)
        return ' '.join([lemmatizer.lemmatize(word) for word in tokens])
    except:
        return text

# Preprocessing pipeline
df['cleaned_text'] = df['text'].apply(clean_text)
df['processed_text'] = df['cleaned_text'].apply(lemmatize_text)

# EDA Visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df)
plt.title('Emotion Class Distribution')
plt.xticks(rotation=45)
plt.savefig('class_distribution.png')
plt.close()

# Prepare data
X = df['processed_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Save vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Basic ML Models
def train_evaluate_model(model, model_name):
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=df['label'].unique(), 
                yticklabels=df['label'].unique())
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    
    return model

# Logistic Regression
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    C=1.0
)
lr_model = train_evaluate_model(lr_model, "Logistic Regression")

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_model = train_evaluate_model(rf_model, "Random Forest")

# Save best model
joblib.dump(lr_model, 'emotion_classifier_lr.pkl')

# Prediction Example
def predict_emotion(text):
    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    vectorized = tfidf.transform([lemmatized])
    prediction = lr_model.predict(vectorized)[0]
    return prediction

# Test prediction
sample_texts = [
    "I'm feeling ecstatic and overjoyed!",
    "This situation makes me furious and angry",
    "I feel so alone and hopeless..."
]

print("\nSample Predictions:")
for text in sample_texts:
    print(f"Text: {text}")
    print(f"Predicted Emotion: {predict_emotion(text)}\n")

# Feature Importance Visualization (for Logistic Regression)
feature_names = tfidf.get_feature_names_out()
coefs = lr_model.coef_[0]
top_features = pd.DataFrame({
    'feature': feature_names,
    'weight': coefs
}).sort_values(by='weight', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='weight', y='feature', data=top_features)
plt.title('Top 20 Important Features for Emotion Detection')
plt.savefig('feature_importance.png')
plt.close()