# emotion_detection_final.py
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
import sounddevice as sd
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ======================
# Configuration
# ======================
LABEL_MAP = {
    0: "joy",
    1: "sadness",
    2: "anger",
    3: "fear",
    4: "love",
    5: "surprise"
}

# ======================
# NLTK Setup
# ======================
nltk.download('punkt', force=True)
nltk.download('wordnet', force=True)
nltk.download('omw-1.4', force=True)

# ======================
# Data Processing
# ======================
def load_data():
    df = pd.read_csv('emotions.csv')
    df['text'] = df['text'].str.lower()
    return df

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

# ======================
# Model Training
# ======================
def train_model(df):
    # Preprocess
    df['cleaned'] = df['text'].apply(clean_text)
    df['processed'] = df['cleaned'].apply(lemmatize_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Vectorize
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf = tfidf.transform(X_test)
    
    # Train
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_tf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tf)
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred, target_names=list(LABEL_MAP.values())))
    
    # Save artifacts
    joblib.dump(model, 'emotion_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    
    return model, tfidf

# ======================
# Audio Processing
# ======================
def record_audio(duration=5):
    fs = 16000  # Sample rate
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording.flatten()

def audio_to_text(audio_data):
    r = sr.Recognizer()
    audio = sr.AudioData(audio_data.tobytes(), 16000, 2)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None

# ======================
# Prediction
# ======================
def predict_emotion(text, model, tfidf):
    cleaned = lemmatize_text(clean_text(text))
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return LABEL_MAP[prediction]

# ======================
# Main Application
# ======================
def main():
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        print("Error: emotions.csv not found in current directory")
        return
    
    # Load or train model
    try:
        model = joblib.load('emotion_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        print("Loaded trained model")
    except FileNotFoundError:
        print("Training new model...")
        model, tfidf = train_model(df)
    
    # Interaction loop
    while True:
        choice = input("\nChoose input type:\n1. Text\n2. Audio\n3. Exit\n> ")
        
        if choice == '1':
            text = input("Enter text: ")
            emotion = predict_emotion(text, model, tfidf)
            print(f"Predicted emotion: {emotion}")
            
        elif choice == '2':
            try:
                duration = int(input("Recording duration (seconds): "))
                audio = record_audio(duration)
                text = audio_to_text(audio)
                if text:
                    emotion = predict_emotion(text, model, tfidf)
                    print(f"Recognized text: {text}\nPredicted emotion: {emotion}")
            except Exception as e:
                print(f"Audio error: {e}")
                
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()