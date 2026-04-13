import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train():
    print("Loading simulated IoT data...")
    df = pd.read_csv('data/raw/sensor_data.csv')
    
    # 1. Feature Engineering & Setup
    # Input features: what the model learns from
    X = df[['temperature', 'vibration', 'pressure']]
    # Target variable: what the model tries to predict
    y = df['failure']
    
    # 2. Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Evaluate the model
    predictions = model.predict(X_test)
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_test, predictions))
    
    # 4. Generate and save Confusion Matrix Visualization
    print("Generating visual output (Confusion Matrix)...")
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Failure Prediction Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    print("Graph saved to: outputs/confusion_matrix.png")
    
    # 5. Save the trained model
    joblib.dump(model, 'models/predictive_model.pkl')
    print("Trained model saved to: models/predictive_model.pkl")

if __name__ == "__main__":
    train()