import pandas as pd
import joblib

def predict_live_data():
    print("Loading the trained AI model...")
    try:
        model = joblib.load('models/predictive_model.pkl')
    except FileNotFoundError:
        print("Error: Model not found. Please run train_model.py first.")
        return

    # Simulating 5 live, incoming IoT readings
    # Machines 1, 2, and 3 are normal. Machines 4 and 5 are degrading.
    live_data = pd.DataFrame({
        'temperature': [71.2, 69.5, 70.1, 88.5, 92.0],
        'vibration': [1.4, 1.6, 1.5, 3.8, 4.1],
        'pressure': [101.0, 99.5, 100.2, 75.0, 70.0]
    })
    
    print("\n--- Live IoT Sensor Feed ---")
    
    # Get the raw prediction (0 or 1)
    predictions = model.predict(live_data)
    # Get the exact mathematical probability (e.g., 0.85 = 85% chance of failure)
    probabilities = model.predict_proba(live_data)[:, 1]
    
    for i in range(len(live_data)):
        prob_percentage = probabilities[i] * 100
        
        if predictions[i] == 1:
            status = "🚨 CRITICAL: MAINTENANCE REQUIRED"
        elif prob_percentage > 30:
            status = "⚠️ WARNING: MONITOR CLOSELY"
        else:
            status = "✅ HEALTHY"
            
        print(f"Machine {i+1} | Failure Risk: {prob_percentage:05.1f}% | Status: {status}")

if __name__ == "__main__":
    predict_live_data()