import pandas as pd
import numpy as np
import os

def generate_iot_data(num_records=5000):
    np.random.seed(42)
    
    # Simulate normal operational data
    temperature = np.random.normal(loc=70, scale=5, size=num_records)
    vibration = np.random.normal(loc=1.5, scale=0.3, size=num_records)
    pressure = np.random.normal(loc=100, scale=10, size=num_records)
    
    # Introduce anomalies (failures)
    # Let's say 10% of the time, the machine is failing
    failure_labels = np.random.choice([0, 1], size=num_records, p=[0.9, 0.1])
    
    # Modify sensor readings where failure is 1 (Machine is breaking down)
    for i in range(num_records):
        if failure_labels[i] == 1:
            temperature[i] += np.random.normal(loc=15, scale=5) # Overheating
            vibration[i] += np.random.normal(loc=2, scale=0.5)  # Shaking
            pressure[i] -= np.random.normal(loc=20, scale=5)    # Pressure drop
            
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='1/1/2026', periods=num_records, freq='h'),
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'failure': failure_labels
    })
    
    # Save to the raw data folder
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/sensor_data.csv', index=False)
    print("Virtual IoT Data Generated successfully at: data/raw/sensor_data.csv")

if __name__ == "__main__":
    generate_iot_data()