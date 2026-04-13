import pandas as pd
import numpy as np

# 1. Setup Parameters
num_samples = 1000
np.random.seed(42)

# 2. Generate Base Normal Data
# Normal Temp: ~70, Vibration: ~1.5, Current: ~100
timestamps = pd.date_range(start='2026-01-01', periods=num_samples, freq='min')
temp = np.random.normal(70, 3, num_samples)
vib = np.random.normal(1.5, 0.2, num_samples)
curr = np.random.normal(100, 5, num_samples)
failure = np.zeros(num_samples)

# 3. Inject "Mechanical Failure" (Samples 400 to 449 = 50 samples)
# Pattern: Vibration spikes, Temp rises slowly
vib[400:450] += np.random.uniform(2.0, 3.5, 50)
temp[400:450] += np.random.uniform(10, 15, 50)
failure[400:450] = 1

# 4. Inject "Electrical Failure" (Samples 800 to 849 = 50 samples)
# Pattern: Current spikes, Temp spikes rapidly
# FIXED: Changed 51 to 50 to match the slice size
curr[800:850] += np.random.uniform(30, 50, 50) 
temp[800:850] += np.random.uniform(20, 30, 50)
failure[800:850] = 1

# 5. Create DataFrame and Save
test_df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temp,
    'vibration': vib,
    'current': curr,
    'failure': failure.astype(int)
})

test_df.to_csv('test_sensor_data.csv', index=False)
print("Success: 'test_sensor_data.csv' has been created in your root folder!")