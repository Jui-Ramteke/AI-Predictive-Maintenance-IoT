import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def generate_dashboard():
    print("Generating Visual Analytics Dashboard...")
    df = pd.read_csv('data/raw/sensor_data.csv')
    model = joblib.load('models/predictive_model.pkl')
    
    # Set the visual style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Sensor Correlation Heatmap
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', ax=axes[0, 0])
    axes[0, 0].set_title('Sensor Correlation Heatmap')

    # 2. Feature Importance (What the AI values most)
    importances = model.feature_importances_
    features = ['Temperature', 'Vibration', 'Pressure']
    sns.barplot(x=features, y=importances, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title('Feature Importance (AI Decision Drivers)')

    # 3. Sensor Distributions (Failure vs Healthy)
    sns.kdeplot(data=df, x='temperature', hue='failure', fill=True, ax=axes[1, 0])
    axes[1, 0].set_title('Temperature Distribution (Failure Overlap)')

    # 4. Vibration vs Pressure (Failure Clusters)
    sns.scatterplot(data=df.sample(500), x='vibration', y='pressure', hue='failure', ax=axes[1, 1], alpha=0.6)
    axes[1, 1].set_title('Vibration vs Pressure Cluster Analysis')

    plt.tight_layout()
    plt.savefig('outputs/analytics_dashboard.png')
    print("Dashboard saved to: outputs/analytics_dashboard.png")

if __name__ == "__main__":
    generate_dashboard()