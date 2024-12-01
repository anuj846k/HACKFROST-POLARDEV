# HACKFROST 2024 - POLARDEV 

## THE PROBLEM

In machine learning systems where real-time data continuously flows into the system, models can degrade in performance over time. This degradation happens because the new incoming data might not align with the data the model was originally trained on. This phenomenon is often referred to as data drift or concept drift.

To maintain the accuracy and reliability of the deployed model, I automated the process of monitoring data drift, retraining the model, and redeploying it with **kestra** which is an **orchestation tool**. This ensures that the system adapts to the new data environment without requiring constant manual intervention.


Links of the repo where Ml models are being deployed whenever retraining happens - https://github.com/anuj846k/kestra_demo

YouTubeVideo Link - https://youtu.be/VPWFfgOpi6E?si=0lHNaS3QMsBwmRxs


## Workflow

The Kestra workflow consists of four key steps:

1. **Data Drift Detection**  
   - Compares the incoming real-time data from https://www.data.gov.in/resource/real-time-air-quality-index-various-locations with the reference dataset (`reference_data.csv`).
   - If data drift is detected (based on statistical differences or other criteria), the workflow proceeds to retrain the model.

2. **Model Training**  
   - Processes the updated dataset for training.
   - Trains a Random Forest Regressor to predict pollution levels (`pollutant_avg`).
   - Outputs the trained model (`model.joblib`) and metrics (`metrics.json`).

3. **GitHub Integration**  
   - Clones a GitHub repository to store the retrained model and metrics files.
   - Pushes the outputs (`model.joblib` and `metrics.json`) back to the repository for version control.

4. **Dynamic Inputs and Secure Handling**  
   - Uses inputs and secrets for flexibility and security.

---

## Key Features

### **1. Data Drift Detection**
- A Python script computes statistical differences between the real-time data and the reference data.
- Checks include:
  - Mean and standard deviation comparisons.
  - Distribution overlap using metrics like Wasserstein Distance (optional).
- Retraining is triggered only if drift exceeds predefined thresholds.

### **2. Conditional Model Training**
- Ensures resources are used efficiently by retraining the model only when necessary.
- Retraining uses both the reference and real-time datasets for improved performance.

### **3. Secure GitHub Integration**
- Credentials (`GITHUB_USERNAME` and `GITHUB_TOKEN`) are passed as secrets.
- Files are version-controlled for reproducibility.

---

## Workflow Inputs

| Input            | Type   | Description                                   |
|-------------------|--------|-----------------------------------------------|
| `reference_data`  | File   | The original dataset for training the model.  |
| `real_time_data`  | File   | Incoming data for drift detection.            |
| `GITHUB_USERNAME` | String | GitHub username, passed as a secret.          |
| `GITHUB_TOKEN`    | String | GitHub personal access token, passed as a secret. |
| `branch`          | String | Git branch for pushing the files (default: `main`). |

---

## Workflow Steps

### **1. Data Drift Detection**
The `check_drift` task runs a Python script to analyze data differences.  

Example:
```python
import subprocess
import os
import pandas as pd
from scipy import stats
import numpy as np
import requests
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
import json
from pymongo import MongoClient

# Load reference and current data
reference_data = pd.read_csv("data/reference_data.csv", parse_dates=['last_update'])
api_url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=YOUR_API_KEY&format=csv"
response = requests.get(api_url)
with open('new_data.csv', 'w') as f:
    f.write(response.text)

current_data = pd.read_csv("new_data.csv", parse_dates=['last_update'])

# Drift detection function
def detect_drift(ref_data, curr_data, features):
    drift_score = 0
    for feature in features:
        ref_mean = ref_data[feature].mean()
        curr_mean = curr_data[feature].mean()
        diff = abs(ref_mean - curr_mean) / ref_mean
        drift_score += diff
    return drift_score / len(features)

# Features for drift detection
numerical_features = ['pollutant_min', 'pollutant_max', 'pollutant_avg']
drift_score = detect_drift(reference_data, current_data, numerical_features)

# Decide on retraining
needs_retraining = drift_score > 0.1

# Save drift results
results = {
    "drift_score": drift_score,
    "needs_retraining": needs_retraining
}
with open('drift_report.json', 'w') as f:
    json.dump(results, f, indent=4)

# Trigger retraining if necessary
if needs_retraining:
    subprocess.run(["python", "train_model.py"])

# Save updated data to MongoDB if drift detected
if needs_retraining:
    MONGODB_URI = os.getenv("MONGODB_URI")
    client = MongoClient(MONGODB_URI)
    db = client["drift_detection"]
    historical_data = db["reference_data_history"]
    historical_data.insert_one({"timestamp": pd.to_datetime("now"), "drift_score": drift_score})


```

### **2. Conditional Model Training**
If data drift is detected, the `train_model` task is executed to retrain the model. This step is skipped if no drift is detected.

### **3. GitHub Push**
The retrained model and updated metrics are pushed to the specified GitHub repository.

---

## Dependencies

The workflow uses the following Python packages:
```plaintext
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.1
scipy==1.11.0
joblib==1.2.0
requests==2.31.0
```

---

## Example Outputs

- **Trained Model:** `model.joblib`
- **Performance Metrics:** `metrics.json`  

Example of `metrics.json`:
```json
{
    "mse": 93.58324999999999,
    "r2_score": -92.58324999999999,
    "feature_importance": {
        "latitude": 0.09072690891402097,
        "longitude": 0.10329695033286054,
        "pollutant_min": 0.17495759633176836,
        "pollutant_max": 0.6310185444213502,
        "hour": 0.0,
        "day": 0.0,
        "month": 0.0
    },
    "data_stats": {
        "total_samples": 10,
        "training_samples": 8,
        "test_samples": 2
    }
}
```

---

## Running the Workflow

1. Upload both the `reference_data.csv` file and fetch the real time data from https://www.data.gov.in/resource/real-time-air-quality-index-various-locations .
2. Set the required secrets (`GITHUB_USERNAME`, `GITHUB_TOKEN`) in Kestra.
3. Trigger the workflow. The workflow will:
   - Check for data drift.
   - Retrain the model if drift is detected.
   - Push the outputs to GitHub.

---
