import subprocess
import sys
import warnings
import os
from datetime import datetime,timezone
import pandas as pd
from scipy import stats
import numpy as np
import requests
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
import json
import shutil
from pymongo import MongoClient

warnings.filterwarnings('ignore')


os.makedirs("data", exist_ok=True)

MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client["drift_detection"]
historical_data = db["reference_data_history"]

api_url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b&format=csv"
response = requests.get(api_url)
if response.status_code != 200:
    print(f"Error fetching data from API: {response.status_code}")
    raise Exception("Failed to fetch data from API")

with open('new_data.csv', 'w') as f:
    f.write(response.text)

reference_data = pd.read_csv("data/reference_data.csv", sep=',', parse_dates=['last_update'])
current_data = pd.read_csv("new_data.csv", sep=',', parse_dates=['last_update'])
epsilon = 1e-10
numerical_features = [
    'pollutant_min', 'pollutant_max', 'pollutant_avg'
]
for col in numerical_features:
    reference_data[col] = reference_data[col].replace(0, epsilon)
    current_data[col] = current_data[col].replace(0, epsilon)

print("Reference Data Columns:", reference_data.columns.tolist())
print("Current Data Columns:", current_data.columns.tolist())

column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_features

def clean_data_for_mongodb(data):
    """Clean pandas data for MongoDB insertion"""
    if isinstance(data, dict):
        return {k: clean_data_for_mongodb(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_mongodb(v) for v in data]
    elif isinstance(data, pd.Timestamp):
        return data.isoformat() if pd.notna(data) else None
    elif pd.isna(data):
        return None
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    return data

def calculate_drift_score(ref_data, curr_data, features):
    feature_drifts = {}
    overall_drift = 0
    
    for feature in features:
        ref_stats = {
            'mean': ref_data[feature].mean(),
            'std': ref_data[feature].std(),
            'median': ref_data[feature].median()
        }
        
        curr_stats = {
            'mean': curr_data[feature].mean(),
            'std': curr_data[feature].std(),
            'median': curr_data[feature].median()
        }
        
        mean_change = abs(curr_stats['mean'] - ref_stats['mean']) / (ref_stats['mean'] + epsilon)
        std_change = abs(curr_stats['std'] - ref_stats['std']) / (ref_stats['std'] + epsilon)
        
    
        feature_drift = (mean_change + std_change) / 2
        feature_drifts[feature] = feature_drift
        
        print(f"\n{feature} drift analysis:")
        print(f"Mean change: {mean_change:.2%}")
        print(f"Std change: {std_change:.2%}")
        print(f"Feature drift score: {feature_drift:.2%}")
        
    overall_drift = np.mean(list(feature_drifts.values()))
    return overall_drift, feature_drifts


drift_score, feature_drifts = calculate_drift_score(
    reference_data, 
    current_data, 
    numerical_features
)

print(f"\nOverall drift score: {drift_score:.2%}")
needs_retraining = drift_score > 0.1  

validation_outputs = {
    "drift_score": float(drift_score),
    "needs_retraining": bool(needs_retraining),
    "feature_drifts": {k: float(v) for k, v in feature_drifts.items()},
    "feature_stats": {
        feature: {
            "ref_mean": float(reference_data[feature].mean()),
            "curr_mean": float(current_data[feature].mean()),
            "ref_std": float(reference_data[feature].std()),
            "curr_std": float(current_data[feature].std())    
        } for feature in numerical_features
    }
}      
with open('validation_outputs.json', 'w') as f:
    json.dump(validation_outputs, f, indent=4)



def get_formatted_timestamp():
    """Get current UTC timestamp in a consistent format"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")

try:
    print("Backing up current reference data...")
    timestamp = get_formatted_timestamp()
    backup_data = {
        "timestamp": timestamp,
        "data": clean_data_for_mongodb(reference_data.to_dict(orient='records')),
        "drift_score": float(drift_score),
        "feature_drifts": {k: float(v) for k, v in feature_drifts.items()}
    }
    historical_data.insert_one(backup_data)
    print(f"Backed up reference data to MongoDB Atlas with timestamp {timestamp}")
except Exception as e:
    print(f"Error during backup process: {e}")


try:
    current_data.to_csv("data/reference_data.csv", index=False)
    print("Successfully updated reference data file.")
except Exception as e:
    print(f"Error updating reference data file: {e}")

try:
    print("\nRecent versions in MongoDB Atlas:")
    for record in historical_data.find().sort("timestamp", -1).limit(5):
        timestamp = record['timestamp']
        drift = record['drift_score']
        print(f"- Version from {timestamp} (drift: {drift:.2%})")
except Exception as e:
    print(f"Error fetching version history: {e}")


if needs_retraining:
    print("Significant drift detected. Initiating model retraining process...")

results = {
    "drift_score": float(drift_score),
    "needs_retraining": bool(needs_retraining),
    "timestamp": datetime.now(timezone.utc).isoformat()
}

with open('drift_report.json', 'w') as f:
    json.dump(results, f, indent=4)
