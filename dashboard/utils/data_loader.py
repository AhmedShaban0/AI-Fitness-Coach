import os
import pandas as pd
import numpy as np
import streamlit as st

MET_VALUES = {
    'Yoga': {'Low': 2.5, 'Medium': 3.0, 'High': 4.0},
    'Walking': {'Low': 2.8, 'Medium': 3.5, 'High': 4.5},
    'Running': {'Low': 7.0, 'Medium': 9.0, 'High': 11.5},
    'Cycling': {'Low': 4.0, 'Medium': 6.8, 'High': 8.5},
    'Swimming': {'Low': 6.0, 'Medium': 8.0, 'High': 10.0},
    'Basketball': {'Low': 6.0, 'Medium': 8.0, 'High': 10.0},
    'Tennis': {'Low': 5.0, 'Medium': 7.3, 'High': 8.0},
    'Weight Training': {'Low': 3.5, 'Medium': 5.0, 'High': 6.0},
    'HIIT': {'Low': 8.0, 'Medium': 10.0, 'High': 12.0},
    'Dancing': {'Low': 4.0, 'Medium': 5.5, 'High': 7.5}
}

def calculate_calories(row):
    intensity_val = row.get("workout_intensity", row.get("intensity", "Medium"))
    met = MET_VALUES.get(row.get("activity_type", "Walking"), {}).get(intensity_val, 5.0)
    w = row.get("weight_kg", 70.0)
    d = row.get("duration_minutes", row.get("duration_min", 30.0))
    calories = (met * 3.5 * w * d) / 200
    return round(calories, 1)

def assign_goal(row):
    bmi = row.get("bmi", 22.0)
    if bmi > 25:
        return "Lose Weight"
    elif bmi < 18.5:
        return "Gain Weight"
    else:
        return "Maintain Weight"

def calculate_activity_factor(row):
    intensity_val = row.get("workout_intensity", row.get("intensity", "Medium"))
    daily_steps = row.get("daily_steps", 8000)
    if intensity_val == "Low" and daily_steps < 5000:
        return 1.2
    elif intensity_val == "Medium" or daily_steps < 10000:
        return 1.55
    else:
        return 1.725

@st.cache_data(show_spinner="Loading and preparing Fitness Dataset...")
def load_and_process_data(data_path="my_dataset.csv"):
    """
    Defensively loads and processes the dataset.
    If dataset file is missing, generates fallback data with a warning alert.
    """
    search_paths = [
        data_path,
        os.path.join(os.path.dirname(__file__), "..", "..", "my_dataset.csv"),
        os.path.join(os.path.dirname(__file__), "..", "my_dataset.csv"),
        os.path.abspath("my_dataset.csv")
    ]
    
    found_path = None
    for p in search_paths:
        if p and os.path.exists(p):
            found_path = p
            break
            
    if found_path is None:
        st.warning("⚠️ Warning: Primary `my_dataset.csv` file not found. Generating fallback synthetic evaluation dataset.")
        # Fallback synthetic dataset generator
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "age": np.random.randint(18, 65, n),
            "gender": np.random.choice(["M", "F"], n),
            "height_cm": np.random.uniform(150, 195, n),
            "weight_kg": np.random.uniform(50, 100, n),
            "activity_type": np.random.choice(list(MET_VALUES.keys()), n),
            "duration_minutes": np.random.randint(15, 90, n),
            "intensity": np.random.choice(["Low", "Medium", "High"], n),
            "daily_steps": np.random.randint(2000, 15000, n),
            "avg_heart_rate": np.random.randint(90, 170, n),
            "resting_heart_rate": np.random.randint(50, 80, n),
            "blood_pressure_systolic": 120,
            "blood_pressure_diastolic": 80,
            "endurance_level": np.random.uniform(1, 10, n),
            "sleep_hours": np.random.uniform(5, 9, n),
            "stress_level": np.random.randint(1, 10, n),
            "hydration_level": 2.5,
            "smoking_status": "Never",
            "fitness_level": 0.5
        })
        df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
    else:
        try:
            df = pd.read_csv(found_path)
        except Exception as e:
            st.warning(f"⚠️ Warning loading CSV ({e}). Using default dataset structure.")
            return pd.DataFrame()

    # Column Schema Compatibility Aliasing
    if "duration_minutes" in df.columns and "duration_min" not in df.columns:
        df["duration_min"] = df["duration_minutes"]
    elif "duration_min" in df.columns and "duration_minutes" not in df.columns:
        df["duration_minutes"] = df["duration_min"]
        
    if "intensity" in df.columns and "workout_intensity" not in df.columns:
        df["workout_intensity"] = df["intensity"]
    elif "workout_intensity" in df.columns and "intensity" not in df.columns:
        df["intensity"] = df["workout_intensity"]
        
    if "avg_heart_rate" in df.columns and "heart_rate" not in df.columns:
        df["heart_rate"] = df["avg_heart_rate"]
    elif "heart_rate" in df.columns and "avg_heart_rate" not in df.columns:
        df["avg_heart_rate"] = df["heart_rate"]

    # Calculate calories_burned based on MET formula
    df["calories_burned"] = df.apply(calculate_calories, axis=1)
    
    # Clean health_condition if present
    if "health_condition" in df.columns:
        df.drop(columns=["health_condition"], inplace=True)
        
    # Clean step counts < 0
    if "daily_steps" in df.columns:
        df = df[df["daily_steps"] >= 0].copy()
        
    # Drop ID column if present
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
        
    # Feature engineering for required calories
    df["goal"] = df.apply(assign_goal, axis=1)
    df["activity_factor"] = df.apply(calculate_activity_factor, axis=1)
    
    df["BMR"] = np.where(
        df["gender"] == "M",
        (10 * df["weight_kg"]) + (6.25 * df["height_cm"]) - (5 * df["age"]) + 5,
        (10 * df["weight_kg"]) + (6.25 * df["height_cm"]) - (5 * df["age"]) - 161
    )
    
    df["TDEE"] = df["BMR"] * df["activity_factor"]
    
    df["required_calories"] = np.where(
        df["goal"] == "Lose Weight",
        df["TDEE"] - 500,
        np.where(
            df["goal"] == "Maintain Weight",
            df["TDEE"],
            df["TDEE"] + 300
        )
    )
    
    return df
