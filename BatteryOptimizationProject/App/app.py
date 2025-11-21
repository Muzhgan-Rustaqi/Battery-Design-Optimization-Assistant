import pandas as pd
import joblib
import streamlit as st
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_degradation.joblib")

# Load the model
model = joblib.load(MODEL_PATH)
st.title("Battery Design Optimization Assistant")

# Inputs
battery_temp_start = st.number_input("Battery Temperature (Start) [°C]", value=25.0)
battery_temp_end = st.number_input("Battery Temperature (End) [°C]", value=30.0)
soc_start = st.number_input("Battery State of Charge (Start) [%]", value=80.0)
soc_end = st.number_input("Battery State of Charge (End) [%]", value=60.0)
ambient_temp = st.number_input("Ambient Temperature (Start) [°C]", value=22.0)
distance = st.number_input("Distance [km]", value=10.0)
duration = st.number_input("Duration [min]", value=15.0)
target_cabin_temp = st.number_input("Target Cabin Temperature [°C]", value=22.0)

# Prepare data
X = pd.DataFrame([{
    "Battery Temperature (Start) [°C]": battery_temp_start,
    "Battery Temperature (End)": battery_temp_end,
    "Battery State of Charge (Start)": soc_start,
    "Battery State of Charge (End)": soc_end,
    "Ambient Temperature (Start) [°C]": ambient_temp,
    "Distance [km]": distance,
    "Duration [min]": duration,
    "Target Cabin Temperature": target_cabin_temp,
}])

# Predict
pred = model.predict(X)[0]
st.metric("Predicted Battery Consumed (%)", f"{pred:.3f}%")

# Feature importance plot
st.subheader("Feature Importance")
importances = model.feature_importances_
feature_names = X.columns

fig, ax = plt.subplots()
sns.barplot(x=importances, y=feature_names, ax=ax)
st.pyplot(fig)

# LLM explanation
from openai import OpenAI
client = OpenAI()

prompt = f"""
Battery consumption predicted: {pred:.3f}%.

Explain why this might be happening given:
Battery Temperature Start={battery_temp_start},
Battery Temperature End={battery_temp_end},
SOC Start={soc_start}, End={soc_end},
Ambient Temp={ambient_temp},
Distance={distance},
Duration={duration},
Target Cabin Temp={target_cabin_temp}.
"""

if st.button("Explain"):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    st.write(response.choices[0].message.content)
if st.button("Recommend"):
    prompt = f"""
    Battery predicted degradation: {pred:.3f}%.
    Given the inputs (Battery Temp Start={battery_temp_start}, ...)
    Give 3 prioritized, actionable changes to reduce degradation, including expected impact.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    st.write(response.choices[0].message.content)

if st.button("Scenario Simulation"):
    prompt = f"""
    Battery predicted degradation: {pred:.3f}%.
    Simulate what happens if charge_95F_kW increased by 20%, keeping other params constant.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}]
    )
    st.write(response.choices[0].message.content)





