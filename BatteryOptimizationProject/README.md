# Battery Design Optimization Assistant

## Overview
This project is a prototype AI assistant that predicts battery degradation and provides engineer-ready recommendations using ML + LLMs.  
It demonstrates an end-to-end workflow from data → model → predictions → natural language explanations, focused on battery performance in real driving cycles.

## Dataset
- Source: Kaggle – **Battery and Heating Data in Real Driving Cycles**  
- Columns used for this project:  
  - `Battery Temperature (Start) [°C]`  
  - `Battery Temperature (End)`  
  - `Battery State of Charge (Start)`  
  - `Battery State of Charge (End)`  
  - `Battery Consumed (%)`  
  - `Ambient Temperature (Start) [°C]`  
  - `Target Cabin Temperature`  
  - `Distance [km]`  
  - `Duration [min]`  

These features were used to predict battery performance and degradation.

## Model
- **RandomForestRegressor** for predicting battery degradation  
- ML model output is explained by an **LLM** (OpenAI GPT) to generate engineer-ready recommendations

## Demo

   Enter battery parameters in the Streamlit app
   Get predicted degradation and LLM explanations


Author
Muzhgan Rustaqi



