import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction + Explanation System")
st.write("This AI predicts survival probability and explains the top 3 reasons.")

st.sidebar.header("Enter Passenger Details")

# User inputs
pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 80, 25)
fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=500.0, value=50.0)
family_size = st.sidebar.slider("Family Size", 1, 10, 1)

embarked = st.sidebar.selectbox("Embarked", ["S", "Q", "C"])

# Encoding input (must match cleaned dataset columns)
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Input format must match training dataset feature order
input_data = np.array([[pclass, age, fare, family_size, sex_male, embarked_Q, embarked_S]])

if st.button("Predict Survival"):
    # Survival probability
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Survival Probability")
    st.write(f"**{prob:.2f}**")

    # Priority Category
    if prob < 0.4:
        priority = "High Priority 🚨"
    elif prob < 0.7:
        priority = "Medium Priority ⚠️"
    else:
        priority = "Low Priority ✅"

    st.subheader("🚑 Rescue Priority Category")
    st.write(priority)

    # Feature importance (top 3 reasons)
    feature_names = ["Pclass", "Age", "Fare", "FamilySize", "Sex_male", "Embarked_Q", "Embarked_S"]
    importances = model.feature_importances_

    # Sort top 3
    top_indices = np.argsort(importances)[::-1][:3]

    st.subheader("🧠 Top 3 Reasons (Most Important Features)")
    for idx in top_indices:
        st.write(f"✅ **{feature_names[idx]}** (importance = {importances[idx]:.3f})")
