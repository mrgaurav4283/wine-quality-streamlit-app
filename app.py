# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("wine_quality_model.pkl")

# App title
st.title("🍷 Wine Quality Predictor")

st.markdown("""
Predict whether a wine is **Good or Not Good** based on its chemical properties.  
This app uses a **Random Forest Classifier** trained on UCI Wine Quality data.
""")

st.sidebar.header("Input Wine Properties")

def user_input_features():
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.0)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.5)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.0, 15.0, 2.5)
    chlorides = st.sidebar.slider("Chlorides", 0.01, 0.2, 0.05)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1, 72, 15)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 6, 289, 50)
    density = st.sidebar.slider("Density", 0.9900, 1.0040, 0.996)
    pH = st.sidebar.slider("pH", 2.9, 4.0, 3.3)
    sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.6)
    alcohol = st.sidebar.slider("Alcohol %", 8.0, 15.0, 10.0)

    data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("🔍 Input Parameters")
st.write(input_df)

# Predict
prediction = model.predict(input_df)
proba = model.predict_proba(input_df)

st.subheader("📌 Prediction Result")
st.write("Quality: **{}**".format("Good 🍷" if prediction[0] == 1 else "Not Good ❌"))

st.subheader("📊 Prediction Probability")
st.bar_chart(proba[0])

# Feature importance
st.subheader("📈 Feature Importance")
importances = model.feature_importances_
features = input_df.columns
imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax)
st.pyplot(fig)
