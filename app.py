import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

st.title("My Machine Learning Prediction App")
st.write("Enter the input values to get a prediction.")

# Example input fields
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)

if st.button("Predict"):
    # Keep feature names and order consistent with training
    input_data = pd.DataFrame(
        [[feature1, feature2]],
        columns=["feature1", "feature2"]
    )

    prediction = model.predict(input_data)

    st.success(f"Prediction: {prediction[0]}")