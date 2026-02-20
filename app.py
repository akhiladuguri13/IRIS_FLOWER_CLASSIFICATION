import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("iris_model.pkl")

st.title("🌸 Iris Flower Classification App")

st.write("Enter the flower measurements below:")

# User inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)

    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Species: {species[prediction[0]]}")