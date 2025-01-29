import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib  # For loading machine learning models

# Setting the page title and layout
st.set_page_config(page_title="Income Predictor App", layout ="wide")

# Load your pre-trained classifier
classifier = joblib.load('models/income_predictor.joblib')

# Title of the app
st.title("Income Predictor App")
st.subheader("Predict your income level based on demographic and employment details.")
st.write("Fill out the form below to see your predicted income group.")

#Add a picture
image = Image.open("Income Predictor.png")
st.image(image, caption = "Welcome to the Income Predictor App", use_container_width = True)

# Create the form
with st.form("prediction_form", clear_on_submit=True):
    st.write("### Personal Information")
    age = st.slider("Age", min_value=18, max_value=100, value=25, step=1)
    gender = st.radio("Gender", options=["Male", "Female"])

    st.write("### Employment Details")
    workclass = st.selectbox("Workclass", options=["Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"])
    occupation = st.selectbox("Occupation", options=["Machine-op-ins", "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "other-service","Priv-house-serv", "Prof-specialty", "Sales", "Tech-support", "Transport-moving"])
    hours_per_week = st.slider("Hours per Week", min_value=1, max_value=80, value=40)

    st.write("### Education and Financial Metrics")
    education = st.selectbox("Education Level", options=["Bachelors", "HS-grad", "11th", "Assoc-acdm", "Some-college", "10th", "Prof-school", "7th-8th", "Masters", "Doctorate", "Assoc-voc", "9th", "12th", "5th-6th", "1st-4th"])
    education_num = st.slider("Years of Education", min_value=1, max_value=20, value=12)


 # Submit button
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame({"age": [age],"gender": [gender], "workclass": [workclass],"occupation": [occupation],"hours_per_week": [hours_per_week],"education": [education],"education_num": [education_num],})

        try:
            # Perform prediction
            prediction = classifier.transform(input_data)

            # Display prediction result
            st.balloons()
            if prediction[0] == 0:
               st.success("Predicted Income: Less than or equal to $50,000", icon="✅")
            else:
                st.success("Predicted Income: Greater than $50,000", icon="✅")

            # Optional: Add a visual component or explanation
            st.write("Your inputs have been successfully processed!")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")