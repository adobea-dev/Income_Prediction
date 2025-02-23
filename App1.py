import streamlit as st
import joblib
import numpy as np
from PIL import Image
st.set_page_config(page_title="Income Predictor App", layout ="wide")
def load_model():
    return joblib.load('models/income_predictor.joblib')
def predict_income(data, model_params):
    features = np.array([data[col] for col in model_params['numerical_columns']])
    z = np.dot(features, model_params['coefficients']) + model_params['intercept']
    probability = 1 / (1 + np.exp(-z))
    
    return probability > 0.5
def main():
    st.title("Income Predictor App")
    st.subheader("Predict your income level based on demographic and employment details.")
    st.write("Fill out the form below to see your predicted income group.")
    image = Image.open("Income Predictor.png")
    st.image(image, caption = "Welcome to the Income Predictor App", use_container_width = True)
    
    model_params = load_model()
    
    with st.form("prediction_form", clear_on_submit=True):
        st.write("### Personal Information")
        age = st.slider("Age", min_value=18, max_value=100, value=25, step=1)
        gender = st.radio("Gender", options=["Male", "Female"])
        relationship = st.selectbox("Relationship", options=["Not-in-family", "Husband", "Own-child", "Unmarried", "Wife"])
        race = st.selectbox("Race", options=["White", "Black", "Asian-pac-islander", "Amer-indian-eskimo", "Other"])
        native_country = st.selectbox("Native-country", options=["Cambodia","Canada", "China", "Columbia", "Cuba", "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece", "Guatemala", "Haiti", "Holand-Nertherlands", "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Puerto-Rico", "Poland", "Portugal", "Scotland", "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"])
        marital_status = st.selectbox("marital-status", options=["Divorced", "Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent", "Never-married", "Separated", "Widowed"])
        st.write("### Employment Details")
        workclass = st.selectbox("Workclass", options=["Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"])
        occupation = st.selectbox("Occupation", options=["Machine-op-inspct", "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Other-service","Priv-house-serv","Protective-serv","Prof-specialty", "Sales", "Tech-support", "Transport-moving"])
        hours_per_week = st.slider("Hours per Week", min_value=1, max_value=80, value=40)
        st.write("### Education and Financial Metrics")
        education = st.selectbox("Education Level", options=["Bachelors", "HS-grad", "11th", "Assoc-acdm", "Some-college", "10th", "Prof-school", "7th-8th", "Masters", "Doctorate", "Assoc-voc", "9th", "12th", "5th-6th", "1st-4th"])
        education_num = st.slider("Years of Education", min_value = 1, max_value=20, value=12)
        final_weight = st.slider("Final weight", min_value = 13492, max_value = 1490400, value = 0)
        capital_gain = st.slider("Capital gain", min_value = 0, max_value = 99999, value = 0)
        capital_loss = st.slider("Capital loss", min_value = 0, max_value = 99999, value = 0)
        # Submit button
        submitted = st.form_submit_button("Predict")
    
        if submitted:
            input_data = {
                'Age': age,
                'gender': gender,
                'education-num': education,
                'workclass': workclass,
                'occupation': occupation,
                'hours_per_week': hours_per_week,
                'education': education_num,
                'fnlwgt' : final_weight,
                'capital-gain' : capital_gain,
                'relationship' : relationship,
                'race' : race
            }        
            prediction = predict_income(input_data, model_params)
            result = ">50K" if prediction else "≤50K"
            st.write(f"Predicted income: {result}")
if __name__ == "__main__":
    main()