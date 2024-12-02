import streamlit as st
import pandas as pd
import pickle

# Load the trained model and preprocessor
with open('artifacts/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('artifacts/preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# App Title
st.title("Math Score Predictor")

# Input fields for user data
st.header("Enter Student Details")
gender = st.selectbox("Gender", ["male", "female"])
race_ethnicity = st.selectbox(
    "Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"]
)
parental_education = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ],
)
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation = st.selectbox(
    "Test Preparation Course", ["none", "completed"]
)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)

# Prediction Button
if st.button("Predict Math Score"):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame(
        {
            "gender": [gender],
            "race_ethnicity": [race_ethnicity],
            "parental_level_of_education": [parental_education],
            "lunch": [lunch],
            "test_preparation_course": [test_preparation],
            "reading_score": [reading_score],
            "writing_score": [writing_score],
        }
    )

    # Preprocess and predict
    transformed_data = preprocessor.transform(input_data)
    prediction = model.predict(transformed_data)

    # Display the result
    st.success(f"Predicted Math Score: {prediction[0]:.2f}")
