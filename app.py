import streamlit as st
import pandas as pd
import joblib

# --- Configuration ---
st.set_page_config(page_title="Mental Health Prediction", page_icon="🧠", layout="centered")

# --- Asset Loading ---
try:
    # Load BOTH the model and the columns list
    model = joblib.load('mental_health_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model or column file not found. Please run the `project.ipynb` notebook first.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()


# --- UI Setup ---
st.title("🧠 Mental Health Treatment Prediction")
st.write(
    "This app predicts the likelihood of an individual seeking mental health treatment. "
    "Please answer the questions below."
)

# --- Input Widgets ---
col1, col2 = st.columns(2)
# Using the same full UI as before
with col1:
    gender = st.selectbox('Gender', ('Male', 'Female', 'Other'), key='gender')
    country = st.selectbox('Country', ('United States', 'United Kingdom', 'Canada', 'Australia', 'Other'), key='country')
    occupation = st.selectbox('Occupation', ('Corporate', 'Business', 'Student', 'Housewife', 'Others'), key='occupation')
    self_employed = st.selectbox('Are you self-employed?', ('No', 'Yes'), key='self_employed')
    family_history = st.selectbox('Do you have a family history of mental illness?', ('No', 'Yes'), key='family_history')
    days_indoors = st.selectbox('How much time do you spend indoors?', ('1-14 days', '15-30 days', '31-60 days', 'More than 60 days', 'Go out Every day'), key='days_indoors')
    growing_stress = st.selectbox('Are you experiencing growing stress?', ('Yes', 'No', 'Maybe'), key='growing_stress')
    changes_habits = st.selectbox('Have you experienced changes in your habits?', ('Yes', 'No', 'Maybe'), key='changes_habits')

with col2:
    mental_health_history = st.selectbox('Do you have a history of mental health conditions?', ('No', 'Yes', 'Maybe'), key='mental_health_history')
    mood_swings = st.selectbox('Do you experience mood swings?', ('Low', 'Medium', 'High'), key='mood_swings')
    coping_struggles = st.selectbox('Are you struggling to cope with day-to-day life?', ('No', 'Yes'), key='coping_struggles')
    work_interest = st.selectbox('Have you lost interest in your work?', ('No', 'Yes', 'Maybe'), key='work_interest')
    social_weakness = st.selectbox('Do you feel socially weak or isolated?', ('No', 'Yes', 'Maybe'), key='social_weakness')
    mental_health_interview = st.selectbox('Would you be willing to discuss your mental health with an interviewer?', ('No', 'Yes', 'Maybe'), key='mental_health_interview')
    care_options = st.selectbox('Are you aware of care options for mental health?', ('Yes', 'No', 'Not sure'), key='care_options')


# --- Prediction Logic ---
if st.button('Predict Likelihood', type="primary"):
    # Create a DataFrame from the user's inputs
    input_data = {
        'Gender': [gender], 'Country': [country], 'Occupation': [occupation],
        'self_employed': [self_employed], 'family_history': [family_history],
        'Days_Indoors': [days_indoors], 'Growing_Stress': [growing_stress],
        'Changes_Habits': [changes_habits], 'Mental_Health_History': [mental_health_history],
        'Mood_Swings': [mood_swings], 'Coping_Struggles': [coping_struggles],
        'Work_Interest': [work_interest], 'Social_Weakness': [social_weakness],
        'mental_health_interview': [mental_health_interview], 'care_options': [care_options]
    }
    input_df = pd.DataFrame(input_data)

    # 1. Manually one-hot encode the user's input
    input_encoded = pd.get_dummies(input_df)

    # 2. **CRITICAL STEP**: Align the columns of the input with the model's columns
    # All columns the model was trained on will be present.
    # Missing columns in the user's input will be added and filled with 0.
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict using the perfectly aligned data
    prediction = model.predict(final_input)[0]
    prediction_proba = model.predict_proba(final_input)[0]

    # --- Display Results ---
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"This individual is LIKELY to seek treatment.", icon="✅")
        confidence = prediction_proba[1] * 100
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        st.progress(int(confidence))
    else:
        st.warning(f"This individual is NOT LIKELY to seek treatment.", icon="❌")
        confidence = prediction_proba[0] * 100
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        st.progress(int(confidence))