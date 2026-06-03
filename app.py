import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Page config
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")
st.write("Fill in the details below to predict the student's score.")

# Input fields
study_hours = st.slider("📚 Study Hours per Day", 1, 12, 5)
attendance = st.slider("🏫 Attendance (%)", 40, 100, 75)
previous_score = st.slider("📝 Previous Exam Score", 30, 100, 60)
sleep_hours = st.slider("😴 Sleep Hours per Day", 4, 10, 7)

# Predict button
if st.button("🔍 Predict Grade"):
    features = np.array([[study_hours, attendance, previous_score, sleep_hours]])
    prediction = model.predict(features)[0]
    prediction = round(min(max(prediction, 0), 100), 2)

    st.subheader(f"📊 Predicted Score: **{prediction} / 100**")

    if prediction >= 85:
        st.success("🏆 Excellent! Keep it up!")
    elif prediction >= 70:
        st.info("👍 Good performance!")
    elif prediction >= 55:
        st.warning("⚠️ Average. Study a bit more!")
    else:
        st.error("❌ Needs improvement. Focus more!")