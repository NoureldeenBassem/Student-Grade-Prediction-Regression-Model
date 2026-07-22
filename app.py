"""
Streamlit deployment app for the Student Final Grade (G3) Prediction project.

Loads the single saved pipeline (preprocessing + best regression model) and
predicts a student's final grade (G3, 0-20) from their profile.

Run with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "student_g3_pipeline.joblib"

st.set_page_config(page_title="Student Final Grade Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

st.title("🎓 Student Final Grade (G3) Predictor")
st.write(
    "This app predicts a student's **final grade (G3, on a 0-20 scale)** "
    "from demographic, family, lifestyle and academic information, using a "
    "model trained on the UCI Student Alcohol Consumption (Math course) dataset."
)

st.divider()

with st.form("student_form"):
    st.subheader("Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        school = st.selectbox("School", ["GP", "MS"], help="GP = Gabriel Pereira, MS = Mousinho da Silveira")
        sex = st.selectbox("Sex", ["F", "M"])
    with col2:
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home address type", ["U", "R"], help="U = Urban, R = Rural")
    with col3:
        famsize = st.selectbox("Family size", ["LE3", "GT3"], help="LE3 = <=3 members, GT3 = >3 members")
        parent_status = st.selectbox("Parents' cohabitation status", ["T", "A"], help="T = living together, A = apart")

    st.subheader("Family Background")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mother_edu = st.slider("Mother's education", 0, 4, 2, help="0 = none, 4 = higher education")
    with col2:
        father_edu = st.slider("Father's education", 0, 4, 2, help="0 = none, 4 = higher education")
    with col3:
        mother_job = st.selectbox("Mother's job", ["at_home", "health", "other", "services", "teacher"])
    with col4:
        father_job = st.selectbox("Father's job", ["at_home", "health", "other", "services", "teacher"])

    col1, col2 = st.columns(2)
    with col1:
        reason = st.selectbox("Reason to choose this school", ["course", "home", "reputation", "other"])
    with col2:
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])

    st.subheader("School-Related")
    col1, col2, col3 = st.columns(3)
    with col1:
        traveltime = st.slider("Home-to-school travel time", 1, 4, 1, help="1 = <15min, 4 = >1hour")
        weekly_studytime = st.slider("Weekly study time", 1, 4, 2, help="1 = <2h, 4 = >10h")
    with col2:
        failures = st.slider("Past class failures", 0, 3, 0)
        absences = st.slider("Number of school absences", 0, 93, 4)
    with col3:
        g1 = st.slider("1st period grade (G1)", 0, 20, 10)
        g2 = st.slider("2nd period grade (G2)", 0, 20, 10)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        extra_edu_supp = st.selectbox("Extra school educational support", ["yes", "no"])
    with col2:
        family_edu_supp = st.selectbox("Family educational support", ["yes", "no"])
    with col3:
        extra_paid_class = st.selectbox("Extra paid classes", ["yes", "no"])
    with col4:
        extra_curr_activities = st.selectbox("Extra-curricular activities", ["yes", "no"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        nursery = st.selectbox("Attended nursery school", ["yes", "no"])
    with col2:
        higher = st.selectbox("Wants higher education", ["yes", "no"])
    with col3:
        internet = st.selectbox("Internet access at home", ["yes", "no"])
    with col4:
        romantic = st.selectbox("In a romantic relationship", ["yes", "no"])

    st.subheader("Lifestyle & Health")
    col1, col2, col3 = st.columns(3)
    with col1:
        famrel = st.slider("Quality of family relationships", 1, 5, 4)
        freetime = st.slider("Free time after school", 1, 5, 3)
    with col2:
        goout = st.slider("Going out with friends", 1, 5, 3)
        health = st.slider("Health status", 1, 5, 4)
    with col3:
        dalc = st.slider("Workday alcohol consumption", 1, 5, 1)
        walc = st.slider("Weekend alcohol consumption", 1, 5, 1)

    submitted = st.form_submit_button("Predict Final Grade (G3)", use_container_width=True)

if submitted:
    raw_input = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Parrent_status": parent_status,
        "Mother_edu": mother_edu,
        "Father_edu": father_edu,
        "Mother_job": mother_job,
        "Father_job": father_job,
        "reason_to_chose_school": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "weekly_studytime": weekly_studytime,
        "failures": failures,
        "extra_edu_supp": extra_edu_supp,
        "family_edu_supp": family_edu_supp,
        "extra_paid_class": extra_paid_class,
        "extra_curr_activities": extra_curr_activities,
        "nursery": nursery,
        "Interested_in_higher_edu": higher,
        "internet_access": internet,
        "romantic_relationship": romantic,
        "Family_quality_reln": famrel,
        "freetime_after_school": freetime,
        "goout_with_friends": goout,
        "workday_alcohol_consum": dalc,
        "weekend_alcohol_consum": walc,
        "health_status": health,
        "absences": absences,
        "G1": g1,
        "G2": g2,
    }

    # Recreate the same engineered features used during training.
    # These must be computed here because they are built BEFORE the
    # saved pipeline (the pipeline itself only imputes/encodes/scales).
    raw_input["alcohol_total"] = raw_input["workday_alcohol_consum"] + raw_input["weekend_alcohol_consum"]
    raw_input["parent_edu_avg"] = (raw_input["Mother_edu"] + raw_input["Father_edu"]) / 2
    raw_input["study_efficiency"] = raw_input["weekly_studytime"] / (raw_input["failures"] + 1)
    raw_input["grade_trend"] = raw_input["G2"] - raw_input["G1"]
    raw_input["early_grade_avg"] = (raw_input["G1"] + raw_input["G2"]) / 2

    input_df = pd.DataFrame([raw_input])

    prediction = pipeline.predict(input_df)[0]
    prediction = float(np.clip(prediction, 0, 20))

    st.divider()
    st.subheader("Prediction")
    st.metric("Predicted Final Grade (G3)", f"{prediction:.2f} / 20")

    if prediction >= 16:
        st.success("Excellent predicted performance.")
    elif prediction >= 12:
        st.info("Good predicted performance.")
    elif prediction >= 10:
        st.warning("Passing, but there is room to improve.")
    else:
        st.error("At risk — predicted grade is below the passing threshold.")

st.divider()
st.caption(
    "Model: single scikit-learn Pipeline (imputation + encoding + scaling + regressor) "
    "trained on the UCI Student Alcohol Consumption (Math) dataset."
)
