import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------
# TITLE
# -----------------------
st.title("Student Mental Distress Risk Prediction")
st.markdown("Predict whether a student is at risk based on lifestyle and social media usage.")

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("Student_mental_health.csv")

# -----------------------
# PREPROCESSING
# -----------------------

# Encode categorical variables
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# -----------------------
# FEATURE ENGINEERING (same as notebook)
# -----------------------

df["MentalHealthRiskScore"] = (
    df["Depression"] +
    df["Anxiety"] +
    df["PanicAttack"] +
    (df["StressLevel"] >= 4).astype(int)
)

df["Target"] = (df["MentalHealthRiskScore"] >= 2).astype(int)

# -----------------------
# SELECT FEATURES (NO LEAKAGE)
# -----------------------

features = [
    "Age",
    "Gender",
    "CGPA",
    "SocialMediaHoursPerDay",
    "UsageFrequency",
    "MostUsedPlatform",
    "EngagementBehavior"
]

features = [col for col in features if col in df.columns]

X = df[features]
y = df["Target"]

# -----------------------
# TRAIN TEST SPLIT
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------
# SCALING (ONLY NUMERIC)
# -----------------------

scaler = StandardScaler()

scale_cols = ["Age", "CGPA", "SocialMediaHoursPerDay"]
existing_scale_cols = [col for col in scale_cols if col in X_train.columns]

# Convert to float
X_train[existing_scale_cols] = X_train[existing_scale_cols].astype(float)
X_test[existing_scale_cols] = X_test[existing_scale_cols].astype(float)

# Apply scaling
X_train[existing_scale_cols] = scaler.fit_transform(X_train[existing_scale_cols])
X_test[existing_scale_cols] = scaler.transform(X_test[existing_scale_cols])

# -----------------------
# MODEL
# -----------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------
# EVALUATION
# -----------------------

st.subheader("Model Evaluation")

y_pred = model.predict(X_test)

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Feature importance
st.subheader("Feature Importance")

importance = pd.Series(model.feature_importances_, index=X.columns)

fig2, ax2 = plt.subplots()
importance.sort_values().plot(kind="barh", ax=ax2)
plt.title("Feature Importance")
st.pyplot(fig2)

# -----------------------
# USER INPUT
# -----------------------

st.subheader("Make a Prediction")

with st.form("prediction_form"):

    age = st.slider("Age", 16, 30, 20)
    gender = st.selectbox("Gender", list(label_encoders["Gender"].classes_))
    cgpa = st.slider("CGPA", 0.0, 4.0, 3.0)
    sm_hours = st.slider("Social Media Hours Per Day", 0.0, 12.0, 3.0)
    usage = st.selectbox("Usage Frequency", list(label_encoders["UsageFrequency"].classes_))
    platform = st.selectbox("Most Used Platform", list(label_encoders["MostUsedPlatform"].classes_))
    engagement = st.selectbox("Engagement Behavior", list(label_encoders["EngagementBehavior"].classes_))

    submit = st.form_submit_button("Predict")

# -----------------------
# PREDICTION
# -----------------------

if submit:

    input_data = pd.DataFrame([[
        age,
        label_encoders["Gender"].transform([gender])[0],
        cgpa,
        sm_hours,
        label_encoders["UsageFrequency"].transform([usage])[0],
        label_encoders["MostUsedPlatform"].transform([platform])[0],
        label_encoders["EngagementBehavior"].transform([engagement])[0]
    ]], columns=features)

    # Scale numeric
    input_data[existing_scale_cols] = input_data[existing_scale_cols].astype(float)
    input_data[existing_scale_cols] = scaler.transform(input_data[existing_scale_cols])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    result = "At Risk of Mental Distress" if prediction == 1 else "Not At Risk"

    st.success(f"Prediction: {result}")
    st.info(f"Probability: {prob:.2%}")