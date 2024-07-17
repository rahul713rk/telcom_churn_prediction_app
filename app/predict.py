import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv('app/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    data['SeniorCitizen'] = data['SeniorCitizen'].map({1:'Yes' , 0 : 'No'})
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    return data

# Train the model and save it
@st.cache_resource
def train_model(data):
    # Select features and target
    X = data.drop(columns=['Churn', 'customerID'])
    y = data['Churn']

    # Create a preprocessing pipeline
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'Partner' , 'SeniorCitizen', 'Dependents', 'PhoneService', 
                            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                            'Contract', 'PaperlessBilling', 'PaymentMethod']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    X = preprocessor.fit_transform(X)
    over = SMOTE(sampling_strategy = 1 , random_state = 42)
    X , y = over.fit_resample(X, y)
    model = LGBMClassifier(verbose=-1, random_state=42)
    model.fit(X , y)

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])

    # Ensure the model directory exists
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    joblib.dump(clf, os.path.join(model_dir, 'lgbm_model.pkl'))

    return clf

# Streamlit Interface
def app():
    st.title("Churn Prediction")

    # Load data
    data = load_data()

    # Train the model (or load if it already exists)
    if not st.session_state.get('model'):
        st.session_state.model = train_model(data)

    # User input for prediction
    input_data = {}

    # Categorize features
    personal_info = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
    billing_info = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    service_info = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    st.header("Personal Information")
    for feature in personal_info:
        input_data[feature] = st.radio(f'**:blue[{feature}]**', data[feature].unique(), horizontal=True)

    st.header("Billing Information")
    for feature in billing_info:
        if feature in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            input_data[feature] = st.slider(f'**:orange[{feature}]**', min_value=float(data[feature].min()), max_value=float(data[feature].max()), value=float(data[feature].median()))
        else:
            input_data[feature] = st.radio(f'**:blue[{feature}]**', data[feature].unique(), horizontal=True)

    st.header("Service Information")
    for feature in service_info:
        input_data[feature] = st.radio(f'**:blue[{feature}]**', data[feature].unique(), horizontal=True)

    input_df = pd.DataFrame([input_data])

    # Predictions
    if st.button('Predict'):
        y_pred = st.session_state.model.predict(input_df)
        prediction = "Churn" if y_pred[0] == 1 else "No Churn"

        if prediction == "Churn":
            st.error("The model predicts: **Churn**")
            st.snow()
        else:
            st.success("The model predicts: **No Churn**")
            st.balloons()
