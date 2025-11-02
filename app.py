import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    return model

model = load_model()

# Title and Team Information
st.title("ML-MINIPROJECT-HEART DISEASE PREDICTION")
st.subheader("by ShyamSunder, Surya, Narayana Reddy")
st.markdown("---")

st.markdown("""
This AI-powered tool predicts heart disease risk based on clinical parameters.
**Disclaimer**: This is for educational purposes only. Consult a healthcare professional for medical advice.
""")

# Sidebar for input
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", 
                           ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.sidebar.selectbox("Resting ECG", 
                                ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
thalach = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST", ["Upsloping", "Flat", "Downsloping"])
ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Prediction button
if st.sidebar.button("Predict Risk", type="primary"):
    # Mapping dictionaries for consistent encoding
    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3
    }
    
    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    
    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    
    thal_map = {
        "Normal": 0,
        "Fixed Defect": 1,
        "Reversible Defect": 2
    }
    
    try:
        # Calculate categorical features
        bp_category = 0 if trestbps < 120 else 1 if trestbps < 140 else 2 if trestbps < 180 else 3
        chol_risk = 0 if chol < 200 else 1 if chol < 240 else 2
        age_group = 0 if age < 40 else 1 if age < 55 else 2 if age < 70 else 3
        
        # Create input data in EXACT order from training
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == "Male" else 0],
            'cp': [cp_map[cp]],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [1 if fbs == "Yes" else 0],
            'restecg': [restecg_map[restecg]],
            'thalach': [thalach],
            'exang': [1 if exang == "Yes" else 0],
            'oldpeak': [oldpeak],
            'slope': [slope_map[slope]],
            'ca': [float(ca)],
            'thal': [thal_map[thal]],
            'heart_rate_age_ratio': [thalach / age],
            'bp_category': [bp_category],
            'chol_risk': [chol_risk],
            'age_group': [age_group],
            'age_chol_interaction': [age * chol],
            'bp_chol_interaction': [trestbps * chol]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")
        
        with col2:
            st.metric("Risk Probability", f"{probability[1]*100:.1f}%")
        
        with col3:
            risk_level = "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.4 else "Low"
            st.metric("Risk Level", risk_level)
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability[1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Heart Disease Risk Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if probability[1] > 0.7 else "orange" if probability[1] > 0.4 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Recommendations")
        if prediction == 1:
            st.warning("""
            **High Risk Detected!** 
            
            **Immediate Actions:**
            - Consult a cardiologist as soon as possible
            - Monitor blood pressure and cholesterol regularly
            - Get a complete cardiac workup
            
            **Lifestyle Changes:**
            - Adopt a heart-healthy diet (DASH or Mediterranean diet)
            - Increase physical activity gradually (consult doctor first)
            - Practice stress management techniques
            - Stop smoking immediately if applicable
            - Ensure 7-9 hours of quality sleep
            - Limit alcohol consumption
            """)
        else:
            st.success("""
            **Low Risk - Great Job!**
            
            **Maintain Your Heart Health:**
            - Regular exercise (150 min/week moderate activity)
            - Continue balanced, nutritious diet
            - Annual health checkups and screenings
            - Avoid smoking and excessive alcohol
            - Maintain good sleep hygiene (7-9 hours)
            - Manage stress through meditation or hobbies
            - Maintain healthy weight (BMI 18.5-24.9)
            """)
        
        # Risk Factor Analysis
        st.subheader("Your Risk Factor Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Clinical Measurements:**")
            measurements = {
                'Age': f"{age} years",
                'Blood Pressure': f"{trestbps} mm Hg {'[High]' if trestbps > 140 else '[Normal]'}",
                'Cholesterol': f"{chol} mg/dl {'[High]' if chol > 240 else '[Normal]'}",
                'Max Heart Rate': f"{thalach} bpm",
                'ST Depression': f"{oldpeak}"
            }
            for key, value in measurements.items():
                st.text(f"{key}: {value}")
        
        with col2:
            st.markdown("**Risk Indicators:**")
            indicators = {
                'Chest Pain Type': cp,
                'Exercise Angina': f"{exang} {'[Warning]' if exang == 'Yes' else '[Normal]'}",
                'Fasting Blood Sugar': f"{fbs} {'[High]' if fbs == 'Yes' else '[Normal]'}",
                'ECG Results': restecg,
                'Major Vessels': f"{ca} {'[Abnormal]' if ca > 0 else '[Normal]'}"
            }
            for key, value in indicators.items():
                st.text(f"{key}: {value}")
        
        # Feature Importance Visualization
        st.subheader("Key Contributing Factors")
        
        risk_factors = {
            'Age Factor': min(age / 100, 1.0),
            'Cholesterol Level': min(chol / 300, 1.0),
            'Blood Pressure': min(trestbps / 180, 1.0),
            'Heart Rate Response': 1 - min(thalach / 220, 1.0),
            'Chest Pain Severity': cp_map[cp] / 3,
            'Exercise Tolerance': 1 if exang == "Yes" else 0
        }
        
        feature_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Risk Score'])
        feature_df = feature_df.sort_values('Risk Score', ascending=False)
        
        st.bar_chart(feature_df.set_index('Factor'))
        
        # Additional Information
        with st.expander("Understanding Your Results"):
            st.markdown("""
            **Risk Probability Interpretation:**
            - **0-40%**: Low risk - Continue healthy habits
            - **40-70%**: Moderate risk - Consider lifestyle modifications
            - **70-100%**: High risk - Seek medical attention
            
            **Important Notes:**
            - This is a screening tool, not a diagnostic tool
            - Results should be discussed with a healthcare provider
            - Multiple factors contribute to heart disease risk
            - Early detection and intervention can significantly reduce risk
            """)
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please ensure all fields are filled correctly and try again.")
        import traceback
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())