
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('diabetes_multiclass_model.pkl')
    scaler = joblib.load('diabetes_scaler.pkl')
    return model, scaler

# Feature engineering function
def create_features(df):
    df_feat = df.copy()
    
    # Log transformations
    df_feat['Urea_log'] = np.log1p(df['Urea'])
    df_feat['Cr_log'] = np.log1p(df['Cr'])
    df_feat['VLDL_log'] = np.log1p(df['VLDL'])
    df_feat['TG_log'] = np.log1p(df['TG'])
    
    # Risk indicators
    df_feat['HbA1c_Risk'] = (df['HbA1c'] >= 6.5).astype(int)
    df_feat['BMI_Risk'] = (df['BMI'] >= 30).astype(int)
    df_feat['Age_Risk'] = (df['AGE'] >= 45).astype(int)
    df_feat['TG_Risk'] = (df['TG'] > 150).astype(int)
    df_feat['Chol_Risk'] = (df['Chol'] > 200).astype(int)
    
    # Composite risk score
    df_feat['Total_Risk_Score'] = (df_feat['HbA1c_Risk'] + df_feat['BMI_Risk'] + 
                                   df_feat['Age_Risk'] + df_feat['TG_Risk'] + 
                                   df_feat['Chol_Risk'])
    
    # Ratios
    df_feat['Chol_HDL_Ratio'] = df['Chol'] / (df['HDL'] + 1e-5)
    df_feat['LDL_HDL_Ratio'] = df['LDL'] / (df['HDL'] + 1e-5)
    df_feat['TG_HDL_Ratio'] = df['TG'] / (df['HDL'] + 1e-5)
    
    return df_feat

# Prediction function
def predict_diabetes_risk(patient_data, model, scaler):
    # Feature columns
    feature_columns = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'VLDL', 'BMI',
                      'Urea_log', 'Cr_log', 'VLDL_log', 'TG_log',
                      'HbA1c_Risk', 'BMI_Risk', 'Age_Risk', 'TG_Risk', 'Chol_Risk',
                      'Total_Risk_Score', 'Chol_HDL_Ratio', 'LDL_HDL_Ratio', 'TG_HDL_Ratio']
    
    # Create dataframe
    patient_df = pd.DataFrame([patient_data])
    
    # Engineer features
    patient_enhanced = create_features(patient_df)
    
    # Select features
    patient_features = patient_enhanced[feature_columns]
    
    # Scale
    patient_scaled = scaler.transform(patient_features)
    
    # Predict
    prediction = model.predict(patient_scaled)[0]
    probabilities = model.predict_proba(patient_scaled)[0]
    
    class_labels = ['Non-Diabetic', 'Diabetic', 'Pre-Diabetic']
    
    return {
        'prediction': class_labels[prediction],
        'prediction_code': prediction,
        'probabilities': {class_labels[i]: prob for i, prob in enumerate(probabilities)},
        'confidence': max(probabilities)
    }

# Main app
def main():
    # Load model
    model, scaler = load_model()
    
    # Header
    st.title("🏥 Diabetes Risk Prediction System")
    st.markdown("### AI-powered diabetes risk assessment based on clinical parameters")
    
    # Sidebar for input
    with st.sidebar:
        st.header("Patient Information")
        st.markdown("Enter the patient's clinical parameters below:")
        
        # Demographics
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_code = 0 if gender == "Female" else 1
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=45)
        
        # Clinical Parameters
        st.subheader("Clinical Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=5.5, step=0.1)
            urea = st.number_input("Urea (mg/dL)", min_value=5.0, max_value=50.0, value=15.0)
            cr = st.number_input("Creatinine (mg/dL)", min_value=0.5, max_value=3.0, value=0.9, step=0.1)
            bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        
        with col2:
            chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
            tg = st.number_input("Triglycerides (mg/dL)", min_value=30, max_value=500, value=120)
            hdl = st.number_input("HDL (mg/dL)", min_value=20, max_value=100, value=50)
            ldl = st.number_input("LDL (mg/dL)", min_value=50, max_value=300, value=100)
        
        vldl = st.number_input("VLDL (mg/dL)", min_value=5, max_value=100, value=25)
        
        predict_button = st.button("🔍 Predict Risk", use_container_width=True)
    
    # Main content area
    if predict_button:
        # Prepare patient data
        patient_data = {
            'Gender': gender_code,
            'AGE': age,
            'Urea': urea,
            'Cr': cr,
            'HbA1c': hba1c,
            'Chol': chol,
            'TG': tg,
            'HDL': hdl,
            'LDL': ldl,
            'VLDL': vldl,
            'BMI': bmi
        }
        
        # Make prediction
        result = predict_diabetes_risk(patient_data, model, scaler)
        
        # Display results
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("Prediction Results")
            
            # Color code based on prediction
            colors = {
                'Non-Diabetic': 'green',
                'Pre-Diabetic': 'orange',
                'Diabetic': 'red'
            }
            
            st.markdown(f"""
                <div style="background-color: {colors[result['prediction']]}20; 
                            padding: 20px; border-radius: 10px; border: 2px solid {colors[result['prediction']]}">
                    <h2 style="color: {colors[result['prediction']]}; margin: 0;">
                        {result['prediction']}
                    </h2>
                    <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                        Confidence: {result['confidence']:.1%}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Probability gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['probabilities'][result['prediction']] * 100,
                title = {'text': "Prediction Confidence"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': colors[result['prediction']]},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=200)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            # Risk factors summary
            st.metric("Total Risk Factors", 
                     sum([hba1c >= 6.5, bmi >= 30, age >= 45, tg > 150, chol > 200]))
        
        # Probability breakdown
        st.subheader("Risk Probability Breakdown")
        prob_df = pd.DataFrame([result['probabilities']]).T.reset_index()
        prob_df.columns = ['Condition', 'Probability']
        prob_df['Probability'] = prob_df['Probability'] * 100
        
        fig_bar = px.bar(prob_df, x='Condition', y='Probability', 
                        color='Condition',
                        color_discrete_map={
                            'Non-Diabetic': 'green',
                            'Pre-Diabetic': 'orange',
                            'Diabetic': 'red'
                        },
                        title="Probability Distribution (%)")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Clinical parameter analysis
        with st.expander("📊 Clinical Parameter Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Indicators")
                risk_data = {
                    'Parameter': ['HbA1c', 'BMI', 'Age', 'Triglycerides', 'Cholesterol'],
                    'Status': [
                        '⚠️ High' if hba1c >= 6.5 else '✅ Normal',
                        '⚠️ High' if bmi >= 30 else '✅ Normal',
                        '⚠️ Risk Age' if age >= 45 else '✅ Normal',
                        '⚠️ High' if tg > 150 else '✅ Normal',
                        '⚠️ High' if chol > 200 else '✅ Normal'
                    ],
                    'Value': [f"{hba1c}%", f"{bmi:.1f}", f"{age} years", f"{tg} mg/dL", f"{chol} mg/dL"]
                }
                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, hide_index=True)
            
            with col2:
                st.subheader("Recommendations")
                if result['prediction'] == 'Diabetic':
                    st.warning("⚠️ **High Risk Detected**")
                    st.markdown("""
                    - Consult with an endocrinologist immediately
                    - Start monitoring blood glucose regularly
                    - Consider lifestyle modifications
                    - Review medication options with healthcare provider
                    """)
                elif result['prediction'] == 'Pre-Diabetic':
                    st.info("ℹ️ **Moderate Risk Detected**")
                    st.markdown("""
                    - Schedule regular check-ups (every 3-6 months)
                    - Implement dietary changes
                    - Increase physical activity
                    - Monitor weight and BMI
                    """)
                else:
                    st.success("✅ **Low Risk**")
                    st.markdown("""
                    - Maintain healthy lifestyle
                    - Annual health check-ups recommended
                    - Continue balanced diet and exercise
                    - Monitor any changes in health
                    """)
        
        # Generate report
        with st.expander("📄 Generate Report"):
            report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            report = f"""
# Diabetes Risk Assessment Report

**Date:** {report_date}

## Patient Information
- **Age:** {age} years
- **Gender:** {gender}
- **BMI:** {bmi:.1f} kg/m²

## Clinical Parameters
- **HbA1c:** {hba1c}%
- **Fasting Glucose:** Not provided
- **Total Cholesterol:** {chol} mg/dL
- **Triglycerides:** {tg} mg/dL
- **HDL:** {hdl} mg/dL
- **LDL:** {ldl} mg/dL

## Risk Assessment
- **Prediction:** {result['prediction']}
- **Confidence:** {result['confidence']:.1%}

## Risk Probabilities
- Non-Diabetic: {result['probabilities']['Non-Diabetic']:.1%}
- Pre-Diabetic: {result['probabilities']['Pre-Diabetic']:.1%}
- Diabetic: {result['probabilities']['Diabetic']:.1%}

---
*This is an AI-generated assessment. Please consult with healthcare professionals for medical advice.*
            """
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"diabetes_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
    
    else:
        # Welcome screen
        st.info("👈 Please enter patient information in the sidebar and click 'Predict Risk' to start the assessment.")
        
        # Display information about the model
        with st.expander("ℹ️ About This Model"):
            st.markdown("""
            This diabetes risk prediction system uses machine learning to assess the likelihood of diabetes based on clinical parameters.
            
            ### Key Features Analyzed:
            - **HbA1c**: Glycated hemoglobin (diabetes indicator)
            - **BMI**: Body Mass Index (obesity risk factor)
            - **Age**: Risk increases with age, especially after 45
            - **Lipid Profile**: Cholesterol, Triglycerides, HDL, LDL
            - **Kidney Function**: Urea and Creatinine levels
            
            ### Model Performance:
            - Accuracy: >90% on test data
            - Validated on clinical datasets
            - Multi-class classification: Non-Diabetic, Pre-Diabetic, Diabetic
            
            ### Important Note:
            This tool is for screening purposes only and should not replace professional medical consultation.
            """)

if __name__ == "__main__":
    main()
