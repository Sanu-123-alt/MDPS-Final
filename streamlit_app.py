import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib
import os

def load_model_safe(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {str(e)}")
        return None

# Load the models and scalers (using direct paths)
try:
    models_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load models and scalers from the same directory as the script
    model_files = {
        'diabetes': ('diabetespred_model.sav', 'diabetes_scaler.sav'),
        'heart': ('heartdisease_model.sav', 'heart_scaler.sav'),
        'parkinsons': ('parkinsons_model.sav', 'parkinsons_scaler.sav'),
        'lung_cancer': ('lung_cancer_model.joblib', 'lung_cancer_scaler.joblib')
    }
    
    # Load all models
    diabetes_model = load_model_safe(os.path.join(models_dir, model_files['diabetes'][0]))
    heart_disease_model = load_model_safe(os.path.join(models_dir, model_files['heart'][0]))
    parkinsons_model = load_model_safe(os.path.join(models_dir, model_files['parkinsons'][0]))
    lung_cancer_model = load_model_safe(os.path.join(models_dir, model_files['lung_cancer'][0]))
    
    # Load scalers
    diabetes_scaler = joblib.load(os.path.join(models_dir, model_files['diabetes'][1]))
    heart_scaler = joblib.load(os.path.join(models_dir, model_files['heart'][1]))
    parkinsons_scaler = joblib.load(os.path.join(models_dir, model_files['parkinsons'][1]))
    lung_cancer_scaler = joblib.load(os.path.join(models_dir, model_files['lung_cancer'][1]))

    if not all([diabetes_model, heart_disease_model, parkinsons_model, lung_cancer_model,
                diabetes_scaler, heart_scaler, parkinsons_scaler, lung_cancer_scaler]):
        st.error("Failed to load one or more models/scalers")
        st.stop()

except Exception as e:
    st.error(f"Initialization error: {str(e)}")
    st.stop()

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Lung Cancer Prediction'],
                          icons=['activity', 'heart', 'person', 'lungs'],
                          default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # page title
    st.title('Diabetes Prediction using ML')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
        
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
        
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
        
    with col2:
        Insulin = st.text_input('Insulin Level')
        
    with col3:
        BMI = st.text_input('BMI value')
        
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
    with col2:
        Age = st.text_input('Age of the Person')
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert input to float
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), 
                         float(SkinThickness), float(Insulin), float(BMI), 
                         float(DiabetesPedigreeFunction), float(Age)]
            
            # Scale the features
            features_scaled = diabetes_scaler.transform([user_input])
            diab_prediction = diabetes_model.predict(features_scaled)
            
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
            
            st.success(diab_diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values for all fields")

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])
        
    with col3:
        cp = st.selectbox('Chest Pain types', 
                         ['Typical Angina', 'Atypical Angina', 
                          'Non-anginal Pain', 'Asymptomatic'])
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
        
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results',
                              ['Normal', 'ST-T Wave Abnormality', 
                               'Left Ventricular Hypertrophy'])
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment',
                           ['Upsloping', 'Flat', 'Downsloping'])
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.selectbox('Thalassemia',
                           ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            # Convert categorical variables
            sex = 1 if sex == 'Male' else 0
            cp_dict = {'Typical Angina': 0, 'Atypical Angina': 1, 
                      'Non-anginal Pain': 2, 'Asymptomatic': 3}
            cp = cp_dict[cp]
            fbs = 1 if fbs == 'Yes' else 0
            restecg_dict = {'Normal': 0, 'ST-T Wave Abnormality': 1, 
                           'Left Ventricular Hypertrophy': 2}
            restecg = restecg_dict[restecg]
            exang = 1 if exang == 'Yes' else 0
            slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
            slope = slope_dict[slope]
            thal_dict = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}
            thal = thal_dict[thal]
            
            # Convert input to float
            features = [float(age), sex, cp, float(trestbps), float(chol), 
                       fbs, restecg, float(thalach), exang, float(oldpeak), 
                       slope, float(ca), thal]
            
            # Scale the features
            features_scaled = heart_scaler.transform([features])
            heart_prediction = heart_disease_model.predict(features_scaled)
            
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
            
            st.success(heart_diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values for all fields")

# Parkinsons Prediction Page
elif selected == 'Parkinsons Prediction':
    # page title
    st.title('Parkinsons Prediction using ML')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        try:
            # Convert input to float
            features = [float(fo), float(fhi), float(flo), float(Jitter_percent),
                       float(Jitter_Abs), float(RAP), float(PPQ), float(DDP),
                       float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                       float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE),
                       float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
            
            # Scale the features
            features_scaled = parkinsons_scaler.transform([features])
            parkinsons_prediction = parkinsons_model.predict(features_scaled)
            
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
            
            st.success(parkinsons_diagnosis)
        except ValueError:
            st.error("Please enter valid numerical values for all fields")

# Lung Cancer Prediction Page
elif selected == "Lung Cancer Prediction":
    # page title
    st.title('Lung Cancer Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=0)
    with col2:
        smoking = st.number_input("Smoking (0-2)", min_value=0, max_value=2, value=0, step=1)
    with col3:
        yellow_fingers = st.number_input("Yellow Fingers (0-2)", min_value=0, max_value=2, value=0, step=1)
    
    with col1:
        anxiety = st.number_input("Anxiety (0-2)", min_value=0, max_value=2, value=0, step=1)
    with col2:
        peer_pressure = st.number_input("Peer Pressure (0-2)", min_value=0, max_value=2, value=0, step=1)
    with col3:
        chronic_disease = st.number_input("Chronic Disease (0-2)", min_value=0, max_value=2, value=0, step=1)
    
    with col1:
        fatigue = st.number_input("Fatigue (0-2)", min_value=0, max_value=2, value=0, step=1)
    with col2:
        allergy = st.number_input("Allergy (0-2)", min_value=0, max_value=2, value=0, step=1)
    with col3:
        wheezing = st.number_input("Wheezing (0-2)", min_value=0, max_value=2, value=0, step=1)
    
    with col1:
        alcohol = st.number_input("Alcohol Consuming (0-2)", min_value=0, max_value=2, value=0, step=1)
    with col2:
        coughing = st.number_input("Coughing (0-2)", min_value=0, max_value=2, value=0, step=1)
    with col3:
        shortness_of_breath = st.number_input("Shortness of Breath (0-2)", min_value=0, max_value=2, value=0, step=1)
    
    with col1:
        swallowing_difficulty = st.number_input("Swallowing Difficulty (0-2)", min_value=0, max_value=2, value=0, step=1)
    with col2:
        chest_pain = st.number_input("Chest Pain (0-2)", min_value=0, max_value=2, value=0, step=1)
    with col3:
        gender = st.number_input("Gender (0-1)", min_value=0, max_value=1, value=0, step=1,
                               help="0: Female, 1: Male")
    

    
    # code for Prediction
    lung_cancer_diagnosis = ''
    
    # creating a button for Prediction
    if st.button("Lung Cancer Test Result"):
        try:
            # Create features array directly from numerical inputs
            features = [age, smoking, yellow_fingers, anxiety, peer_pressure, 
                       chronic_disease, fatigue, allergy, wheezing, alcohol,
                       coughing, shortness_of_breath, swallowing_difficulty, 
                       chest_pain, gender]  # Added gender as the 15th feature
            
            # Scale features and make prediction
            features_scaled = lung_cancer_scaler.transform([features])
            prediction = lung_cancer_model.predict(features_scaled)
            probability = lung_cancer_model.predict_proba(features_scaled)[0][1]
            
            if prediction[0] == 1:
                lung_cancer_diagnosis = "High risk of lung cancer detected"
                st.error(lung_cancer_diagnosis)
            else:
                lung_cancer_diagnosis = "Low risk of lung cancer"
                st.success(lung_cancer_diagnosis)
            
            st.write(f"Probability of lung cancer: {probability:.2%}")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please ensure all fields are filled correctly")
