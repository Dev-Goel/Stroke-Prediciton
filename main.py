import sklearn


import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def stroke_prediction(input_data):
   
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person does not have a stroke'
    else:
        return 'The person have a stroke'

def main():
    st.title('Heart Stroke Prediction')
    #gender	age	hypertension	heart_disease	ever_married	work_type	Residence_type	avg_glucose_level	bmi	smoking_status
    
    gender = st.radio("Gender",('Male', 'Female', 'Other'))
    age = st.text_input('Age')
    hypertension = st.text_input('Hypertension')
    heart_disease = st.text_input('Heart Disease')
    ever_married = st.text_input('Ever Married')
    work_type = st.text_input('Work Type')
    Residence_type = st.text_input('Residence Type')
    avg_glucose_level = st.text_input('Average Glucose Level')
    bmi = st.text_input('BMI')
    smoking_status = st.text_input('Smoking Status')

    diagnosis = ''

    if st.button('Stroke Test Result'):
        diagnosis = stroke_prediction([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status])

    st.success(diagnosis)

if __name__ == '__main__':
    main()

