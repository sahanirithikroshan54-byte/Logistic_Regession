#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#6. Deployment with Streamlit:
#Create app.py
import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('logestic_model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
st.title('Diabetes Prediction App')
preg=st.number_input('Pregnancies')
glucose=st.number_input('Glucose')
bp=st.number_input('Blood Pressure')
skin_thickness=st.number_input('Skin Thickness')
insulin=st.number_input('Insulin')
bmi=st.number_input('BMI')
dpf=st.number_input('Diabetes Pedigree Function')
age=st.number_input('Age')
if st.button('Predict'):
  data=np.array([[preg,
                  glucose,
                  bp,skin_thickness,insulin,
                  bmi,dpf,age]])
  data=scaler.transform(data)
  prediction=model.predict(data)
  if prediction[0]==1:
    st.error('The person is having diabetes')
  else:
    st.success('The person is not having diabetes')

