import pandas as pd
import streamlit as st
from PIL import Image
import keras
from keras import Sequential
from keras.utils import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scal=MinMaxScaler()


#1.loading model
model_heart = keras.models.load_model(r'C:\Users\shazi\OneDrive\Desktop\developer academy\model_NN')

#2. get Data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

#3. set page configuration
st.set_page_config(page_title= 'Heathy Heart Prediction ', layout = 'wide')

st.title('Heart Disease Prediction using ANN')


#5. add image
image = Image.open(r"C:\Users\shazi\OneDrive\Desktop\developer academy\heart.jfif")
st.image(image, use_column_width = True)

#4. Set a subheader
st.subheader('Data Information:')


def preprocess(age,anaemia,creatinine_phophokinase,diabetes,ejection_fration,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking, time):
#5.preprocessing user input
    if sex=="male":
        sex=1 
    else: sex=0
    
    
    if anaemia=="Yes":
        anaemia=1
    elif anaemia=="No":
        anaemia=0
 
    if diabetes=="Yes":
        diabetes=1
    elif diabetes=="No":
        diabetes=0
 
    if high_blood_pressure=="Yes":
        high_blood_pressure=1
    elif high_blood_pressure=="No":
          high_blood_pressure=0
    
    if smoking=="Yes":
        smoking=1
    elif smoking=="No":
        smoking=0

    user_input=[age,anaemia,creatinine_phophokinase,diabetes,ejection_fration,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking, time]
    user_input=np.array(user_input)
    user_input=user_input.reshape(1,-1)
    user_input=scal.fit_transform(user_input)
    prediction = model_heart.predict(user_input)               
    

    return prediction

st.sidebar.header('User Input Features')
#6. get user input by creating boxes in which user can enter data required to make prediction

age = st.sidebar.slider("age", 40, 95, 45)
anaemia = st.sidebar.selectbox("anaemia(1=yes, 0=No)", ["1", "0"])
diabetes = st.sidebar.selectbox("diabetic(1=yes, 0=No)", ["1", "0"])
high_blood_pressure = st.sidebar.selectbox("high_blood_pressure(1=yes , 0=no)", ["1", "0"])
smoking = st.sidebar.selectbox("smoking(1=yes, 0=no)", ["1", "0"])
sex = st.sidebar.selectbox("(1=Male, 0=Female)", ["1", "0"])
creatinine_phosphokinase = st.sidebar.slider("creatinine_phosphokinase", 23, 7861, 33)

ejection_fraction = st.sidebar.slider("ejection_fraction", 14, 80, 20)

platelets = st.sidebar.slider("Splatelets", 2510, 85000, 3000)
serum_creatinine = st.sidebar.slider("serum_creatinine", 0.5, 9.5, 0.9)
serum_sodium = st.sidebar.slider("serum_sodium",  113, 148, 120)


time = st.sidebar.slider("follow up days", 4, 285, 15)
 

#user_input=preprocess
pred = preprocess(age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking, time) 

#7.make prediction
if st.button("Predict"):    
  
#8.print result
  if pred[0] == 0:
    st.write('Warning! You have high risk of getting a heart attack!')
    
  else:
    st.write('You have lower risk of getting a heart disease!')

    


