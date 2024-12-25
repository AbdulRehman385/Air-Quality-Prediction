# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

import warnings
warnings.filterwarnings('ignore')

st.title('Air Quality Prediction')

# define the function for dataset
def load_data():
  # Creating the slider for the feature inputs
  temperature = st.slider('Temperature', 0.0, 70.0, 0.0)
  himidity = st.slider('Humidity', 10.0, 140.0, 0.0)
  PM_2 = st.slider('PM2.5', -10.0, 300.0, 0.0)
  PM_10 = st.slider('PM10', -10.0, 320.0, 0.0)
  NO2 = st.slider('NO2', -10.0, 70.0, 0.0)
  SO2 = st.slider('SO2', -10.0, 50.0, 0.0)
  CO2 = st.slider('CO2', -10.0, 20.0, 0.0)
  industrial_area = st.slider('Proximity to indistrail Area', 0.0, 50.0, 0.0)
  pop_density = st.slider('Population Density', 100.0, 2000.0, 0.0)


  data = {
      'Temperature' : temperature,
      'Humidity' : himidity,
      'PM2.5' : PM_2,
      'PM10' : PM_10,
      'NO2' : NO2,
      'SO2' : SO2,
      'CO2' : CO2,
      'Proximity to Industrial Area' : industrial_area,
      'Population Density' : pop_density
  }
  features = pd.DataFrame(data, index = [0])

  return features

# Load input Features
df = load_data()

# Noralize the features
std_scaler = StandardScaler()
df_scaled = std_scaler.fit_transform(df)

# Load the model
model = pickle.load(open(r'model.pkl', 'rb'))

if st.button('Predict'):
  pred = model.predict(df_scaled)
  prob = model.predict_proba(df_scaled)

  st.subheader('Prediction')

  if pred == 0:
    st.write('Air Quality is Poor')
  elif pred == 1:
    st.write('Air Quality is Moderate')
  elif pred == 2:
    st.write('Air Quality is Good')
  else:
    st.write('Air Quality is Very Good')

  st.subheader('Probability')

  if pred == 0:
    st.write(f'Probability : {prob[0][0] : .3f}')
  elif pred == 1:
    st.write(f'Probability : {prob[0][1] : .3f}')
  elif pred == 2:
    st.write(f'Probability : {prob[0][2] : .3f}')
  else:
    st.write(f'Probability : {prob[0][3] : .3f}')

