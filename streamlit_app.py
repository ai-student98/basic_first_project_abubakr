import streamlit as st
import pandas as pd

st.title('😁😂 My first website')

st.write('Тут я задеплою модель классификации')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

with st.expander('Data'):
  st.write("X")
  X_raw = df.drop('species', axis=1)
  st.dataframe(X_raw)

  st.write("y")
  y_raw = df.species
  st.dataframe(y_raw)

with st.sidebar:
  st.header("Введите признаки: ")
  island = st.selectbox('Island', ('Torgersen', 'Dream', 'Biscoe'))
  bill_lenght_mm = st.slider('Bill length (mm)', 32.1, 59.6, 44.5)
