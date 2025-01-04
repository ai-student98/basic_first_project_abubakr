import streamlit as st

st.title('😁😂 My first website')

st.write('Тут я задеплою модель классификации')

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

with st.expander('Data'):
  st.write("X")
