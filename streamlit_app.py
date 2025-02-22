import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.title('🐧 Penguin Species Classification')
st.write('### Исследование и предсказание видов пингвинов')

# Загрузка данных
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

# Визуализация данных
st.sidebar.header("Фильтр данных")
island_filter = st.sidebar.multiselect('Острова', df['island'].unique(), default=df['island'].unique())
gender_filter = st.sidebar.multiselect('Пол', df['sex'].unique(), default=df['sex'].unique())

filtered_df = df[(df['island'].isin(island_filter)) & (df['sex'].isin(gender_filter))]

st.write("### Данные пингвинов после фильтрации")
st.dataframe(filtered_df)

# Графики
st.subheader("📊 Визуализация данных")
fig1 = px.scatter(filtered_df, x='flipper_length_mm', y='body_mass_g', color='species', title='Размер крыла vs Масса тела')
st.plotly_chart(fig1)

fig2 = px.box(filtered_df, x='species', y='bill_length_mm', color='species', title='Длина клюва по видам')
st.plotly_chart(fig2)

# Подготовка данных
X = df.drop(columns=['species'])
y = df['species']
label_encoders = {}

for col in ['island', 'sex']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Ввод пользовательских данных
st.sidebar.header("Введите параметры для предсказания")
island_input = st.sidebar.selectbox('Остров', df['island'].unique())
bill_length_input = st.sidebar.slider('Длина клюва (мм)', float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
bill_depth_input = st.sidebar.slider('Глубина клюва (мм)', float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
flipper_length_input = st.sidebar.slider('Длина крыла (мм)', float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
body_mass_input = st.sidebar.slider('Масса тела (г)', float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))
gender_input = st.sidebar.selectbox('Пол', df['sex'].unique())

# Создание DataFrame с пользовательским вводом
input_data = pd.DataFrame({
    'island': [label_encoders['island'].transform([island_input])[0]],
    'bill_length_mm': [bill_length_input],
    'bill_depth_mm': [bill_depth_input],
    'flipper_length_mm': [flipper_length_input],
    'body_mass_g': [body_mass_input],
    'sex': [label_encoders['sex'].transform([gender_input])[0]]
})

# Предсказание
prediction = model.predict(input_data)[0]
prediction_proba = model.predict_proba(input_data)

# Вывод результата
st.subheader("🔍 Результаты предсказания")
st.write(f"**Предсказанный вид:** {prediction}")

df_prediction_proba = pd.DataFrame(prediction_proba, columns=model.classes_)
st.bar_chart(df_prediction_proba.T)
