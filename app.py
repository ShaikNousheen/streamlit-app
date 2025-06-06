import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Smart Grid Forecasting', layout='wide')
st.title('🔌 Smart Grid Forecasting App')

@st.cache_data
def load_data():
    df = pd.read_csv('smart_grid_dataset.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df

df = load_data()

target = 'Power Consumption (kW)'
X = df.drop(columns=[target])
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)
preds = model.predict(X_scaled)
df['Predicted Load'] = preds

st.line_chart(df[[target, 'Predicted Load']])

st.subheader('SHAP Feature Importance')
explainer = shap.Explainer(model)
shap_values = explainer(X_scaled[:100])
shap.summary_plot(shap_values, features=X.iloc[:100], feature_names=X.columns, show=False)
st.pyplot(bbox_inches='tight')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
shap.summary_plot(shap_values, features=X.iloc[:100], feature_names=X.columns, show=False)
st.pyplot(fig)

