# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Streamlit configuration
st.set_page_config(page_title="Bike Sharing Data Analysis", layout="wide")

# Sidebar details
st.sidebar.title("Bike Sharing Analysis")
st.sidebar.markdown("### Data Options")

# Load Data
@st.cache
def load_data():
    df_day = pd.read_csv('day.csv')
    df_hour = pd.read_csv('hour.csv')
    return df_day, df_hour

df_day, df_hour = load_data()

# Sidebar options
st.sidebar.subheader("Dataset Overview")
show_day_data = st.sidebar.checkbox("Show Day Dataset")
show_hour_data = st.sidebar.checkbox("Show Hour Dataset")

# Display datasets in the sidebar
if show_day_data:
    st.subheader("Day Dataset")
    st.write(df_day.head())

if show_hour_data:
    st.subheader("Hour Dataset")
    st.write(df_hour.head())

# Data Preprocessing
df_day['dteday'] = pd.to_datetime(df_day['dteday'])
df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])

season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
df_day['season'] = df_day['season'].map(season_map)
df_hour['season'] = df_hour['season'].map(season_map)

weather_map = {1: 'Clear', 2: 'Mist', 3: 'Light Snow', 4: 'Heavy Rain'}
df_day['weathersit'] = df_day['weathersit'].map(weather_map)
df_hour['weathersit'] = df_hour['weathersit'].map(weather_map)

# Analysis & Insights
st.title("Bike Sharing Data Analysis")
st.write("This dashboard provides insights into bike sharing patterns based on time, weather, and other factors.")

# Question 1: Pattern of Usage by Time and Day of Week
st.header("Pola Penggunaan Sepeda Berdasarkan Waktu dan Hari")
usage_pattern = df_hour.groupby(['weekday', 'hr'])['cnt'].mean().unstack()

# Display Heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(usage_pattern, cmap="YlGnBu", cbar=True, ax=ax)
ax.set_title("Pola Penggunaan Sepeda Berdasarkan Waktu dan Hari dalam Seminggu")
ax.set_xlabel("Jam")
ax.set_ylabel("Hari dalam Seminggu (0 = Minggu, 6 = Sabtu)")
st.pyplot(fig)

# Hourly Usage by Day of Week Line Chart
st.subheader("Penggunaan Sepeda per Jam pada Setiap Hari")
plt.figure(figsize=(12, 6))
for day in range(7):
    daily_usage = df_hour[df_hour['weekday'] == day].groupby('hr')['cnt'].mean()
    plt.plot(daily_usage, label=f'Hari {day}')
plt.title("Pola Penggunaan Sepeda Berdasarkan Jam untuk Setiap Hari")
plt.xlabel("Jam")
plt.ylabel("Rata-rata Penggunaan Sepeda")
plt.legend(['Minggu', 'Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu'], title="Hari")
st.pyplot(plt)

# Question 2: Impact of Weather on Usage
st.header("Pengaruh Cuaca Terhadap Peminjaman Sepeda")

# Boxplot of Usage by Weather
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='weathersit', y='cnt', data=df_day, ax=ax)
ax.set_title('Jumlah Peminjaman Sepeda Berdasarkan Situasi Cuaca')
ax.set_xlabel('Situasi Cuaca')
ax.set_ylabel('Jumlah Peminjaman Sepeda')
st.pyplot(fig)

# Scatter plot for Temp vs. Bike Rentals
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='temp', y='cnt', data=df_day, ax=ax)
ax.set_title('Hubungan antara Suhu dan Jumlah Peminjaman Sepeda')
ax.set_xlabel('Suhu')
ax.set_ylabel('Jumlah Peminjaman Sepeda')
st.pyplot(fig)

# Display Correlation
st.subheader("Korelasi Antara Variabel Cuaca dan Jumlah Peminjaman")
correlation = df_day[['temp', 'hum', 'windspeed', 'cnt']].corr()
st.write(correlation)

# Simple Regression Model
st.subheader("Regresi untuk Memprediksi Peminjaman Berdasarkan Cuaca")
X = df_day[['temp', 'hum', 'windspeed']]
y = df_day['cnt']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
st.write(model.summary())
