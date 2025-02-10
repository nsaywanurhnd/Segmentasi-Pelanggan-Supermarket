import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score

# ---- 🎨 SET PAGE CONFIG ----
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")

# ---- 🎨 CSS CUSTOM ----
custom_css = """
<style>
    body {
        background-color: #f4f4f4;
    }
    .stApp {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .css-1d391kg, .css-1v3fvcr {
        background-color: white !important;
        border-radius: 10px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("📊 Segmentasi Pelanggan Toserba")
st.markdown("Menganalisis dan mengelompokkan pelanggan berdasarkan pola pembelian menggunakan **K-Means** dan **Random Forest**.")

# ---- 📂 UPLOAD DATA ----
st.header("📂 Upload Data")
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data berhasil diunggah!")
else:
    try:
        df = pd.read_csv("data_customer.csv")
        st.info("Menggunakan data bawaan: data_customer.csv")
    except FileNotFoundError:
        st.error("File data_customer.csv tidak ditemukan. Silakan upload file terlebih dahulu.")
        st.stop()

st.write("**Data yang digunakan:**")
st.dataframe(df, use_container_width=True)

# Data Cleaning & Preparation
df.columns = df.columns.str.strip()
df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)

# Scaling Data
X = df[['income', 'score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.session_state.df = df
st.session_state.X_scaled = X_scaled

# ---- 📈 K-MEANS CLUSTERING ----
st.header("📈 K-Means Clustering")

# Pilihan jumlah klaster hanya angka ganjil
ganjil_values = [i for i in range(3, 11, 2)]
n_clusters = st.selectbox("Pilih Jumlah Cluster (Ganjil Saja):", ganjil_values)

if st.button("Jalankan K-Means Clustering"):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Visualisasi Hasil K-Means
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['income'], df['score'], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel("Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("K-Means Clustering")
    st.pyplot(fig)

    st.session_state.df = df
    st.session_state.kmeans_model = kmeans
