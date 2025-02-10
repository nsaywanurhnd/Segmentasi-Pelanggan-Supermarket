import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")
st.title("📊 Segmentasi Pelanggan Toserba")
st.markdown("**Tujuan Website:** Menganalisis dan mengelompokkan pelanggan berdasarkan pola pembelian mereka menggunakan K-Means dan Random Forest.")

# Buttons for navigation
st.sidebar.markdown("## 📌 Navigasi")
if st.sidebar.button("Upload Data"):
    selected_tab = "Upload Data"
elif st.sidebar.button("Visualisasi Data"):
    selected_tab = "Visualisasi Data"
elif st.sidebar.button("K-Means Clustering"):
    selected_tab = "K-Means Clustering"
elif st.sidebar.button("Random Forest Classification"):
    selected_tab = "Random Forest Classification"
elif st.sidebar.button("Input Manual Data"):
    selected_tab = "Input Manual Data"
else:
    selected_tab = "Upload Data"

# Upload Data Section
if selected_tab == "Upload Data":
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
    
    st.write("Data yang digunakan:")
    st.dataframe(df, use_container_width=True)
    
    # Data cleaning and preparation
    df.columns = df.columns.str.strip()
    df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)
    
    st.session_state.df = df
    X = df[['income', 'score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.session_state.X_scaled = X_scaled

# Visualisasi Data Section
elif selected_tab == "Visualisasi Data":
    st.header("📊 Visualisasi Data")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df
        df_numeric = df.select_dtypes(include=[np.number])
        st.subheader("📌 Correlation Heatmap")
        if df_numeric.empty:
            st.warning("Tidak ada kolom numerik dalam dataset.")
        else:
            corr_matrix = df_numeric.corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
