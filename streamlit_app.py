import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score, mean_absolute_error, mean_squared_error

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")
st.title("📊 Segmentasi Pelanggan Toserba")
st.markdown("**Tujuan Website:** Menganalisis dan mengelompokkan pelanggan berdasarkan pola pembelian mereka menggunakan K-Means dan Random Forest.")

# Menyimpan state navigasi jika belum ada
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Upload Data"

# Buttons for navigation
st.markdown("## 📌 Navigasi")
col1, col2, col3, col4, col5 = st.columns(5)
if col1.button("Upload Data"):
    st.session_state.selected_tab = "Upload Data"
elif col2.button("Visualisasi Data"):
    st.session_state.selected_tab = "Visualisasi Data"
elif col3.button("K-Means Clustering"):
    st.session_state.selected_tab = "K-Means Clustering"
elif col4.button("Random Forest Classification"):
    st.session_state.selected_tab = "Random Forest Classification"
elif col5.button("Input Manual Data"):
    st.session_state.selected_tab = "Input Manual Data"

selected_tab = st.session_state.selected_tab

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
        
        st.subheader("📌 Histogram")
        for col in df_numeric.columns:
            fig = px.histogram(df, x=col, nbins=30, title=f'Distribusi {col}')
            st.plotly_chart(fig, use_container_width=True)
        
        categorial_features = [col for col in ['gender', 'preferred_category'] if col in df.columns]
        if categorial_features:
            st.subheader("📌 Kategori Data")
            fig, axes = plt.subplots(1, len(categorial_features), figsize=(15,6))
            for i, feature in enumerate(categorial_features):
                sns.countplot(x=feature, data=df, ax=axes[i], palette='terrain')
                axes[i].set_title(f'Distribution of {feature.capitalize()}')
                axes[i].set_xlabel(feature.capitalize())
            plt.tight_layout()
            st.pyplot(fig)
        
        if 'income' in df.columns:
            st.subheader("📌 Density Plot for Income")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(df['income'], color="yellow", shade=True)
            plt.title("Density Plot for Annual Income")
            st.pyplot(fig)
