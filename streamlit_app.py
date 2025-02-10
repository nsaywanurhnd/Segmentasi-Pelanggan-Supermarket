import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")
st.title("📊 Segmentasi Pelanggan Toserba")

# Tabs for navigation
tabs = ["Upload Data", "Visualisasi Data", "K-Means Clustering", "Random Forest Classification", "Input Manual Data"]
selected_tab = st.selectbox("Pilih Halaman:", tabs)

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
        
        # Filter hanya kolom numerik
        df_numeric = df.select_dtypes(include=[np.number])
        
        st.subheader("📌 Correlation Heatmap")
        if df_numeric.empty:
            st.warning("Tidak ada kolom numerik dalam dataset.")
        else:
            corr_matrix = df_numeric.corr()
            fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu', title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📌 Histogram")
        numerical_features = df_numeric.columns.tolist()
        fig, axes = plt.subplots(len(numerical_features), 1, figsize=(10, 5 * len(numerical_features)))
        
        if len(numerical_features) == 1:
            axes = [axes]
        
        for ax, feature in zip(axes, numerical_features):
            df[feature].hist(bins=15, ax=ax, grid=False)
            ax.set_title(f'Distribution of {feature.capitalize()}')
            ax.set_xlabel(feature.capitalize())
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)

# K-Means Clustering Section
elif selected_tab == "K-Means Clustering":
    st.header("📈 K-Means Clustering")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        X_scaled = st.session_state.X_scaled
        
        st.subheader("Elbow Method")
        k_range = range(2, 11)
        inertia = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=inertia, mode='lines+markers', marker=dict(color='blue')))
        fig.update_layout(title="Elbow Method", xaxis_title="Number of Clusters (K)", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)
        
        n_clusters = st.slider("Pilih Jumlah Cluster:", min_value=2, max_value=10, value=3, step=1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df = st.session_state.df.copy()
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        st.session_state.df = df
        
        st.subheader("Visualisasi Hasil K-Means")
        fig = px.scatter(df, x='income', y='score', color='Cluster', title="K-Means Clustering", color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

# Input Manual Data Section
elif selected_tab == "Input Manual Data":
    st.header("✍️ Input Data Manual")
    if 'df' in st.session_state and 'Cluster' in st.session_state.df.columns:
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("💰 Masukkan Income (dalam ribuan $)", min_value=0, max_value=200, value=50, step=1)
        with col2:
            score = st.number_input("📈 Masukkan Spending Score (0-100)", min_value=0, max_value=100, value=50, step=1)
        
        if st.button("Cek Klaster"):
            df = st.session_state.df
            X = df[['income', 'score']]
            kmeans = KMeans(n_clusters=df['Cluster'].nunique(), random_state=42, n_init=10)
            kmeans.fit(X)
            cluster = kmeans.predict([[income, score]])[0]
            st.success(f"Data yang Anda masukkan termasuk dalam klaster: **{cluster}**")
