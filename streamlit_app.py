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

# K-Means Clustering Section
elif selected_tab == "K-Means Clustering":
    st.header("📈 K-Means Clustering")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df
        X_scaled = st.session_state.X_scaled
        
        k_range = range(3, 12, 2)  # Menggunakan angka ganjil mulai dari 3
        inertia, silhouette = [], []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertia.append(kmeans.inertia_)
            silhouette.append(silhouette_score(X_scaled, labels))
        
        st.subheader("Elbow Method")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, inertia, marker='o', linestyle='-', color='blue')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig)
        
        st.subheader("Silhouette Score")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, silhouette, marker='o', linestyle='-', color='green')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score")
        st.pyplot(fig)
        
        n_clusters = st.sidebar.slider("Pilih Jumlah Cluster:", min_value=3, max_value=11, value=3, step=2)
        if st.sidebar.button("Run Clustering"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            st.session_state.df = df
            
            st.subheader("Visualisasi Hasil K-Means")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['income'], df['score'], c=df['Cluster'], cmap='viridis')
            ax.set_xlabel("Income")
            ax.set_ylabel("Spending Score")
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)
            
            st.subheader("Deskripsi Tiap Cluster")
            for i in range(n_clusters):
                cluster_data = df[df['Cluster'] == i]
                st.write(f"**Cluster {i}:**")
                st.write(f"- Rata-rata Income: ${cluster_data['income'].mean():,.2f}")
                st.write(f"- Rata-rata Spending Score: {cluster_data['score'].mean():.2f}")
                st.write(f"- Jumlah Anggota: {len(cluster_data)} orang")

# Perbandingan K-Means dan Random Forest
elif selected_tab == "Perbandingan Metode":
    st.header("📊 Perbandingan Metode K-Means vs Random Forest")
    if 'df' in st.session_state and 'Cluster' in st.session_state.df.columns:
        df = st.session_state.df
        X_train, X_test, y_train, y_test = train_test_split(df[['income', 'score']], df['Cluster'], test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        st.subheader("Evaluasi Metode")
        accuracy = accuracy_score(y_test, y_pred)
        silhouette_avg = silhouette_score(df[['income', 'score']], df['Cluster'])
        st.write(f"Silhouette Score (K-Means): {silhouette_avg:.2f}")
        st.write(f"Accuracy Score (Random Forest): {accuracy:.2f}")
        st.write("Kesimpulan: K-Means lebih baik untuk segmentasi, sedangkan Random Forest lebih baik untuk prediksi berdasarkan cluster.")
