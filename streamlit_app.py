import streamlit as st  # 🔹 Import Streamlit HARUS setelah set_page_config
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
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score

# 🔥 Harus di baris paling awal!
st.set_page_config(page_title="Segmentasi Pelanggan", page_icon="📊", layout="wide")

# Simpan navigasi di session state
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Upload Data"

# Sidebar navigasi
st.sidebar.title("📌 Navigasi")
menu = ["Upload Data", "Visualisasi Data", "K-Means Clustering", "Random Forest Classification", "Perbandingan Metode"]
selected_tab = st.sidebar.radio("Pilih Halaman", menu)
st.session_state.selected_tab = selected_tab

# 📂 **Upload Data**
if selected_tab == "Upload Data":
    st.title("📂 Upload Data")
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

    # **Preprocessing Data**
    df.columns = df.columns.str.strip()
    df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)

    st.session_state.df = df
    X = df[['income', 'score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.session_state.X_scaled = X_scaled

# 📊 **Visualisasi Data**
elif selected_tab == "Visualisasi Data":
    st.title("📊 Visualisasi Data")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df

        col1, col2 = st.columns([2, 1])

        # **Line Chart - Tren Income**
        with col1:
            fig = px.line(df, x=df.index, y='income', title="📈 Tren Income Pelanggan")
            st.plotly_chart(fig, use_container_width=True)

        # **Pie Chart - Distribusi Spending Score**
        with col2:
            fig = px.pie(df, names="score", title="🎯 Distribusi Spending Score", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        # **Metrik Utama**
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pelanggan", len(df))
        col2.metric("Rata-rata Income", f"${df['income'].mean():,.2f}")
        col3.metric("Rata-rata Spending Score", f"{df['score'].mean():.2f}")
        col4.metric("Max Spending Score", f"{df['score'].max()}")

# 🎯 **K-Means Clustering**
elif selected_tab == "K-Means Clustering":
    st.title("📈 K-Means Clustering")
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df
        X_scaled = st.session_state.X_scaled

        # **Elbow Method & Silhouette Score**
        k_range = range(3, 12, 2)
        inertia, silhouette = [], []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertia.append(kmeans.inertia_)
            silhouette.append(silhouette_score(X_scaled, labels))

        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(x=k_range, y=inertia, markers=True, title="🔍 Elbow Method")
            fig.update_xaxes(title="Jumlah Cluster (K)")
            fig.update_yaxes(title="Inertia")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(x=k_range, y=silhouette, markers=True, title="✨ Silhouette Score")
            fig.update_xaxes(title="Jumlah Cluster (K)")
            fig.update_yaxes(title="Silhouette Score")
            st.plotly_chart(fig, use_container_width=True)

        # **Pilih jumlah cluster**
        n_clusters = st.slider("Pilih Jumlah Cluster:", min_value=3, max_value=11, value=3, step=2)

        if st.button("Run Clustering"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            st.session_state.df = df

            # **Visualisasi Clustering**
            fig = px.scatter(df, x='income', y='score', color=df['Cluster'].astype(str),
                             title="🎨 Hasil K-Means Clustering", labels={"Cluster": "Cluster"})
            st.plotly_chart(fig, use_container_width=True)

# 🌲 **Random Forest Classification**
elif selected_tab == "Random Forest Classification":
    st.title("🌲 Random Forest Classification")
    if 'df' not in st.session_state or 'Cluster' not in st.session_state.df.columns:
        st.warning("Silakan jalankan K-Means Clustering terlebih dahulu.")
    else:
        df = st.session_state.df
        X_train, X_test, y_train, y_test = train_test_split(df[['income', 'score']], df['Cluster'], test_size=0.3, random_state=42)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        fig = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

# 📊 **Perbandingan K-Means vs Random Forest**
elif selected_tab == "Perbandingan Metode":
    st.title("📊 Perbandingan K-Means vs Random Forest")
    df = st.session_state.df
    X_train, X_test, y_train, y_test = train_test_split(df[['income', 'score']], df['Cluster'], test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    silhouette_avg = silhouette_score(df[['income', 'score']], df['Cluster'])
    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Silhouette Score (K-Means)", f"{silhouette_avg:.2f}")
    st.metric("Accuracy Score (Random Forest)", f"{accuracy:.2f}")
