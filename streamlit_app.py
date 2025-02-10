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

# ---- 🎨 CSS KUSTOM ----
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #4CAF50;
        font-size: 30px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")

# Judul Aplikasi
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

# Simpan data ke session state
st.session_state.df = df
st.session_state.X_scaled = X_scaled

# ---- 📊 VISUALISASI DATA SEBELUM PENGOLAHAN ----
st.header("📊 Visualisasi Data Sebelum Pengolahan")

col1, col2 = st.columns([2, 1])

# Line Chart - Tren Income
with col1:
    fig = px.line(df, x=df.index, y='income', title="📈 Tren Income Pelanggan")
    st.plotly_chart(fig, use_container_width=True)

# Pie Chart - Distribusi Spending Score
with col2:
    fig = px.pie(df, names="score", title="🎯 Distribusi Spending Score", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# Metrik Utama
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pelanggan", len(df), help="Jumlah total pelanggan dalam dataset.")
col2.metric("Rata-rata Income", f"${df['income'].mean():,.2f}", help="Rata-rata income pelanggan.")
col3.metric("Rata-rata Spending Score", f"{df['score'].mean():.2f}", help="Rata-rata spending score pelanggan.")
col4.metric("Max Spending Score", f"{df['score'].max()}", help="Spending score tertinggi dalam dataset.")

# ---- 📈 K-MEANS CLUSTERING ----
st.header("📈 K-Means Clustering")

# Slider untuk memilih jumlah cluster
n_clusters = st.slider("Pilih Jumlah Cluster:", min_value=2, max_value=10, value=3)

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

    # Simpan hasil ke session state
    st.session_state.df = df
    st.session_state.kmeans_model = kmeans

# ---- 🌲 RANDOM FOREST CLASSIFICATION ----
st.header("🌲 Random Forest Classification")

if st.button("Jalankan Random Forest Classification"):
    if 'Cluster' not in df.columns:
        st.warning("Silakan jalankan K-Means Clustering terlebih dahulu.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Cluster'], test_size=0.3, random_state=42)

        # Membuat dan Melatih Model Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Prediksi
        y_pred = rf.predict(X_test)

        # Simpan hasil ke session state
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.accuracy = accuracy_score(y_test, y_pred)

        # Menampilkan Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Menampilkan Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

# ---- 📊 DASHBOARD ----
st.header("📊 Dashboard Segmentasi Pelanggan")

if 'Cluster' in df.columns:
    col1, col2 = st.columns((2, 1))

    # Grafik Line Chart untuk Tren
    with col1:
        st.subheader("Tren Kunjungan Pelanggan")
        fig, ax = plt.subplots(figsize=(10, 4))
        df.reset_index(inplace=True)

        if "income" in df.columns and "Cluster" in df.columns:
            sns.lineplot(data=df, x=df.index, y=df["income"], hue=df["Cluster"].astype(str), palette="tab10", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Kolom 'income' atau 'Cluster' tidak ditemukan di dataset.")

    # Pie Chart untuk Proporsi Klaster
    with col2:
        st.subheader("Distribusi Klaster Pelanggan")
        fig, ax = plt.subplots()
        cluster_counts = df["Cluster"].value_counts()
        ax.pie(cluster_counts, labels=cluster_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
        st.pyplot(fig)

    # METRIK PENTING
    col3, col4, col5, col6 = st.columns(4)

    col3.metric("Total Pelanggan", df.shape[0])
    col4.metric("Jumlah Klaster", df["Cluster"].nunique())
    col5.metric("Akurasi Random Forest", f"{st.session_state.accuracy*100:.2f}%")
    col6.metric("Fitur yang Dipakai", ", ".join(['income', 'score']))

    # TABEL HASIL SEGMENTASI
    st.subheader("📋 Hasil Segmentasi Pelanggan")
    st.dataframe(df.head(10))

    # DISTRIBUSI DATA
    st.subheader("📊 Distribusi Data Pelanggan")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df[['income', 'score']], ax=ax)
    st.pyplot(fig)

    # PERBANDINGAN METODE
    st.header("📊 Perbandingan Metode K-Means vs Random Forest")

    silhouette_avg = silhouette_score(X_scaled, df['Cluster'])

    st.write(f"Silhouette Score (K-Means): {silhouette_avg:.2f}")
    st.write(f"Accuracy Score (Random Forest): {st.session_state.accuracy:.2f}")
    st.write("Kesimpulan: **K-Means lebih baik untuk segmentasi, sedangkan Random Forest lebih baik untuk prediksi berdasarkan cluster.**")
else:
    st.warning("Silakan jalankan K-Means Clustering terlebih dahulu.")
