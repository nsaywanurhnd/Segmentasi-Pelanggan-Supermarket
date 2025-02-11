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

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")

# Fungsi untuk membaca data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)
    return df

# Data sampel
sample_data = pd.DataFrame({
    'income': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'score': [15, 25, 35, 45, 55, 65, 75, 85, 95, 5]
})

# Memuat data
try:
    df = load_data("data_customer.csv")
    st.info("Menggunakan data bawaan: data_customer.csv")
except FileNotFoundError:
    df = sample_data
    st.warning("Menampilkan data sampel. Silakan upload file untuk menggantinya.")

# Validasi kolom
required_columns = {'income', 'score'}
if not required_columns.issubset(df.columns):
    st.error(f"Dataset harus memiliki kolom: {', '.join(required_columns)}")
    st.stop()

X = df[['income', 'score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tabs sebagai navbar
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualisasi Data", "📈 K-Means", "🌲 Random Forest", "📋 Dashboard", "📊 Perbandingan Metode"])

# ---- Tab 1: Visualisasi Data ----
with tab1:
    st.header("📊 Visualisasi Data")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.line(df, x=df.index, y='income', title="📈 Tren Income Pelanggan", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x="score", title="🎯 Distribusi Spending Score", nbins=10, color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig, use_container_width=True)
    
    # Menambahkan metrik utama
    st.subheader("📊 Statistik Utama")
    col3, col4, col5, col6 = st.columns(4)
    col3.metric("Total Users", len(df))
    col4.metric("Rata-rata Income", f"${df['income'].mean():.2f}K")
    col5.metric("Rata-rata Score", f"{df['score'].mean():.2f}")
    col6.metric("Max Score", df['score'].max())

# ---- Tab 2: K-Means Clustering ----
with tab2:
    st.header("📈 K-Means Clustering")
    st.markdown("K-Means adalah algoritma unsupervised learning yang mengelompokkan data berdasarkan kesamaan fitur.")
    
    inertia = []
    silhouette_scores = []
    for k in range(3, 12, 2):  # Hanya angka ganjil
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    fig = px.line(x=range(3, 12, 2), y=inertia, markers=True, title="Elbow Method untuk Menentukan K")
    fig.update_layout(xaxis_title="Jumlah Cluster", yaxis_title="Inertia")
    st.plotly_chart(fig, use_container_width=True)
    
    fig_silhouette = px.line(x=range(3, 12, 2), y=silhouette_scores, markers=True, title="Silhouette Score")
    fig_silhouette.update_layout(xaxis_title="Jumlah Cluster", yaxis_title="Silhouette Score")
    st.plotly_chart(fig_silhouette, use_container_width=True)
    
    num_clusters = st.slider("Pilih jumlah cluster (hanya ganjil):", 3, 11, step=2, value=3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    fig = px.scatter(df, x='income', y='score', color=df['Cluster'].astype(str), title="K-Means Clustering", labels={'color': 'Cluster'})
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: Random Forest ----
with tab3:
    st.header("🌲 Random Forest Classification")
    st.markdown("Random Forest adalah metode supervised learning yang menggunakan banyak pohon keputusan untuk meningkatkan akurasi.")
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Cluster'], test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 5: Perbandingan Metode ----
with tab5:
    st.header("📊 Perbandingan Akurasi Metode")
    kmeans_silhouette = silhouette_score(X_scaled, kmeans.labels_)
    rf_accuracy = accuracy_score(y_test, y_pred)
    
    st.metric("Silhouette Score K-Means", f"{kmeans_silhouette:.2f}")
    st.metric("Akurasi Random Forest", f"{rf_accuracy * 100:.2f}%")
    
    st.markdown("Dari hasil di atas, **K-Means** digunakan untuk clustering, sedangkan **Random Forest** memiliki akurasi lebih tinggi untuk klasifikasi data hasil clustering.")
