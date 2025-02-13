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
import io

# Set page config
st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")

# Fungsi untuk membaca data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)
    return df

# Sidebar untuk upload data
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

# Data sampel
sample_data = pd.DataFrame({
    'income': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'score': [15, 25, 35, 45, 55, 65, 75, 85, 95, 5]
})

# Memuat data
if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success("Data berhasil diunggah!")
else:
    try:
        df = load_data("data_customer.csv")
        st.sidebar.info("Menggunakan data bawaan: data_customer.csv")
    except FileNotFoundError:
        df = sample_data
        st.sidebar.warning("Menampilkan data sampel. Silakan upload file untuk menggantinya.")

# Validasi kolom
required_columns = {'income', 'score'}
if not required_columns.issubset(df.columns):
    st.error(f"Dataset harus memiliki kolom: {', '.join(required_columns)}")
    st.stop()

X = df[['income', 'score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tabs sebagai navbar
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visualisasi Data", "📈 K-Means", "🌲 Random Forest", "📋 Dashboard", "📊 Perbandingan Metode"
])

# ---- Tab 1: Visualisasi Data ----
with tab1:
    st.header("📊 Visualisasi Data")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.histogram(df, x='income', title="📈 Distribusi Income Pelanggan")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(df, names="score", title="🎯 Distribusi Spending Score", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: K-Means Clustering ----
with tab2:
    st.header("📈 K-Means Clustering")
    num_clusters = st.slider("Pilih jumlah cluster:", 2, 10, value=3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
    st.metric("Silhouette Score", f"{silhouette_avg:.2f}")
    
    fig = px.scatter(df, x='income', y='score', color=df['Cluster'].astype(str), title="K-Means Clustering")
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: Random Forest ----
with tab3:
    st.header("🌲 Random Forest Classification")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Cluster'], test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Akurasi Random Forest", f"{accuracy * 100:.2f}%")
    
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    feature_importance = pd.DataFrame({'Feature': ['income', 'score'], 'Importance': rf.feature_importances_})
    fig = px.bar(feature_importance, x='Feature', y='Importance', title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 4: Dashboard ----
with tab4:
    st.header("📋 Dashboard Segmentasi Pelanggan")
    selected_cluster = st.selectbox("Pilih Cluster:", df['Cluster'].unique())
    filtered_df = df[df['Cluster'] == selected_cluster]
    
    st.dataframe(filtered_df)
    
    buffer = io.BytesIO()
    filtered_df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button("📥 Unduh Data", data=buffer, file_name="segmentasi_pelanggan.csv", mime="text/csv")

# ---- Tab 5: Perbandingan Metode ----
with tab5:
    st.header("📊 Perbandingan Metode K-Means vs Random Forest")
    st.metric("Silhouette Score K-Means", f"{silhouette_avg:.2f}")
    st.metric("Akurasi Random Forest", f"{accuracy * 100:.2f}%")
