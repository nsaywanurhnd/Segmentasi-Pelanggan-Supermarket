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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Visualisasi Data", "📈 K-Means", "🌲 Random Forest", "📋 Dashboard", "📊 Perbandingan Metode", "📝 Input Data Manual"])

# ---- Tab 5: Perbandingan Metode ----
with tab5:
    st.header("📊 Perbandingan Metode K-Means vs Random Forest")
    
    st.subheader("Silhouette Score untuk K-Means")
    if 'silhouette_scores' in locals():
        fig_silhouette = px.line(x=range(3, 12, 2), y=silhouette_scores, markers=True, title="Silhouette Score")
        fig_silhouette.update_layout(xaxis_title="Jumlah Cluster", yaxis_title="Silhouette Score")
        st.plotly_chart(fig_silhouette, use_container_width=True)
    else:
        st.warning("Silhouette Score belum dihitung. Silakan jalankan K-Means terlebih dahulu.")

    st.subheader("Akurasi Random Forest")
    if 'y_test' in locals() and 'y_pred' in locals():
        st.metric("Akurasi", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
    else:
        st.warning("Random Forest belum dijalankan. Silakan jalankan model terlebih dahulu.")

# ---- Tab 6: Input Data Manual ----
with tab6:
    st.header("📝 Input Data Manual untuk Prediksi Cluster")
    
    income_input = st.number_input("Masukkan Pendapatan (k$)", min_value=0, max_value=500, step=1, value=50)
    score_input = st.number_input("Masukkan Spending Score", min_value=0, max_value=100, step=1, value=50)
    
    input_data = np.array([[income_input, score_input]])
    input_scaled = scaler.transform(input_data)
    
    if st.button("Prediksi Cluster K-Means"):
        cluster_pred = kmeans.predict(input_scaled)
        st.success(f"Data termasuk ke dalam Cluster: {cluster_pred[0]}")
    
    if st.button("Prediksi dengan Random Forest"):
        rf_pred = rf.predict(input_scaled)
        st.success(f"Data diprediksi masuk ke Cluster: {rf_pred[0]}")
