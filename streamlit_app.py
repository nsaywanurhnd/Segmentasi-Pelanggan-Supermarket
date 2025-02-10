import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, mean_absolute_error, mean_squared_error, accuracy_score

st.set_page_config(page_title="Segmentasi Pelanggan Toserba", page_icon="📊", layout="wide")
st.title("📊 Segmentasi Pelanggan Toserba")

menu = st.sidebar.radio("Navigasi", ["Upload Data", "Visualisasi Data", "K-Means Clustering", "Random Forest Classification", "Perbandingan Metode", "Input Manual Data"])

# Load data
df = None
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File berhasil diunggah!")
else:
    st.warning("Silakan upload file CSV terlebih dahulu.")

if df is not None:
    df.columns = df.columns.str.strip()
    df.rename(columns={'spending_score': 'score', 'Annual Income (k$)': 'income'}, inplace=True)
    X = df[['income', 'score']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.session_state.df = df
    st.session_state.X_scaled = X_scaled

    if menu == "Visualisasi Data":
        st.header("📊 Visualisasi Data")
        column = st.sidebar.selectbox("Pilih Kolom untuk Visualisasi", df.columns)
        
        st.subheader("📌 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='RdBu')
        st.pyplot(fig)

        st.subheader("📌 Histogram")
        numerical_features = ['income', 'score']
        fig, axes = plt.subplots(1, len(numerical_features), figsize=(15, 5))
        for i, feature in enumerate(numerical_features):
            ax = axes[i]
            df[feature].hist(bins=15, ax=ax, grid=False)
            ax.set_title(f'Distribution of {feature.capitalize()}')
        plt.tight_layout()
        st.pyplot(fig)

    elif menu == "K-Means Clustering":
        st.header("📈 K-Means Clustering")
        k_range = range(2, 11)
        inertia, silhouette = [], []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertia.append(kmeans.inertia_)
            silhouette.append(silhouette_score(X_scaled, labels))

        st.subheader("Elbow Method")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, inertia, marker='o')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        st.pyplot(fig)

        st.subheader("Silhouette Score")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, silhouette, marker='o')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        st.pyplot(fig)

        n_clusters = st.sidebar.slider("Pilih Jumlah Cluster:", 2, 10, 3)
        if st.sidebar.button("Run Clustering"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            st.write("Hasil Clustering:")
            st.write(df[['income', 'score', 'Cluster']].head())

else:
    st.info("Menu tidak aktif. Silakan upload file CSV terlebih dahulu.")
